"""LangGraph Checkpointer 工厂 + 自定义 RedisSaver（同步实现）。

后端由 ``settings.checkpoint_backend`` 选择：

- ``memory``：``langgraph.checkpoint.memory.MemorySaver``（仅调试/测试）；
- ``sqlite``：``langgraph.checkpoint.sqlite.SqliteSaver``（需额外安装
  ``langgraph-checkpoint-sqlite``）；
- ``redis``：``RedisSaver``（本模块实现，只依赖已安装的 ``redis`` 包，
  不需要 ``langgraph-checkpoint-redis``）。

RedisSaver 的存储模型（全部 key 共享 ``settings.redis_key_prefix`` 前缀，
TTL=``settings.checkpoint_ttl_seconds``）::

    {pfx}:ckpt:latest:{thread_id}:{ns}                 -> checkpoint_id
    {pfx}:ckpt:{thread_id}:{ns}:{checkpoint_id}        -> {checkpoint, metadata}
    {pfx}:ckpt:parent:{thread_id}:{ns}:{checkpoint_id} -> parent_checkpoint_id
    {pfx}:ckpt:writes:{thread_id}:{ns}:{checkpoint_id}:{task_id} -> writes

与 ``SqliteSaver`` 一致的语义：latest 指针 + 完整 checkpoint blob + 父指针 +
按 task 保存的 writes。checkpoint 整体序列化（含 ``channel_values``），不做
按 channel 的 blob 拆分。

简化点 / 与官方 ``langgraph-checkpoint-redis`` 的差异：

1. **不区分 pending writes 生命周期**：writes 只按 ``(task_id, idx)`` 去重后
   保存，不维护「已应用 / 待应用」两阶段状态；基础 interrupt/resume 可以工作
   （依赖 pending_writes 回放），但没有官方包对 pending sends / 跨 namespace
   分片写入的专门处理；
2. **不实现 list / get_state_history**：历史列表返回空；
3. **key 结构与官方包不兼容**：官方 ``langgraph-checkpoint-redis`` 使用
   另一套 key/JSON 结构，两者不能互读；
4. 仅同步实现（顶层图同步编译；异步方法只是把同步调用丢到线程池）。
"""
from __future__ import annotations

import asyncio
import base64
import json
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    PendingWrite,
    get_checkpoint_metadata,
)

from app.core.config import settings
from app.core.logger import get_logger
from app.store.redis import get_redis

logger = get_logger("store.checkpointer")


# ---------------------------------------------------------------------------
# 序列化：优先 LangGraph 自带 JsonPlusSerializer（可处理 LangChain 消息、
# pydantic、numpy 等类型）；极端情况下退化为 json.dumps(default=str)。
# ---------------------------------------------------------------------------

try:
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    _SERDE: Any = JsonPlusSerializer()
except Exception:  # pragma: no cover - 正常安装 langgraph-checkpoint 时不会走到

    class _JsonSerde:
        """最小可用 fallback：只保证不崩，复杂对象会被转成字符串。"""

        def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
            return "json", json.dumps(
                obj, ensure_ascii=False, default=str
            ).encode("utf-8")

        def loads_typed(self, data: tuple[str, bytes]) -> Any:
            return json.loads(data[1].decode("utf-8"))

    _SERDE = _JsonSerde()


def _encode(value: Any) -> dict[str, str]:
    """任意对象 → 可 JSON 化的 {t: 类型标签, d: base64 字节}。"""
    type_tag, payload = _SERDE.dumps_typed(value)
    return {
        "t": str(type_tag),
        "d": base64.b64encode(payload).decode("ascii"),
    }


def _decode(data: Any) -> Any:
    if not isinstance(data, dict) or "t" not in data or "d" not in data:
        raise ValueError("无效的序列化字段（缺少 t/d）")
    return _SERDE.loads_typed(
        (str(data["t"]), base64.b64decode(str(data["d"])))
    )


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


# ---------------------------------------------------------------------------
# RedisSaver
# ---------------------------------------------------------------------------

class RedisSaver(BaseCheckpointSaver[str]):
    """把 LangGraph checkpoint 存到 Redis 的简化实现（同步）。

    参数：
        client: 可注入 redis.Redis（测试用）；默认使用 ``app.store.redis.get_redis()``。
        serde:  可注入序列化器；默认 ``JsonPlusSerializer()``。
        ttl:    所有 checkpoint key 的过期秒数；默认
                ``settings.checkpoint_ttl_seconds``，<=0 表示不过期。
    """

    def __init__(
        self,
        client: Any | None = None,
        *,
        serde: Any | None = None,
        ttl: int | None = None,
    ) -> None:
        super().__init__(serde=serde or _SERDE)
        self._client = client if client is not None else get_redis()
        self._prefix = settings.redis_key_prefix
        self._ttl = int(
            ttl if ttl is not None else (settings.checkpoint_ttl_seconds or 0)
        )

    # ---------------------------------------------------------------- key

    def _latest_key(self, thread_id: str, checkpoint_ns: str) -> str:
        return f"{self._prefix}:ckpt:latest:{thread_id}:{checkpoint_ns}"

    def _blob_key(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> str:
        return f"{self._prefix}:ckpt:{thread_id}:{checkpoint_ns}:{checkpoint_id}"

    def _parent_key(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> str:
        return (
            f"{self._prefix}:ckpt:parent:{thread_id}:{checkpoint_ns}:{checkpoint_id}"
        )

    def _writes_key(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        task_id: str,
    ) -> str:
        return (
            f"{self._prefix}:ckpt:writes:{thread_id}:{checkpoint_ns}:"
            f"{checkpoint_id}:{task_id}"
        )

    def _writes_pattern(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> str:
        return (
            f"{self._prefix}:ckpt:writes:{thread_id}:{checkpoint_ns}:"
            f"{checkpoint_id}:*"
        )

    def _set(self, key: str, value: str) -> None:
        if self._ttl > 0:
            self._client.set(key, value, ex=self._ttl)
        else:
            self._client.set(key, value)

    @staticmethod
    def _config_parts(
        config: RunnableConfig,
    ) -> tuple[str, str, str | None]:
        configurable = config.get("configurable") or {}
        thread_id = configurable.get("thread_id")
        if not thread_id:
            raise ValueError("checkpoint config 缺少 configurable.thread_id")
        checkpoint_ns = str(configurable.get("checkpoint_ns", "") or "")
        parent_id = configurable.get("checkpoint_id")
        return str(thread_id), checkpoint_ns, (
            str(parent_id) if parent_id else None
        )

    # ---------------------------------------------------------------- 读

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """按 config 读取 checkpoint；无 checkpoint_id 时读 latest 指针。"""
        thread_id, checkpoint_ns, requested_id = self._config_parts(config)

        if requested_id:
            checkpoint_id = requested_id
            blob_raw = self._client.get(
                self._blob_key(thread_id, checkpoint_ns, checkpoint_id)
            )
            if blob_raw is None:
                return None
        else:
            latest_raw = self._client.get(
                self._latest_key(thread_id, checkpoint_ns)
            )
            if latest_raw is None:
                return None
            checkpoint_id = _text(latest_raw)
            blob_raw = self._client.get(
                self._blob_key(thread_id, checkpoint_ns, checkpoint_id)
            )
            if blob_raw is None:
                logger.warning(
                    "latest 指针指向不存在的 checkpoint: thread=%s ns=%s id=%s",
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                )
                return None

        try:
            blob = json.loads(_text(blob_raw))
            if isinstance(blob, dict) and "checkpoint" in blob:
                checkpoint: Checkpoint = _decode(blob["checkpoint"])
                metadata: CheckpointMetadata = (
                    _decode(blob["metadata"])
                    if blob.get("metadata") is not None
                    else {}
                )
            else:  # 兼容直接存 checkpoint 的旧格式
                checkpoint = _decode(blob)
                metadata = {}
        except Exception as exc:
            raise RuntimeError(f"Redis checkpoint 解析失败: {exc}") from exc

        parent_raw = self._client.get(
            self._parent_key(thread_id, checkpoint_ns, checkpoint_id)
        )
        parent_config: RunnableConfig | None = None
        if parent_raw is not None:
            parent_id = _text(parent_raw)
            if parent_id:
                parent_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_id,
                    }
                }

        pending_writes = self._load_writes(
            thread_id, checkpoint_ns, checkpoint_id
        )

        if requested_id:
            result_config = config
        else:
            result_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            }

        return CheckpointTuple(
            config=result_config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    def _load_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> list[PendingWrite]:
        pattern = self._writes_pattern(thread_id, checkpoint_ns, checkpoint_id)
        keys = sorted(_text(key) for key in self._client.scan_iter(match=pattern))
        writes: list[PendingWrite] = []
        for key in keys:
            raw = self._client.get(key)
            if raw is None:
                continue
            try:
                payload = json.loads(_text(raw))
            except (TypeError, ValueError) as exc:
                logger.warning("checkpoint writes 数据损坏 key=%s: %s", key, exc)
                continue
            if not isinstance(payload, dict):
                continue
            task_id = str(payload.get("task_id") or "")
            for entry in payload.get("entries") or []:
                if not isinstance(entry, dict):
                    continue
                try:
                    value = _decode(entry.get("value"))
                except Exception as exc:
                    logger.warning(
                        "checkpoint write 反序列化失败 key=%s: %s", key, exc
                    )
                    continue
                writes.append((task_id, str(entry.get("channel") or ""), value))
        return writes

    # ---------------------------------------------------------------- 写

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """保存 checkpoint，并更新 latest 指针与 parent 指针。

        注意：checkpoint_id 优先使用 ``checkpoint["id"]``（LangGraph 的
        ``_loop`` 随后会用内存中的同一个 id 调 ``put_writes``，两者必须一致），
        仅在缺失时生成 uuid4。
        """
        thread_id, checkpoint_ns, parent_id = self._config_parts(config)
        checkpoint_id = str(checkpoint.get("id") or uuid4())
        if checkpoint.get("id") != checkpoint_id:
            checkpoint = {**checkpoint, "id": checkpoint_id}

        try:
            blob = {
                "checkpoint": _encode(checkpoint),
                "metadata": _encode(get_checkpoint_metadata(config, metadata)),
            }
            self._set(
                self._blob_key(thread_id, checkpoint_ns, checkpoint_id),
                json.dumps(blob, ensure_ascii=False),
            )
            if parent_id:
                self._set(
                    self._parent_key(thread_id, checkpoint_ns, checkpoint_id),
                    parent_id,
                )
            self._set(
                self._latest_key(thread_id, checkpoint_ns), checkpoint_id
            )
        except Exception as exc:
            raise RuntimeError(f"Redis checkpoint 写入失败: {exc}") from exc

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """保存某 task 的写入列表；同一 ``(task_id, idx)`` 只保留一条。"""
        thread_id, checkpoint_ns, _ = self._config_parts(config)
        configurable = config.get("configurable") or {}
        checkpoint_id = configurable.get("checkpoint_id")
        if not checkpoint_id:
            raise ValueError("put_writes 缺少 configurable.checkpoint_id")
        checkpoint_id = str(checkpoint_id)
        if not writes:
            return

        key = self._writes_key(
            thread_id, checkpoint_ns, checkpoint_id, str(task_id)
        )
        try:
            raw = self._client.get(key)
            payload: dict[str, Any] = {}
            if raw is not None:
                try:
                    loaded = json.loads(_text(raw))
                    if isinstance(loaded, dict):
                        payload = loaded
                except (TypeError, ValueError) as exc:
                    logger.warning("checkpoint writes 数据损坏，覆盖重写: %s", exc)

            entries: dict[int, dict[str, Any]] = {}
            for entry in payload.get("entries") or []:
                if not isinstance(entry, dict):
                    continue
                try:
                    entries[int(entry.get("idx"))] = entry
                except (TypeError, ValueError):
                    continue

            for position, (channel, value) in enumerate(writes):
                idx = WRITES_IDX_MAP.get(channel, position)
                # 正索引代表普通 channel 的写入，已存在则跳过（缓存命中重复写）；
                # 负索引是 ERROR/SCHEDULED/INTERRUPT/RESUME 等特殊通道，允许覆盖。
                if idx >= 0 and idx in entries:
                    continue
                entries[idx] = {
                    "idx": idx,
                    "channel": str(channel),
                    "value": _encode(value),
                }

            payload = {
                "task_id": str(task_id),
                "entries": list(entries.values()),
            }
            self._set(key, json.dumps(payload, ensure_ascii=False))
        except Exception as exc:
            raise RuntimeError(f"Redis checkpoint writes 写入失败: {exc}") from exc

    # ---------------------------------------------------------------- 维护

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """简化实现：不维护历史列表，返回空迭代器。"""
        return iter(())

    def delete_thread(self, thread_id: str) -> None:
        """删除某 thread 的全部 checkpoint / writes（按 key 模式 SCAN）。"""
        thread_id = str(thread_id)
        patterns = (
            f"{self._prefix}:ckpt:latest:{thread_id}:*",
            f"{self._prefix}:ckpt:{thread_id}:*",
            f"{self._prefix}:ckpt:parent:{thread_id}:*",
            f"{self._prefix}:ckpt:writes:{thread_id}:*",
        )
        keys: set[str] = set()
        for pattern in patterns:
            for key in self._client.scan_iter(match=pattern):
                keys.add(_text(key))
        if keys:
            self._client.delete(*sorted(keys))

    # ---------------------------------------------------------------- 异步

    async def aget_tuple(
        self, config: RunnableConfig
    ) -> CheckpointTuple | None:
        return await asyncio.to_thread(self.get_tuple, config)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await asyncio.to_thread(
            self.put, config, checkpoint, metadata, new_versions
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await asyncio.to_thread(
            self.put_writes, config, writes, task_id, task_path
        )

    async def adelete_thread(self, thread_id: str) -> None:
        await asyncio.to_thread(self.delete_thread, thread_id)


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------

_checkpointer: Any | None = None
_checkpointer_lock = threading.Lock()


def _build_checkpointer(backend: str) -> Any:
    backend = str(backend or "").strip().lower()

    if backend == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        logger.info("Checkpointer 后端: memory（仅进程内，不持久化）")
        return MemorySaver()

    if backend == "sqlite":
        import sqlite3

        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise RuntimeError(
                "SqliteSaver 不可用：请先安装 langgraph-checkpoint-sqlite"
                "（G:\\anaconda\\envs\\rag_env\\python.exe -m pip install "
                "langgraph-checkpoint-sqlite）"
            ) from exc

        db_path = Path(settings.checkpoint_db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        saver = SqliteSaver(conn)
        # SqliteSaver 构造器不会自动建表，必须显式 setup（不同小版本均有该方法）
        setup = getattr(saver, "setup", None)
        if callable(setup):
            setup()
        logger.info("Checkpointer 后端: sqlite path=%s", db_path)
        return saver

    if backend == "redis":
        logger.info("Checkpointer 后端: redis（自定义 RedisSaver）")
        return RedisSaver()

    raise ValueError(
        f"未知的 checkpoint_backend: {backend!r}（可选 memory / sqlite / redis）"
    )


def get_checkpointer() -> Any:
    """按 ``settings.checkpoint_backend`` 返回进程内单例 checkpointer。"""
    global _checkpointer
    if _checkpointer is None:
        with _checkpointer_lock:
            if _checkpointer is None:
                _checkpointer = _build_checkpointer(settings.checkpoint_backend)
    return _checkpointer
