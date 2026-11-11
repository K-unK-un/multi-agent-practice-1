"""Chroma 向量库访问层（Docker HTTP 模式）。

- 只存子块文本 + 元数据，不承担「元数据主库」职责（MySQL 才是）；
- 客户端、collection 均为进程内单例懒加载；
- 连接失败统一抛 ``RuntimeError("Chroma 连接失败: ...")``；
- metadata 只允许 str/int/float/bool：写入前做归一化，空 list 转 ""，
  单元素 list 取唯一值，多元素 list 直接报错（Chroma where 无法对多值做集合
  匹配，必须由入库侧拆成多条记录）。
"""
from __future__ import annotations

import threading
from datetime import date, datetime
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.errors import NotFoundError

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger("store.chroma")


def _sanitize_value(key: str, value: Any) -> str | int | float | bool:
    """把任意 metadata 值收敛到 Chroma 支持的基础类型。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int, float)):
        return value
    if value is None:
        # filters.py 约定：缺失 key 在 Chroma where 里永远不匹配，因此显式写空串
        return ""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))
        if not items:
            return ""
        if len(items) == 1:
            return _sanitize_value(key, items[0])
        raise ValueError(
            f"Chroma metadata 字段 {key!r} 不支持多值 {items!r}；"
            "请在入库侧拆成多条记录（where 只能做单值比较/$in）"
        )
    return str(value)


def _sanitize_metadata(metadata: dict | None) -> dict | None:
    if not metadata:
        return None
    return {str(key): _sanitize_value(str(key), value) for key, value in metadata.items()}


class ChromaStore:
    """Chroma HTTP 客户端封装。"""

    def __init__(self, client: Any | None = None) -> None:
        self._client = client
        self._collections: dict[str, Collection] = {}
        self._lock = threading.RLock()

    # ---------------------------------------------------------------- 连接

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is None:
                try:
                    client = chromadb.HttpClient(
                        host=settings.chroma_host,
                        port=settings.chroma_port,
                        tenant=settings.chroma_tenant,
                        database=settings.chroma_database,
                    )
                    # 主动探活：HttpClient 构造是延迟连接，不 heartbeat 的话
                    # 「服务没起来」会伪装成「集合为空」。
                    client.heartbeat()
                    self._client = client
                except Exception as exc:
                    raise RuntimeError(f"Chroma 连接失败: {exc}") from exc
        return self._client

    def get_or_create_collection(self, name: str | None = None) -> Collection:
        """获取/创建 collection；默认使用 ``settings.chroma_collection``。"""
        collection_name = name or settings.chroma_collection
        with self._lock:
            cached = self._collections.get(collection_name)
            if cached is not None:
                return cached
            try:
                collection = self._ensure_client().get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"},
                    # 向量由 Ollama 在应用侧生成，Chroma 不需要默认 embedding function
                    embedding_function=None,
                )
            except RuntimeError:
                raise
            except Exception as exc:
                raise RuntimeError(f"Chroma 获取 collection 失败: {exc}") from exc
            self._collections[collection_name] = collection
            logger.info("Chroma collection 就绪: %s", collection_name)
            return collection

    # ---------------------------------------------------------------- 写入

    def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        """批量写入子块。四组数据长度必须一致。"""
        if not ids:
            return
        if not (len(ids) == len(documents) == len(embeddings) == len(metadatas)):
            raise ValueError(
                "Chroma add 参数长度不一致: "
                f"ids={len(ids)}, documents={len(documents)}, "
                f"embeddings={len(embeddings)}, metadatas={len(metadatas)}"
            )
        clean_metadatas = [_sanitize_metadata(item) for item in metadatas]
        collection = self.get_or_create_collection()
        collection.add(
            ids=[str(item) for item in ids],
            documents=[str(item) for item in documents],
            embeddings=[list(item) for item in embeddings],
            metadatas=clean_metadatas,
        )

    # ---------------------------------------------------------------- 查询

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict | None = None,
    ) -> list[dict]:
        """向量检索，返回平铺的命中列表。

        每条：``{id, doc_id, article_no, page, title, distance, document, ...}``，
        其中 metadata 的其他键（status/version/scope 等）也会展开保留。
        """
        if not query_embeddings or n_results <= 0:
            return []
        collection = self.get_or_create_collection()
        kwargs: dict[str, Any] = {
            "query_embeddings": [list(item) for item in query_embeddings],
            "n_results": int(n_results),
            "include": ["metadatas", "documents", "distances"],
        }
        if where:
            kwargs["where"] = where

        result = collection.query(**kwargs)
        id_batches = result.get("ids") or []
        meta_batches = result.get("metadatas") or []
        doc_batches = result.get("documents") or []
        dist_batches = result.get("distances") or []

        rows: list[dict] = []
        for batch_index, id_batch in enumerate(id_batches):
            metas = meta_batches[batch_index] if batch_index < len(meta_batches) else []
            docs = doc_batches[batch_index] if batch_index < len(doc_batches) else []
            dists = dist_batches[batch_index] if batch_index < len(dist_batches) else []
            for item_index, item_id in enumerate(id_batch or []):
                meta = metas[item_index] if item_index < len(metas) else None
                row = dict(meta or {})
                row["id"] = item_id
                row["doc_id"] = row.get("doc_id")
                row["article_no"] = row.get("article_no")
                row["page"] = row.get("page")
                row["title"] = row.get("title")
                row["distance"] = (
                    dists[item_index] if item_index < len(dists) else None
                )
                row["document"] = (
                    docs[item_index] if item_index < len(docs) else None
                )
                rows.append(row)
        return rows

    def count(self) -> int:
        return int(self.get_or_create_collection().count())

    def delete_collection(self) -> None:
        """删除默认 collection（幂等：不存在时静默）。"""
        collection_name = settings.chroma_collection
        try:
            self._ensure_client().delete_collection(name=collection_name)
        except NotFoundError:
            pass
        finally:
            with self._lock:
                self._collections.pop(collection_name, None)


_chroma_store: ChromaStore | None = None
_chroma_lock = threading.Lock()


def get_chroma() -> ChromaStore:
    """进程内 ChromaStore 单例；首次调用时探活，失败抛 RuntimeError。"""
    global _chroma_store
    if _chroma_store is None:
        with _chroma_lock:
            if _chroma_store is None:
                store = ChromaStore()
                store._ensure_client()  # 提前暴露连接错误；失败时不缓存坏单例
                _chroma_store = store
    return _chroma_store
