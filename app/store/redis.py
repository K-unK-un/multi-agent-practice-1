"""Redis 缓存层（同步 redis-py）。

定位：**加速层，不是正确性依赖**。Redis 不可用时所有缓存操作降级为
miss / 静默失败，业务继续走 MySQL/Chroma 等真实数据源。

键前缀统一由 ``settings.redis_key_prefix`` 自动添加；调用方只传业务键，
例如 ``cache.get_json("route:abc")`` 实际访问 ``policymind:route:abc``。
Checkpointer 的键由 ``app.store.checkpointer`` 自行拼装 ``:ckpt:`` 命名空间，
同样共享该前缀，互不冲突。
"""
from __future__ import annotations

import json
import threading

import redis

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger("store.redis")

_redis_client: redis.Redis | None = None
_redis_lock = threading.Lock()


def get_redis() -> redis.Redis:
    """进程内 Redis 客户端单例（懒加载 + 双检锁）。

    ``redis.Redis.from_url`` 只解析 URL 并构造客户端，不会立即建立连接；
    真正不可用要到第一次 get/set 才会暴露，因此这里不做 ping（避免启动期
    强依赖 Redis）。
    """
    global _redis_client
    if _redis_client is None:
        with _redis_lock:
            if _redis_client is None:
                _redis_client = redis.Redis.from_url(
                    settings.redis_url, decode_responses=True
                )
    return _redis_client


class RedisCache:
    """JSON 缓存封装：值统一 json 序列化，key 自动加全局前缀。"""

    def __init__(self, client: redis.Redis | None = None) -> None:
        self._client = client if client is not None else get_redis()
        self._prefix = settings.redis_key_prefix
        self._default_ttl = int(settings.checkpoint_ttl_seconds or 0)

    # ---------------------------------------------------------------- 内部

    def _key(self, key: str) -> str:
        return f"{self._prefix}:{key}"

    # ---------------------------------------------------------------- 读写

    def get_json(self, key: str) -> dict | None:
        """读 JSON 缓存；未命中、Redis 不可用或值损坏均返回 None。"""
        full_key = self._key(key)
        try:
            raw = self._client.get(full_key)
        except Exception as exc:
            logger.warning("Redis 读取失败（降级为 miss）key=%s: %s", full_key, exc)
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError) as exc:
            logger.warning("Redis 缓存值不是合法 JSON key=%s: %s", full_key, exc)
            return None

    def set_json(self, key: str, value: dict, ttl: int | None = None) -> None:
        """写 JSON 缓存；Redis 不可用或序列化失败时只记 warning，不向上抛。"""
        full_key = self._key(key)
        expire = self._default_ttl if ttl is None else int(ttl)
        try:
            payload = json.dumps(value, ensure_ascii=False, default=str)
            if expire > 0:
                self._client.set(full_key, payload, ex=expire)
            else:
                self._client.set(full_key, payload)
        except Exception as exc:
            logger.warning("Redis 写入失败（已忽略）key=%s: %s", full_key, exc)

    def delete(self, key: str) -> None:
        """删除缓存键；失败只记 warning。"""
        full_key = self._key(key)
        try:
            self._client.delete(full_key)
        except Exception as exc:
            logger.warning("Redis 删除失败（已忽略）key=%s: %s", full_key, exc)


_cache: RedisCache | None = None
_cache_lock = threading.Lock()


def get_cache() -> RedisCache:
    """进程内 RedisCache 单例。"""
    global _cache
    if _cache is None:
        with _cache_lock:
            if _cache is None:
                _cache = RedisCache()
    return _cache
