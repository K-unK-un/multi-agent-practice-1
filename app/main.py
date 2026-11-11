"""FastAPI 应用入口：路由装配、CORS、静态前端与五项健康检查。"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import auth as auth_api
from app.api import chat as chat_api
from app.api import feedback as feedback_api
from app.api import kb as kb_api
from app.core.config import settings
from app.core.logger import get_logger
from app.store.chroma import get_chroma
from app.store.mysql import get_mysql
from app.store.redis import get_redis

logger = get_logger("main")

# 探活总时限：外部依赖不可达时健康检查也必须快速返回。
_HEALTH_TIMEOUT_SECONDS = 3.0

# app/main.py -> app/ -> 项目根目录
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_WEB_DIR = _PROJECT_ROOT / "web"


def _check_mysql() -> bool:
    """MySQL 探活：优先用存储层现成的 ping/执行入口，最后兜底独立短连接。

    存储层的具体执行方法名未在契约中固定，这里逐个尝试常见入口；
    任意一种成功即视为健康，避免因方法签名差异误报。
    """
    store = get_mysql()

    for name in ("ping", "check_health", "health_check"):
        method = getattr(store, name, None)
        if callable(method):
            try:
                return method() is not False
            except Exception:
                logger.debug("MySQL 探活方法 %s 失败", name, exc_info=True)

    for name in ("execute", "execute_sql", "fetch_one", "fetch_all", "scalar"):
        method = getattr(store, name, None)
        if callable(method):
            try:
                method("SELECT 1")
                return True
            except Exception:
                logger.debug("MySQL 探活方法 %s 失败", name, exc_info=True)

    engine = getattr(store, "engine", None) or getattr(store, "_engine", None)
    if engine is not None:
        try:
            from sqlalchemy import text

            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except Exception:
            logger.debug("存储层 engine 探活失败，尝试独立连接", exc_info=True)

    session_factory = getattr(store, "SessionLocal", None) or getattr(store, "session_factory", None)
    if callable(session_factory):
        try:
            from sqlalchemy import text

            session = session_factory()
            try:
                session.execute(text("SELECT 1"))
            finally:
                session.close()
            return True
        except Exception:
            logger.debug("存储层 session 探活失败，尝试独立连接", exc_info=True)

    # 兜底：不依赖存储层实现细节，直接建短连接探活（3s 连接超时）。
    from sqlalchemy import create_engine, text

    engine = create_engine(
        settings.database_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 3},
    )
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    finally:
        engine.dispose()
    return True


def _check_redis() -> bool:
    return bool(get_redis().ping())


def _check_chroma() -> bool:
    """Chroma 探活：优先调用底层 HttpClient 的 heartbeat()。"""
    store = get_chroma()
    candidates: list[Any] = []
    for attr in ("client", "_client", "http_client", "chroma_client"):
        candidate = getattr(store, attr, None)
        if candidate is not None:
            candidates.append(candidate)
    candidates.append(store)

    for candidate in candidates:
        heartbeat = getattr(candidate, "heartbeat", None)
        if callable(heartbeat):
            try:
                heartbeat()
                return True
            except Exception:
                logger.debug("Chroma heartbeat 失败，尝试下一个入口", exc_info=True)
    return False


def _check_ollama() -> bool:
    """Ollama 探活：/api/tags 中必须能找到配置的 embedding 模型。"""
    url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"
    response = httpx.get(url, timeout=3.0)
    response.raise_for_status()
    payload = response.json()
    target = (settings.embedding_model or "").strip()
    if not target:
        return True
    target_base = target.split(":")[0]
    for item in payload.get("models") or []:
        name = str(item.get("name") or item.get("model") or "")
        if not name:
            continue
        if name == target or name.split(":")[0] == target_base:
            return True
    return False


def _run_health_checks() -> dict[str, Any]:
    """并行执行四项外部探活，整体最多等 3 秒；BM25 只检查本地索引文件。"""
    checks: dict[str, Any] = {}
    probes = {
        "mysql": _check_mysql,
        "redis": _check_redis,
        "chroma": _check_chroma,
        "ollama": _check_ollama,
    }

    executor = ThreadPoolExecutor(max_workers=len(probes), thread_name_prefix="health")
    futures = {name: executor.submit(probe) for name, probe in probes.items()}
    deadline = time.monotonic() + _HEALTH_TIMEOUT_SECONDS
    for name, future in futures.items():
        remaining = max(0.0, deadline - time.monotonic())
        try:
            checks[name] = bool(future.result(timeout=remaining))
        except FuturesTimeoutError:
            logger.warning("健康检查超时：%s", name)
            checks[name] = False
        except Exception:
            logger.warning("健康检查失败：%s", name, exc_info=True)
            checks[name] = False
    # 不等待仍在阻塞的探活线程，保证 /health 快速返回。
    executor.shutdown(wait=False, cancel_futures=True)

    index_path = Path(settings.bm25_index_path)
    checks["bm25"] = True if index_path.exists() else "missing"
    return checks


def create_app() -> FastAPI:
    app = FastAPI(
        title="PolicyMind",
        description="企业规章制度智能问答（LangGraph 多 Agent + RAG）",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins or ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth_api.router, prefix="/api")
    app.include_router(chat_api.router, prefix="/api")
    app.include_router(kb_api.router, prefix="/api")
    app.include_router(feedback_api.router, prefix="/api")

    @app.get("/")
    def root() -> dict[str, str]:
        return {"service": "PolicyMind", "docs": "/docs", "web": "/web"}

    @app.get("/health")
    def health() -> dict[str, Any]:
        checks = _run_health_checks()
        failed = [
            name
            for name in ("mysql", "redis", "chroma", "ollama")
            if not checks.get(name)
        ]
        if checks.get("bm25") == "missing":
            logger.info("BM25 索引尚未建库：%s", settings.bm25_index_path)
        return {
            "status": "degraded" if failed else "ok",
            "checks": checks,
        }

    if _WEB_DIR.is_dir():
        app.mount("/web", StaticFiles(directory=str(_WEB_DIR), html=True), name="web")
    else:
        logger.warning("前端目录不存在，未挂载 /web：%s", _WEB_DIR)

    return app


app = create_app()
