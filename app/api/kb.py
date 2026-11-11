"""知识库接口：文档上传（异步建库）、任务状态、重建索引、文档列表。

除健康检查外的 KB 管理操作统一依赖 require_auth（开发模式自动匿名放行）。
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import aiofiles
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status

from app.core.config import settings
from app.core.logger import get_logger
from app.core.security import require_auth
from app.ingest.pipeline import create_task, get_task, rebuild_all, run_ingest, update_task
from app.store.mysql import get_mysql

logger = get_logger("api.kb")
router = APIRouter(prefix="/kb", tags=["kb"], dependencies=[Depends(require_auth)])

_CHUNK_SIZE = 1024 * 1024


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """兼容存储层返回 dict 或 ORM 对象。"""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _json_safe(value: Any) -> Any:
    """任务结果转成可 JSON 序列化的结构。"""
    if value is None:
        return None
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return str(value)


def _safe_filename(filename: str | None) -> str:
    """去掉路径与非法字符，防止目录穿越 / Windows 非法文件名。"""
    raw = Path(filename or "upload.bin").name
    cleaned = "".join(ch for ch in raw if ch not in '\\/:*?"<>|').strip().strip(".")
    if not cleaned:
        cleaned = "upload.bin"
    if len(cleaned) > 120:
        suffix = Path(cleaned).suffix[:16]
        cleaned = Path(cleaned).stem[: max(1, 120 - len(suffix))] + suffix
    return cleaned


def _ingest_worker(task_id: str, path: str) -> None:
    """后台建库任务：running → done/failed，状态写回任务注册表。"""
    try:
        update_task(task_id, status="running")
        result = run_ingest([path])
        update_task(task_id, status="done", result=_json_safe(result))
        logger.info("文档建库完成 task_id=%s path=%s", task_id, path)
    except Exception as exc:
        logger.exception("文档建库失败 task_id=%s path=%s", task_id, path)
        try:
            update_task(task_id, status="failed", error=str(exc))
        except Exception:
            logger.exception("更新任务失败状态出错 task_id=%s", task_id)


def _reindex_worker(task_id: str) -> None:
    """后台全量重建索引任务：data/docs 全量扫描 + data/uploads 上传文件补扫。"""
    try:
        update_task(task_id, status="running")
        result = rebuild_all()
        upload_dir = Path(settings.upload_dir)
        upload_files = (
            sorted(
                f for f in upload_dir.glob("*")
                if f.suffix.lower() in {".pdf", ".docx", ".md", ".txt"}
            )
            if upload_dir.exists()
            else []
        )
        if upload_files:
            extra = run_ingest([str(f) for f in upload_files])
            for key in ("documents", "clauses", "rules"):
                result[key] = result.get(key, 0) + extra.get(key, 0)
            result.setdefault("warnings", []).extend(extra.get("warnings", []))
            if "chroma_count" in extra:
                result["chroma_count"] = extra["chroma_count"]
        update_task(task_id, status="done", result=_json_safe(result))
        logger.info("全量重建索引完成 task_id=%s", task_id)
    except Exception as exc:
        logger.exception("全量重建索引失败 task_id=%s", task_id)
        try:
            update_task(task_id, status="failed", error=str(exc))
        except Exception:
            logger.exception("更新任务失败状态出错 task_id=%s", task_id)


@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> dict[str, str]:
    """上传制度文档并提交后台建库任务，返回 task_id。"""
    safe_name = _safe_filename(file.filename)
    upload_dir = Path(settings.upload_dir)
    try:
        upload_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.exception("创建上传目录失败 dir=%s", upload_dir)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="上传目录不可用")

    stored_name = f"{uuid.uuid4().hex[:12]}_{safe_name}"
    dest = upload_dir / stored_name
    max_bytes = max(0, int(settings.max_upload_mb)) * 1024 * 1024
    size = 0
    too_large = False

    try:
        async with aiofiles.open(dest, "wb") as handle:
            while True:
                chunk = await file.read(_CHUNK_SIZE)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    too_large = True
                    break
                await handle.write(chunk)
    except Exception:
        # 读取/写盘中断时清理半成品，避免污染上传目录。
        try:
            dest.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    finally:
        await file.close()

    if too_large:
        try:
            dest.unlink(missing_ok=True)
        except OSError:
            pass
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"文件超过 {settings.max_upload_mb} MB 上限",
        )

    task_id = create_task("upload", {"path": str(dest), "filename": safe_name})
    background_tasks.add_task(_ingest_worker, task_id, str(dest))
    logger.info("上传完成，已提交建库任务 task_id=%s file=%s size=%d", task_id, stored_name, size)
    return {"task_id": task_id}


@router.get("/tasks/{task_id}")
def task_status(task_id: str) -> dict[str, Any]:
    """查询建库/重建任务状态。"""
    try:
        task = get_task(task_id)
    except Exception:
        logger.exception("读取任务状态失败 task_id=%s", task_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="读取任务状态失败",
        )
    if task is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="任务不存在")

    if isinstance(task, dict):
        payload = dict(task)
    else:
        payload = {"result": _json_safe(task)}
    payload.setdefault("task_id", task_id)
    return payload


@router.post("/reindex")
def reindex(background_tasks: BackgroundTasks) -> dict[str, str]:
    """提交全量重建索引任务（走同一任务注册表）。"""
    task_id = create_task("reindex", {})
    background_tasks.add_task(_reindex_worker, task_id)
    logger.info("已提交重建索引任务 task_id=%s", task_id)
    return {"task_id": task_id}


@router.get("/docs")
def list_documents() -> list[dict[str, Any]]:
    """返回制度文档列表（不含条款数，保持查询轻量）。"""
    try:
        rows = get_mysql().list_documents() or []
    except Exception:
        logger.exception("查询制度文档列表失败")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="查询制度文档列表失败",
        )

    documents: list[dict[str, Any]] = []
    for row in rows:
        effective_date = _field(row, "effective_date")
        documents.append(
            {
                "id": _field(row, "id"),
                "title": _field(row, "title") or "",
                "version": _field(row, "version"),
                "effective_date": str(effective_date) if effective_date is not None else None,
                "status": _field(row, "status"),
            }
        )
    return documents
