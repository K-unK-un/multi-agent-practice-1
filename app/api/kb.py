"""知识库接口：文档上传（异步建库）、任务状态、重建索引、文档列表。

权限模型：
- 文档列表（只读）：管理员 + 4 级用户（require_kb_viewer）；
- 上传 / 重建索引 / 任务状态：仅管理员（require_admin）。

上传后的文档由后台任务自动完成解析 → 清洗 → 切分 → 入库 → 双路索引，
前端通过 /tasks/{task_id} 轮询进度。所有建库任务串行执行，避免并发写
Chroma / BM25 造成索引错乱。
"""
from __future__ import annotations

import json
import re
import threading
import uuid
from pathlib import Path
from typing import Any

import aiofiles
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status

from app.core.config import settings
from app.core.logger import get_logger
from app.core.security import require_admin, require_kb_viewer
from app.ingest.pipeline import (
    SUPPORTED_SUFFIXES,
    create_task,
    get_task,
    rebuild_all,
    run_ingest,
    update_task,
)
from app.store.mysql import get_mysql

logger = get_logger("api.kb")
router = APIRouter(prefix="/kb", tags=["kb"])

_CHUNK_SIZE = 1024 * 1024
# 建库任务串行锁：Chroma 写入与 BM25 全量重建都不是并发安全的。
_INGEST_LOCK = threading.Lock()
# 上传落盘名形如 {uuid12}_{原文件名}，展示时去掉随机前缀。
_UPLOAD_PREFIX_RE = re.compile(r"^[0-9a-fA-F]{12}_")


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


def _source_name(source_path: Any) -> str:
    """从 source_path 提取展示用文件名（去掉上传随机前缀）。"""
    if not source_path:
        return ""
    name = Path(str(source_path)).name
    return _UPLOAD_PREFIX_RE.sub("", name)


def _ingest_worker(task_id: str, path: str, operator: dict[str, Any]) -> None:
    """后台建库任务：pending → running → done/failed，状态写回任务注册表。

    等待串行锁期间保持 pending，前端显示为“排队中”。
    """
    try:
        with _INGEST_LOCK:
            update_task(task_id, status="running")
            result = run_ingest(
                [path],
                task_id=task_id,
                task_kind="upload",
                operator=operator,
            )
        update_task(task_id, status="done", result=_json_safe(result))
        logger.info("文档建库完成 task_id=%s path=%s", task_id, path)
    except Exception as exc:
        logger.exception("文档建库失败 task_id=%s path=%s", task_id, path)
        try:
            update_task(task_id, status="failed", error=str(exc))
        except Exception:
            logger.exception("更新任务失败状态出错 task_id=%s", task_id)


def _reindex_worker(task_id: str, operator: dict[str, Any]) -> None:
    """后台全量重建索引任务：data/docs 全量扫描 + data/uploads 上传文件补扫。"""
    try:
        with _INGEST_LOCK:
            update_task(task_id, status="running")
            result = rebuild_all(
                task_id=task_id,
                task_kind="reindex",
                operator=operator,
            )
            upload_dir = Path(settings.upload_dir)
            upload_files = (
                sorted(
                    f for f in upload_dir.glob("*")
                    if f.suffix.lower() in SUPPORTED_SUFFIXES
                )
                if upload_dir.exists()
                else []
            )
            if upload_files:
                extra = run_ingest(
                    [str(f) for f in upload_files],
                    task_id=task_id,
                    task_kind="reindex",
                    operator=operator,
                )
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
    admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, str]:
    """上传制度文档并提交后台建库任务，返回 task_id。仅管理员可用。"""
    safe_name = _safe_filename(file.filename)
    if Path(safe_name).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="仅支持 PDF / DOCX / MD / TXT 格式",
        )
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

    if size == 0:
        try:
            dest.unlink(missing_ok=True)
        except OSError:
            pass
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="文件内容为空",
        )

    task_id = create_task(
        "upload",
        {
            "path": str(dest),
            "filename": safe_name,
            "operator": admin.get("username"),
        },
    )
    background_tasks.add_task(_ingest_worker, task_id, str(dest), admin)
    logger.info(
        "上传完成，已提交建库任务 task_id=%s file=%s size=%d admin=%s",
        task_id, stored_name, size, admin.get("username"),
    )
    return {"task_id": task_id}


@router.get("/tasks/{task_id}")
def task_status(
    task_id: str,
    admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """查询建库/重建任务状态。仅管理员可用（任务均由管理员发起）。"""
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
def reindex(
    background_tasks: BackgroundTasks,
    admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, str]:
    """提交全量重建索引任务（走同一任务注册表）。仅管理员可用。"""
    task_id = create_task("reindex", {"operator": admin.get("username")})
    background_tasks.add_task(_reindex_worker, task_id, admin)
    logger.info("已提交重建索引任务 task_id=%s admin=%s", task_id, admin.get("username"))
    return {"task_id": task_id}


@router.get("/docs")
def list_documents(
    include_archived: bool = False,
    viewer: dict[str, Any] = Depends(require_kb_viewer),
) -> list[dict[str, Any]]:
    """返回制度文档列表（含条款数）。管理员与 4 级用户可查看。

    ``include_archived=true`` 时同时返回已归档的历史版本，供入库管理界面展示。
    """
    try:
        rows = get_mysql().list_documents_with_counts(active_only=not include_archived) or []
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
                "doc_id": _field(row, "id"),
                "title": _field(row, "title") or "",
                "version": _field(row, "version"),
                "effective_date": str(effective_date) if effective_date is not None else None,
                "status": _field(row, "status") or "active",
                "scope": list(_field(row, "scope") or []),
                "clause_count": int(_field(row, "clause_count") or 0),
                # 新入库会写 source_name；历史数据回退到从 source_path 推导
                "source_name": _field(row, "source_name")
                or _source_name(_field(row, "source_path")),
                "content_fingerprint": _field(row, "content_fingerprint"),
                "created_at": _field(row, "created_at"),
                "updated_at": _field(row, "updated_at"),
            }
        )
    return documents


@router.get("/ingest-records")
def list_ingest_records(
    limit: int = 100,
    doc_id: str | None = None,
    viewer: dict[str, Any] = Depends(require_kb_viewer),
) -> list[dict[str, Any]]:
    """入库台账：每份文档的 doc_id、来源、操作人、时间与动作。管理员与 4 级用户可查看。"""
    try:
        size = max(1, min(int(limit or 100), 500))
    except (TypeError, ValueError):
        size = 100
    try:
        rows = get_mysql().list_ingest_records(limit=size, doc_id=doc_id) or []
    except Exception:
        logger.exception("查询入库记录失败")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="查询入库记录失败",
        )
    return [_json_safe(row) for row in rows]
