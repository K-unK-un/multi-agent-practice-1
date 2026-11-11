"""反馈接口：用户评分 / 纠错写入，以及供实验分析的全量读取。"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.core.logger import get_logger
from app.store.mysql import get_mysql

logger = get_logger("api.feedback")
router = APIRouter(prefix="/feedback", tags=["feedback"])

_FEEDBACK_FIELDS = (
    "id",
    "session_id",
    "message_id",
    "rating",
    "comment",
    "correction",
    "created_at",
)


class FeedbackIn(BaseModel):
    session_id: str | None = Field(default=None, max_length=36)
    message_id: int | None = None
    rating: int = Field(ge=1, le=5)
    comment: str | None = None
    correction: str | None = None


def _row_to_dict(row: Any) -> dict[str, Any]:
    """兼容 dict / SQLAlchemy Row / ORM 对象。"""
    if isinstance(row, dict):
        data = dict(row)
    else:
        mapping = getattr(row, "_mapping", None)
        if mapping is not None:
            data = dict(mapping)
        else:
            data = {name: getattr(row, name, None) for name in _FEEDBACK_FIELDS}

    created_at = data.get("created_at")
    if created_at is not None and hasattr(created_at, "isoformat"):
        data["created_at"] = created_at.isoformat()
    return data


def _load_feedback_rows() -> list[Any]:
    """读取反馈：优先存储层公开方法，其次原生 SQL，最后复用其会话作用域。"""
    store = get_mysql()

    list_fn = getattr(store, "list_feedback", None)
    if callable(list_fn):
        return list(list_fn() or [])

    fetch_all = getattr(store, "fetch_all", None)
    if callable(fetch_all):
        rows = fetch_all(
            "SELECT id, session_id, message_id, rating, comment, correction, created_at "
            "FROM feedback ORDER BY id DESC"
        )
        return list(rows or [])

    # 当前 MySQLStore 只提供写入接口：复用其事务作用域 + Feedback ORM 只读查询。
    session_scope = getattr(store, "_session_scope", None)
    if callable(session_scope):
        from sqlalchemy import select

        from app.store.schemas import Feedback

        with session_scope() as session:
            entities = (
                session.execute(select(Feedback).order_by(Feedback.id.desc()))
                .scalars()
                .all()
            )
            return [_row_to_dict(entity) for entity in entities]

    raise NotImplementedError("存储层暂未提供反馈查询入口")


@router.post("")
def submit_feedback(body: FeedbackIn) -> dict[str, bool]:
    """写入一条反馈（评分 1~5，可选评论与纠正答案）。"""
    try:
        get_mysql().insert_feedback(
            {
                "session_id": body.session_id,
                "message_id": body.message_id,
                "rating": body.rating,
                "comment": body.comment,
                "correction": body.correction,
            }
        )
    except Exception:
        logger.exception("反馈写入失败 session_id=%s", body.session_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="反馈写入失败",
        )
    return {"ok": True}


@router.get("")
def list_feedback() -> list[dict[str, Any]]:
    """全部反馈列表（实验分析用）。"""
    try:
        rows = _load_feedback_rows()
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="存储层暂未提供反馈列表接口",
        )
    except Exception:
        logger.exception("读取反馈列表失败")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="读取反馈列表失败",
        )

    return [_row_to_dict(row) for row in (rows or [])]
