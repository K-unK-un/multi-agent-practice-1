"""get_clause 工具：按 doc_id + 条款号从 MySQL 取回父块全文，并写入共享黑板。

检索子图返回的是条款片段；生成/核查阶段需要原文时用本工具取全文（父子块回填）。
"""
from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from app.core.blackboard import record_evidence
from app.core.logger import get_logger

logger = get_logger("tools.clause")

_NOT_FOUND = "知识库中不存在该条款或该版本已失效"


def _row_to_dict(row: Any) -> dict:
    """兼容 dict / SQLAlchemy Row / ORM 对象。"""
    if isinstance(row, dict):
        return row
    mapping = getattr(row, "_mapping", None)
    if mapping is not None:
        return dict(mapping)
    if hasattr(row, "__dict__"):
        return {k: v for k, v in vars(row).items() if not k.startswith("_")}
    return {}


@tool
def get_clause_tool(
    doc_id: str, article_no: str, include_archived: bool = False
) -> str:
    """按文档 ID 与条款号取回条款全文（父块），用于引用原文或核对细节。

    Args:
        doc_id: 制度文档 ID。
        article_no: 条款号，如 "12" 或 "12.3"。
        include_archived: 是否允许取已归档旧版本（跨版本对比时置 True）。
    """
    try:
        from app.store.mysql import get_mysql

        row = get_mysql().get_clause(
            doc_id, article_no, include_archived=include_archived
        )
    except Exception as exc:
        logger.warning("get_clause 查询失败：doc_id=%s article=%s err=%s", doc_id, article_no, exc)
        return _NOT_FOUND

    if not row:
        return _NOT_FOUND

    data = _row_to_dict(row)
    text = str(data.get("content") or data.get("text") or "")
    if not text:
        return _NOT_FOUND

    article = str(data.get("article_no") or article_no)
    record_evidence(
        [
            {
                "doc_id": str(data.get("doc_id") or doc_id),
                "article_no": article,
                "title": data.get("title") or "",
                "page": data.get("page"),
                "text": text,
                "full_text": text,
                "channel": "get_clause",
                "query": f"条款{doc_id}/{article_no}",
            }
        ]
    )
    return text
