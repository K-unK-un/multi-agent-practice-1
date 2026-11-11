"""list_versions 工具：按制度名称关键词列出所有版本（含归档旧版）。

跨版本对比场景由 compare_agent 使用；只做元数据查询，不写证据池。
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

from langchain_core.tools import tool

from app.core.logger import get_logger

logger = get_logger("tools.version")


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


def _fmt_date(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value).strip()


@tool
def list_versions_tool(title_keyword: str) -> str:
    """按制度名称关键词列出该制度的所有版本及生效日期、状态。

    Args:
        title_keyword: 制度名称关键词，如 "差旅" 或 "员工手册"。
    """
    keyword = (title_keyword or "").strip()
    try:
        from app.store.mysql import get_mysql

        rows = get_mysql().list_versions(keyword)
    except Exception as exc:
        logger.warning("list_versions 查询失败：keyword=%s err=%s", keyword, exc)
        return "查询制度版本失败，请稍后重试"

    versions = [row for row in (rows or []) if row]
    if not versions:
        return "未找到相关制度"

    lines: list[str] = []
    for index, row in enumerate(versions, 1):
        data = _row_to_dict(row)
        title = data.get("title") or "未命名制度"
        version = data.get("version") or "未标注"
        effective = _fmt_date(data.get("effective_date")) or "未标注"
        status = data.get("status") or "未知"
        doc_id = data.get("id") or data.get("doc_id") or "-"
        lines.append(
            f"{index}. 《{title}》｜版本：{version}｜生效日期：{effective}"
            f"｜状态：{status}｜id：{doc_id}"
        )
    return "\n".join(lines)
