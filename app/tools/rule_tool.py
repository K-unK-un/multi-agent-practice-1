"""rule_table 工具：查询结构化规则表（阶梯/比例/上限/门槛），供 eligibility 专家使用。

数据来源：MySQL rule_table 表（建库时由 ingest/rule_extract.py 抽取）。
纪律：规则表只提供结构化候选，最终结论仍须核对依据条款原文；
needs_review=True 的行运行时标 ⚠，提示调用方降级为"引用原文由人工判断"。
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from app.core.blackboard import emit
from app.core.logger import get_logger
from app.store.mysql import get_mysql

logger = get_logger("tools.rule")

# 单次返回的最大行数与单元格长度，避免工具输出撑爆专家上下文
_MAX_ROWS = 50
_MAX_CELL = 120

_FIELDS = (
    "doc_id",
    "article_no",
    "title",
    "condition",
    "conclusion",
    "value",
    "unit",
    "basis_text",
    "needs_review",
)


def _to_dict(row: Any) -> dict:
    """兼容 dict / SQLAlchemy 行 / 对象三种返回形态。"""
    if isinstance(row, dict):
        return row
    data: dict[str, Any] = {}
    for field in _FIELDS:
        if hasattr(row, field):
            try:
                data[field] = getattr(row, field)
            except Exception:  # pragma: no cover - 属性访问异常极少见
                continue
    return data


def _as_text(value: Any) -> str:
    """把 JSON 结构化字段压缩成单行文本。"""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _cell(value: Any, limit: int = _MAX_CELL) -> str:
    text = _as_text(value).replace("\r", " ").replace("\n", " ").replace("|", "／").strip()
    if len(text) > limit:
        text = text[:limit] + "…"
    return text or "-"


def _condition_text(row: dict) -> str:
    return _cell(row.get("condition"))


def _conclusion_text(row: dict) -> str:
    conclusion = row.get("conclusion")
    if isinstance(conclusion, dict):
        text = conclusion.get("text")
        if text:
            return _cell(text)
    value = row.get("value")
    unit = row.get("unit")
    if value is not None:
        return _cell(f"{value}{unit or ''}")
    return _cell(conclusion)


def _basis_text(row: dict) -> str:
    basis = _as_text(row.get("basis_text")).strip()
    title = _as_text(row.get("title")).strip()
    article = _as_text(row.get("article_no")).strip()
    prefix = ""
    if title:
        prefix += f"《{title}》"
    if article:
        prefix += f"第{article}条"
    if prefix and basis:
        return _cell(f"{prefix}：{basis}", 160)
    if prefix:
        return _cell(prefix)
    if basis:
        return _cell(basis, 160)
    return _cell(row.get("doc_id"))


def _match_keyword(row: dict, keyword: str) -> bool:
    needle = keyword.strip().lower()
    if not needle:
        return True
    haystack = " ".join(_as_text(row.get(field)) for field in _FIELDS).lower()
    return needle in haystack


def _filter_rows(rows: list[dict], article_no: str, keyword: str) -> list[dict]:
    result = rows
    article_no = (article_no or "").strip()
    if article_no:
        result = [
            row
            for row in result
            if str(row.get("article_no") or "").strip() == article_no
        ]
    if keyword and keyword.strip():
        result = [row for row in result if _match_keyword(row, keyword)]
    return result


def _format_table(rows: list[dict]) -> str:
    shown = rows[:_MAX_ROWS]
    lines = [
        "| 条件 | 结论 | 依据条款 | 待复核 |",
        "| --- | --- | --- | --- |",
    ]
    for row in shown:
        flag = "⚠" if row.get("needs_review") else ""
        lines.append(
            f"| {_condition_text(row)} | {_conclusion_text(row)} | "
            f"{_basis_text(row)} | {flag} |"
        )
    if len(rows) > len(shown):
        lines.append(f"\n（共 {len(rows)} 条匹配，仅显示前 {len(shown)} 条）")
    return "\n".join(lines)


@tool
def rule_table_tool(doc_id: str = "", article_no: str = "", keyword: str = "") -> str:
    """查询企业规章制度的结构化规则表（条件区间 → 结论值 → 依据条款）。

    适用场景：资格判定、天数/金额/比例/审批层级推导。规则表条目仍须回到
    get_clause_tool 核对依据条款原文；needs_review 为 True 的条目会标 ⚠，
    表示抽取值待复核，不得直接用于计算。

    Args:
        doc_id: 制度文档 ID，空字符串表示不限制度。
        article_no: 条款号（如 "12" / "12.3"），空字符串表示不限条款。
        keyword: 关键词过滤（匹配条件、结论、依据原文、单位等），空字符串表示不过滤。
    """
    try:
        rows = [_to_dict(row) for row in (get_mysql().get_rule_rows(doc_id or None) or [])]
    except Exception as exc:
        logger.warning("规则表查询失败：%s", exc)
        emit("tool", tool="rule_table", hits=0, error=str(exc)[:200])
        return f"规则表查询失败：{exc}"

    matched = _filter_rows(rows, article_no, keyword)
    emit("tool", tool="rule_table", hits=len(matched))

    if not matched:
        return "规则表暂无匹配条目"

    return _format_table(matched)
