"""check_citations 工具：按 doc_id + article_no 回库校验引用是否真实存在。

供 critic 的规则核查使用，也可被专家在提交结论前自检。
默认 include_archived=False：已归档（失效）版本的引用同样判为不通过。
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from app.core.blackboard import emit
from app.core.logger import get_logger
from app.store.mysql import get_mysql

logger = get_logger("tools.cite_check")

_MAX_DETAIL = 20


def _normalize_items(payload: Any) -> list[dict] | None:
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return payload
    return None


def _label(item: dict) -> str:
    doc_id = str(item.get("doc_id") or "").strip()
    article_no = str(item.get("article_no") or "").strip()
    ref_id = str(item.get("ref_id") or "").strip()
    base = f"{doc_id} 第{article_no}条" if doc_id else "未知条款"
    return f"{ref_id}（{base}）" if ref_id else base


@tool
def check_citations_tool(citations_json: str) -> str:
    """校验引用列表是否真实存在（引用真实性规则核查）。

    输入为 JSON 数组，元素形如：
    [{"doc_id": "abc123", "article_no": "12"}, {"doc_id": "abc123", "article_no": "13.2"}]
    可选 ref_id 字段仅用于结果展示。

    Args:
        citations_json: JSON 数组字符串；每个元素必须包含 doc_id 与 article_no。
    """
    try:
        payload = json.loads(citations_json)
    except (json.JSONDecodeError, TypeError) as exc:
        return f"引用校验失败：无法解析 JSON（{exc}）"

    items = _normalize_items(payload)
    if items is None:
        return "引用校验失败：JSON 必须是对象数组，元素形如 {\"doc_id\": \"...\", \"article_no\": \"...\"}"
    if not items:
        emit("tool", tool="check_citations", total=0, bad=0)
        return "通过 0 条 / 伪造或失效 0 条：引用列表为空"

    good = 0
    bad: list[str] = []
    try:
        mysql = get_mysql()
        for item in items:
            doc_id = str(item.get("doc_id") or "").strip()
            article_no = str(item.get("article_no") or "").strip()
            if not doc_id or not article_no:
                bad.append(f"{_label(item)}：缺少 doc_id 或 article_no")
                continue
            row = mysql.get_clause(doc_id, article_no)
            if row:
                good += 1
            else:
                bad.append(f"{_label(item)}：条款不存在或已失效")
    except Exception as exc:
        logger.warning("引用校验失败：%s", exc)
        emit("tool", tool="check_citations", total=len(items), bad=len(bad), error=str(exc)[:200])
        return f"引用校验失败：{exc}"

    emit("tool", tool="check_citations", total=len(items), bad=len(bad))

    if not bad:
        return f"通过 {good} 条 / 伪造或失效 0 条：全部引用均可在库中命中"

    shown = "；".join(bad[:_MAX_DETAIL])
    if len(bad) > _MAX_DETAIL:
        shown += f"；……另有 {len(bad) - _MAX_DETAIL} 条"
    return f"通过 {good} 条 / 伪造或失效 {len(bad)} 条：{shown}"
