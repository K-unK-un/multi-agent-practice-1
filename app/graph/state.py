"""LangGraph 状态定义：顶层多 Agent 状态（SupervisorState）+ 检索子图状态（RetrievalState）。"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def merge_evidence(left: list[dict], right: list[dict]) -> list[dict]:
    """证据池 reducer：按 doc_id+article_no 去重；合并通道/查询/分数，取更长文本。"""
    pool: dict[str, dict] = {}
    for item in left + right:
        if not item.get("doc_id"):
            continue
        key = f"{item.get('doc_id')}::{item.get('article_no')}"
        if key in pool:
            old = pool[key]
            if len(item.get("text", "")) > len(old.get("text", "")):
                old["text"] = item["text"]
                old["page"] = item.get("page", old.get("page"))
                old["title"] = item.get("title", old.get("title"))
            old["full_text"] = old.get("full_text") or item.get("full_text")
            for field in ("channels", "queries"):
                old[field] = list(
                    dict.fromkeys([*(old.get(field) or []), *(item.get(field) or [])])
                )
            try:
                old["score"] = max(float(old.get("score") or 0), float(item.get("score") or 0))
            except (TypeError, ValueError):
                pass
        else:
            pool[key] = dict(item)
    return list(pool.values())


def merge_usage(left: dict, right: dict) -> dict:
    """token 用量累加 reducer。"""
    total: dict[str, Any] = {}
    for key in set(left) | set(right):
        a, b = left.get(key, 0), right.get(key, 0)
        total[key] = a + b if isinstance(a, (int, float)) and isinstance(b, (int, float)) else b
    return total


class SupervisorState(TypedDict, total=False):
    # ---------------- 会话 ----------------
    question: str
    session_id: str
    user_scope: list[str]
    history_summary: str  # 历史压缩摘要，不给原始消息流

    # ---------------- 调度 ----------------
    plan: dict[str, Any]  # {intent, confidence, subtasks, next, done, reason}
    next_agent: str
    dispatch_round: int
    review_round: int

    # ---------------- 黑板 ----------------
    evidence: Annotated[list[dict], merge_evidence]  # 条款级证据池，全 Agent 共享
    agent_results: Annotated[list[dict], operator.add]  # 各专家结论与置信度

    # ---------------- 专家独立消息（上下文隔离）----------------
    # 纯 list（写覆盖语义，不用 add_messages，避免重复累积）；元素 {role, content}
    qa_messages: list[dict]
    cmp_messages: list[dict]
    eli_messages: list[dict]

    # ---------------- 审核与产出 ----------------
    review: dict[str, Any]  # {verdict, unsupported[], bad_citations[], numeric_mismatches[]}
    final_answer: str
    citations: list[dict]  # [{ref_id, doc_id, article_no, title, page, snippet}]

    # ---------------- 观测 ----------------
    trace: Annotated[list[dict], operator.add]  # 节点/工具事件，SSE 数据源
    token_usage: Annotated[dict, merge_usage]


class RetrievalState(TypedDict, total=False):
    """检索子图状态。"""
    question: str
    mode: str  # auto | compare（compare 允许 archived）
    user_scope: list[str]
    include_archived: bool
    queries: list[str]  # rewrite 产出的子查询
    candidates: list[dict]  # 融合后的候选条款
    coverage: float
    missing: list[str]
    loop: int
    contexts: list[dict]  # 精排后的最终上下文
    notice: str  # 缺口说明
