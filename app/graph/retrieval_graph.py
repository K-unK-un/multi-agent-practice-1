"""检索子图（Agentic RAG）：rewrite → 双路召回 → RRF → grade ⟲ → rerank。

对外只暴露 invoke_retrieval() 一个函数，作为各专家 Agent 的 search 工具底座。
图内每个节点都自带异常降级：任何一路检索/模型调用失败，返回已有结果 + notice，不抛出。

拓扑：
    START → rewrite → retrieve → grade ─┬─(coverage 不足且 loop < max)→ refine → retrieve
                                        └─(充分 / 超限)→ final → END

说明：RetrievalState 是契约文件，不能新增字段；grade 产出的补查建议暂时存放在本模块
内部扩展状态 _GraphState.suggestions 中，不污染对外状态结构。
"""
from __future__ import annotations

import time
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.core.blackboard import emit
from app.core.config import settings
from app.core.logger import get_logger
from app.graph.state import RetrievalState
from app.llm.client import get_structured_llm
from app.llm.prompts.grade import GRADE_PROMPT, GradeSchema
from app.llm.prompts.rewrite import REWRITE_PROMPT, RewriteSchema
from app.retrieval.bm25 import get_bm25_index
from app.retrieval.filters import apply_metadata_filters
from app.retrieval.fusion import rrf_fuse
from app.retrieval.reranker import get_reranker
from app.retrieval.vector import get_vector_retriever

logger = get_logger("graph.retrieval")


class _GraphState(RetrievalState, total=False):
    """内部图状态：在契约 RetrievalState 基础上增加 suggestions（不对外暴露）。"""

    suggestions: list[str]


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def _field(obj: Any, name: str, default: Any) -> Any:
    """兼容 pydantic 对象与 dict 两种结构化输出形态。"""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _hit_key(hit: dict) -> str:
    return f"{hit.get('doc_id') or ''}::{hit.get('article_no') or ''}"


def _clean_queries(values: Any, limit: int = 3) -> list[str]:
    if not values:
        return []
    if isinstance(values, str):  # 防御结构化输出退化为单个字符串
        values = [values]
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    return list(dict.fromkeys(cleaned))[:limit]


# ---------------------------------------------------------------------------
# 节点：rewrite
# ---------------------------------------------------------------------------

def _rewrite_node(state: _GraphState) -> dict:
    """口语化问题 → 1~3 个检索子查询；开关关闭或 LLM 失败时用原始问题。"""
    question = (state.get("question") or "").strip()
    if not question:
        return {"queries": []}
    if not settings.enable_query_rewrite:
        return {"queries": [question]}

    try:
        llm = get_structured_llm("light", RewriteSchema)
        result = (REWRITE_PROMPT | llm).invoke({"question": question})
        queries = _clean_queries(_field(result, "queries", []))
        return {"queries": queries or [question]}
    except Exception as exc:
        logger.warning("查询改写失败（降级为原始问题）：%s", exc)
        return {"queries": [question]}


# ---------------------------------------------------------------------------
# 节点：retrieve（双路召回 + RRF + 元数据过滤）
# ---------------------------------------------------------------------------

def _apply_doc_filters(
    fused: list[dict],
    user_scope: list[str] | None,
    include_archived: bool,
) -> list[dict]:
    """用 MySQL 元数据做与 Chroma where 同源的过滤，按 doc_id 集合取交集。

    BM25 索引不存元数据，只能拿到候选 doc_id 后回查 MySQL。MySQL 不可用 / 无对应
    元数据时，退化为只保留向量路命中：向量路已在 Chroma where 中执行过同一套过滤，
    不会越权；BM25-only 命中无法验权，直接丢弃（安全优先）。
    """
    if not fused:
        return []
    doc_ids = sorted({str(hit.get("doc_id")) for hit in fused if hit.get("doc_id")})
    if not doc_ids:
        return []

    try:
        from app.store.mysql import get_mysql

        rows = get_mysql().get_clauses(
            doc_ids=doc_ids, include_archived=include_archived
        )
        rows = apply_metadata_filters(
            rows or [], user_scope=user_scope, include_archived=include_archived
        )
        allowed = {
            str(row.get("doc_id"))
            for row in rows
            if isinstance(row, dict) and row.get("doc_id")
        }
        if allowed:
            return [hit for hit in fused if str(hit.get("doc_id")) in allowed]
        logger.warning(
            "MySQL 未返回候选条款元数据（doc_ids=%d），仅保留向量路命中", len(doc_ids)
        )
    except Exception as exc:
        logger.warning("元数据过滤失败，仅保留向量路命中：%s", exc)

    return [hit for hit in fused if "vector" in (hit.get("channels") or [])]


def _retrieve_node(state: _GraphState) -> dict:
    """逐个 query 双路召回 → RRF → 元数据过滤，按 (doc_id, article_no) 累加去重。"""
    started = time.perf_counter()
    queries = _clean_queries(state.get("queries"), limit=3)
    if not queries:
        queries = [(state.get("question") or "").strip()]
    user_scope = state.get("user_scope") or []
    include_archived = bool(state.get("include_archived"))

    merged: dict[str, dict] = {
        _hit_key(candidate): dict(candidate)
        for candidate in (state.get("candidates") or [])
        if isinstance(candidate, dict) and candidate.get("doc_id")
    }

    for query in queries:
        if not query:
            continue

        vector_hits: list[dict] = []
        bm25_hits: list[dict] = []
        try:
            vector_hits = get_vector_retriever().search(
                query, user_scope=user_scope, include_archived=include_archived
            )
        except Exception as exc:
            logger.warning("向量召回异常（已跳过该路）：%s", exc)
        try:
            bm25_hits = get_bm25_index().search(query)
        except Exception as exc:
            logger.warning("BM25 召回异常（已跳过该路）：%s", exc)

        fused = _apply_doc_filters(
            rrf_fuse(vector_hits, bm25_hits), user_scope, include_archived
        )

        for hit in fused:
            key = _hit_key(hit)
            if key in merged:
                old = merged[key]
                try:
                    old["score"] = max(
                        _as_float(old.get("score")), _as_float(hit.get("score"))
                    )
                except (TypeError, ValueError):
                    pass
                old["channels"] = list(
                    dict.fromkeys(
                        [*(old.get("channels") or []), *(hit.get("channels") or [])]
                    )
                )
                if len(hit.get("text") or "") > len(old.get("text") or ""):
                    old["text"] = hit.get("text") or ""
                    old["page"] = hit.get("page", old.get("page"))
                    old["title"] = hit.get("title") or old.get("title") or ""
            else:
                merged[key] = dict(hit)

    candidates = list(merged.values())
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    emit(
        "retrieve",
        source="vector+bm25",
        count=len(candidates),
        elapsed_ms=elapsed_ms,
    )
    return {"candidates": candidates}


# ---------------------------------------------------------------------------
# 节点：grade
# ---------------------------------------------------------------------------

def _grade_node(state: _GraphState) -> dict:
    """LLM 评估候选条款对原始问题的覆盖度；失败时视为已覆盖以避免无效回环。"""
    question = state.get("question") or ""
    candidates = [c for c in (state.get("candidates") or []) if isinstance(c, dict)]
    loop = int(state.get("loop") or 0)

    if not candidates:
        missing = ["未检索到候选条款"]
        emit("grade", coverage=0.0, missing=missing, loop=loop)
        return {"coverage": 0.0, "missing": missing, "suggestions": []}

    lines = []
    for candidate in candidates:
        text = str(candidate.get("text") or "").replace("\n", " ").strip()[:60]
        lines.append(
            f"- doc_id={candidate.get('doc_id')}，"
            f"第{candidate.get('article_no') or '?'}条：{text}"
        )
    candidates_text = "\n".join(lines)

    try:
        llm = get_structured_llm("light", GradeSchema)
        result = (GRADE_PROMPT | llm).invoke(
            {"question": question, "candidates": candidates_text}
        )
        coverage = _clamp_unit(_as_float(_field(result, "coverage", 0.0)))
        missing = _clean_queries(_field(result, "missing", []), limit=10)
        suggestions = _clean_queries(_field(result, "suggestions", []))
    except Exception as exc:
        # 评估器不可用不应阻塞检索：直接放行到精排，由 notice 之外的环节兜底
        logger.warning("覆盖度评估失败（降级为视为已覆盖）：%s", exc)
        coverage = 1.0
        missing = []
        suggestions = []

    emit("grade", coverage=coverage, missing=missing, loop=loop)
    return {"coverage": coverage, "missing": missing, "suggestions": suggestions}


def _route_after_grade(state: _GraphState) -> str:
    """覆盖不足且未达回环上限、且确有补查建议 → refine；否则 final。"""
    coverage = _as_float(state.get("coverage"), 0.0)
    loop = int(state.get("loop") or 0)
    suggestions = _clean_queries(state.get("suggestions"))
    if (
        coverage < settings.grade_min_coverage
        and loop < settings.max_retrieval_loop
        and suggestions
    ):
        return "refine"
    return "final"


# ---------------------------------------------------------------------------
# 节点：refine / final
# ---------------------------------------------------------------------------

def _refine_node(state: _GraphState) -> dict:
    """用 grade 的补查建议替换 queries，并累加回环计数。"""
    suggestions = _clean_queries(state.get("suggestions"))
    queries = suggestions or [(state.get("question") or "").strip()]
    return {"queries": queries, "loop": int(state.get("loop") or 0) + 1}


def _final_node(state: _GraphState) -> dict:
    """面向原始问题精排，输出 contexts + 缺口说明。"""
    question = state.get("question") or ""
    candidates = [
        dict(c) for c in (state.get("candidates") or []) if isinstance(c, dict)
    ]

    try:
        contexts = get_reranker().rerank(question, candidates)
    except Exception as exc:
        logger.warning("精排异常（降级为融合顺序）：%s", exc)
        contexts = candidates[: settings.rerank_top_k]

    coverage = _as_float(state.get("coverage"), 0.0)
    missing = _clean_queries(state.get("missing"), limit=10)

    if not contexts:
        notice = "知识库可能未建索引或范围过滤后无结果"
    elif coverage < settings.grade_min_coverage:
        gap = "；".join(missing) or "未识别具体缺口"
        notice = f"已检索 {len(contexts)} 条条款，但覆盖可能不足：{gap}"
    else:
        notice = ""

    emit(
        "rerank",
        top_k=len(contexts),
        items=[
            {
                "doc_id": ctx.get("doc_id"),
                "article_no": ctx.get("article_no"),
                "title": ctx.get("title"),
                "score": ctx.get("score"),
            }
            for ctx in contexts
        ],
    )
    return {"contexts": contexts, "notice": notice}


# ---------------------------------------------------------------------------
# 图组装（模块级编译单例）
# ---------------------------------------------------------------------------

def _build_graph():
    builder = StateGraph(_GraphState)
    builder.add_node("rewrite", _rewrite_node)
    builder.add_node("retrieve", _retrieve_node)
    builder.add_node("grade", _grade_node)
    builder.add_node("refine", _refine_node)
    builder.add_node("final", _final_node)

    builder.add_edge(START, "rewrite")
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade",
        _route_after_grade,
        {"refine": "refine", "final": "final"},
    )
    builder.add_edge("refine", "retrieve")
    builder.add_edge("final", END)
    return builder.compile()


_graph = _build_graph()


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------

def invoke_retrieval(
    question: str,
    mode: str = "auto",
    user_scope: list[str] | None = None,
    include_archived: bool | None = None,
) -> dict:
    """运行检索子图。

    Args:
        question: 用户原始问题（精排与覆盖度评估都面向它）。
        mode: "auto"（默认只查现行版本）| "compare"（允许 archived 旧版本）。
        user_scope: 用户可见范围；None/空时由 filters 默认收窄为 ["employee"]。
        include_archived: 默认 (mode == "compare")。

    Returns:
        {contexts, coverage, missing, loops, notice}
    """
    question = (question or "").strip()
    if include_archived is None:
        include_archived = mode == "compare"

    initial: _GraphState = {
        "question": question,
        "mode": mode,
        "user_scope": list(user_scope) if user_scope else [],
        "include_archived": bool(include_archived),
        "queries": [],
        "candidates": [],
        "coverage": 0.0,
        "missing": [],
        "loop": 0,
        "contexts": [],
        "notice": "",
        "suggestions": [],
    }

    try:
        result = _graph.invoke(initial)
    except Exception as exc:
        logger.warning("检索子图执行失败（降级为空结果）：%s", exc)
        return {
            "contexts": [],
            "coverage": 0.0,
            "missing": [],
            "loops": 0,
            "notice": f"检索失败：{exc}",
        }

    contexts = [
        dict(context)
        for context in (result.get("contexts") or [])
        if isinstance(context, dict)
    ]
    missing = _clean_queries(result.get("missing"), limit=10)
    notice = result.get("notice") or ""
    if not contexts and not notice:
        notice = "知识库可能未建索引或范围过滤后无结果"

    return {
        "contexts": contexts,
        "coverage": _as_float(result.get("coverage"), 0.0),
        "missing": missing,
        "loops": int(result.get("loop") or 0),
        "notice": notice,
    }
