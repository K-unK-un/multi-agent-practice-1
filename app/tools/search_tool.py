"""search 工具：各专家 Agent 通过它调用检索子图（双路召回 + grade 回环 + 精排）。

工具写入共享黑板：
- record_evidence：每条精排上下文写入条款级证据（channel 记录命中的召回通道）；
- emit("tool", ...)  记录一次工具调用，供 SSE 链路可视化。
"""
from __future__ import annotations

from langchain_core.tools import tool

from app.core.blackboard import emit, record_evidence
from app.graph.retrieval_graph import invoke_retrieval


def _format_score(value) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "-"


@tool
def search_tool(query: str, mode: str = "auto") -> str:
    """检索企业规章制度知识库，返回与问题最相关的条款摘要。

    Args:
        query: 检索问题（可口语化，内部会做查询改写与双路召回）。
        mode: "auto" 只检索现行有效版本；"compare" 允许检索已归档旧版本（跨版本对比时使用）。
    """
    # 黑板（DocStore）只保存证据与 trace，不携带 user_scope；此处不传，
    # 由 retrieval.filters 默认收窄为 ["employee"]（最小权限）。若未来需要按登录用户
    # scope 检索，可通过 contextvar 或工具闭包注入，而不是修改黑板结构。
    result = invoke_retrieval(query, mode=mode)
    contexts = result.get("contexts") or []

    if contexts:
        evidence = []
        for context in contexts:
            channels = context.get("channels") or []
            if isinstance(channels, str):
                channels = [channels]
            evidence.append(
                {
                    "doc_id": context.get("doc_id") or "",
                    "article_no": context.get("article_no") or "",
                    "title": context.get("title") or "",
                    "page": context.get("page"),
                    "text": context.get("text") or "",
                    "channel": "/".join(str(ch) for ch in channels),
                    "score": context.get("score"),
                    "query": query,
                }
            )
        record_evidence(evidence)

    emit(
        "tool",
        agent="retrieval",
        tool="search",
        query=query,
        hits=len(contexts),
    )

    if not contexts:
        return "未检索到相关条款"

    lines: list[str] = []
    for index, context in enumerate(contexts, 1):
        title = context.get("title") or context.get("doc_id") or "制度"
        article = context.get("article_no") or "?"
        snippet = (
            str(context.get("text") or "").replace("\n", " ").replace("\r", " ").strip()
        )[:80]
        lines.append(
            f"{index}. 《{title}》第{article}条"
            f"（score={_format_score(context.get('score'))}）：{snippet}"
        )
    return "\n".join(lines)
