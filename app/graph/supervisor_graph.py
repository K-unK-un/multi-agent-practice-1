"""顶层图组装：supervisor ⇄ {policy_qa, compare, eligibility} → synthesize → critic → END。

同时提供：
- run_answer_once：同步跑一次问答（评估脚本用）；
- stream_answer：SSE 事件流（只消费 updates），前端链路可视化数据源。

流式说明（langgraph 1.2.10 实测）：
    astream(input, config, stream_mode=[...], subgraphs=True)
    - stream_mode 为列表且 subgraphs=True 时，chunk 为 3 元组
      (namespace, mode, data)；
    - stream_mode 为列表且 subgraphs=False 时，chunk 为 2 元组 (mode, data)；
    - stream_mode 为单个字符串时，chunk 直接是 data（本模块不使用该形态）。
    本模块统一用 _split_stream_chunk 归一化。
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, AsyncIterator

from langgraph.graph import END, START, StateGraph

from app.agents.compare import compare_node
from app.agents.criticize import critic_node
from app.agents.eligibility import eligibility_node
from app.agents.policy_qa import policy_qa_node
from app.agents.supervisor import clarify_node, supervisor_node
from app.agents.synthesize import synthesize_node
from app.core.config import settings
from app.core.logger import get_logger
from app.graph.state import SupervisorState
from app.llm.client import get_llm
from app.store.checkpointer import get_checkpointer
from app.store.mysql import get_mysql

logger = get_logger("graph.supervisor")

_graph: Any = None


# ---------------------------------------------------------------------------
# 图构建
# ---------------------------------------------------------------------------

def build_graph(checkpointer: Any = None):
    """组装并编译顶层图。checkpointer=None 时退化为无记忆的临时图（测试用）。"""
    builder = StateGraph(SupervisorState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("policy_qa", policy_qa_node)
    builder.add_node("compare", compare_node)
    builder.add_node("eligibility", eligibility_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("critic", critic_node)
    builder.add_node("clarify", clarify_node)

    builder.add_edge(START, "supervisor")
    # 专家返回后控制权回到中枢（专家之间不直接通信）
    builder.add_edge("policy_qa", "supervisor")
    builder.add_edge("compare", "supervisor")
    builder.add_edge("eligibility", "supervisor")
    builder.add_edge("synthesize", "critic")
    builder.add_edge("clarify", END)

    return builder.compile(checkpointer=checkpointer)


def get_graph():
    """进程内单例：首次调用时构建并挂载 checkpointer。"""
    global _graph
    if _graph is None:
        _graph = build_graph(checkpointer=get_checkpointer())
    return _graph


def reset_graph() -> None:
    """测试用：清空单例，下次 get_graph() 重新构建。"""
    global _graph
    _graph = None


# ---------------------------------------------------------------------------
# 输入 / 历史摘要
# ---------------------------------------------------------------------------

def _initial_state(
    question: str,
    user_scope: list[str] | None,
    session_id: str,
    history_summary: str,
    user_level: int | None = None,
) -> dict:
    return {
        "question": question,
        "session_id": session_id,
        "user_scope": list(user_scope or ["employee"]),
        "user_level": int(user_level or 1),
        "history_summary": history_summary or "",
        "plan": {},
        "next_agent": "",
        "dispatch_round": 0,
        "review_round": 0,
        "review": {},
        "final_answer": "",
        "citations": [],
        "evidence": [],
        "agent_results": [],
        "trace": [],
        "token_usage": {},
    }


def _row_field(row: Any, name: str, default: Any = "") -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return "" if content is None else str(content)


def _build_history_summary(session_id: str, current_question: str = "") -> str:
    """取最近 6 条会话消息拼摘要；超长用 light 模型压缩到 200 字，失败则截断。"""
    try:
        rows = get_mysql().list_messages(session_id) or []
    except Exception as exc:
        logger.warning("读取会话历史失败，按无历史处理：%s", exc)
        return ""

    lines: list[str] = []
    for row in list(rows)[-6:]:
        role = str(_row_field(row, "role", "user") or "user")
        content = _content_text(_row_field(row, "content", "")).strip()
        if not content:
            continue
        if current_question and content == current_question.strip():
            # 部分接口会先落库本轮问题，避免把当前问题当历史重复注入
            continue
        lines.append(f"{role}: {content}")

    text = "\n".join(lines)
    if not text:
        return ""
    if len(text) <= 800:
        return text

    try:
        ai = get_llm("light").invoke(
            "请把以下多轮对话历史压缩为不超过 200 字的中文摘要，"
            "只保留与后续追问相关的制度主题、已确认结论和未解决问题：\n\n"
            f"{text[:4000]}"
        )
        summary = _content_text(getattr(ai, "content", "")).strip()
        if summary:
            return summary[:200]
    except Exception as exc:
        logger.warning("历史摘要压缩失败，退回截断拼接：%s", exc)
    return text[:800]


# ---------------------------------------------------------------------------
# 同步问答（评估 / 脚本）
# ---------------------------------------------------------------------------

def run_answer_once(
    question: str,
    user_scope: list[str] | None = None,
    session_id: str = "eval",
    history_summary: str = "",
    user_level: int | None = None,
) -> dict:
    """同步跑一次完整问答，返回最终状态中的关键字段。"""
    graph = get_graph()
    if not history_summary:
        history_summary = _build_history_summary(session_id, question)
    inputs = _initial_state(question, user_scope, session_id, history_summary, user_level)
    config = {"configurable": {"thread_id": session_id}}
    final_state = graph.invoke(inputs, config=config)
    return {
        "final_answer": final_state.get("final_answer") or "",
        "citations": final_state.get("citations") or [],
        "review": final_state.get("review") or {},
        "plan": final_state.get("plan") or {},
        "evidence": final_state.get("evidence") or [],
        "agent_results": final_state.get("agent_results") or [],
        "trace": final_state.get("trace") or [],
        "token_usage": final_state.get("token_usage") or {},
        "dispatch_round": final_state.get("dispatch_round") or 0,
    }


# ---------------------------------------------------------------------------
# 流式问答（SSE 数据源）
# ---------------------------------------------------------------------------

def _split_stream_chunk(chunk: Any) -> tuple[tuple[str, ...], str | None, Any]:
    """归一化 astream 输出：兼容 (ns, mode, data) / (mode, data) 两种形态。"""
    if isinstance(chunk, tuple):
        if len(chunk) == 3:
            namespace, mode, data = chunk
            if isinstance(namespace, str):
                namespace = (namespace,)
            return tuple(namespace or ()), str(mode), data
        if len(chunk) == 2:
            mode, data = chunk
            return (), str(mode), data
    return (), None, None


def _update_events(node: str, update: Any) -> list[dict]:
    """把一次节点 update 转成 SSE 事件（trace 逐条 + route/review 专项）。"""
    events: list[dict] = []
    if not isinstance(update, dict):
        return events

    for item in update.get("trace") or []:
        if isinstance(item, dict) and item.get("event"):
            events.append({"event": str(item["event"]), "data": item})

    if node == "supervisor" and isinstance(update.get("plan"), dict):
        plan = update["plan"]
        events.append(
            {
                "event": "route",
                "data": {
                    "intent": plan.get("intent"),
                    "confidence": plan.get("confidence"),
                    "subtasks": plan.get("subtasks") or [],
                    "plan": plan,
                },
            }
        )

    if isinstance(update.get("review"), dict) and update["review"]:
        events.append({"event": "review", "data": update["review"]})
    return events


def _observe_update(observed: dict, update: Any) -> None:
    """旁路累积节点 update 中的产出字段。

    用途：checkpointer 不可用或 get_state 读取失败时，done 事件仍能带回答案；
    get_state 成功时以终态为准，这里只做兜底。
    """
    if not isinstance(update, dict):
        return
    for key in ("final_answer", "citations", "review", "plan", "dispatch_round", "token_usage"):
        value = update.get(key)
        if value not in (None, "", [], {}):
            observed[key] = value
    results = update.get("agent_results")
    if isinstance(results, list):
        bucket = observed.setdefault("agent_results", [])
        bucket.extend(item for item in results if isinstance(item, dict))


def _merge_observed(final_state: dict | None, observed: dict) -> dict:
    """终态优先，缺失字段用流中观察到的 update 兜底。"""
    merged = dict(observed or {})
    for key, value in (final_state or {}).items():
        if value not in (None, "", [], {}):
            merged[key] = value
    return merged


def _done_events(final_state: dict, started: float) -> list[dict]:
    citations = [c for c in (final_state.get("citations") or []) if isinstance(c, dict)]
    review = final_state.get("review") if isinstance(final_state.get("review"), dict) else {}
    agents_used: list[str] = []
    for item in final_state.get("agent_results") or []:
        if isinstance(item, dict):
            agent = item.get("agent")
            if agent and agent not in agents_used:
                agents_used.append(agent)

    events = [{"event": "citation", "data": citation} for citation in citations]
    events.append(
        {
            "event": "done",
            "data": {
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "token_usage": final_state.get("token_usage") or {},
                "agents_used": agents_used,
                "verdict": review.get("verdict"),
                "final_answer": final_state.get("final_answer") or "",
                "citations": citations,
                "dispatch_round": final_state.get("dispatch_round") or 0,
            },
        }
    )
    return events


def _final_state_sync(graph: Any, config: dict) -> dict:
    try:
        return graph.get_state(config).values or {}
    except Exception as exc:
        logger.warning("读取最终状态失败：%s", exc)
        return {}


async def stream_answer(
    question: str,
    user_scope: list[str] | None = None,
    session_id: str | None = None,
    user_level: int | None = None,
) -> AsyncIterator[dict]:
    """流式执行一次问答，按 SSE 事件协议逐个 yield。

    事件：route / agent_start / agent_end / tool / synthesize / review /
    citation / clarify / done / error。
    专家 Agent 的中间 token 不再下发，答案只在 done.final_answer 中给出。
    """
    session_id = session_id or uuid.uuid4().hex
    started = time.perf_counter()
    graph = get_graph()
    config = {"configurable": {"thread_id": session_id}}

    try:
        history_summary = await asyncio.to_thread(
            _build_history_summary, session_id, question
        )
    except Exception as exc:  # 历史读取失败不影响本轮问答
        logger.warning("历史摘要生成失败：%s", exc)
        history_summary = ""

    inputs = _initial_state(question, user_scope, session_id, history_summary, user_level)
    emitted = False
    observed: dict = {}

    try:
        try:
            async for chunk in graph.astream(
                inputs,
                config,
                stream_mode=["updates"],
                subgraphs=True,
            ):
                _namespace, mode, data = _split_stream_chunk(chunk)
                if mode != "updates":
                    # 专家 Agent 的中间 token 不再下发。最终答案只在 done.final_answer
                    # 中给出，避免前端先渲染“未检索到”等中间稿、随后被终稿反复覆盖。
                    continue
                if not isinstance(data, dict):
                    continue
                for node, update in data.items():
                    _observe_update(observed, update)
                    for event in _update_events(node, update):
                        emitted = True
                        yield event
        except Exception as exc:
            # astream 签名/行为不兼容或中途失败：降级处理，保证前端仍能收到收尾事件。
            # - 若尚未产出任何事件（通常是签名不兼容，失败发生在执行前），一次性 invoke；
            # - 若已产出部分事件，不重复执行模型，直接读取终态补发 citation/done。
            logger.warning("astream 失败，降级处理：%s", exc)
            if not emitted:
                try:
                    final_state = await asyncio.to_thread(graph.invoke, inputs, config)
                except Exception as invoke_exc:
                    logger.exception("降级 invoke 也失败：%s", invoke_exc)
                    yield {"event": "error", "data": {"message": str(invoke_exc)}}
                    return
                final_state = _merge_observed(final_state, observed)
                plan = final_state.get("plan") if isinstance(final_state.get("plan"), dict) else {}
                yield {
                    "event": "route",
                    "data": {
                        "intent": plan.get("intent"),
                        "confidence": plan.get("confidence"),
                        "subtasks": plan.get("subtasks") or [],
                        "plan": plan,
                    },
                }
            else:
                final_state = await asyncio.to_thread(_final_state_sync, graph, config)

            for event in _done_events(_merge_observed(final_state, observed), started):
                yield event
            return

        final_state = await asyncio.to_thread(_final_state_sync, graph, config)
        for event in _done_events(_merge_observed(final_state, observed), started):
            yield event
    except Exception as exc:  # 顶层兜底：任何异常都不向调用方抛出
        logger.exception("stream_answer 顶层异常：%s", exc)
        yield {"event": "error", "data": {"message": str(exc)}}
