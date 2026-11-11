"""compare 对比专家：独立的 ReAct 子图（create_agent），负责版本/制度/群体差异对照。

模型使用 expert 档（跨版本对比需要更强推理）；工具允许显式检索归档版本。
注意：create_agent 在当前环境由 `langchain.agents` 提供（langgraph.prebuilt 只有旧版
create_react_agent）。
"""
from __future__ import annotations

import time
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.core.blackboard import blackboard_context
from app.core.logger import get_logger
from app.graph.state import SupervisorState
from app.llm.client import get_llm
from app.llm.prompts.compare import COMPARE_PROMPT
from app.store.docstore import DocStore
from app.tools.clause_tool import get_clause_tool
from app.tools.search_tool import search_tool
from app.tools.version_tool import list_versions_tool

logger = get_logger("agents.compare")

AGENT_NAME = "compare"

_graph = create_agent(
    model=get_llm("expert"),
    tools=[search_tool, get_clause_tool, list_versions_tool],
    system_prompt=COMPARE_PROMPT,
    name=AGENT_NAME,
)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return "" if content is None else str(content)


def _to_messages(raw: list[dict] | None) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if content is None:
            continue
        if str(item.get("role") or "user").lower() in ("user", "human"):
            messages.append(HumanMessage(content=_content_text(content)))
        else:
            messages.append(AIMessage(content=_content_text(content)))
    return messages


def _serialize_messages(messages: list[BaseMessage]) -> list[dict]:
    role_map = {"human": "user", "ai": "assistant"}
    serialized: list[dict] = []
    for message in messages:
        role = role_map.get(getattr(message, "type", ""), getattr(message, "type", "assistant"))
        serialized.append({"role": role, "content": message.content})
    return serialized


def _last_ai_message(messages: list[BaseMessage]) -> AIMessage | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and _content_text(message.content).strip():
            return message
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


def _extract_usage(message: AIMessage | None) -> dict:
    if message is None:
        return {}
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict) and usage:
        out = {
            key: usage[key]
            for key in ("input_tokens", "output_tokens", "total_tokens")
            if isinstance(usage.get(key), (int, float))
        }
        if out:
            return out

    metadata = getattr(message, "response_metadata", None) or {}
    raw = metadata.get("token_usage") or metadata.get("usage") or {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, int | float] = {}
    if isinstance(raw.get("prompt_tokens"), (int, float)):
        out["input_tokens"] = raw["prompt_tokens"]
    elif isinstance(raw.get("input_tokens"), (int, float)):
        out["input_tokens"] = raw["input_tokens"]
    if isinstance(raw.get("completion_tokens"), (int, float)):
        out["output_tokens"] = raw["completion_tokens"]
    elif isinstance(raw.get("output_tokens"), (int, float)):
        out["output_tokens"] = raw["output_tokens"]
    if isinstance(raw.get("total_tokens"), (int, float)):
        out["total_tokens"] = raw["total_tokens"]
    elif "input_tokens" in out or "output_tokens" in out:
        out["total_tokens"] = out.get("input_tokens", 0) + out.get("output_tokens", 0)
    return out


def compare_node(state: SupervisorState) -> dict:
    """对比专家节点：消息转换 → ReAct 子图 → 结论/证据/轨迹回写。"""
    started = time.perf_counter()
    raw_messages = state.get("cmp_messages") or []
    messages = _to_messages(raw_messages)
    if not messages:
        question = str(state.get("question") or "")
        messages = [HumanMessage(content=question or "请对比相关制度条款")]

    ds = DocStore()
    try:
        with blackboard_context(ds):
            result = _graph.invoke({"messages": messages})
        result_messages = list(result.get("messages") or messages)
        ai = _last_ai_message(result_messages)
        conclusion = _content_text(ai.content).strip() if ai is not None else ""
        if not conclusion:
            conclusion = "（专家未返回有效结论）"
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return {
            "cmp_messages": _serialize_messages(result_messages),
            "agent_results": [
                {"agent": AGENT_NAME, "conclusion": conclusion, "confidence": None}
            ],
            "evidence": ds.get_evidence(),
            "trace": ds.snapshot_trace()
            + [
                {
                    "event": "agent_end",
                    "agent": AGENT_NAME,
                    "elapsed_ms": elapsed_ms,
                }
            ],
            "token_usage": _extract_usage(ai),
        }
    except Exception as exc:
        logger.exception("compare 专家执行失败：%s", exc)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return {
            "cmp_messages": _serialize_messages(messages),
            "agent_results": [
                {
                    "agent": AGENT_NAME,
                    "conclusion": f"（专家执行失败：{exc}）",
                    "confidence": 0.0,
                }
            ],
            "evidence": ds.get_evidence(),
            "trace": ds.snapshot_trace()
            + [
                {
                    "event": "agent_end",
                    "agent": AGENT_NAME,
                    "elapsed_ms": elapsed_ms,
                    "error": str(exc)[:300],
                }
            ],
            "token_usage": {},
        }
