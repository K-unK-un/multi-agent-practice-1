"""synthesize 汇总生成：证据 + 专家结论 → final_answer + 条款级 citations。

引用编号规则：证据文本由 DocStore.snapshot_text() 生成，按顺序标【E1】【E2】…
final_answer 中出现哪些【EX】，citations 就按出现顺序回填哪些证据（ref_id 对齐）。
LLM 失败时用证据原文模板兜底，保证图不崩。
"""
from __future__ import annotations

import re
import time
from typing import Any

from app.core.config import settings
from app.core.logger import get_logger
from app.graph.state import SupervisorState
from app.llm.client import get_llm
from app.llm.prompts.synthesize import SYNTHESIZE_PROMPT
from app.store.docstore import DocStore

logger = get_logger("agents.synthesize")

_REF_RE = re.compile(r"【E(\d+)】")


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


def _extract_usage(message: Any) -> dict:
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


def _format_expert_results(results: list[dict]) -> str:
    lines: list[str] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        agent = str(item.get("agent") or "专家")
        conclusion = str(item.get("conclusion") or "").strip()
        if not conclusion:
            continue
        lines.append(f"【{agent}】{conclusion}")
    return "\n\n".join(lines)[:4000]


def _build_citations(final_answer: str, evidence: list[dict]) -> list[dict]:
    """按答案中【EX】首次出现的顺序回填证据，保证引用编号与证据一一对应。"""
    seen: set[int] = set()
    citations: list[dict] = []
    for match in _REF_RE.finditer(final_answer or ""):
        index = int(match.group(1))
        if index in seen:
            continue
        seen.add(index)
        if not 1 <= index <= len(evidence):
            continue
        item = evidence[index - 1]
        if not isinstance(item, dict):
            continue
        citations.append(
            {
                "ref_id": f"E{index}",
                "doc_id": item.get("doc_id") or "",
                "article_no": item.get("article_no") or "",
                "title": item.get("title") or "",
                "page": item.get("page"),
                "snippet": str(item.get("text") or "")[:120],
            }
        )
    return citations


def _fallback_answer(evidence: list[dict]) -> str:
    """LLM 失败时的模板兜底：只拼接证据原文，不新增任何推断。"""
    if not evidence:
        return (
            "制度未明确：当前知识库未检索到可支撑该问题的条款。\n"
            "建议补充制度名称或具体条件后再提问，或向人力资源部门确认。"
        )
    blocks: list[str] = []
    for index, item in enumerate(evidence[:3], 1):
        if not isinstance(item, dict):
            continue
        title = item.get("title") or item.get("doc_id") or "制度"
        article = item.get("article_no") or "?"
        text = str(item.get("text") or "").strip()
        blocks.append(f"【E{index}】《{title}》第{article}条：{text[:180]}")
    return (
        "以下是检索到的相关条款，供参考：\n\n"
        + "\n\n".join(blocks)
        + "\n\n说明：以上为条款摘录，未经过模型归纳；制度未明确处请以原文及人力资源解释为准。"
    )


def synthesize_node(state: SupervisorState) -> dict:
    """汇总节点：证据编号文本 + 专家结论 → 最终答案与引用。"""
    started = time.perf_counter()
    question = str(state.get("question") or "").strip()
    evidence = [item for item in (state.get("evidence") or []) if isinstance(item, dict)]
    expert_text = _format_expert_results(state.get("agent_results") or [])

    ds = DocStore()
    try:
        ds.add_evidence(evidence)
        evidence_text = ds.snapshot_text(settings.context_max_chars)
    except Exception as exc:
        logger.warning("证据文本构建失败：%s", exc)
        evidence_text = ""

    prompt = (
        f"{SYNTHESIZE_PROMPT}\n\n"
        f"【用户问题】\n{question or '（空）'}\n\n"
        f"【专家结论】\n{expert_text or '（无专家结论）'}\n\n"
        f"【证据】\n{evidence_text or '（无可用条款证据）'}"
    )

    fallback_used = False
    token_usage: dict = {}
    try:
        llm = get_llm("main")
        if settings.synthesize_max_tokens > 0:
            # 只限制最终答案生成长度，不影响专家/规则抽取等其他调用
            llm = llm.bind(max_tokens=settings.synthesize_max_tokens)
        ai = llm.invoke(prompt)
        final_answer = _content_text(getattr(ai, "content", "")).strip()
        token_usage = _extract_usage(ai)
        if not final_answer:
            raise ValueError("模型返回空答案")
    except Exception as exc:
        logger.warning("synthesize 生成失败，使用证据模板兜底：%s", exc)
        final_answer = _fallback_answer(evidence)
        fallback_used = True
        token_usage = {}

    citations = _build_citations(final_answer, evidence)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    trace = [
        {
            "event": "synthesize",
            "elapsed_ms": elapsed_ms,
            "evidence_count": len(evidence),
            "expert_results": len(state.get("agent_results") or []),
            "citations": len(citations),
            "fallback": fallback_used,
        }
    ]
    return {
        "final_answer": final_answer,
        "citations": citations,
        "trace": trace,
        "token_usage": token_usage,
    }
