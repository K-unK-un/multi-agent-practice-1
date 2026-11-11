"""supervisor 调度中枢：意图分类 → 任务分解 → 专家分派 → 结束判定。

控制流用 Command 显式跳转；所有回环受 settings.max_dispatch_round 硬约束。
supervisor 只做路由与收口判断，不直接回答用户问题。
"""
from __future__ import annotations

import time
from typing import Literal

from langgraph.types import Command
from pydantic import BaseModel, Field

from app.core.blackboard import emit
from app.core.config import settings
from app.core.logger import get_logger
from app.graph.state import SupervisorState
from app.llm.client import get_structured_llm
from app.llm.prompts.supervisor import SUPERVISOR_PROMPT
from app.store.docstore import DocStore

logger = get_logger("agents.supervisor")

EXPERT_AGENTS: tuple[str, ...] = ("policy_qa", "compare", "eligibility")
# 专家 → 独立消息通道（上下文隔离，覆盖写语义）
EXPERT_MESSAGE_KEYS: dict[str, str] = {
    "policy_qa": "qa_messages",
    "compare": "cmp_messages",
    "eligibility": "eli_messages",
}


class Subtask(BaseModel):
    """一个可执行子任务。"""

    agent: Literal["policy_qa", "compare", "eligibility"] = Field(
        description="承接该子任务的专家"
    )
    goal: str = Field(description="要查什么 / 要算什么，一句话说清")
    hint: str = Field(description="检索关键词、制度名、版本或已知条款线索")
    confidence: float = Field(description="该子任务需要执行的置信度，0~1")


class SupervisorPlan(BaseModel):
    """调度决策（结构化输出）。"""

    intent: Literal["fact", "compare", "eligibility", "clarify", "refuse"] = Field(
        description="用户问题的意图类别"
    )
    confidence: float = Field(description="意图与路由的综合置信度，0~1")
    subtasks: list[Subtask] = Field(description="完整任务分解，可为空")
    next: Literal["policy_qa", "compare", "eligibility", "synthesize", "clarify"] = Field(
        description="下一步立即执行的节点"
    )
    done: bool = Field(description="证据是否已足以收口")
    reason: str = Field(description="一句中文说明分派或收口理由")


def _truncate(text: str, limit: int) -> str:
    text = str(text or "").strip()
    return text[:limit] if len(text) > limit else text


def _review_brief(review: dict | None) -> str:
    """把 critic 的结构化反馈压缩成给调度模型看的短文本。"""
    if not review:
        return "（无）"
    if not isinstance(review, dict):
        return _truncate(str(review), 500)

    parts: list[str] = [f"上一版答案未通过核查（verdict={review.get('verdict')}）"]
    agent = review.get("agent")
    if agent:
        parts.append(f"原专家：{agent}")

    bad = review.get("bad_citations") or []
    if bad:
        labels = []
        for item in bad[:5]:
            if isinstance(item, dict):
                labels.append(f"{item.get('doc_id', '?')} 第{item.get('article_no', '?')}条")
            else:
                labels.append(str(item))
        parts.append(f"伪造/失效引用 {len(bad)} 条：" + "、".join(labels))

    numbers = review.get("numeric_mismatches") or []
    if numbers:
        raws = []
        for item in numbers[:8]:
            if isinstance(item, dict):
                raws.append(str(item.get("raw") or item.get("value") or ""))
            else:
                raws.append(str(item))
        parts.append("数值不符：" + "、".join(raws))

    unsupported = review.get("unsupported") or []
    if unsupported:
        samples = []
        for item in unsupported[:3]:
            if isinstance(item, dict):
                samples.append(str(item.get("sentence") or "")[:80])
            else:
                samples.append(str(item)[:80])
        parts.append(f"无支撑句 {len(unsupported)} 句：" + "；".join(samples))

    return _truncate("；".join(parts), 1200)


def decide_plan(
    question: str,
    user_scope: list[str] | None = None,
    history_summary: str = "",
    evidence_summary: str = "",
    dispatch_round: int = 0,
    review: dict | None = None,
) -> SupervisorPlan:
    """调用轻量模型生成调度计划；模型失败时保守降级到事实条款专家。"""
    dispatch_round = max(0, int(dispatch_round or 0))
    rounds_left = max(0, settings.max_dispatch_round - dispatch_round)
    scope_text = "、".join(user_scope or []) or "employee（默认）"

    prompt = (
        f"{SUPERVISOR_PROMPT}\n\n"
        "【本轮输入】\n"
        f"用户问题：{_truncate(question, 2000)}\n"
        f"用户范围：{scope_text}\n"
        f"历史摘要：{_truncate(history_summary, 1000) or '（无）'}\n"
        f"已有证据摘要：{_truncate(evidence_summary, 500) or '（暂无证据）'}\n"
        f"已完成派发轮次：{dispatch_round} / 上限 {settings.max_dispatch_round}"
        f"（剩余 {rounds_left} 轮）\n"
        f"核查反馈：{_review_brief(review)}\n"
    )

    try:
        raw = get_structured_llm("light", SupervisorPlan).invoke(prompt)
        plan = raw if isinstance(raw, SupervisorPlan) else SupervisorPlan.model_validate(raw)
    except Exception as exc:  # LLM 超时、function calling 不兼容、输出不合法等
        logger.warning("调度模型失败，保守降级为事实条款专家：%s", exc)
        # 降级原则：路由不确定时回到最通用的取证专家，而不是直接收口或拒答；
        # 后续由 supervisor 再次决策，且仍受 max_dispatch_round 硬上限约束。
        return SupervisorPlan(
            intent="fact",
            confidence=0.0,
            subtasks=[
                Subtask(
                    agent="policy_qa",
                    goal="定位并摘录与用户问题直接相关的制度条款",
                    hint="优先检索制度名称与核心关键词；若涉及计算或对比，先返回相关条款依据",
                    confidence=0.3,
                )
            ],
            next="policy_qa",
            done=False,
            reason=f"调度模型调用失败，降级派发事实条款专家补证：{_truncate(str(exc), 200)}",
        )

    # 核查打回场景：只要还有派发额度，就派回原专家补证，避免直接重新 synthesize 蒙混过关。
    if review and review.get("verdict") == "reject" and dispatch_round < settings.max_dispatch_round:
        target = review.get("agent") or "policy_qa"
        if target not in EXPERT_AGENTS:
            target = "policy_qa"
        issues = _truncate(_review_brief(review), 300)
        matched = [task for task in plan.subtasks if task.agent == target]
        if matched:
            matched = [
                task.model_copy(update={"hint": _truncate(f"{task.hint}；核查问题：{issues}", 500)})
                for task in matched
            ]
        else:
            matched = [
                Subtask(
                    agent=target,
                    goal="根据核查反馈补充证据并修正结论",
                    hint=f"核查问题：{issues}",
                    confidence=0.5,
                )
            ]
        plan = plan.model_copy(
            update={
                "next": target,
                "done": False,
                "subtasks": matched,
                "reason": _truncate(f"{plan.reason}；核查打回，派回 {target} 补证", 500),
            }
        )

    return plan


def _evidence_from_state(state: SupervisorState) -> tuple[DocStore, str, int]:
    """把 state 证据装进临时 DocStore，得到与 synthesize 同口径的摘要。"""
    ds = DocStore()
    evidence = state.get("evidence") or []
    try:
        ds.add_evidence([item for item in evidence if isinstance(item, dict)])
    except Exception as exc:  # 证据结构异常不应中断调度
        logger.warning("证据摘要构建失败：%s", exc)
    return ds, ds.summary(), len(ds.get_evidence())


def _route_event(plan: SupervisorPlan, dispatch_round: int) -> dict:
    return {
        "event": "route",
        "intent": plan.intent,
        "confidence": plan.confidence,
        "subtasks": [task.model_dump() for task in plan.subtasks],
        "next": plan.next,
        "done": plan.done,
        "reason": plan.reason,
        "dispatch_round": dispatch_round,
    }


def supervisor_node(state: SupervisorState) -> Command:
    """调度中枢节点：读黑板摘要 → 决策 → Command(goto=..., update=...)。"""
    started = time.perf_counter()
    question = str(state.get("question") or "").strip()
    user_scope = state.get("user_scope") or []
    history_summary = str(state.get("history_summary") or "")
    dispatch_round = max(0, int(state.get("dispatch_round") or 0))

    raw_review = state.get("review")
    review = dict(raw_review) if isinstance(raw_review, dict) and raw_review else None
    if review is not None and state.get("next_agent"):
        # 记录产生上一版答案的专家，供打回时定向补证
        review.setdefault("agent", state.get("next_agent"))

    _, evidence_summary, evidence_count = _evidence_from_state(state)

    plan = decide_plan(
        question,
        user_scope=user_scope,
        history_summary=history_summary,
        evidence_summary=evidence_summary,
        dispatch_round=dispatch_round,
        review=review,
    )

    # ---------------- 硬上限：达到最大派发轮次后强制收口 ----------------
    forced = False
    if dispatch_round >= settings.max_dispatch_round and not plan.done:
        forced = True
        if evidence_count > 0:
            plan = plan.model_copy(
                update={
                    "next": "synthesize",
                    "done": True,
                    "reason": _truncate(
                        f"已达到最大派发轮次 {settings.max_dispatch_round}，"
                        f"停止补证并基于现有证据收口；原计划：{plan.reason}",
                        500,
                    ),
                }
            )
        else:
            plan = plan.model_copy(
                update={
                    "intent": "clarify",
                    "next": "clarify",
                    "done": True,
                    "reason": _truncate(
                        f"已达到最大派发轮次 {settings.max_dispatch_round} 且没有可用证据，"
                        f"降级为澄清/未明确；原计划：{plan.reason}",
                        500,
                    ),
                }
            )

    # 全程只 emit 一次最终路由（黑板未挂载时静默）
    emit(
        "route",
        intent=plan.intent,
        confidence=plan.confidence,
        subtasks=[task.model_dump() for task in plan.subtasks],
        next=plan.next,
        done=plan.done,
        forced=forced,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )

    route_event = _route_event(plan, dispatch_round)
    if forced:
        route_event["forced"] = True
    trace = [route_event]
    base_update = {
        "plan": plan.model_dump(),
        "review": {},  # review 是一次性打回信号，消费后清空，避免下一轮重复补证
        "trace": trace,
    }

    # ---------------- 澄清 / 拒答优先 ----------------
    if plan.intent in ("clarify", "refuse") or plan.next == "clarify":
        if plan.next != "clarify":
            plan = plan.model_copy(update={"intent": plan.intent if plan.intent == "refuse" else "clarify", "next": "clarify"})
            trace[0] = _route_event(plan, dispatch_round)
            trace[0]["forced"] = forced
            base_update["plan"] = plan.model_dump()
        base_update["next_agent"] = "clarify"
        return Command(goto="clarify", update=base_update)

    # ---------------- 派发专家 ----------------
    if plan.next in EXPERT_AGENTS:
        target = plan.next
        subtasks = [task for task in plan.subtasks if task.agent == target]
        if not subtasks:
            # LLM 给了 next 但没给对应子任务：用问题本身兜底，保证专家有明确目标
            subtasks = [
                Subtask(
                    agent=target,
                    goal=question or "检索并回答用户问题",
                    hint=plan.reason,
                    confidence=plan.confidence,
                )
            ]
        packages = [
            {
                "role": "user",
                "content": (
                    f"任务目标：{task.goal}\n"
                    f"提示：{task.hint}\n"
                    f"已有证据：{evidence_summary}\n"
                    f"用户问题：{question}"
                ),
            }
            for task in subtasks
        ]
        message_key = EXPERT_MESSAGE_KEYS[target]
        base_update.update(
            {
                "next_agent": target,
                "dispatch_round": dispatch_round + 1,
                "trace": trace
                + [
                    {
                        "event": "agent_start",
                        "agent": target,
                        "goal": subtasks[0].goal,
                        "dispatch_round": dispatch_round + 1,
                    }
                ],
                message_key: packages,
            }
        )
        return Command(goto=target, update=base_update)

    # ---------------- 收口：synthesize ----------------
    if evidence_count == 0:
        # 绝不允许在没有证据时让 synthesize 凭空生成
        plan = plan.model_copy(
            update={
                "intent": "clarify",
                "next": "clarify",
                "done": True,
                "reason": "没有任何可用证据，降级为澄清/未明确，避免编造答案",
            }
        )
        trace[0] = _route_event(plan, dispatch_round)
        trace[0]["forced"] = True
        base_update["plan"] = plan.model_dump()
        base_update["next_agent"] = "clarify"
        return Command(goto="clarify", update=base_update)

    base_update["next_agent"] = "synthesize"
    return Command(goto="synthesize", update=base_update)


def clarify_node(state: SupervisorState) -> dict:
    """澄清 / 拒答节点：不调用 LLM，按 plan.intent 与已有证据生成中文说明。"""
    plan = state.get("plan") if isinstance(state.get("plan"), dict) else {}
    intent = str(plan.get("intent") or "clarify")
    evidence = [item for item in (state.get("evidence") or []) if isinstance(item, dict)]
    review = state.get("review") if isinstance(state.get("review"), dict) else {}
    draft = str(state.get("final_answer") or "").strip()

    citations: list[dict] = []
    labels: list[str] = []
    for index, item in enumerate(evidence[:8], 1):
        title = item.get("title") or item.get("doc_id") or "制度"
        article = item.get("article_no") or "?"
        label = f"《{title}》第{article}条"
        if label not in labels:
            labels.append(label)
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

    degraded = bool(draft) and review.get("verdict") == "reject"
    if intent == "refuse" and not degraded:
        answer = (
            "抱歉，该问题超出公司制度范围，现有制度知识库中没有相关依据，无法给出回答。\n"
            "建议咨询人力资源部门或相关业务负责人确认；如问题涉及外部法律法规，请以正式法律意见为准。"
        )
        citations = []
        labels = []
    elif degraded:
        # critic 超限降级：保留已核实部分，缺口显式标注
        answer = (
            "审核未能完全通过，以下为部分回答的降级输出，请谨慎参考：\n\n"
            f"{draft}\n\n"
            "【制度未明确】上述内容中仍有引用或数值未通过核查，缺失依据的部分制度未明确，"
            "请以制度原文及人力资源解释为准。"
        )
        if labels:
            answer += "\n\n已检索到的相关条款：" + "、".join(labels)
    elif evidence:
        answer = (
            "信息不足或制度未明确，暂时无法给出确定答案。\n"
            "以下是知识库中与问题最相关的条款，供人工判断：\n"
            + "\n".join(f"- {label}" for label in labels)
            + "\n\n如果上述条款不适用于你的情况，说明制度未覆盖该情形；"
            "建议补充具体条件（如工龄、岗位、时间、制度版本）后再提问，或向人力资源部门确认。"
        )
    else:
        answer = (
            "信息不足或制度未明确：现有知识库中未检索到可支撑该问题的制度条款。\n"
            "请补充具体条件（如涉及哪项制度、工龄/岗位/时间等），或向人力资源部门确认。"
        )

    trace = [
        {
            "event": "clarify",
            "intent": intent,
            "degraded": degraded,
            "evidence_count": len(evidence),
        }
    ]
    return {"final_answer": answer, "citations": citations, "trace": trace}
