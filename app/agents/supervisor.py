"""supervisor 调度中枢：意图分类 → 任务分解 → 专家分派 → 结束判定。

控制流用 Command 显式跳转；所有回环受 settings.max_dispatch_round 硬约束。
supervisor 只做路由与收口判断，不直接回答用户问题。
"""
from __future__ import annotations

import re
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

    intent: Literal[
        "fact", "compare", "eligibility", "clarify", "refuse", "smalltalk", "out_of_scope"
    ] = Field(description="用户问题的意图类别")
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


# 纯寒暄/感谢/告别/询问助手身份的整句匹配。用 ^...$ 全匹配，避免把
# “你好，我想问年假”这类带寒暄前缀的制度问题误判成闲聊。
_SMALLTALK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "greeting",
        re.compile(
            r"^(你好|您好|嗨|哈喽|hello|hi|hey|早上好|上午好|下午好|晚上好|在吗|在不在)"
            r"(\s+there)?[\s,，!！。.、?？~～呀啊哦啦哈]*$",
            re.IGNORECASE,
        ),
    ),
    (
        "thanks",
        re.compile(
            r"^(谢谢|感谢|多谢|thanks|thank you|thanks a lot)"
            r"[\s,，!！。.、?？~～呀啊哦啦哈你您]*$",
            re.IGNORECASE,
        ),
    ),
    (
        "bye",
        re.compile(
            r"^(再见|拜拜|bye|goodbye)[\s,，!！。.、?？~～呀啊哦]*$",
            re.IGNORECASE,
        ),
    ),
    (
        "identity",
        re.compile(
            r"^(你是谁|你叫什么|你叫什么名字|介绍一下你自己|你是干嘛的|你能做什么|"
            r"你会什么|你能帮我做什么|你可以做什么|你是什么模型|你用的什么模型|"
            r"你是gpt吗|你是ai吗|你是什么ai|你是机器人吗)"
            r"[\s,，!！。.、?？~～呀啊哦]*$",
            re.IGNORECASE,
        ),
    ),
    (
        "ack",
        re.compile(
            r"^(好的|好|嗯|嗯嗯|收到|明白了|知道了|ok|okay)"
            r"[\s,，!！。.、?？~～呀啊哦]*$",
            re.IGNORECASE,
        ),
    ),
)


def _smalltalk_kind(question: str) -> str:
    """返回 greeting/thanks/bye/identity/ack；不是纯寒暄返回空串。"""
    text = str(question or "").strip()
    if not text:
        return ""
    for kind, pattern in _SMALLTALK_PATTERNS:
        if pattern.match(text):
            return kind
    return ""


def smalltalk_reply(question: str) -> str:
    """寒暄场景的固定短回复，不调用 LLM、不检索知识库。"""
    kind = _smalltalk_kind(question)
    if kind == "thanks":
        return "不客气！如果还有其他制度问题，随时问我。"
    if kind == "bye":
        return "再见！有制度问题随时来问。"
    if kind == "identity":
        return (
            "我是 PolicyMind，一个基于企业制度知识库的问答助手，"
            "可以回答考勤、休假、报销、权限等制度问题，并给出条款引用。"
        )
    if kind == "ack":
        return "好的，如果还有制度问题随时问我。"
    return (
        "你好！我是 PolicyMind，可以帮你查询企业制度，"
        "比如考勤、休假、报销、权限等。请直接输入你的问题。"
    )


# 制度领域关键词：命中任意一个就不再按“非制度问题”拦截，交给正常调度/检索。
# 这样“公司在高温天气有哪些补贴”仍会按制度问题处理。
_POLICY_HINTS = re.compile(
    r"制度|规定|办法|细则|条例|政策|规范|流程|公司|企业|单位|部门|员工|职工|同事|"
    r"领导|上级|下属|岗位|职级|职位|编制|招聘|入职|转正|试用期|离职|辞职|解聘|"
    r"合同|续签|培训|晋升|调岗|考核|绩效|考勤|打卡|迟到|早退|旷工|请假|休假|"
    r"年假|病假|事假|婚假|产假|陪产假|丧假|调休|加班|工时|作息|上班|下班|"
    r"工作时间|排班|值班|工资|薪资|薪酬|奖金|提成|补贴|津贴|报销|差旅|出差|"
    r"福利|社保|公积金|医保|体检|退休|食堂|宿舍|停车|门禁|办公用品|权限|审批|"
    r"签批|用印|档案|保密|违纪|处分|奖惩|劳动|聘用|工资条|个税|年终奖|"
    r"入职培训|职业道德|行为规范"
)

# 非制度领域：只有整句不带制度关键词时才判定为 out_of_scope。
_OUT_OF_SCOPE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "weather",
        re.compile(r"天气|气温|下雨|下雪|台风|空气质量|雾霾|冷空气|weather", re.IGNORECASE),
    ),
    (
        "time",
        re.compile(
            r"现在几点|几点了|当前时间|今天几号|今天星期几|今天周几|"
            r"现在是什么时间|今天的日期|日期是"
        ),
    ),
    (
        "news",
        re.compile(
            r"新闻|头条|热搜|时事|最近发生|股票|股市|基金|彩票|汇率|房价|黄金价格"
        ),
    ),
    (
        "entertainment",
        re.compile(
            r"笑话|电影|电视剧|综艺|音乐|歌曲|游戏|小说|故事|诗|星座|运势|"
            r"明星|八卦|动漫"
        ),
    ),
    (
        "general",
        re.compile(
            r"量子|物理|化学|历史|地理|宇宙|天文|生物|医学|英语|数学|编程|代码|"
            r"python|javascript|java|c\+\+|翻译|作文|作业|论文怎么写|养生|减肥|"
            r"健身|旅游|景点|美食|菜谱|做饭|买车|买房|贷款|手机推荐|电脑推荐",
            re.IGNORECASE,
        ),
    ),
    (
        "math",
        re.compile(
            r"^\s*[\d\s.+\-*/×÷=()（）]+\s*(等于)?\s*(几|多少)?\s*[？?]?\s*$"
        ),
    ),
)


def _out_of_scope_kind(question: str) -> str:
    """非制度问题识别：命中非制度主题且全文没有制度关键词时返回类别。"""
    text = str(question or "").strip()
    if not text or _POLICY_HINTS.search(text):
        return ""
    for kind, pattern in _OUT_OF_SCOPE_PATTERNS:
        if pattern.search(text):
            return kind
    return ""


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
        logger.warning("调度模型失败，按问题是否属于制度领域降级处理：%s", exc)
        # 降级原则：不再“无论什么问题都派专家检索”。
        # 1) 明显非制度问题 → 直接说明不检索；
        # 2) 没有任何制度关键词 → 先澄清，避免盲目检索；
        # 3) 有制度关键词 → 回到最通用的事实条款专家补证。
        out_kind = _out_of_scope_kind(question)
        if out_kind:
            return SupervisorPlan(
                intent="out_of_scope",
                confidence=0.0,
                subtasks=[],
                next="clarify",
                done=True,
                reason=f"调度模型调用失败，且问题为非制度主题（{out_kind}），不检索",
            )
        if not _POLICY_HINTS.search(question):
            return SupervisorPlan(
                intent="clarify",
                confidence=0.0,
                subtasks=[],
                next="clarify",
                done=True,
                reason=(
                    "调度模型调用失败，且问题未包含制度关键词，先请用户补充信息，"
                    f"不盲目检索：{_truncate(str(exc), 200)}"
                ),
            )
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

    # 纯寒暄/感谢/告别/询问助手身份：直接短回复，不调用调度模型、不检索知识库。
    # 用整句正则匹配，用户问“你好，年假怎么算”仍会正常进入制度问答。
    smalltalk_kind = _smalltalk_kind(question)
    out_of_scope_kind = "" if smalltalk_kind else _out_of_scope_kind(question)
    forced = False
    if smalltalk_kind:
        plan = SupervisorPlan(
            intent="smalltalk",
            confidence=1.0,
            subtasks=[],
            next="clarify",
            done=True,
            reason=f"纯寒暄/闲聊（{smalltalk_kind}），不检索制度知识库",
        )
    elif out_of_scope_kind:
        plan = SupervisorPlan(
            intent="out_of_scope",
            confidence=1.0,
            subtasks=[],
            next="clarify",
            done=True,
            reason=f"非制度问题（{out_of_scope_kind}），不检索知识库",
        )
    else:
        plan = decide_plan(
            question,
            user_scope=user_scope,
            history_summary=history_summary,
            evidence_summary=evidence_summary,
            dispatch_round=dispatch_round,
            review=review,
        )

        # ---------------- 硬上限：达到最大派发轮次后强制收口 ----------------
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

    # ---------------- 寒暄 / 非制度问题 / 澄清 / 拒答优先 ----------------
    if plan.intent in ("smalltalk", "out_of_scope", "clarify", "refuse") or plan.next == "clarify":
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
    for index, item in enumerate(evidence[:5], 1):
        title = item.get("title") or item.get("doc_id") or "制度"
        article = item.get("article_no") or "?"
        label = f"《{title}》第{article}条"
        # 答案正文最多列 3 条，避免证据多时回复臃肿；完整引用仍在右侧引用面板。
        if label not in labels and len(labels) < 3:
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
    if intent == "smalltalk":
        # 寒暄不涉及制度内容，直接短回复，不允许出现引用或“未检索到”措辞。
        answer = smalltalk_reply(str(state.get("question") or ""))
        citations = []
        labels = []
    elif intent == "out_of_scope":
        # 天气/新闻/娱乐/通用知识等非制度问题：不检索，直接说明能力边界。
        answer = (
            "这个问题和公司制度无关，我不提供天气、新闻、娱乐等通用问答。\n"
            "你可以问我考勤、休假、报销、权限、绩效等制度问题。"
        )
        citations = []
        labels = []
    elif intent == "refuse" and not degraded:
        answer = (
            "抱歉，这个问题超出公司制度范围，知识库中没有相关依据。\n"
            "建议咨询人力资源部门或相关业务负责人确认。"
        )
        citations = []
        labels = []
    elif degraded:
        # critic 超限降级：保留已核实部分，缺口显式标注
        answer = (
            "以下是部分可用回答，引用或数值尚未完全通过核查，请谨慎参考：\n\n"
            f"{draft}\n\n"
            "缺失或存疑的部分请以制度原文及人力资源解释为准。"
        )
        if labels:
            answer += "\n\n相关条款：" + "、".join(labels)
    elif evidence:
        answer = (
            "现有制度中没有找到足够明确的依据。\n"
            "最相关的条款：" + "、".join(labels) + "\n"
            "建议补充具体条件（如制度名称、工龄、岗位、时间）后再提问，或向人力资源部门确认。"
        )
    else:
        answer = (
            "现有知识库中没有检索到可支撑该问题的制度条款。\n"
            "可以补充制度名称或具体条件（工龄、岗位、时间等）后再问，或向人力资源部门确认。"
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
