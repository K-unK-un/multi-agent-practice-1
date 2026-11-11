"""规则表抽取：把"阶梯 / 比例 / 上限 / 下限 / 门槛"类条款抽成结构化行。

纪律（对应《风险与建议》1.3）：
- 只抽有明确数值依据的条款；无依据、纯原则性表述一律不抽；
- basis_text 必须是条款原文摘录，事后校验它确实出现在源条款中，否则丢弃；
- conclusion.value 用正则回原文二次校验，不一致 → needs_review=True（保留待人工核对）；
- confidence < 0.7 丢弃；
- 单批 LLM 异常只记 warning，返回已抽取结果，不中断建库。
"""
from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from app.core.logger import get_logger
from app.llm.client import get_structured_llm

logger = get_logger("ingest.rule_extract")

BATCH_SIZE = 5              # 每批 ≤5 条条款
MIN_CONFIDENCE = 0.7        # 低于该置信度丢弃
NUMERIC_TOLERANCE = 0.01    # 数值二次校验允许误差
MAX_PROMPT_CLAUSE_CHARS = 1500  # 单条条款进 prompt 的截断长度，控制 token


class RuleSchema(BaseModel):
    """单条规则。condition/conclusion 用 JSON 承载，便于直接落 MySQL JSON 列。"""

    model_config = ConfigDict(extra="ignore")

    article_no: str = Field(description="条款号，必须与输入一致，如 '12'")
    condition: dict = Field(
        description='条件区间，如 {"工龄": {"min": 10, "max": 20, "unit": "年"}}'
    )
    conclusion: dict = Field(
        description='结论，如 {"value": 10, "unit": "天", "text": "年休假10天"}'
    )
    basis_text: str = Field(description="条款原文摘录，不得改写")
    confidence: float = Field(description="0~1 置信度")


class RuleListSchema(BaseModel):
    """结构化输出包一层 list：DeepSeek function_calling 对顶层数组支持不稳。"""

    model_config = ConfigDict(extra="ignore")

    rules: list[RuleSchema] = Field(default_factory=list)


_SYSTEM_PROMPT = """你是企业规章制度的结构化抽取器。请从给定条款中抽取"可计算的规则"。

只抽取同时满足以下条件的条款：
1. 明确包含数值门槛 / 阶梯区间 / 比例 / 上限 / 下限 / 资格条件；
2. 结论是可计算的数值（天数、金额、比例、次数、年限等）。

不要抽取：原则性表述、职责描述、流程说明、只有"按规定执行"而没有具体数值的条款。

输出字段要求：
- article_no：必须与输入条款号完全一致（只填 "12" 这类标识，不要带"第/条"）。
- condition：JSON 对象。条件区间示例：
  {"工龄": {"min": 10, "max": 20, "unit": "年"}}
  单侧条件只写 min 或 max；无条件门槛时给 {}。
- conclusion：必须含 value（数字）、unit（单位）、text（结论原文简述）。
  百分比请使用原文数字，例如原文"80%" → value=80, unit="%"（不要写成 0.8）。
- basis_text：必须从输入【原文】中原样摘录支撑该规则的句子（可截取，禁止改写）。
- confidence：0~1，依据越明确越高。

如果条款没有明确数值依据，宁可不输出，也不要猜测。"""


def _format_batch(batch: list[dict]) -> str:
    parts: list[str] = []
    for clause in batch:
        content = (clause.get("content") or "").strip()
        if len(content) > MAX_PROMPT_CLAUSE_CHARS:
            content = content[:MAX_PROMPT_CLAUSE_CHARS] + "……"
        parts.append(
            f"【条款号】{clause.get('article_no')}\n"
            f"【标题】{clause.get('title') or ''}\n"
            f"【原文】\n{content}"
        )
    return "\n\n---\n\n".join(parts)


def _squash(text: str) -> str:
    """去掉全部空白，用于 basis_text 是否为原文摘录的宽松比对。"""
    return re.sub(r"\s+", "", text or "")


def _extract_numbers(text: str) -> list[float]:
    """提取原文中的阿拉伯数字，用于 value 二次校验。

    百分比按原文数字比对（原文 "80%" → 80），不自动折算成 0.8：
    prompt 已要求 value 使用原文数字，若模型擅自归一化成 0.8，应判为不一致并
    needs_review，避免把 "0.8%" 这种差 100 倍的结论静默入库。
    """
    return [float(match.group()) for match in re.finditer(r"\d+(?:\.\d+)?", text or "")]


def _coerce_number(raw) -> float | None:
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        match = re.search(r"-?\d+(?:\.\d+)?", raw.strip().replace("％", "%"))
        if match:
            return float(match.group())
    return None


_CN_DIGITS = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_CN_UNITS = {"十": 10, "百": 100}


def _cn_to_int(value: str) -> int | None:
    """中文数字转 int（支持十/百）；纯阿拉伯数字直接转换。"""
    if value.isdigit():
        return int(value)
    total = 0
    current = 0
    for char in value:
        if char in _CN_DIGITS:
            current = _CN_DIGITS[char]
        elif char in _CN_UNITS:
            total += (current or 1) * _CN_UNITS[char]
            current = 0
        else:
            return None
    return total + current


def _normalize_article_no(raw: str) -> str:
    """把 "第12条" / "第十二条" 统一成 "12"，便于与切分结果对齐。"""
    text = (raw or "").strip()
    match = re.search(r"第?\s*([一二三四五六七八九十百零〇\d]+)\s*条?", text)
    if match:
        token = match.group(1)
        number = _cn_to_int(token)
        return str(number) if number is not None else token
    return text


def _validate_rule(rule: RuleSchema, clause: dict) -> dict | None:
    """把 LLM 输出校验/转换为可落库行；不通过返回 None。"""
    if rule.confidence < MIN_CONFIDENCE:
        logger.debug(
            "丢弃低置信度规则 article=%s confidence=%s",
            rule.article_no,
            rule.confidence,
        )
        return None

    basis_text = (rule.basis_text or "").strip()
    if not basis_text:
        logger.debug("丢弃无依据规则 article=%s", rule.article_no)
        return None

    # basis_text 必须是原文摘录（去空白后包含判定，容忍模型复制时多一个空格）
    source_content = clause.get("content") or ""
    if _squash(basis_text) not in _squash(source_content):
        logger.warning(
            "丢弃依据不在原文中的规则 doc=%s article=%s basis=%s",
            clause.get("doc_id"),
            rule.article_no,
            basis_text[:60],
        )
        return None

    conclusion = dict(rule.conclusion or {})
    value = _coerce_number(conclusion.get("value"))
    if value is None:
        logger.debug(
            "丢弃无有效 value 的规则 article=%s conclusion=%s",
            rule.article_no,
            conclusion,
        )
        return None

    unit = str(conclusion.get("unit") or "")
    # 百分比兜底：value 写成 "80%" 或 conclusion.unit 缺失时补单位
    if not unit and "%" in str(conclusion.get("value") or ""):
        unit = "%"

    needs_review = not any(
        abs(value - candidate) <= NUMERIC_TOLERANCE
        for candidate in _extract_numbers(basis_text)
    )
    if needs_review:
        logger.warning(
            "规则数值二次校验不一致 doc=%s article=%s value=%s basis=%s",
            clause.get("doc_id"),
            rule.article_no,
            value,
            basis_text[:60],
        )

    condition = rule.condition if isinstance(rule.condition, dict) else {}
    return {
        "doc_id": clause.get("doc_id"),
        "article_no": clause.get("article_no"),
        "condition": condition,
        "conclusion": {
            "value": value,
            "unit": unit,
            "text": str(conclusion.get("text") or ""),
        },
        "value": value,
        "unit": unit,
        "basis_text": basis_text,
        "needs_review": needs_review,
    }


def extract_rules(parent_clauses: list[dict]) -> list[dict]:
    """按批抽取规则。LLM 异常返回已抽结果（可能为空），不抛出。"""
    clauses = [c for c in (parent_clauses or []) if (c.get("content") or "").strip()]
    if not clauses:
        return []

    try:
        structured_llm = get_structured_llm("main", RuleListSchema)
    except Exception as exc:
        logger.warning("规则抽取模型初始化失败，跳过: %s", exc)
        return []

    # 按 doc_id 分组后再分批：不同制度的条款号会重复（如都有"第1条"），
    # 混批会导致模型返回的 article_no 无法确定属于哪份文档。
    groups: dict[str, list[dict]] = {}
    for clause in clauses:
        groups.setdefault(str(clause.get("doc_id") or ""), []).append(clause)

    rows: list[dict] = []
    seen: set[tuple] = set()

    for group in groups.values():
        for offset in range(0, len(group), BATCH_SIZE):
            batch = group[offset: offset + BATCH_SIZE]
            by_article = {str(c.get("article_no")): c for c in batch}
            messages = [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=_format_batch(batch)),
            ]
            try:
                result = structured_llm.invoke(messages)
            except Exception as exc:
                logger.warning(
                    "规则抽取批次失败（offset=%s, size=%s），跳过: %s",
                    offset,
                    len(batch),
                    exc,
                )
                continue

            rules = getattr(result, "rules", None)
            if rules is None and isinstance(result, dict):
                try:
                    rules = [RuleSchema(**item) for item in result.get("rules", [])]
                except Exception as exc:
                    logger.warning("规则抽取结果解析失败: %s", exc)
                    continue
            if not rules:
                continue

            for rule in rules:
                article_key = _normalize_article_no(rule.article_no)
                clause = by_article.get(article_key)
                if clause is None:
                    # 模型可能返回 "第12条" / "12"，再宽松匹配一次
                    clause = by_article.get(str(rule.article_no).strip())
                if clause is None:
                    logger.debug("模型返回了批次外条款号，忽略: %s", rule.article_no)
                    continue

                row = _validate_rule(rule, clause)
                if row is None:
                    continue
                dedup_key = (
                    row["doc_id"],
                    row["article_no"],
                    round(row["value"], 4),
                    row["unit"],
                    row["basis_text"][:60],
                )
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                rows.append(row)

    logger.info("规则抽取完成：输入条款 %s，入库规则 %s", len(clauses), len(rows))
    return rows
