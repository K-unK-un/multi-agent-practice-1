"""critic 核查 Agent（以规则为主，不参与生成、不改写答案）。

三步核查：
1. 引用真实性：citations 每条回 MySQL 按 doc_id + article_no 命中；
2. 数值一致性：final_answer 的数值必须出现在引用条款原文中；
3. 逐句支撑（可选）：settings.enable_semantic_check=True 时用句向量相似度检查。

判定 pass → END；reject 且未超 review 上限 → 回 supervisor 由原专家补证；
reject 超限 → 降级到 clarify（部分回答 + 制度未明确）。
"""
from __future__ import annotations

import re
import time
from typing import Any

from langgraph.graph import END
from langgraph.types import Command

from app.core.blackboard import emit
from app.core.config import settings
from app.core.logger import get_logger
from app.graph.state import SupervisorState
from app.llm.client import apply_hf_offline_env
from app.store.mysql import get_mysql

logger = get_logger("agents.critic")

# 数字（含千分位/小数）与可选百分号
_NUM_RE = re.compile(r"(?<![\d.])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(\s*[%％])?")
# 日期整体排除：2025-01-01 / 2025/1/1 / 2025年1月1日 中的数字不作数值一致性比对
_DATE_RE = re.compile(r"\d{4}\s*[-/年]\s*\d{1,2}\s*[-/月]\s*\d{1,2}\s*日?")
_ARTICLE_SUFFIXES = ("条", "款", "章", "节", "项", "编")
_SENT_SPLIT_RE = re.compile(r"[。！？!?；;\n]+")
_MAX_SENTENCES = 40


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------

def _row_content(row: Any) -> str:
    """兼容 dict / ORM 对象两种 get_clause 返回形态。"""
    if row is None:
        return ""
    if isinstance(row, dict):
        return str(row.get("content") or row.get("text") or "")
    return str(getattr(row, "content", "") or getattr(row, "text", "") or "")


def _is_article_reference(text: str, start: int, end: int) -> bool:
    """判断数字是否是条款号（第X条 / X款 / X章…），这类数字不参与数值比对。"""
    before = text[max(0, start - 4) : start].rstrip()
    after = text[end : end + 4].lstrip()
    if before.endswith("第"):
        return True
    return after.startswith(_ARTICLE_SUFFIXES)


def _is_citation_marker(text: str, start: int, end: int) -> bool:
    """排除【E1】这类引用编号中的数字。"""
    return text[max(0, start - 2) : start] == "【E" and text[end : end + 1] == "】"


def _is_year_like(text: str, start: int, end: int, raw: str) -> bool:
    """排除 2025年 / 2025版 / 2022 年度这类年份与版本号。"""
    digits = raw.replace(",", "")
    if "." in digits or len(digits) != 4:
        return False
    try:
        value = int(digits)
    except ValueError:
        return False
    if not 1900 <= value <= 2100:
        return False
    before = text[max(0, start - 2) : start]
    after = text[end : end + 3].lstrip()
    return before.endswith("版") or after.startswith(("年", "版", "年度"))


def _num_key(raw: str) -> str:
    """数值归一化键：去千分位、统一小数表示；百分比与裸数字共用同一数值键。"""
    try:
        value = float(raw.replace(",", ""))
    except ValueError:
        return raw
    if value == 0:
        return "0"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _extract_numbers(text: str) -> dict[str, str]:
    """提取文本中的数值：{归一化数值: 首次出现的原文}。"""
    found: dict[str, str] = {}
    text = text or ""
    date_spans = [(m.start(), m.end()) for m in _DATE_RE.finditer(text)]
    for match in _NUM_RE.finditer(text):
        start, end = match.start(1), match.end(1)
        raw = match.group(1)
        if any(span_start <= start and end <= span_end for span_start, span_end in date_spans):
            continue
        if _is_citation_marker(text, start, end):
            continue
        if _is_article_reference(text, start, end):
            continue
        if _is_year_like(text, start, end, raw):
            continue
        display = raw + ("%" if match.group(2) else "")
        try:
            key = _num_key(raw)
        except Exception:
            continue
        found.setdefault(key, display)
    return found


def _split_sentences(text: str) -> list[str]:
    sentences = [part.strip() for part in _SENT_SPLIT_RE.split(text or "")]
    return [sentence for sentence in sentences if len(sentence) >= 8][:_MAX_SENTENCES]


# ---------------------------------------------------------------------------
# 三步核查
# ---------------------------------------------------------------------------

def _check_citations(citations: list[dict], notes: list[str]) -> tuple[list[dict], list[str]]:
    """步骤 1：引用真实性。返回 (bad_citations, 有效引用条款原文列表)。"""
    bad: list[dict] = []
    clause_texts: list[str] = []
    try:
        mysql = get_mysql()
    except Exception as exc:
        logger.warning("引用核查降级：MySQL 不可用：%s", exc)
        notes.append(f"引用真实性核查降级：MySQL 不可用（{exc}），未核验引用")
        return bad, clause_texts

    for citation in citations:
        if not isinstance(citation, dict):
            bad.append({"doc_id": "", "article_no": "", "reason": "引用结构不合法"})
            continue
        doc_id = str(citation.get("doc_id") or "").strip()
        article_no = str(citation.get("article_no") or "").strip()
        if not doc_id or not article_no:
            bad.append(
                {
                    "ref_id": citation.get("ref_id"),
                    "doc_id": doc_id,
                    "article_no": article_no,
                    "reason": "缺少 doc_id 或 article_no",
                }
            )
            continue
        try:
            row = mysql.get_clause(doc_id, article_no)
        except Exception as exc:
            logger.warning("引用 %s 第%s条 核查异常：%s", doc_id, article_no, exc)
            notes.append(f"引用核查降级：{doc_id} 第{article_no}条 查询异常（{exc}）")
            continue
        if row is None:
            bad.append(
                {
                    "ref_id": citation.get("ref_id"),
                    "doc_id": doc_id,
                    "article_no": article_no,
                    "reason": "条款不存在或已失效（含归档版本）",
                }
            )
        else:
            clause_texts.append(_row_content(row))
    return bad, clause_texts


def _check_numbers(final_answer: str, clause_texts: list[str]) -> list[dict]:
    """步骤 2：答案中的数值必须能在所引用条款原文中找到。"""
    answer_numbers = _extract_numbers(final_answer)
    if not answer_numbers:
        return []
    clause_numbers: set[str] = set()
    for text in clause_texts:
        clause_numbers.update(_extract_numbers(text).keys())
    return [
        {"value": key, "raw": display, "reason": "引用条款原文中未出现该数值"}
        for key, display in answer_numbers.items()
        if key not in clause_numbers
    ]


def _check_semantics(
    final_answer: str, clause_texts: list[str], notes: list[str]
) -> tuple[list[dict], bool]:
    """步骤 3（可选）：逐句与引用条款做句向量余弦相似度。模型不可用时降级跳过。"""
    if not settings.enable_semantic_check:
        return [], False
    if not clause_texts or not final_answer.strip():
        return [], False

    sentences = _split_sentences(final_answer)
    if not sentences:
        return [], False

    try:
        apply_hf_offline_env()
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(settings.semantic_model, cache_folder=settings.hf_home)
        sentence_vectors = model.encode(
            sentences,
            normalize_embeddings=True,
            batch_size=16,
            show_progress_bar=False,
        )
        clause_vectors = model.encode(
            clause_texts,
            normalize_embeddings=True,
            batch_size=16,
            show_progress_bar=False,
        )
        unsupported: list[dict] = []
        for sentence, sentence_vector in zip(sentences, sentence_vectors):
            best = max(
                sum(float(a) * float(b) for a, b in zip(sentence_vector, clause_vector))
                for clause_vector in clause_vectors
            )
            if best < settings.semantic_threshold:
                unsupported.append(
                    {"sentence": sentence[:150], "max_similarity": round(float(best), 4)}
                )
        return unsupported, True
    except Exception as exc:
        logger.warning("句向量核查不可用，跳过本步：%s", exc)
        notes.append(f"逐句支撑核查降级：模型加载/推断失败（{exc}）")
        return [], False


# ---------------------------------------------------------------------------
# 节点
# ---------------------------------------------------------------------------

def critic_node(state: SupervisorState) -> Command:
    """独立核查节点：规则优先，给出 pass / reject 并决定下一跳。"""
    started = time.perf_counter()
    final_answer = str(state.get("final_answer") or "")
    raw_citations = state.get("citations") or []
    citations = [item for item in raw_citations if isinstance(item, dict)]
    review_round = max(0, int(state.get("review_round") or 0))

    notes: list[str] = []
    bad_citations, clause_texts = _check_citations(citations, notes)
    numeric_mismatches = _check_numbers(final_answer, clause_texts)
    unsupported, semantic_checked = _check_semantics(final_answer, clause_texts, notes)

    sentence_count = len(_split_sentences(final_answer)) if semantic_checked else 0
    unsupported_ratio = (
        len(unsupported) / sentence_count if sentence_count else 0.0
    )

    reject_reasons: list[str] = []
    if not final_answer.strip():
        reject_reasons.append("最终答案为空")
    if bad_citations:
        reject_reasons.append(f"存在伪造/失效引用 {len(bad_citations)} 条")
    if numeric_mismatches:
        reject_reasons.append(f"存在数值不一致 {len(numeric_mismatches)} 处")
    if semantic_checked and unsupported_ratio > 0.10:
        reject_reasons.append(
            f"无支撑句占比 {unsupported_ratio:.0%} 超过 10%"
        )
    # README 4.4：pass 需要"有引用支撑"。空答案/无引用一律视为无支撑。
    if final_answer.strip() and not citations:
        reject_reasons.append("答案未提供任何条款级引用")

    verdict = "reject" if reject_reasons else "pass"
    review = {
        "verdict": verdict,
        "unsupported": unsupported,
        "bad_citations": bad_citations,
        "numeric_mismatches": numeric_mismatches,
        "semantic_checked": semantic_checked,
        "reasons": reject_reasons,
        "notes": notes,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }

    emit(
        "review",
        verdict=verdict,
        unsupported=len(unsupported),
        bad_citations=len(bad_citations),
        numeric_mismatches=len(numeric_mismatches),
    )

    if verdict == "pass":
        return Command(goto=END, update={"review": review})

    if review_round < settings.max_review_round:
        # 回 supervisor 打回原专家补证（supervisor 会根据 review 定向派发）
        return Command(
            goto="supervisor",
            update={"review": review, "review_round": review_round + 1},
        )

    degraded_plan = {
        "intent": "clarify",
        "confidence": 0.0,
        "subtasks": [],
        "next": "clarify",
        "done": True,
        "degraded": True,
        "reason": (
            f"critic 核查未通过且已达最大复核轮次 {settings.max_review_round}，"
            f"降级输出已生成内容；问题：{'；'.join(reject_reasons)}"
        ),
    }
    return Command(
        goto="clarify",
        update={"review": review, "plan": degraded_plan},
    )
