"""PolicyMind 端到端 RAG 评估：答案覆盖、引用、审核、延迟、token 与失败归因。

用途
----
读取 eval/dataset.jsonl，逐题调用
run_answer_once(question, user_scope=["employee"], session_id=f"eval-{id}")，
统计生成/审核/系统层指标，并按“路由错 / 召回缺 / 引用伪造 / 拒答误判 / 计算错”
五类归档失败案例，输出 Markdown + JSON 双份报告。

指标定义
--------
- 答案覆盖率 = 命中关键词数 / gold_answer 关键词总数。
  关键词提取：先按中英文标点/空白/算式符号切分，保留长度 >= 2 的片段；再额外抽取
  “数字+单位”（如 10天、3个月、200%）作为独立关键词；去重后逐一判断是否
  以子串形式出现在 final_answer 中（NFKC 归一化并去空白）。
- 引用命中率（micro）= Σ 命中金标 article_ids 的 citation 数 / Σ citation 总数。
  article_ids 为空的题记为 skipped，不参与该指标。
- 引用命中率（逐题平均）= 各题引用命中率的算术平均。
- 审核通过率 = review.verdict == "pass" 的题数 / 总题数。
- 引用伪造数 = review.bad_citations 条目总数（审核器判定）；
  另报告“引用未命中金标数” = 金标可用的题中，citation 的 (doc_id, article_no)
  不在 article_ids 内的条目数。
- 延迟：单次 run_answer_once 墙钟耗时，报告平均 / P50 / P95。
- token：从 token_usage 提取 total_tokens（兼容 prompt+completion 分项），
  报告平均 / P50 / P95 / 最小 / 最大。
- 回环触发率 = dispatch_round > 1 的题数 / 总题数。
- 端到端通过率 = 未被归入任何失败类别的题数 / 总题数。

失败案例归类规则（按优先级取 primary category，同时保留全部命中类别）
--------------------------------------------------------------------
1. 路由错：refuse 题 intent 不是 refuse（next=="clarify" 且 intent=="refuse" 除外）；
   其他题 plan.next 与 expected_agent 不一致。
2. 拒答误判：refuse 题路由未拒答且答案不含拒答话术（漏拒答）；非 refuse 题
   intent=="refuse"，或答案命中拒答话术且答案覆盖率 < 0.3（误拒答）。
3. 引用伪造：review.bad_citations 非空；或 article_ids 非空时，citations 中存在
   不在金标集合内的 (doc_id, article_no)。
4. 计算错：family/意图为 eligibility，且 review.numeric_mismatches 非空，或
   答案覆盖率 < 0.5（关键数值/结论未命中）。
5. 召回缺：article_ids 非空且 citations/evidence 均未命中任一金标条款；或非拒答
   题答案覆盖率 < 0.5 且未被以上类别覆盖（兜底）；或审核 verdict 非 pass 且未
   被以上类别覆盖（通常意味着答案缺少条款支撑）。

用法示例
--------
    python scripts/eval_rag.py
    python scripts/eval_rag.py --limit 5
    python scripts/eval_rag.py --dataset eval/dataset.jsonl --report-dir eval/reports

依赖
----
app.graph.supervisor_graph.run_answer_once（其他开发者并行实现，本脚本只按签名调用）。
需要可用的 DEEPSEEK_API_KEY、已建库的 Chroma/MySQL/BM25 索引与 Ollama embedding 服务。
异常（key 未配、LLM 调用失败等）打印中文提示并以 exit 1 退出。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# 统一工作目录：让 pydantic-settings 读取项目根目录的 .env，并让相对路径按项目根解析。
os.chdir(PROJECT_ROOT)

from app.core.config import settings  # noqa: E402

FAMILY_ORDER = ["fact", "cross", "eligibility", "compare", "refuse"]
CATEGORY_ORDER = ["路由错", "拒答误判", "引用伪造", "计算错", "召回缺"]
EXPERT_AGENTS = {"policy_qa", "compare", "eligibility"}

# 关键词切分：中英文标点、括号、引号、空白、斜杠、算式符号都视为分隔符。
# 小数点单独处理：只在不是“数字.数字”的位置切分，避免把 21.75 拆成 21 和 75。
_PUNCT_SPLIT = re.compile(
    r"[，。；、！？：,!?:;()（）\[\]【】<>《》\"“”'‘’\s/\\|=＝×÷＋+－-]+"
    r"|(?<!\d)\.|\.(?!\d)"
)
_NUMBER_UNIT = re.compile(
    r"\d+(?:\.\d+)?\s*(?:个工作日|个?月|天|日|年|小时|周|次|元|%|％|倍)"
)
_REFUSAL_MARKERS = (
    "无法回答",
    "无法提供",
    "无法查询",
    "不能提供",
    "不予回答",
    "不便提供",
    "无可奉告",
    "不掌握",
    "未收录",
    "未找到相关",
    "不在知识库",
    "超出知识库",
    "超出制度范围",
    "涉及隐私",
    "个人隐私",
    "无权提供",
)


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------
def _resolve(path_value: str | os.PathLike[str]) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """同时兼容 dict 与 pydantic 模型/对象的字段读取。"""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    for method in ("model_dump", "dict"):
        fn = getattr(value, method, None)
        if callable(fn):
            try:
                return _jsonable(fn())
            except Exception:  # pragma: no cover - 仅兜底
                pass
    return str(value)


def _list_len(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    return 1 if str(value).strip() else 0


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    if not denominator:
        return None
    return round(float(numerator) / float(denominator), 4)


def _mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 4) if values else None


def _percentile(values: list[float], percentile: float) -> float | None:
    """线性插值分位数（避免额外依赖；空列表返回 None）。"""
    if not values:
        return None
    data = sorted(float(v) for v in values)
    if len(data) == 1:
        return round(data[0], 2)
    rank = (len(data) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return round(data[lower], 2)
    fraction = rank - lower
    return round(data[lower] * (1 - fraction) + data[upper] * fraction, 2)


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_md(cell) for cell in row) + " |")
    return "\n".join(lines)


def _display_width(text: str) -> int:
    return sum(
        2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in str(text)
    )


def _pad(text: Any, width: int) -> str:
    value = str(text)
    return value + " " * max(0, width - _display_width(value))


def _print_table(headers: list[str], rows: list[list[Any]]) -> None:
    widths = [
        max([_display_width(headers[i])] + [_display_width(row[i]) for row in rows])
        for i in range(len(headers))
    ]
    print("  ".join(_pad(headers[i], widths[i]) for i in range(len(headers))))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  ".join(_pad(row[i], widths[i]) for i in range(len(headers))))


def normalize_agent(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text.endswith("_agent"):
        text = text[: -len("_agent")]
    aliases = {
        "qa": "policy_qa",
        "policyqa": "policy_qa",
        "fact": "policy_qa",
        "factual": "policy_qa",
        "fact_qa": "policy_qa",
        "single_fact": "policy_qa",
        "comparison": "compare",
        "compare_versions": "compare",
        "version_compare": "compare",
        "eligibility_check": "eligibility",
        "calc": "eligibility",
        "calculation": "eligibility",
        "refusal": "refuse",
        "reject": "refuse",
        "out_of_scope": "refuse",
        "off_topic": "refuse",
        "拒答": "refuse",
        "finish": "end",
        "finished": "end",
        "done": "end",
    }
    return aliases.get(text, text)


def _routing_correct(expected: str, intent: str, next_agent: str) -> bool:
    if expected == "refuse":
        return intent == "refuse" or (next_agent == "clarify" and intent == "refuse")
    return next_agent == expected


# ---------------------------------------------------------------------------
# 文本 / 引用 / 条款工具
# ---------------------------------------------------------------------------
def _normalize_text(text: Any) -> str:
    return re.sub(
        r"\s+", "", unicodedata.normalize("NFKC", str(text or ""))
    ).lower()


def extract_keywords(gold_answer: Any) -> list[str]:
    """按标点拆出 >=2 字的关键词，并补充“数字+单位”关键词（去重保序）。"""
    text = unicodedata.normalize("NFKC", str(gold_answer or ""))
    raw_keywords: list[str] = []
    for part in _PUNCT_SPLIT.split(text):
        part = part.strip()
        if len(part) >= 2:
            raw_keywords.append(part)
    for match in _NUMBER_UNIT.finditer(text):
        raw_keywords.append(re.sub(r"\s+", "", match.group(0)))

    keywords: list[str] = []
    seen: set[str] = set()
    for keyword in raw_keywords:
        key = re.sub(r"\s+", "", keyword)
        if len(key) >= 2 and key not in seen:
            seen.add(key)
            keywords.append(key)
    return keywords


def answer_coverage(
    gold_answer: Any, final_answer: Any
) -> tuple[float | None, list[str], list[str]]:
    keywords = extract_keywords(gold_answer)
    if not keywords:
        return None, [], []
    normalized_answer = _normalize_text(final_answer)
    hits = [keyword for keyword in keywords if keyword in normalized_answer]
    missing = [keyword for keyword in keywords if keyword not in normalized_answer]
    return round(len(hits) / len(keywords), 4), hits, missing


def _normalize_article_no(article_no: Any) -> str:
    text = str(article_no or "").strip()
    if text.startswith("第"):
        text = text[1:]
    if text.endswith("条"):
        text = text[:-1]
    return text.strip()


def _article_key(doc_id: Any, article_no: Any) -> tuple[str, str]:
    return str(doc_id or "").strip(), _normalize_article_no(article_no)


def parse_article_ids(raw: Any) -> list[tuple[str, str]]:
    """把 dataset.article_ids 归一化为 [(doc_id, article_no), ...]（去重保序）。"""
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple, set)):
        raw = [raw]

    result: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw:
        doc_id = article_no = ""
        if isinstance(item, dict):
            doc_id = str(item.get("doc_id", "") or "")
            article_no = str(item.get("article_no", "") or "")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            doc_id = str(item[0] or "")
            article_no = str(item[1] or "")
        else:
            text = str(item or "").strip()
            for separator in ("::", "#", "|"):
                if separator in text:
                    left, right = text.split(separator, 1)
                    doc_id, article_no = left, right
                    break
            else:
                print(
                    f"警告：无法解析 article_id：{item!r}；"
                    "支持 doc_id::article_no、doc_id#article_no、doc_id|article_no "
                    '或 {"doc_id": ..., "article_no": ...}',
                    file=sys.stderr,
                )
                continue
        key = _article_key(doc_id, article_no)
        if key[0] and key[1] and key not in seen:
            seen.add(key)
            result.append(key)
    return result


def _keys_from_items(items: Any) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if isinstance(items, dict):
        items = [items]
    elif not isinstance(items, (list, tuple, set)):
        items = []
    for item in items:
        key = _article_key(_get(item, "doc_id"), _get(item, "article_no"))
        if key[0] and key[1]:
            keys.add(key)
    return keys


def _normalize_citations(citations: Any) -> list[dict[str, Any]]:
    if isinstance(citations, dict):
        citations = [citations]
    elif not isinstance(citations, (list, tuple, set)):
        citations = []
    result: list[dict[str, Any]] = []
    for citation in citations:
        key = _article_key(_get(citation, "doc_id"), _get(citation, "article_no"))
        result.append(
            {
                "ref_id": str(_get(citation, "ref_id", "") or ""),
                "doc_id": key[0],
                "article_no": key[1],
                "title": str(_get(citation, "title", "") or ""),
                "page": _jsonable(_get(citation, "page")),
                "snippet": str(_get(citation, "snippet", "") or "")[:300],
            }
        )
    return result


def _total_tokens(token_usage: Any) -> int:
    """从 token_usage 提取总 token，兼容 total_tokens / prompt+completion 两种口径。"""
    if not isinstance(token_usage, dict):
        token_usage = _jsonable(token_usage)
    if not isinstance(token_usage, dict):
        return 0

    def _as_int(value: Any) -> int | None:
        return int(value) if isinstance(value, (int, float)) else None

    for key in ("total_tokens", "total", "tokens"):
        value = _as_int(token_usage.get(key))
        if value is not None:
            return value

    prompt = _as_int(token_usage.get("prompt_tokens"))
    completion = _as_int(token_usage.get("completion_tokens"))
    if prompt is not None or completion is not None:
        return int(prompt or 0) + int(completion or 0)

    total = 0
    for key, value in token_usage.items():
        if isinstance(value, (int, float)) and "token" in str(key).lower():
            total += int(value)
    return total


def _looks_refused(text: Any) -> bool:
    answer = str(text or "")
    return any(marker in answer for marker in _REFUSAL_MARKERS)


# ---------------------------------------------------------------------------
# 数据集与评估
# ---------------------------------------------------------------------------
def _ensure_api_key() -> None:
    key = (settings.deepseek_api_key or "").strip()
    if not key or key.lower().startswith("sk-your"):
        print("错误：未检测到可用的 DEEPSEEK_API_KEY。", file=sys.stderr)
        print(
            "请先配置 DEEPSEEK_API_KEY（可复制 .env.example 为 .env 后填写），再运行本脚本。",
            file=sys.stderr,
        )
        raise SystemExit(1)


def load_dataset(path: Path, limit: int | None) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"错误：测试集不存在：{path}", file=sys.stderr)
        raise SystemExit(1)
    required = {"id", "question", "expected_agent", "gold_answer"}
    questions: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"错误：{path} 第 {line_no} 行不是合法 JSON：{exc}",
                    file=sys.stderr,
                )
                raise SystemExit(1)
            missing = required - set(item)
            if missing:
                print(
                    f"错误：{path} 第 {line_no} 行缺少字段：{', '.join(sorted(missing))}",
                    file=sys.stderr,
                )
                raise SystemExit(1)
            questions.append(item)
    if limit is not None:
        questions = questions[: max(0, limit)]
    if not questions:
        print(f"错误：测试集为空：{path}", file=sys.stderr)
        raise SystemExit(1)
    return questions


def classify_failure(question: dict[str, Any], record: dict[str, Any]) -> list[str]:
    """按 README 风险清单的五类规则归档失败；返回命中的全部类别（按优先级排序）。

    规则见模块 docstring。所有判断只依赖可复现的客观字段：
    plan.intent / plan.next / review / 答案覆盖率 / 引用与金标比对。
    """
    expected = record["expected_agent"]
    categories: set[str] = set()

    # 1) 路由错
    if not _routing_correct(expected, record["intent"], record["next"]):
        categories.add("路由错")

    if expected == "refuse":
        # 2) 拒答误判（漏拒答）：路由未拒答且答案没有拒答话术
        refused_ok = (
            record["intent"] == "refuse"
            or record["next"] == "clarify"
            or record["looks_refused"]
        )
        if not refused_ok:
            categories.add("拒答误判")
    else:
        coverage = record["answer_coverage"]
        # 2) 拒答误判（误拒答）
        wrongly_refused = record["intent"] == "refuse" or (
            record["looks_refused"]
            and coverage is not None
            and coverage < 0.3
        )
        if wrongly_refused:
            categories.add("拒答误判")

        # 3) 引用伪造
        if record["review_bad_citations_count"] > 0 or (
            record["gold_count"] > 0
            and (record["citation_invalid_count"] or 0) > 0
        ):
            categories.add("引用伪造")

        # 4) 计算错
        if (record["family"] == "eligibility" or record["intent"] == "eligibility") and (
            record["review_numeric_mismatch_count"] > 0
            or (coverage is not None and coverage < 0.5)
        ):
            categories.add("计算错")

        # 5) 召回缺
        if record["gold_count"] > 0 and record["gold_evidence_hit"] is False:
            categories.add("召回缺")
        if coverage is not None and coverage < 0.5 and not categories:
            categories.add("召回缺")
        if (
            record["review_verdict"]
            and not record["review_pass"]
            and not categories
        ):
            categories.add("召回缺")

    return [category for category in CATEGORY_ORDER if category in categories]


def evaluate_one(
    question: dict[str, Any],
    gold_ids: list[tuple[str, str]],
    run_answer_once: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    question_id = str(question.get("id", ""))
    started = time.perf_counter()
    try:
        raw_result = run_answer_once(
            question["question"],
            user_scope=["employee"],
            session_id=f"eval-{question_id}",
        )
    except Exception as exc:
        print(
            f"错误：调用 run_answer_once 失败（{question_id or '?'}）：{exc}",
            file=sys.stderr,
        )
        print(
            "请先配置 DEEPSEEK_API_KEY（.env 中填写），并确认 DeepSeek 服务、"
            "检索索引与 Ollama 均可用。",
            file=sys.stderr,
        )
        raise SystemExit(1)
    latency_ms = round((time.perf_counter() - started) * 1000.0, 2)

    result = raw_result or {}
    final_answer = str(_get(result, "final_answer", "") or "")
    citations_raw = _get(result, "citations", []) or []
    if isinstance(citations_raw, dict):
        citations_raw = [citations_raw]
    elif not isinstance(citations_raw, (list, tuple, set)):
        citations_raw = []
    citations = _normalize_citations(citations_raw)

    evidence_raw = _get(result, "evidence", []) or []
    if isinstance(evidence_raw, dict):
        evidence_raw = [evidence_raw]
    elif not isinstance(evidence_raw, (list, tuple, set)):
        evidence_raw = []
    evidence = list(evidence_raw)

    plan_raw = _get(result, "plan", {}) or {}
    intent = normalize_agent(_get(plan_raw, "intent"))
    next_agent = normalize_agent(_get(plan_raw, "next"))
    plan_confidence = _to_float(_get(plan_raw, "confidence"))
    plan_reason = str(_get(plan_raw, "reason", "") or "")

    # supervisor.decide_plan 内部会吞掉 LLM 异常并降级；若本轮出现降级，说明
    # LLM key/服务异常，结果不可作为评估样本，按规格中文提示并 exit 1。
    if "调度模型调用失败" in plan_reason or "降级派发" in plan_reason:
        print(
            f"错误：{question_id or '?'} 的调度模型调用失败（已降级）：{plan_reason}",
            file=sys.stderr,
        )
        print(
            "请先配置 DEEPSEEK_API_KEY（.env 中填写），并确认模型服务与网络可用。",
            file=sys.stderr,
        )
        raise SystemExit(1)

    review_raw = _get(result, "review", {}) or {}
    review_verdict = str(_get(review_raw, "verdict", "") or "").strip().lower()
    review_bad_citations = _get(review_raw, "bad_citations", []) or []
    review_unsupported = _get(review_raw, "unsupported", []) or []
    review_numeric_mismatches = _get(review_raw, "numeric_mismatches", []) or []

    dispatch_round = int(_to_float(_get(result, "dispatch_round"), 0) or 0)
    token_usage = _get(result, "token_usage", {}) or {}
    trace = list(_get(result, "trace", []) or [])
    agent_results = list(_get(result, "agent_results", []) or [])

    coverage, coverage_hits, coverage_missing = answer_coverage(
        question.get("gold_answer", ""), final_answer
    )

    gold_set = set(gold_ids)
    citation_keys = [_article_key(c["doc_id"], c["article_no"]) for c in citations]
    citation_skipped = not gold_set
    if citation_skipped:
        citation_hit_count: int | None = None
        citation_hit_rate: float | None = None
        citation_invalid_count: int | None = None
    else:
        citation_hit_count = sum(1 for key in citation_keys if key in gold_set)
        citation_hit_rate = (
            round(citation_hit_count / len(citation_keys), 4) if citation_keys else 0.0
        )
        citation_invalid_count = len(citation_keys) - citation_hit_count

    evidence_keys = _keys_from_items(evidence)
    retrieval_keys = set(citation_keys) | evidence_keys
    gold_evidence_hit: bool | None = None
    if gold_set:
        gold_evidence_hit = bool(gold_set & retrieval_keys)

    expected_agent = normalize_agent(question.get("expected_agent"))
    record: dict[str, Any] = {
        "id": question_id,
        "family": str(question.get("family", "")),
        "question": question["question"],
        "expected_agent": expected_agent,
        "gold_answer": str(question.get("gold_answer", "")),
        "gold_ids": [f"{doc_id}::{article_no}" for doc_id, article_no in gold_ids],
        "gold_count": len(gold_ids),
        "intent": intent,
        "next": next_agent,
        "plan_confidence": plan_confidence,
        "plan": {
            "intent": intent,
            "next": next_agent,
            "confidence": plan_confidence,
            "subtasks": _jsonable(_get(plan_raw, "subtasks", []) or []),
            "done": _jsonable(_get(plan_raw, "done")),
            "reason": plan_reason,
        },
        "final_answer": final_answer[:4000],
        "answer_length": len(final_answer),
        "looks_refused": _looks_refused(final_answer),
        "answer_coverage": coverage,
        "coverage_keyword_count": len(coverage_hits) + len(coverage_missing),
        "coverage_hits": coverage_hits[:30],
        "coverage_missing": coverage_missing[:30],
        "citations": citations,
        "citation_count": len(citations),
        "citation_skipped": citation_skipped,
        "citation_hit_count": citation_hit_count,
        "citation_hit_rate": citation_hit_rate,
        "citation_invalid_count": citation_invalid_count,
        "evidence_count": len(evidence),
        "evidence_gold_keys": [
            f"{doc_id}::{article_no}"
            for doc_id, article_no in sorted(evidence_keys)[:30]
        ],
        "gold_evidence_hit": gold_evidence_hit,
        "review_verdict": review_verdict,
        "review_pass": review_verdict == "pass",
        "review_missing": not review_verdict,
        "review_bad_citations": _jsonable(review_bad_citations),
        "review_bad_citations_count": _list_len(review_bad_citations),
        "review_unsupported": _jsonable(review_unsupported),
        "review_unsupported_count": _list_len(review_unsupported),
        "review_numeric_mismatches": _jsonable(review_numeric_mismatches),
        "review_numeric_mismatch_count": _list_len(review_numeric_mismatches),
        "latency_ms": latency_ms,
        "token_usage": _jsonable(token_usage),
        "total_tokens": _total_tokens(token_usage),
        "dispatch_round": dispatch_round,
        "loop_triggered": dispatch_round > 1,
        "trace_events": len(trace),
        "agent_result_count": len(agent_results),
    }

    categories = classify_failure(question, record)
    record["failure_categories"] = categories
    record["primary_category"] = categories[0] if categories else None
    record["passed"] = not categories
    return record


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    coverage_values = [
        r["answer_coverage"] for r in records if r["answer_coverage"] is not None
    ]
    coverage_non_refuse = [
        r["answer_coverage"]
        for r in records
        if r["answer_coverage"] is not None and r["expected_agent"] != "refuse"
    ]
    citation_rows = [r for r in records if not r["citation_skipped"]]
    total_citations = sum(r["citation_count"] for r in citation_rows)
    total_citation_hits = sum(r["citation_hit_count"] or 0 for r in citation_rows)
    citation_macro_values = [
        r["citation_hit_rate"]
        for r in citation_rows
        if r["citation_hit_rate"] is not None
    ]

    latencies = [r["latency_ms"] for r in records]
    tokens = [r["total_tokens"] for r in records]
    category_counts = Counter(
        category for r in records for category in r["failure_categories"]
    )
    failures = [r for r in records if not r["passed"]]

    return {
        "total": total,
        "answer_coverage": _mean(coverage_values),
        "answer_coverage_non_refuse": _mean(coverage_non_refuse),
        "answer_coverage_questions": len(coverage_values),
        "citation_evaluated": len(citation_rows),
        "citation_skipped": total - len(citation_rows),
        "citation_hit_rate": _ratio(total_citation_hits, total_citations),
        "citation_hit_rate_macro": _mean(citation_macro_values),
        "citation_total": total_citations,
        "citation_hit_total": total_citation_hits,
        "forged_citations_review": sum(
            r["review_bad_citations_count"] for r in records
        ),
        "invalid_citations_vs_gold": sum(
            r["citation_invalid_count"] or 0 for r in citation_rows
        ),
        "questions_with_citation_issue": sum(
            1
            for r in records
            if r["review_bad_citations_count"] > 0
            or (r["citation_invalid_count"] or 0) > 0
        ),
        "review_pass_count": sum(1 for r in records if r["review_pass"]),
        "review_pass_rate": _ratio(
            sum(1 for r in records if r["review_pass"]), total
        ),
        "review_missing": sum(1 for r in records if r["review_missing"]),
        "avg_latency_ms": _mean(latencies),
        "p50_latency_ms": _percentile(latencies, 50),
        "p95_latency_ms": _percentile(latencies, 95),
        "total_tokens_sum": sum(tokens),
        "avg_tokens": _mean(tokens),
        "p50_tokens": _percentile(tokens, 50),
        "p95_tokens": _percentile(tokens, 95),
        "min_tokens": min(tokens) if tokens else None,
        "max_tokens": max(tokens) if tokens else None,
        "loop_trigger_count": sum(1 for r in records if r["loop_triggered"]),
        "loop_trigger_rate": _ratio(
            sum(1 for r in records if r["loop_triggered"]), total
        ),
        "avg_dispatch_round": _mean(
            [float(r["dispatch_round"]) for r in records]
        ),
        "failure_count": len(failures),
        "pass_count": total - len(failures),
        "pass_rate": _ratio(total - len(failures), total),
        "failure_category_counts": dict(category_counts),
    }


def summarize_by_family(records: list[dict[str, Any]]) -> dict[str, Any]:
    families = FAMILY_ORDER + sorted({r["family"] for r in records} - set(FAMILY_ORDER))
    result: dict[str, Any] = {}
    for family in families:
        family_records = [r for r in records if r["family"] == family]
        if not family_records:
            continue
        item = summarize(family_records)
        result[family] = {
            "total": item["total"],
            "answer_coverage": item["answer_coverage"],
            "answer_coverage_non_refuse": item["answer_coverage_non_refuse"],
            "citation_hit_rate": item["citation_hit_rate"],
            "review_pass_rate": item["review_pass_rate"],
            "avg_latency_ms": item["avg_latency_ms"],
            "p50_latency_ms": item["p50_latency_ms"],
            "avg_tokens": item["avg_tokens"],
            "loop_trigger_rate": item["loop_trigger_rate"],
            "failure_count": item["failure_count"],
            "pass_rate": item["pass_rate"],
        }
    return result


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def build_markdown(
    meta: dict[str, Any],
    summary: dict[str, Any],
    by_family: dict[str, Any],
    results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> str:
    lines = [
        "# PolicyMind 端到端 RAG 评估报告",
        "",
        f"- 生成时间：{meta['generated_at']}",
        f"- 测试集：`{meta['dataset']}`",
        f"- 评估题数：{summary['total']}（--limit={meta['limit']}）",
        "- 调用方式：`run_answer_once(question, user_scope=[\"employee\"], "
        "session_id=\"eval-{id}\")`",
        "",
        "## 概览指标",
        "",
        _md_table(
            ["指标", "数值", "说明"],
            [
                [
                    "答案覆盖率（全部）",
                    _pct(summary["answer_coverage"]),
                    "gold_answer 关键词在 final_answer 中的命中比例",
                ],
                [
                    "答案覆盖率（非拒答）",
                    _pct(summary["answer_coverage_non_refuse"]),
                    "排除 refuse 题后的覆盖率",
                ],
                [
                    "引用命中率（micro）",
                    _pct(summary["citation_hit_rate"]),
                    f"命中 {summary['citation_hit_total']}/{summary['citation_total']} 条 citation",
                ],
                [
                    "引用命中率（逐题平均）",
                    _pct(summary["citation_hit_rate_macro"]),
                    "各题引用命中率的算术平均",
                ],
                [
                    "引用评估/跳过",
                    f"{summary['citation_evaluated']}/{summary['citation_skipped']}",
                    "article_ids 为空的题跳过引用命中率",
                ],
                [
                    "审核通过率",
                    _pct(summary["review_pass_rate"]),
                    f"{summary['review_pass_count']}/{summary['total']} 题 verdict=pass",
                ],
                [
                    "引用伪造数（审核器）",
                    summary["forged_citations_review"],
                    "review.bad_citations 条目总数",
                ],
                [
                    "引用未命中金标数",
                    summary["invalid_citations_vs_gold"],
                    "citation 的 (doc_id, article_no) 不在 article_ids 内",
                ],
                [
                    "平均延迟",
                    f"{_num(summary['avg_latency_ms'], 2)} ms",
                    "单次 run_answer_once 墙钟耗时",
                ],
                [
                    "P50 / P95 延迟",
                    f"{_num(summary['p50_latency_ms'], 2)} / "
                    f"{_num(summary['p95_latency_ms'], 2)} ms",
                    "延迟中位数 / 95 分位",
                ],
                [
                    "平均 token",
                    _num(summary["avg_tokens"], 1),
                    "token_usage.total_tokens 的均值",
                ],
                [
                    "token 分布（P50/P95）",
                    f"{_num(summary['p50_tokens'], 1)} / {_num(summary['p95_tokens'], 1)}",
                    f"最小 {summary['min_tokens']}，最大 {summary['max_tokens']}",
                ],
                [
                    "回环触发率",
                    _pct(summary["loop_trigger_rate"]),
                    f"{summary['loop_trigger_count']}/{summary['total']} 题 dispatch_round > 1",
                ],
                [
                    "平均 dispatch_round",
                    _num(summary["avg_dispatch_round"], 2),
                    "平均调度轮次",
                ],
                [
                    "端到端通过率",
                    _pct(summary["pass_rate"]),
                    f"{summary['pass_count']}/{summary['total']} 题未归入失败类别",
                ],
                [
                    "失败题数",
                    summary["failure_count"],
                    "按下方五类规则归档",
                ],
            ],
        ),
        "",
        "## 分题型明细",
        "",
    ]

    family_rows = []
    for family, item in by_family.items():
        family_rows.append(
            [
                family,
                item["total"],
                _pct(item["answer_coverage"]),
                _pct(item["citation_hit_rate"]),
                _pct(item["review_pass_rate"]),
                _num(item["avg_latency_ms"], 2),
                _num(item["p50_latency_ms"], 2),
                _num(item["avg_tokens"], 1),
                _pct(item["loop_trigger_rate"]),
                f"{item['failure_count']}/{item['total']}",
            ]
        )
    lines.append(
        _md_table(
            [
                "题型",
                "题数",
                "答案覆盖率",
                "引用命中率",
                "审核通过率",
                "平均延迟(ms)",
                "P50 延迟(ms)",
                "平均 token",
                "回环触发率",
                "失败数",
            ],
            family_rows,
        )
    )

    lines.extend(["", "## 逐题明细", ""])
    detail_rows = []
    for item in results:
        detail_rows.append(
            [
                item["id"],
                item["family"],
                item["expected_agent"],
                item["intent"] or "-",
                item["next"] or "-",
                _pct(item["answer_coverage"]),
                "skipped" if item["citation_skipped"] else _pct(item["citation_hit_rate"]),
                item["review_verdict"] or "-",
                _num(item["latency_ms"], 1),
                item["total_tokens"],
                item["dispatch_round"],
                "PASS" if item["passed"] else item["primary_category"],
            ]
        )
    lines.append(
        _md_table(
            [
                "ID",
                "题型",
                "期望",
                "intent",
                "next",
                "覆盖率",
                "引用命中",
                "审核",
                "延迟(ms)",
                "token",
                "轮次",
                "判定",
            ],
            detail_rows,
        )
    )

    lines.extend(["", "## 失败案例清单", ""])
    lines.append(
        "归档规则见脚本 docstring：路由错 > 拒答误判 > 引用伪造 > 计算错 > 召回缺；"
        "一题可命中多个类别，下表按 primary category 分组。"
    )
    lines.append("")
    if not failures:
        lines.append("无失败案例。")
    else:
        primary_counts = Counter(
            r["primary_category"] for r in failures if r["primary_category"]
        )
        hit_counts = Counter(
            category for r in failures for category in r["failure_categories"]
        )
        summary_rows = [
            [category, primary_counts.get(category, 0), hit_counts.get(category, 0)]
            for category in CATEGORY_ORDER
            if primary_counts.get(category, 0) or hit_counts.get(category, 0)
        ]
        lines.append(
            _md_table(["类别", "主类别题数", "命中题数（可多选）"], summary_rows)
        )
        lines.append("")

        for category in CATEGORY_ORDER:
            group = [r for r in failures if r["primary_category"] == category]
            if not group:
                continue
            lines.append(f"### {category}（{len(group)} 题）")
            lines.append("")
            for item in group:
                lines.append(f"#### {item['id']}（{item['family']}）")
                lines.append("")
                lines.append(f"- 问题：{item['question']}")
                lines.append(
                    f"- 期望：{item['expected_agent']}；实际：intent={item['intent'] or '-'}，"
                    f"next={item['next'] or '-'}，dispatch_round={item['dispatch_round']}"
                )
                lines.append(
                    f"- 指标：覆盖率={_pct(item['answer_coverage'])}，"
                    f"引用命中={'skipped' if item['citation_skipped'] else _pct(item['citation_hit_rate'])}，"
                    f"审核={item['review_verdict'] or '-'}，延迟={_num(item['latency_ms'], 1)}ms，"
                    f"token={item['total_tokens']}"
                )
                lines.append(
                    f"- 引用：{item['citation_count']} 条；审核器 bad_citations="
                    f"{item['review_bad_citations_count']}，unsupported="
                    f"{item['review_unsupported_count']}，numeric_mismatches="
                    f"{item['review_numeric_mismatch_count']}"
                )
                if item["gold_ids"]:
                    lines.append(
                        f"- 金标条款：{', '.join(item['gold_ids'])}；"
                        f"证据/引用命中金标：{'是' if item['gold_evidence_hit'] else '否'}"
                    )
                lines.append(f"- 归类：{'、'.join(item['failure_categories'])}")
                gold_preview = item["gold_answer"][:250]
                answer_preview = item["final_answer"][:350].replace("\n", " ")
                lines.append(f"- 标准答案要点：{gold_preview}")
                lines.append(f"- 系统答案摘录：{answer_preview or '（空）'}")
                if item["coverage_missing"]:
                    lines.append(
                        f"- 未命中的标准答案关键词：{', '.join(item['coverage_missing'][:10])}"
                    )
                lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_reports(
    report_dir: Path,
    stem: str,
    markdown: str,
    payload: dict[str, Any],
) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    md_path = report_dir / f"{stem}.md"
    json_path = report_dir / f"{stem}.json"
    md_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return md_path, json_path


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PolicyMind 端到端 RAG 评估：覆盖率 / 引用 / 审核 / 延迟 / token / 失败归因",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python scripts/eval_rag.py\n"
            "  python scripts/eval_rag.py --limit 5\n"
            "  python scripts/eval_rag.py --dataset eval/dataset.jsonl "
            "--report-dir eval/reports"
        ),
    )
    parser.add_argument(
        "--dataset",
        default=settings.eval_dataset_path,
        help=f"测试集路径（默认 {settings.eval_dataset_path}）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只评估前 N 题（调试用，默认全部）",
    )
    parser.add_argument(
        "--report-dir",
        default=settings.eval_report_dir,
        help=f"报告输出目录（默认 {settings.eval_report_dir}）",
    )
    return parser


def _load_run_answer_once() -> Callable[..., dict[str, Any]] | None:
    try:
        from app.graph.supervisor_graph import run_answer_once
    except ImportError as exc:  # pragma: no cover - 依赖尚未就绪时触发
        print(
            f"错误：无法导入 app.graph.supervisor_graph.run_answer_once：{exc}",
            file=sys.stderr,
        )
        print(
            "请确认顶层图模块已完成、项目依赖已安装后再运行本脚本。",
            file=sys.stderr,
        )
        return None
    return run_answer_once


def main() -> int:
    args = build_parser().parse_args()
    _ensure_api_key()

    dataset_path = _resolve(args.dataset)
    report_dir = _resolve(args.report_dir)
    questions = load_dataset(dataset_path, args.limit)

    run_answer_once = _load_run_answer_once()
    if run_answer_once is None:
        return 1

    prepared = [
        (question, parse_article_ids(question.get("article_ids")))
        for question in questions
    ]
    print(f"端到端 RAG 评估开始：{len(prepared)} 题，数据集 {dataset_path}")
    results: list[dict[str, Any]] = []
    for index, (question, gold_ids) in enumerate(prepared, 1):
        print(f"[{index}/{len(prepared)}] {question.get('id', '?')} ...", flush=True)
        results.append(evaluate_one(question, gold_ids, run_answer_once))

    summary = summarize(results)
    by_family = summarize_by_family(results)
    failures = [r for r in results if not r["passed"]]

    console_rows = []
    for item in results:
        console_rows.append(
            [
                item["id"],
                item["family"],
                item["intent"] or "-",
                item["next"] or "-",
                _pct(item["answer_coverage"]),
                "skip" if item["citation_skipped"] else _pct(item["citation_hit_rate"]),
                item["review_verdict"] or "-",
                _num(item["latency_ms"], 0),
                str(item["total_tokens"]),
                str(item["dispatch_round"]),
                "PASS" if item["passed"] else item["primary_category"],
            ]
        )
    print()
    _print_table(
        [
            "ID",
            "题型",
            "intent",
            "next",
            "覆盖率",
            "引用",
            "审核",
            "延迟ms",
            "token",
            "轮次",
            "判定",
        ],
        console_rows,
    )
    print()
    print(
        f"答案覆盖率：{_pct(summary['answer_coverage'])}；"
        f"非拒答覆盖率：{_pct(summary['answer_coverage_non_refuse'])}；"
        f"引用命中率：{_pct(summary['citation_hit_rate'])}；"
        f"审核通过率：{_pct(summary['review_pass_rate'])}"
    )
    print(
        f"延迟：平均 {_num(summary['avg_latency_ms'], 1)} ms，"
        f"P50 {_num(summary['p50_latency_ms'], 1)} ms，"
        f"P95 {_num(summary['p95_latency_ms'], 1)} ms；"
        f"平均 token：{_num(summary['avg_tokens'], 1)}；"
        f"回环触发率：{_pct(summary['loop_trigger_rate'])}"
    )
    print(
        f"失败题数：{summary['failure_count']}/{summary['total']}；"
        f"分类：{summary['failure_category_counts']}"
    )

    meta = {
        "script": "eval_rag",
        "generated_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z"),
        "dataset": str(dataset_path),
        "report_dir": str(report_dir),
        "limit": args.limit,
        "user_scope": ["employee"],
    }
    payload = {
        "meta": meta,
        "summary": summary,
        "by_family": by_family,
        "results": results,
        "failures": failures,
    }
    markdown = build_markdown(meta, summary, by_family, results, failures)
    md_path, json_path = write_reports(report_dir, "rag_report", markdown, payload)
    print(f"报告已写入：{md_path}")
    print(f"报告已写入：{json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
