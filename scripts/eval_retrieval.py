"""PolicyMind 检索层评估：验证 retrieval_graph.invoke_retrieval 的召回与精排质量。

用途
----
读取 eval/dataset.jsonl，逐题调用 invoke_retrieval(question, mode="auto",
user_scope=["employee"])，把返回的 contexts 按 (doc_id, article_no) 与题目
article_ids 比对，统计 Recall@5 / Recall@10 / MRR / Context Precision 等指标，
并输出 Markdown + JSON 双份报告。

指标定义
--------
设某题金标条款集合为 G（去重后的 (doc_id, article_no)），检索返回去重后的
有序条款列表为 R（按精排顺序）：

- Recall@k = |G ∩ R[:k]| / |G|，再对参与评估的题目取算术平均。
  多个金标条款时按比例计分，例如命中 1/3 记 0.333。
- MRR = 首个相关条款排名倒数的平均值；若 R 中无相关条款则记 0。
  即 MRR = mean(1 / rank_first_relevant)。
- Context Precision@5 = |R[:5] 中相关条款数| / |R[:5]|，再取平均。
  分母取实际返回条数（不足 5 条时不按 5 强行惩罚）；R 为空记 0。
- 平均 loops = mean(invoke_retrieval 返回的 loops)。
- 平均延迟 = mean(单次 invoke_retrieval 墙钟耗时，毫秒)；同时报告 P50。

article_ids 格式
----------------
题目中的 article_ids 建议写成字符串 "doc_id::article_no"（也兼容
"doc_id#article_no"、"doc_id|article_no" 以及 {"doc_id": ..., "article_no": ...}）。
article_ids 为空表示该题尚未回填金标，记为 skipped 且不参与指标计算。
全部题目都为空时（尚未建库/回填），脚本打印提示并以 exit 0 正常退出。

用法示例
--------
    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --limit 5
    python scripts/eval_retrieval.py --dataset eval/dataset.jsonl --report-dir eval/reports

依赖
----
app.graph.retrieval_graph.invoke_retrieval（其他开发者并行实现，本脚本只按签名调用）。
需要已运行 scripts/ingest_all.py 建库，且 Chroma / MySQL / Ollama 服务可用。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import unicodedata
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
RECALL_KS = (5, 10)


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


# ---------------------------------------------------------------------------
# 数据集与 article_ids
# ---------------------------------------------------------------------------
def load_dataset(path: Path, limit: int | None) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"错误：测试集不存在：{path}", file=sys.stderr)
        raise SystemExit(1)
    required = {"id", "question", "expected_agent"}
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


def _normalize_article_no(article_no: Any) -> str:
    """条款号归一化：'第12条' -> '12'，与 MySQL clause.article_no 的存法对齐。"""
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


# ---------------------------------------------------------------------------
# 检索评估
# ---------------------------------------------------------------------------
def _load_invoke_retrieval() -> Callable[..., dict[str, Any]] | None:
    """延迟导入：article_ids 全空时无需加载检索依赖（Chroma/Ollama 等）。"""
    try:
        from app.graph.retrieval_graph import invoke_retrieval
    except ImportError as exc:  # pragma: no cover - 依赖尚未就绪时触发
        print(
            f"错误：无法导入 app.graph.retrieval_graph.invoke_retrieval：{exc}",
            file=sys.stderr,
        )
        print(
            "请确认检索子图模块已完成、项目依赖已安装后再运行本脚本。",
            file=sys.stderr,
        )
        return None
    return invoke_retrieval


def _loops_value(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return float(len(value))
    return 0.0


def evaluate_one(
    question: dict[str, Any],
    gold_ids: list[tuple[str, str]],
    invoke_retrieval: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    family = str(question.get("family", ""))
    expected_agent = str(question.get("expected_agent", "")).lower()
    # 版本对比需要看到 archived 旧版；检索子图 mode=compare 会打开 include_archived。
    mode = "compare" if family == "compare" or expected_agent == "compare" else "auto"
    include_archived = True if mode == "compare" else None

    started = time.perf_counter()
    try:
        raw_result = invoke_retrieval(
            question["question"],
            mode=mode,
            user_scope=["employee"],
            include_archived=include_archived,
        )
    except Exception as exc:
        print(
            f"错误：检索调用失败（{question.get('id', '?')}）：{exc}",
            file=sys.stderr,
        )
        print(
            "请确认已完成 scripts/ingest_all.py 建库，且 Chroma / MySQL / Ollama "
            "服务已启动。",
            file=sys.stderr,
        )
        raise SystemExit(1)
    latency_ms = round((time.perf_counter() - started) * 1000.0, 2)

    result = raw_result or {}
    contexts_raw = _get(result, "contexts", []) or []
    if isinstance(contexts_raw, dict):
        contexts_raw = [contexts_raw]
    elif not isinstance(contexts_raw, (list, tuple, set)):
        contexts_raw = []
    contexts = list(contexts_raw)
    notice = str(_get(result, "notice", "") or "")

    # invoke_retrieval 内部会捕获异常并返回空 contexts + "检索失败：..." 的 notice；
    # 这不是“未回填金标”的 skipped 场景，应作为环境/索引故障提示并退出。
    if notice.startswith("检索失败") and not contexts:
        print(
            f"错误：检索子图执行失败（{question.get('id', '?')}）：{notice}",
            file=sys.stderr,
        )
        print(
            "请确认已完成 scripts/ingest_all.py 建库，且 Chroma / MySQL / Ollama "
            "服务已启动。",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # 按 (doc_id, article_no) 去重并保留精排顺序；缺少键的 context 记为空键，不参与命中。
    ranked: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for context in contexts:
        key = _article_key(_get(context, "doc_id"), _get(context, "article_no"))
        if key in seen:
            continue
        seen.add(key)
        ranked.append(
            {
                "doc_id": key[0],
                "article_no": key[1],
                "score": _to_float(_get(context, "score")),
            }
        )

    gold_set = set(gold_ids)
    relevance = [
        1 if (item["doc_id"], item["article_no"]) in gold_set else 0 for item in ranked
    ]

    recall_at: dict[str, float] = {}
    for k in RECALL_KS:
        recall_at[str(k)] = (
            round(sum(relevance[:k]) / len(gold_set), 4) if gold_set else 0.0
        )

    mrr = 0.0
    for rank, flag in enumerate(relevance, 1):
        if flag:
            mrr = round(1.0 / rank, 4)
            break

    precision_at_5 = round(sum(relevance[:5]) / max(1, len(relevance[:5])), 4)
    precision_at_10 = round(sum(relevance[:10]) / max(1, len(relevance[:10])), 4)

    hit_keys = {(item["doc_id"], item["article_no"]) for item in ranked}
    missing_ids = [
        f"{doc_id}::{article_no}"
        for doc_id, article_no in gold_ids
        if (doc_id, article_no) not in hit_keys
    ]

    return {
        "id": str(question.get("id", "")),
        "family": family,
        "question": question["question"],
        "status": "evaluated",
        "gold_ids": [f"{d}::{a}" for d, a in gold_ids],
        "mode": mode,
        "include_archived": include_archived,
        "retrieved_count": len(ranked),
        "contexts": ranked,
        "recall_at_5": recall_at["5"],
        "recall_at_10": recall_at["10"],
        "mrr": mrr,
        "context_precision_at_5": precision_at_5,
        "context_precision_at_10": precision_at_10,
        "loops": _loops_value(_get(result, "loops", 0)),
        "coverage": _to_float(_get(result, "coverage")),
        "missing": _jsonable(_get(result, "missing", []) or []),
        "notice": notice,
        "latency_ms": latency_ms,
        "missing_ids": missing_ids,
        "completed": not missing_ids,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [r for r in records if r["status"] == "evaluated"]
    skipped = [r for r in records if r["status"] == "skipped"]
    latencies = [r["latency_ms"] for r in evaluated]
    return {
        "total_questions": len(records),
        "evaluated": len(evaluated),
        "skipped": len(skipped),
        "recall_at_5": _mean([r["recall_at_5"] for r in evaluated]),
        "recall_at_10": _mean([r["recall_at_10"] for r in evaluated]),
        "mrr": _mean([r["mrr"] for r in evaluated]),
        "context_precision_at_5": _mean(
            [r["context_precision_at_5"] for r in evaluated]
        ),
        "context_precision_at_10": _mean(
            [r["context_precision_at_10"] for r in evaluated]
        ),
        "avg_loops": _mean([r["loops"] for r in evaluated]),
        "avg_latency_ms": _mean(latencies),
        "p50_latency_ms": _percentile(latencies, 50),
        "completed_questions": sum(1 for r in evaluated if r["completed"]),
        "skip_reason": "article_ids 为空，未回填金标条款",
    }


def summarize_by_family(records: list[dict[str, Any]]) -> dict[str, Any]:
    families = FAMILY_ORDER + sorted({r["family"] for r in records} - set(FAMILY_ORDER))
    result: dict[str, Any] = {}
    for family in families:
        family_records = [r for r in records if r["family"] == family]
        if not family_records:
            continue
        evaluated = [r for r in family_records if r["status"] == "evaluated"]
        skipped = [r for r in family_records if r["status"] == "skipped"]
        result[family] = {
            "total": len(family_records),
            "evaluated": len(evaluated),
            "skipped": len(skipped),
            "recall_at_5": _mean([r["recall_at_5"] for r in evaluated]),
            "recall_at_10": _mean([r["recall_at_10"] for r in evaluated]),
            "mrr": _mean([r["mrr"] for r in evaluated]),
            "context_precision_at_5": _mean(
                [r["context_precision_at_5"] for r in evaluated]
            ),
            "avg_loops": _mean([r["loops"] for r in evaluated]),
            "avg_latency_ms": _mean([r["latency_ms"] for r in evaluated]),
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
    skipped: list[dict[str, Any]],
) -> str:
    lines = [
        "# PolicyMind 检索层评估报告",
        "",
        f"- 生成时间：{meta['generated_at']}",
        f"- 测试集：`{meta['dataset']}`",
        f"- 题目数：{summary['total_questions']}，参与评估：{summary['evaluated']}，"
        f"跳过：{summary['skipped']}（article_ids 为空）",
        "- 调用方式：`invoke_retrieval(question, mode=\"auto\", user_scope=[\"employee\"])`"
        "（compare 题使用 mode=\"compare\" 且 include_archived=True）",
        "",
        "## 概览指标",
        "",
        _md_table(
            ["指标", "数值", "说明"],
            [
                [
                    "Recall@5",
                    _pct(summary["recall_at_5"]),
                    "|G ∩ R[:5]| / |G| 的平均值",
                ],
                [
                    "Recall@10",
                    _pct(summary["recall_at_10"]),
                    "|G ∩ R[:10]| / |G| 的平均值",
                ],
                [
                    "MRR",
                    _num(summary["mrr"], 4),
                    "首个相关条款排名倒数的平均值",
                ],
                [
                    "Context Precision@5",
                    _pct(summary["context_precision_at_5"]),
                    "R[:5] 中相关条款占比的平均值",
                ],
                [
                    "Context Precision@10",
                    _pct(summary["context_precision_at_10"]),
                    "R[:10] 中相关条款占比的平均值",
                ],
                [
                    "平均 loops",
                    _num(summary["avg_loops"], 2),
                    "检索子图平均回环次数",
                ],
                [
                    "平均延迟",
                    f"{_num(summary['avg_latency_ms'], 2)} ms",
                    "单次 invoke_retrieval 墙钟耗时",
                ],
                [
                    "P50 延迟",
                    f"{_num(summary['p50_latency_ms'], 2)} ms",
                    "延迟中位数",
                ],
                [
                    "完整召回题数",
                    f"{summary['completed_questions']}/{summary['evaluated']}",
                    "全部金标条款都进入返回列表的题数",
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
                item["evaluated"],
                item["skipped"],
                _pct(item["recall_at_5"]),
                _pct(item["recall_at_10"]),
                _num(item["mrr"], 4),
                _pct(item["context_precision_at_5"]),
                _num(item["avg_loops"], 2),
                _num(item["avg_latency_ms"], 2),
            ]
        )
    lines.append(
        _md_table(
            [
                "题型",
                "题数",
                "评估",
                "跳过",
                "Recall@5",
                "Recall@10",
                "MRR",
                "CP@5",
                "平均 loops",
                "平均延迟(ms)",
            ],
            family_rows,
        )
    )

    lines.extend(["", "## 逐题明细", ""])
    detail_rows = []
    for item in results:
        if item["status"] == "skipped":
            detail_rows.append(
                [
                    item["id"],
                    item["family"],
                    0,
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "skipped（article_ids 为空）",
                ]
            )
        else:
            detail_rows.append(
                [
                    item["id"],
                    item["family"],
                    len(item["gold_ids"]),
                    item["retrieved_count"],
                    _pct(item["recall_at_5"]),
                    _pct(item["recall_at_10"]),
                    _num(item["mrr"], 4),
                    _pct(item["context_precision_at_5"]),
                    _num(item["loops"], 2),
                    _num(item["latency_ms"], 2),
                    "PASS" if item["completed"] else "FAIL",
                ]
            )
    lines.append(
        _md_table(
            [
                "ID",
                "题型",
                "金标数",
                "召回数",
                "Recall@5",
                "Recall@10",
                "MRR",
                "CP@5",
                "loops",
                "延迟(ms)",
                "判定",
            ],
            detail_rows,
        )
    )

    lines.extend(["", "## 失败案例清单", ""])
    if not failures:
        lines.append("无失败案例（所有已评估题目的金标条款均被完整召回）。")
    for item in failures:
        issue = "完全未命中" if item["mrr"] == 0 else "部分未命中"
        lines.append(f"### {item['id']}（{item['family']}，{issue}）")
        lines.append("")
        lines.append(f"- 问题：{item['question']}")
        lines.append(f"- 金标条款：{', '.join(item['gold_ids'])}")
        top = item["contexts"][:10]
        top_text = (
            ", ".join(f"{c['doc_id']}::{c['article_no']}" for c in top) or "无"
        )
        lines.append(f"- Top-10 召回：{top_text}")
        lines.append(f"- 未命中金标：{', '.join(item['missing_ids']) or '无'}")
        lines.append(f"- 指标：Recall@5={_pct(item['recall_at_5'])}，"
                     f"Recall@10={_pct(item['recall_at_10'])}，MRR={_num(item['mrr'], 4)}")
        if item["notice"]:
            lines.append(f"- 检索缺口说明：{item['notice']}")
        lines.append("")

    lines.extend(["", "## 跳过题目（article_ids 为空）", ""])
    if not skipped:
        lines.append("无。")
    else:
        skip_rows = [
            [item["id"], item["family"], item["question"]] for item in skipped
        ]
        lines.append(_md_table(["ID", "题型", "问题"], skip_rows))

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
        description="PolicyMind 检索层评估：Recall@5/10、MRR、Context Precision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python scripts/eval_retrieval.py\n"
            "  python scripts/eval_retrieval.py --limit 5\n"
            "  python scripts/eval_retrieval.py --dataset eval/dataset.jsonl "
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


def main() -> int:
    args = build_parser().parse_args()
    dataset_path = _resolve(args.dataset)
    report_dir = _resolve(args.report_dir)
    questions = load_dataset(dataset_path, args.limit)

    prepared: list[tuple[dict[str, Any], list[tuple[str, str]]]] = [
        (question, parse_article_ids(question.get("article_ids")))
        for question in questions
    ]
    active = [(question, gold) for question, gold in prepared if gold]

    if not active:
        print("提示：本次评估范围内全部题目的 article_ids 为空，已跳过检索层评估。")
        print(
            "请先运行 scripts/ingest_all.py 建库，并在 dataset 回填 article_ids"
            "（建议格式：doc_id::article_no）。"
        )
        return 0

    invoke_retrieval = _load_invoke_retrieval()
    if invoke_retrieval is None:
        return 1

    print(f"检索层评估开始：{len(questions)} 题，其中 {len(active)} 题已回填金标")
    results: list[dict[str, Any]] = []
    for index, (question, gold_ids) in enumerate(prepared, 1):
        if not gold_ids:
            print(
                f"[{index}/{len(prepared)}] {question.get('id', '?')} skipped"
                "（article_ids 为空）"
            )
            results.append(
                {
                    "id": str(question.get("id", "")),
                    "family": str(question.get("family", "")),
                    "question": question["question"],
                    "status": "skipped",
                    "reason": "article_ids 为空，未回填金标条款",
                }
            )
            continue
        print(f"[{index}/{len(prepared)}] {question.get('id', '?')} ...", flush=True)
        results.append(evaluate_one(question, gold_ids, invoke_retrieval))

    summary = summarize(results)
    by_family = summarize_by_family(results)
    skipped = [r for r in results if r["status"] == "skipped"]
    failures = [
        r
        for r in results
        if r["status"] == "evaluated"
        and (
            (r["recall_at_10"] is not None and r["recall_at_10"] < 1.0)
            or r["mrr"] == 0
        )
    ]
    failures.sort(key=lambda r: (r["recall_at_5"], r["mrr"]))

    console_rows = []
    for item in results:
        if item["status"] == "skipped":
            console_rows.append(
                [item["id"], item["family"], "-", "-", "-", "-", "-", "skipped"]
            )
        else:
            console_rows.append(
                [
                    item["id"],
                    item["family"],
                    _pct(item["recall_at_5"]),
                    _pct(item["recall_at_10"]),
                    _num(item["mrr"], 3),
                    _pct(item["context_precision_at_5"]),
                    _num(item["latency_ms"], 1),
                    "PASS" if item["completed"] else "FAIL",
                ]
            )
    print()
    _print_table(
        ["ID", "题型", "R@5", "R@10", "MRR", "CP@5", "延迟ms", "判定"],
        console_rows,
    )
    print()
    print(
        f"Recall@5：{_pct(summary['recall_at_5'])}；"
        f"Recall@10：{_pct(summary['recall_at_10'])}；"
        f"MRR：{_num(summary['mrr'], 4)}；"
        f"Context Precision@5：{_pct(summary['context_precision_at_5'])}"
    )
    print(
        f"平均 loops：{_num(summary['avg_loops'], 2)}；"
        f"平均延迟：{_num(summary['avg_latency_ms'], 2)} ms；"
        f"P50 延迟：{_num(summary['p50_latency_ms'], 2)} ms"
    )
    print(
        f"评估 {summary['evaluated']} 题，跳过 {summary['skipped']} 题，"
        f"失败 {len(failures)} 题"
    )
    if all(r["retrieved_count"] == 0 for r in results if r["status"] == "evaluated"):
        print(
            "警告：所有已评估题目均返回 0 条上下文，请确认已运行 "
            "scripts/ingest_all.py 建库且向量库/BM25 索引可用。"
        )

    meta = {
        "script": "eval_retrieval",
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
        "skipped": skipped,
    }
    markdown = build_markdown(meta, summary, by_family, results, failures, skipped)
    md_path, json_path = write_reports(
        report_dir, "retrieval_report", markdown, payload
    )
    print(f"报告已写入：{md_path}")
    print(f"报告已写入：{json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
