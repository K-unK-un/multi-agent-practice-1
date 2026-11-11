"""PolicyMind 调度层评估：验证 supervisor.decide_plan 的意图识别与专家分派质量。

用途
----
读取 eval/dataset.jsonl，逐题调用 decide_plan(question, user_scope=["employee"])，
统计路由准确率、漏派率、多派率与拒答准确率，并输出 Markdown + JSON 双份报告。

指标定义
--------
- 路由准确率 = 路由判定正确的题数 / 参与评估题数
  * expected_agent == "refuse"：intent 归一化为 refuse 即正确
    （规格允许 next == "clarify" 且 intent == "refuse" 的写法，二者等价）；
  * 其他题型：next 归一化为专家名后与 expected_agent 一致即正确。
- 漏派率 = expected_agent（专家类：policy_qa / compare / eligibility）未出现在
  plan.subtasks[].agent 集合中的题数 / 专家类题数。
- 任务分解完整率 = 1 - 漏派率。
- 多派率 = 专家类题目中出现“多余分派”的题数 / 专家类题数。
  * 期望子任务数 <= 1：派发了 expected_agent 以外的专家，或同一题派发多个子任务；
  * 期望子任务数 > 1：子任务数超过期望值才算多派，合理的第二专家不惩罚；
  * refuse 题派发任何专家都算多派（单独计入该题的 issues）。
- 平均多余子任务数 = Σ max(0, 实际专家子任务数 - 期望子任务数) / 专家类题数。
- 拒答准确率 = refuse 题中 intent 表示拒答的题数 / refuse 题数。

用法示例
--------
    python scripts/eval_routing.py
    python scripts/eval_routing.py --limit 5
    python scripts/eval_routing.py --dataset eval/dataset.jsonl --report-dir eval/reports

依赖
----
app.agents.supervisor.decide_plan（其他开发者并行实现，本脚本只按签名调用）。
需要可用的 DEEPSEEK_API_KEY；未配置或调用失败时打印中文提示并以 exit 1 退出。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# 统一工作目录：让 pydantic-settings 读取项目根目录的 .env，并让相对路径按项目根解析。
os.chdir(PROJECT_ROOT)

from app.core.config import settings  # noqa: E402

try:  # 其他开发者并行实现；缺失时给出可读提示，而不是裸 traceback
    from app.agents.supervisor import decide_plan  # noqa: E402
except ImportError as exc:  # pragma: no cover - 依赖尚未就绪时触发
    print(f"错误：无法导入 app.agents.supervisor.decide_plan：{exc}", file=sys.stderr)
    print("请确认调度模块已完成、项目依赖已安装后再运行本脚本。", file=sys.stderr)
    raise SystemExit(1)

FAMILY_ORDER = ["fact", "cross", "eligibility", "compare", "refuse"]
EXPERT_AGENTS = {"policy_qa", "compare", "eligibility"}


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------
def _resolve(path_value: str | os.PathLike[str]) -> Path:
    """相对路径按项目根目录解析，避免从其他目录运行时写到错误位置。"""
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
    """把 pydantic 模型、枚举、set 等转成 json.dump 可序列化的结构。"""
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


def _plan_dict(plan: Any) -> dict[str, Any]:
    if isinstance(plan, dict):
        return plan
    for method in ("model_dump", "dict"):
        fn = getattr(plan, method, None)
        if callable(fn):
            try:
                return dict(fn())
            except Exception:  # pragma: no cover - 仅兜底
                pass
    return {
        key: getattr(plan, key, None)
        for key in ("intent", "confidence", "subtasks", "next", "done", "reason")
    }


def normalize_agent(value: Any) -> str:
    """把 'policy_qa_agent' / 'qa' / 'refusal' 等写法统一成评估用的短名。"""
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
    """调度正确性判定（与 README 11.1 的期望路由对齐）。"""
    if expected == "refuse":
        # 规格：refuse 题看 intent 是否 refuse；next=="clarify" 且 intent=="refuse" 也接受。
        return intent == "refuse" or (next_agent == "clarify" and intent == "refuse")
    return next_agent == expected


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    if not denominator:
        return None
    return round(float(numerator) / float(denominator), 4)


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _md(value: Any) -> str:
    """Markdown 表格单元转义。"""
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
    required = {"id", "question", "family", "expected_agent"}
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


def evaluate_one(question: dict[str, Any]) -> dict[str, Any]:
    """调用一次 decide_plan，返回单题判定与归一化后的调度结果。"""
    try:
        raw_plan = decide_plan(question["question"], user_scope=["employee"])
    except Exception as exc:  # LLM 调用失败或 key 无效
        print(
            f"错误：调用 decide_plan 失败（{question.get('id', '?')}）：{exc}",
            file=sys.stderr,
        )
        print(
            "请先配置 DEEPSEEK_API_KEY（.env 中填写），并确认模型服务与网络可用。",
            file=sys.stderr,
        )
        raise SystemExit(1)

    plan = _plan_dict(raw_plan)
    intent = normalize_agent(_get(plan, "intent"))
    next_agent = normalize_agent(_get(plan, "next"))
    reason = str(_get(plan, "reason", "") or "")

    # 实际实现 decide_plan 会吞掉 LLM 异常并降级为 policy_qa；此类结果不能作为
    # 调度质量样本，按“调用失败”处理（规格要求中文提示 + exit 1）。
    if "调度模型调用失败" in reason or "降级派发" in reason:
        print(
            f"错误：decide_plan 调用失败（{question.get('id', '?')}），"
            f"已触发降级：{reason}",
            file=sys.stderr,
        )
        print(
            "请先配置 DEEPSEEK_API_KEY（.env 中填写），并确认模型服务与网络可用。",
            file=sys.stderr,
        )
        raise SystemExit(1)

    raw_subtasks = _get(plan, "subtasks", []) or []
    if isinstance(raw_subtasks, dict):
        raw_subtasks = [raw_subtasks]
    elif not isinstance(raw_subtasks, (list, tuple, set)):
        raw_subtasks = []

    subtasks: list[dict[str, Any]] = []
    for item in raw_subtasks:
        subtasks.append(
            {
                "agent": normalize_agent(_get(item, "agent")),
                "goal": str(_get(item, "goal", "") or ""),
                "hint": str(_get(item, "hint", "") or ""),
                "confidence": _to_float(_get(item, "confidence")),
            }
        )

    expected = normalize_agent(question.get("expected_agent"))
    try:
        expected_subtasks = int(question.get("subtasks", 1) or 0)
    except (TypeError, ValueError):
        expected_subtasks = 1

    dispatched = [s["agent"] for s in subtasks if s["agent"] in EXPERT_AGENTS]
    dispatched_set = set(dispatched)
    correct = _routing_correct(expected, intent, next_agent)

    # 漏派：仅专家类题目适用；refuse 题不产生专家分派，记为 None。
    missing_dispatch: bool | None = None
    if expected in EXPERT_AGENTS:
        missing_dispatch = expected not in dispatched_set

    # 多派：refuse 题派任何专家都算多派；专家题按期望子任务数判定。
    extra_subtasks = 0
    over_dispatch = False
    if expected == "refuse":
        extra_subtasks = len(dispatched)
        over_dispatch = extra_subtasks > 0
    elif expected in EXPERT_AGENTS:
        base = max(expected_subtasks, 1)
        extra_subtasks = max(0, len(dispatched) - base)
        unexpected_agents = sorted(dispatched_set - {expected}) if base <= 1 else []
        over_dispatch = extra_subtasks > 0 or bool(unexpected_agents)
    else:
        # 未知 expected_agent：不参与漏派/多派统计，但路由判定仍会失败。
        extra_subtasks = max(0, len(dispatched) - max(expected_subtasks, 1))
        over_dispatch = extra_subtasks > 0

    issues: list[str] = []
    if not correct:
        issues.append("路由错")
    if missing_dispatch:
        issues.append(f"漏派:{expected}")
    if over_dispatch:
        if expected == "refuse":
            issues.append("拒答题派发了专家:" + ",".join(dispatched))
        else:
            unexpected = sorted(dispatched_set - {expected}) if expected_subtasks <= 1 else []
            details: list[str] = []
            if extra_subtasks > 0:
                details.append(f"多派{extra_subtasks}个子任务")
            if unexpected:
                details.append("多派其他专家:" + ",".join(unexpected))
            issues.append("；".join(details) or "多派子任务")

    return {
        "id": str(question.get("id", "")),
        "family": str(question.get("family", "")),
        "question": question["question"],
        "expected_agent": expected,
        "expected_subtasks": expected_subtasks,
        "intent": intent,
        "next": next_agent,
        "confidence": _to_float(_get(plan, "confidence")),
        "subtasks": subtasks,
        "dispatched_agents": [s["agent"] for s in subtasks],
        "dispatched_experts": dispatched,
        "routing_correct": correct,
        "missing_dispatch": missing_dispatch,
        "extra_subtasks": extra_subtasks,
        "over_dispatch": over_dispatch,
        "done": _jsonable(_get(plan, "done")),
        "reason": reason,
        "issues": issues,
        "passed": not issues,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    expert_rows = [r for r in rows if r["expected_agent"] in EXPERT_AGENTS]
    refuse_rows = [r for r in rows if r["expected_agent"] == "refuse"]
    correct = sum(1 for r in rows if r["routing_correct"])
    missing = sum(1 for r in expert_rows if r["missing_dispatch"])
    over = sum(1 for r in expert_rows if r["over_dispatch"])
    extra_total = sum(r["extra_subtasks"] for r in expert_rows)
    refuse_correct = sum(1 for r in refuse_rows if r["routing_correct"])
    return {
        "total": total,
        "routing_correct": correct,
        "routing_accuracy": _ratio(correct, total),
        "expert_questions": len(expert_rows),
        "missing_dispatch": missing,
        "missing_dispatch_rate": _ratio(missing, len(expert_rows)),
        "decomposition_completeness": _ratio(len(expert_rows) - missing, len(expert_rows)),
        "over_dispatch_questions": over,
        "over_dispatch_rate": _ratio(over, len(expert_rows)),
        "avg_extra_subtasks": (
            round(extra_total / len(expert_rows), 4) if expert_rows else None
        ),
        "refuse_questions": len(refuse_rows),
        "refuse_correct": refuse_correct,
        "refuse_accuracy": _ratio(refuse_correct, len(refuse_rows)),
    }


def summarize_by_family(rows: list[dict[str, Any]]) -> dict[str, Any]:
    families = FAMILY_ORDER + sorted({r["family"] for r in rows} - set(FAMILY_ORDER))
    result: dict[str, Any] = {}
    for family in families:
        family_rows = [r for r in rows if r["family"] == family]
        if family_rows:
            result[family] = summarize(family_rows)
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
        "# PolicyMind 调度层评估报告",
        "",
        f"- 生成时间：{meta['generated_at']}",
        f"- 测试集：`{meta['dataset']}`",
        f"- 评估题数：{summary['total']}（--limit={meta['limit']}）",
        f"- 调用方式：`decide_plan(question, user_scope=[\"employee\"])`",
        "",
        "## 概览指标",
        "",
        _md_table(
            ["指标", "数值", "分子/分母", "定义"],
            [
                [
                    "路由准确率",
                    _pct(summary["routing_accuracy"]),
                    f"{summary['routing_correct']}/{summary['total']}",
                    "next（refuse 题看 intent）与 expected_agent 一致",
                ],
                [
                    "漏派率",
                    _pct(summary["missing_dispatch_rate"]),
                    f"{summary['missing_dispatch']}/{summary['expert_questions']}",
                    "专家类题的 expected_agent 不在 subtasks 中",
                ],
                [
                    "任务分解完整率",
                    _pct(summary["decomposition_completeness"]),
                    f"{summary['expert_questions'] - summary['missing_dispatch']}/{summary['expert_questions']}",
                    "1 - 漏派率",
                ],
                [
                    "多派率",
                    _pct(summary["over_dispatch_rate"]),
                    f"{summary['over_dispatch_questions']}/{summary['expert_questions']}",
                    "派发了期望之外的专家或超出期望子任务数",
                ],
                [
                    "平均多余子任务数",
                    _num(summary["avg_extra_subtasks"], 3),
                    "-",
                    "Σ max(0, 实际专家子任务数 - 期望子任务数) / 专家类题数",
                ],
                [
                    "拒答准确率",
                    _pct(summary["refuse_accuracy"]),
                    f"{summary['refuse_correct']}/{summary['refuse_questions']}",
                    "refuse 题 intent 归一化为 refuse",
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
                _pct(item["routing_accuracy"]),
                _pct(item["missing_dispatch_rate"]),
                _pct(item["over_dispatch_rate"]),
                _pct(item["refuse_accuracy"]),
            ]
        )
    lines.append(
        _md_table(
            ["题型", "题数", "路由准确率", "漏派率", "多派率", "拒答准确率"],
            family_rows,
        )
    )

    lines.extend(
        [
            "",
            "## 逐题明细",
            "",
        ]
    )
    detail_rows = []
    for item in results:
        agents = ", ".join(s["agent"] for s in item["subtasks"]) or "-"
        detail_rows.append(
            [
                item["id"],
                item["family"],
                item["expected_agent"],
                item["intent"] or "-",
                item["next"] or "-",
                agents,
                "是" if item["routing_correct"] else "否",
                "-" if item["missing_dispatch"] is None else ("是" if item["missing_dispatch"] else "否"),
                "是" if item["over_dispatch"] else "否",
                "PASS" if item["passed"] else "FAIL",
                "；".join(item["issues"]) or "-",
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
                "子任务 Agent",
                "路由正确",
                "漏派",
                "多派",
                "判定",
                "问题",
            ],
            detail_rows,
        )
    )

    lines.extend(
        [
            "",
            "## 失败案例清单",
            "",
        ]
    )
    if not failures:
        lines.append("无失败案例。")
    for item in failures:
        lines.append(f"### {item['id']}（{item['family']}）")
        lines.append("")
        lines.append(f"- 问题：{item['question']}")
        lines.append(
            f"- 期望：{item['expected_agent']}（期望子任务数 {item['expected_subtasks']}）；"
            f"实际：intent={item['intent'] or '-'}，next={item['next'] or '-'}"
        )
        agents = (
            "；".join(
                f"{s['agent']}（goal={s['goal'] or '无'}）" for s in item["subtasks"]
            )
            or "无"
        )
        lines.append(f"- 实际子任务：{agents}")
        lines.append(f"- 问题：{'；'.join(item['issues']) or '未知'}")
        if item["reason"]:
            lines.append(f"- 调度理由：{item['reason']}")
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
        description="PolicyMind 调度层评估：路由准确率 / 漏派率 / 多派率",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python scripts/eval_routing.py\n"
            "  python scripts/eval_routing.py --limit 5\n"
            "  python scripts/eval_routing.py --dataset eval/dataset.jsonl "
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
    _ensure_api_key()

    dataset_path = _resolve(args.dataset)
    report_dir = _resolve(args.report_dir)
    questions = load_dataset(dataset_path, args.limit)

    print(f"调度层评估开始：{len(questions)} 题，数据集 {dataset_path}")
    results: list[dict[str, Any]] = []
    for index, question in enumerate(questions, 1):
        print(f"[{index}/{len(questions)}] {question.get('id', '?')} ...", flush=True)
        results.append(evaluate_one(question))

    summary = summarize(results)
    by_family = summarize_by_family(results)
    failures = [r for r in results if not r["passed"]]

    console_rows = [
        [
            r["id"],
            r["intent"] or "-",
            r["next"] or "-",
            r["expected_agent"],
            "PASS" if r["passed"] else "FAIL",
            "；".join(r["issues"]) or "-",
        ]
        for r in results
    ]
    print()
    _print_table(["ID", "intent", "next", "期望", "判定", "备注"], console_rows)
    print()
    print(
        f"路由准确率：{_pct(summary['routing_accuracy'])} "
        f"（{summary['routing_correct']}/{summary['total']}）"
    )
    print(
        f"漏派率：{_pct(summary['missing_dispatch_rate'])} "
        f"（{summary['missing_dispatch']}/{summary['expert_questions']}）"
    )
    print(
        f"多派率：{_pct(summary['over_dispatch_rate'])} "
        f"（{summary['over_dispatch_questions']}/{summary['expert_questions']}）"
    )
    print(
        f"拒答准确率：{_pct(summary['refuse_accuracy'])} "
        f"（{summary['refuse_correct']}/{summary['refuse_questions']}）"
    )
    print(f"失败题数：{len(failures)}/{summary['total']}")

    meta = {
        "script": "eval_routing",
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
    md_path, json_path = write_reports(report_dir, "routing_report", markdown, payload)
    print(f"报告已写入：{md_path}")
    print(f"报告已写入：{json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
