"""PolicyMind 知识库构建脚本：解析 → 清洗 → 条款切分 → 规则抽取 → 双路索引。

用法：
    python scripts/ingest_all.py                      # 默认构建 data/docs
    python scripts/ingest_all.py --data-dir data/docs
    python scripts/ingest_all.py --file a.pdf --file b.docx
    python scripts/ingest_all.py --rebuild            # 危险：先清空 Chroma collection
    python scripts/ingest_all.py --no-rules           # 只建检索索引，不抽规则表
    python scripts/ingest_all.py --scope employee --scope manager
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # 某些重定向环境没有 reconfigure
    pass

import argparse  # noqa: E402

from app.core.logger import get_logger  # noqa: E402
from app.ingest.pipeline import SUPPORTED_SUFFIXES, run_ingest  # noqa: E402
from app.store.chroma import get_chroma  # noqa: E402

logger = get_logger("scripts.ingest_all")


def _scan_files(data_dir: str) -> list[str]:
    root = Path(data_dir)
    if not root.is_dir():
        return []
    return [
        str(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and not path.name.startswith("~$")  # Word 临时锁文件
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PolicyMind 知识库构建（解析/切分/规则/索引）")
    parser.add_argument(
        "--data-dir",
        default="data/docs",
        help="制度文档目录（默认 data/docs，递归扫描 pdf/docx/md/txt）",
    )
    parser.add_argument(
        "--file",
        action="append",
        default=None,
        help="指定单个文件，可重复；指定后忽略 --data-dir",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="先删除 Chroma collection 再重建（危险操作，会清空现有向量）",
    )
    parser.add_argument(
        "--no-rules",
        action="store_true",
        help="跳过规则表抽取（不调用 LLM）",
    )
    parser.add_argument(
        "--scope",
        action="append",
        default=None,
        help="文档可见范围标签，可重复（默认 employee）",
    )
    return parser


def _print_summary(result: dict) -> None:
    print("=" * 56)
    print("PolicyMind 知识库构建完成")
    print(f"  文档数：{result.get('documents', 0)}")
    print(f"  条款数：{result.get('clauses', 0)}")
    print(f"  规则数：{result.get('rules', 0)}")
    print(f"  向量数：{result.get('chroma_count', 0)}")
    warnings = result.get("warnings") or []
    if warnings:
        print(f"  警告（{len(warnings)}）：")
        for warning in warnings:
            print(f"    - {warning}")
    else:
        print("  警告：无")
    print("=" * 56)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        if args.rebuild:
            # 危险操作：delete_collection 会清空整个向量库，且不可恢复。
            # 仅建议在数据源变更、嵌入模型/维度调整后使用；生产环境应先确认备份。
            logger.warning("--rebuild：正在清空 Chroma collection ...")
            print("警告：--rebuild 将清空现有向量库并全量重建")
            get_chroma().delete_collection()

        if args.file:
            files = [str(Path(item)) for item in args.file]
        else:
            files = _scan_files(args.data_dir)

        if files:
            result = run_ingest(
                files,
                scope=args.scope,
                with_rules=not args.no_rules,
            )
        else:
            logger.warning("知识库目录为空或不存在: %s", args.data_dir)
            result = {"documents": 0, "warnings": ["目录为空"]}

        _print_summary(result)
        return 0
    except Exception as exc:
        print(f"知识库构建失败: {exc}", file=sys.stderr)
        logger.exception("知识库构建失败")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
