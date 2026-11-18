"""初始化 / 迁移 MySQL 表结构（幂等）。

用法：
    python scripts/init_db.py

会创建缺失的表（含 ingest_records 入库台账），并为已有表补齐新增列
（documents.source_name / documents.content_fingerprint 等），不会删除数据。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # 某些重定向环境没有 reconfigure
    pass

from app.store.mysql import get_mysql  # noqa: E402


def main() -> int:
    try:
        get_mysql().init_db()
    except Exception as exc:
        print(f"MySQL 表结构初始化失败: {exc}", file=sys.stderr)
        return 1
    print("MySQL 表结构已检查/迁移完成（含 ingest_records 入库台账）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
