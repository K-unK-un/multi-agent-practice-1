"""检索元数据硬过滤：Chroma where 下推（向量路）与 MySQL 行级过滤（BM25 路）同源。

三条规则（两路必须完全一致，否则会出现“同一条款向量能查到、BM25 查不到”的诡异现象）：

1. status：默认只保留 active；include_archived=True（对比场景）时 active/archived 都保留。
2. effective_date：<= today 或为空。未填写生效日期的条款视为“不限”，不能被误杀。
   注意：Chroma 1.0.0 的 where 不接受字符串 metadata 的 $lte（会报
   "Expected operand value to be an int or a float"），因此向量路的 effective_date
   改为在 VectorRetriever 拿到结果后按 metadata 在 Python 侧过滤；
   MySQL/BM25 路仍由 apply_metadata_filters 行级过滤，两路语义保持一致。
3. scope：与 user_scope 有交集；scope 为空表示全员可见。

元数据存储约定（ingest 写入时必须遵守，否则 Chroma 过滤会漏/误杀）：

- Chroma metadata：
  * status 为单值字符串 "active" / "archived"；
  * effective_date 为 "YYYY-MM-DD" 字符串；未填写时必须显式写 ""（或 "0000-00-00"），
    不能缺 key —— Chroma 的 where 对缺失 key 一律视为不匹配，$eq "" 也救不回来；
  * scope 为单值字符串 "employee" / "manager" / ...，全员条款写 ""。
    注意不能写成逗号拼接字符串（如 "employee,manager"）：Chroma 的 where 只能做单值
    比较与 $in，无法对逗号串做集合匹配；需要多 scope 时请在入库侧拆成多个条款/文档。
- MySQL documents.scope 为 JSON 数组（如 ["employee"]），空数组 / 空字符串 / NULL 表示全员。
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date, datetime, timedelta, timezone
from typing import Any

from app.core.config import settings

_DEFAULT_SCOPE = "employee"
_EMPTY_DATE_VALUES = {"", "0000-00-00"}


# ---------------------------------------------------------------------------
# 内部工具：日期 / scope 归一化
# ---------------------------------------------------------------------------

def _today_date(today: date | str | None = None) -> date:
    """today 归一化为 date；默认取 settings.tz（Asia/Shanghai）的当天。"""
    if today is not None:
        if isinstance(today, datetime):
            return today.date()
        if isinstance(today, date):
            return today
        text = str(today).strip()[:10].replace("/", "-")
        try:
            return date.fromisoformat(text)
        except ValueError:
            pass
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo(settings.tz)).date()
    except Exception:
        # Windows 缺少 tzdata 时 zoneinfo 可能不可用；Asia/Shanghai 无夏令时，固定 +8 兜底
        return datetime.now(timezone(timedelta(hours=8))).date()


def _as_date_str(value: Any) -> str | None:
    """把 str/date/datetime 统一为 "YYYY-MM-DD"；空值/零日期/无法解析返回 None（表示不限）。"""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if text in _EMPTY_DATE_VALUES or text.startswith("0000-00-00"):
        return None
    text = text[:10].replace("/", "-")
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        # 无法解析的脏数据按“未填写”处理，宁可放宽也不误杀
        return None


def _normalize_user_scope(user_scope: list[str] | str | None) -> list[str]:
    """user_scope 归一化为非空字符串列表；空/None 默认 ["employee"]（最小权限）。"""
    if not user_scope:
        return [_DEFAULT_SCOPE]
    if isinstance(user_scope, str):
        parts = user_scope.replace("，", ",").split(",")
    else:
        parts = [str(item) for item in user_scope]
    cleaned = [part.strip() for part in parts if str(part).strip()]
    deduped = list(dict.fromkeys(cleaned))
    return deduped or [_DEFAULT_SCOPE]


def _row_scopes(value: Any) -> set[str]:
    """MySQL scope 归一化为集合：支持 JSON 数组、JSON 字符串、逗号串、单值。空集 = 全员。"""
    if value is None:
        return set()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return set()
        if text.startswith("["):
            try:
                loaded = json.loads(text)
                if isinstance(loaded, list):
                    return {str(item).strip() for item in loaded if str(item).strip()}
            except (TypeError, ValueError):
                pass
        return {
            part.strip()
            for part in text.replace("，", ",").split(",")
            if part.strip()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return {str(item).strip() for item in value if str(item).strip()}
    text = str(value).strip()
    return {text} if text else set()


def _as_row_dict(row: Any) -> dict | None:
    """兼容 dict / SQLAlchemy Row / ORM 对象；无法转换返回 None。"""
    if isinstance(row, Mapping):
        return dict(row)
    mapping = getattr(row, "_mapping", None)
    if mapping is not None:
        return dict(mapping)
    if hasattr(row, "__dict__"):
        return {k: v for k, v in vars(row).items() if not k.startswith("_")}
    return None


# ---------------------------------------------------------------------------
# 向量路：Chroma where 下推
# ---------------------------------------------------------------------------

def is_effective(value: Any, today: date | str | None = None) -> bool:
    """effective_date 为空/无法解析视为不限；否则必须 <= 今天。

    供向量路在拿到 Chroma 结果后做 Python 侧过滤（Chroma where 不支持字符串 $lte）。
    """
    effective = _as_date_str(value)
    if effective is None:
        return True
    return effective <= _today_date(today).isoformat()


def build_chroma_where(
    user_scope: list[str] | None,
    include_archived: bool = False,
    today: date | str | None = None,
) -> dict:
    """构造 Chroma where 过滤条件（{"$and": [...]} 语法）。

    - status：include_archived → {"$in": ["active", "archived"]}；否则 "active"。
    - scope：$in user_scope，或等于空串（全员条款）。
    - effective_date：不在 where 中下推（Chroma 1.0 对字符串 $lte 会报类型错误），
      由 VectorRetriever 调用 is_effective() 在 Python 侧过滤；`today` 参数保留兼容。

    Chroma 对缺失 metadata key 不匹配任何条件，因此 ingest 必须写空串占位。
    """
    scopes = _normalize_user_scope(user_scope)
    _ = today  # 保留参数：日期过滤已移到 Python 侧

    status_cond: dict = (
        {"status": {"$in": ["active", "archived"]}}
        if include_archived
        else {"status": {"$eq": "active"}}
    )

    # scope 为空串 = 全员可见；非空时必须落在用户可见范围内
    scope_cond = {
        "$or": [
            {"scope": {"$in": scopes}},
            {"scope": {"$eq": ""}},
        ]
    }

    return {"$and": [status_cond, scope_cond]}


# ---------------------------------------------------------------------------
# BM25 路：MySQL 行级过滤（与 build_chroma_where 语义一致）
# ---------------------------------------------------------------------------

def apply_metadata_filters(
    rows: list[dict],
    user_scope: list[str] | None,
    include_archived: bool = False,
    today: date | str | None = None,
) -> list[dict]:
    """对 MySQL 条款行做与 build_chroma_where 同源的硬过滤，保持输入顺序。

    - status：默认仅 active；include_archived 时仅 active/archived（与 Chroma 的
      ``$in ["active","archived"]`` 保持一致，缺失/空 status 两路都丢弃）。
    - effective_date：<= today 或为空/无法解析（视为不限）。
    - scope：与 user_scope 交集非空，或 scope 为空（全员）。
    """
    scopes = set(_normalize_user_scope(user_scope))
    today_str = _today_date(today).isoformat()
    kept: list[dict] = []

    for raw_row in rows or []:
        row = _as_row_dict(raw_row)
        if row is None:
            continue

        status = str(row.get("status") or "").strip().lower()
        if include_archived:
            if status not in ("active", "archived"):
                continue
        elif status != "active":
            continue

        effective = _as_date_str(row.get("effective_date"))
        if effective is not None and effective > today_str:
            continue

        row_scopes = _row_scopes(row.get("scope"))
        if row_scopes and not (row_scopes & scopes):
            continue

        kept.append(row)

    return kept
