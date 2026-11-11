"""RRF（Reciprocal Rank Fusion）融合：抹平向量分与 BM25 分的量纲差异。

按 (doc_id, article_no) 去重合并两路召回：
    rrf_score = Σ 1 / (k + rank_in_channel)
融合后 channel 字段升级为 channels 列表（记录命中的通道），score 回写为 rrf_score。
"""
from __future__ import annotations

from app.core.config import settings


def _hit_key(hit: dict) -> tuple[str, str]:
    return str(hit.get("doc_id") or ""), str(hit.get("article_no") or "")


def rrf_fuse(
    vector_hits: list[dict],
    bm25_hits: list[dict],
    k: int | None = None,
) -> list[dict]:
    """融合两路召回，返回按 rrf_score 降序、截断到 retrieval_top_k 的候选条款。"""
    k_value = int(k if k is not None else settings.rrf_k)
    if k_value < 0:
        k_value = 0

    merged: dict[tuple[str, str], dict] = {}

    for channel, hits in (("vector", vector_hits or []), ("bm25", bm25_hits or [])):
        seen: set[tuple[str, str]] = set()
        rank = 0
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            key = _hit_key(hit)
            if not key[0]:
                continue
            if key in seen:  # 同一通道内重复命中只按首次名次计分
                continue
            seen.add(key)
            rank += 1
            contribution = 1.0 / (k_value + rank)

            if key not in merged:
                # 去掉单通道字段，保留条款基础信息
                base = {
                    field: value
                    for field, value in hit.items()
                    if field not in ("channel", "channels", "score", "rrf_score")
                }
                base["channels"] = [channel]
                base["score"] = contribution
                base["rrf_score"] = contribution
                merged[key] = base
            else:
                item = merged[key]
                item["score"] += contribution
                item["rrf_score"] = item["score"]
                if channel not in item["channels"]:
                    item["channels"].append(channel)
                # 同一条款两路文本可能一致；取更长文本，避免子块被截断
                if len(hit.get("text") or "") > len(item.get("text") or ""):
                    item["text"] = hit.get("text") or ""
                    item["page"] = hit.get("page", item.get("page"))
                if hit.get("title") and not item.get("title"):
                    item["title"] = hit["title"]

    ranked = sorted(
        merged.values(),
        key=lambda item: float(item.get("score") or 0.0),
        reverse=True,
    )
    return ranked[: settings.retrieval_top_k]
