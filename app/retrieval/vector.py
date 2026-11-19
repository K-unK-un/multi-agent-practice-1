"""向量召回：Ollama nomic-embed-text 编码 → Chroma where 下推 → 统一 Hit。

Hit 结构：{doc_id, article_no, title, page, text, score, channel="vector"}。
score = 1 - distance（Chroma 余弦距离越小越相关）。
任何查询/编码异常都降级为空结果，不向上抛。
"""
from __future__ import annotations

import threading

from app.core.config import settings
from app.core.logger import get_logger
from app.llm.client import get_embeddings
from app.retrieval.filters import build_chroma_where, is_effective

logger = get_logger("retrieval.vector")


def _to_hit(item: dict) -> dict:
    """Chroma 返回行 → 统一 Hit。"""
    try:
        score = 1.0 - float(item.get("distance"))
    except (TypeError, ValueError):
        score = 0.0
    return {
        "doc_id": str(item.get("doc_id") or ""),
        "article_no": str(item.get("article_no") or ""),
        "title": item.get("title") or "",
        "page": item.get("page"),
        "text": item.get("document") or item.get("text") or "",
        "score": score,
        "channel": "vector",
    }


class VectorRetriever:
    """Chroma 向量召回器（无状态，可复用）。"""

    def search(
        self,
        query: str,
        user_scope: list[str] | None = None,
        include_archived: bool = False,
        top_k: int | None = None,
    ) -> list[dict]:
        query = (query or "").strip()
        if not query:
            return []
        n_results = int(top_k or settings.retrieval_top_k)
        try:
            vector = get_embeddings().embed_query(query)
            where = build_chroma_where(
                user_scope=user_scope, include_archived=include_archived
            )
            # 延迟导入：避免模块导入期就建立 Chroma 连接（单例仍在 app.store.chroma 内维护）
            from app.store.chroma import get_chroma

            rows = get_chroma().query(
                query_embeddings=[vector], n_results=n_results, where=where
            )
        except Exception as exc:  # 编码服务不可用 / Chroma 未就绪 / 集合不存在
            logger.warning("向量召回失败（已降级为空结果）：%s", exc)
            return []

        hits: list[dict] = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            # Chroma 1.0 不支持字符串 effective_date 的 $lte，改在 Python 侧过滤；
            # 未填写生效日期的条款视为不限，与 MySQL/BM25 路语义一致。
            if not is_effective(row.get("effective_date")):
                continue
            hit = _to_hit(row)
            if hit["doc_id"]:
                hits.append(hit)
        return hits


_retriever: VectorRetriever | None = None
_retriever_lock = threading.Lock()


def get_vector_retriever() -> VectorRetriever:
    """进程内单例（无状态，仅避免重复构造）。"""
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                _retriever = VectorRetriever()
    return _retriever
