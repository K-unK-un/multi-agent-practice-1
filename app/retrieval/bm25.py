"""BM25 稀疏召回：jieba 分词 + rank_bm25.BM25Okapi。

索引以 pickle 落盘（data/index/bm25_index.pkl），进程内单例懒加载。
重建顺序：build_from_clauses() 先写 {path}.tmp，再 os.replace 原子替换，最后立即用
落盘产物重建内存索引，保证「建索引的进程」与「检索的进程」看到同一份数据。

注意：若索引由**另一个进程**（如 scripts/ingest_all.py）重建，本进程内的单例不会
自动感知文件变化，需要调用 get_bm25_index().reload() 或重启服务。
"""
from __future__ import annotations

import os
import pickle
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger("retrieval.bm25")

_PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL


def _tokenize(text: str) -> list[str]:
    """按 settings.bm25_tokenizer 分词；jieba 为默认，其他值退化为空白切分。"""
    text = (text or "").strip().lower()
    if not text:
        return []
    if settings.bm25_tokenizer == "jieba":
        return [token.strip() for token in jieba.lcut(text) if token.strip()]
    return [token for token in text.split() if token]


class BM25Index:
    """条款级 BM25 索引。"""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or settings.bm25_index_path)
        self._bm25: BM25Okapi | None = None
        self._entries: list[dict] = []
        self._version: str | None = None
        self._built_at: str | None = None
        self._loaded = False

    # ---------------- 构建 / 加载 ----------------

    def build_from_clauses(self, clauses: list[dict] | None = None) -> None:
        """全量重建索引并原子落盘。

        clauses 元素：{doc_id, article_no, title, page, text|content}。
        空文本条款跳过（无法产生 token，检索也不会命中）。

        兼容两种调用形式：
        - ``get_bm25_index().build_from_clauses(clauses)``（推荐，构建并刷新该实例）；
        - ``BM25Index.build_from_clauses(clauses)``（ingest/pipeline.py 的类式调用，
          此时第一个位置参数实际是 clauses；构建落到模块单例上，保证在线检索立即可见）。
        """
        if not isinstance(self, BM25Index):
            # 类式调用：self 实际是 clauses 列表
            target = get_bm25_index()
            target.build_from_clauses(clauses if clauses is not None else self)
            return

        entries: list[dict] = []
        for clause in clauses or []:
            if not isinstance(clause, dict):
                continue
            doc_id = str(clause.get("doc_id") or "").strip()
            if not doc_id:
                continue
            text = clause.get("text") or clause.get("content") or ""
            tokens = _tokenize(text)
            if not tokens:
                continue
            entries.append(
                {
                    "doc_id": doc_id,
                    "article_no": str(clause.get("article_no") or ""),
                    "title": clause.get("title") or "",
                    "page": clause.get("page"),
                    "tokens": tokens,
                    "text": text,
                }
            )

        payload = {
            "version": uuid.uuid4().hex,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "k1": float(settings.bm25_k1),
            "b": float(settings.bm25_b),
            "entries": entries,
        }

        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = Path(f"{self.path}.tmp")
        try:
            with tmp_path.open("wb") as fh:
                pickle.dump(payload, fh, protocol=_PICKLE_PROTOCOL)
            os.replace(tmp_path, self.path)  # 原子替换：重建期间检索进程不会读到半截文件
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

        # 落盘成功后失效内存缓存并从磁盘重新加载，build 与 search 保持一致
        self.reload()
        logger.info(
            "BM25 索引已重建：%d 条，version=%s，k1=%.2f b=%.2f",
            len(entries),
            payload["version"],
            payload["k1"],
            payload["b"],
        )

    def load(self) -> "BM25Index":
        """懒加载：已加载直接返回 self；文件缺失/损坏降级为空索引（warning 不抛）。"""
        if self._loaded:
            return self
        if not self.path.exists():
            logger.warning(
                "BM25 索引文件不存在：%s（检索将返回空结果，请先运行建索引脚本）",
                self.path,
            )
            self._entries = []
            self._bm25 = None
            self._version = None
            self._built_at = None
            self._loaded = True
            return self

        try:
            with self.path.open("rb") as fh:
                payload = pickle.load(fh)
            if not isinstance(payload, dict):
                raise ValueError("索引文件格式不正确")
        except Exception as exc:
            logger.warning("BM25 索引加载失败（已降级为空索引）：%s", exc)
            self._entries = []
            self._bm25 = None
            self._version = None
            self._built_at = None
            self._loaded = True
            return self

        self._apply_payload(payload)
        logger.info(
            "BM25 索引已加载：%d 条，version=%s", len(self._entries), self._version
        )
        return self

    def reload(self) -> "BM25Index":
        """丢弃内存状态并重新从磁盘加载（供外部进程重建索引后调用）。"""
        self._reset()
        return self.load()

    def _apply_payload(self, payload: dict) -> None:
        entries = payload.get("entries") or []
        self._entries = [entry for entry in entries if isinstance(entry, dict)]
        self._version = payload.get("version")
        self._built_at = payload.get("built_at")
        if self._entries:
            k1 = float(payload.get("k1", settings.bm25_k1))
            b = float(payload.get("b", settings.bm25_b))
            corpus = [entry.get("tokens") or [] for entry in self._entries]
            self._bm25 = BM25Okapi(corpus, k1=k1, b=b)
        else:
            self._bm25 = None
        self._loaded = True

    def _reset(self) -> None:
        self._bm25 = None
        self._entries = []
        self._version = None
        self._built_at = None
        self._loaded = False

    # ---------------- 查询 ----------------

    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def index_version(self) -> str | None:
        return self._version

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """检索并返回统一 Hit：{doc_id, article_no, title, page, text, score, channel="bm25"}。

        未加载时返回空（由 get_bm25_index() 负责首次加载）。
        """
        if not self._loaded:
            return []
        if self._bm25 is None or not self._entries:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        try:
            scores = self._bm25.get_scores(tokens)
        except Exception as exc:
            logger.warning("BM25 打分失败（已降级为空结果）：%s", exc)
            return []

        n_results = int(top_k or settings.bm25_top_k)
        # 只用词项交集判断“命中”：小语料下 rank_bm25 的 IDF 可能为 0（分数全 0），
        # 不能靠 score > 0 过滤，否则会漏掉真实命中的条款；未命中词项的条款也绝不返回。
        query_tokens = set(tokens)
        matched = [
            index
            for index, entry in enumerate(self._entries)
            if query_tokens.intersection(entry.get("tokens") or [])
        ]
        matched.sort(key=lambda i: float(scores[i]), reverse=True)

        hits: list[dict] = []
        for idx in matched[:n_results]:
            score = float(scores[idx])
            entry = self._entries[idx]
            hits.append(
                {
                    "doc_id": entry.get("doc_id") or "",
                    "article_no": entry.get("article_no") or "",
                    "title": entry.get("title") or "",
                    "page": entry.get("page"),
                    "text": entry.get("text") or "",
                    "score": score,
                    "channel": "bm25",
                }
            )
        return hits


_index: BM25Index | None = None
_index_lock = threading.Lock()


def get_bm25_index() -> BM25Index:
    """模块级单例；首次调用触发 load()。"""
    global _index
    if _index is None:
        with _index_lock:
            if _index is None:
                _index = BM25Index().load()
    return _index
