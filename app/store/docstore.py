"""DocStore 共享黑板：单次请求内的条款级证据池（内存）。所有专家工具写入，synthesize/supervisor 读取。"""
from __future__ import annotations

import time
from typing import Optional


class DocStore:
    """证据池纪律：只放条款级证据；按 doc_id+article_no 去重（先写者胜，但取更长文本）；谁写入谁标注用途。"""

    def __init__(self) -> None:
        self._evidence: dict[str, dict] = {}
        self.trace: list[dict] = []

    @staticmethod
    def _key(doc_id: str, article_no: str) -> str:
        return f"{doc_id}::{article_no}"

    # ---------------- 证据写入 ----------------

    def add_evidence(self, items: list[dict]) -> int:
        """写入证据。同一条款合并（更长文本胜、通道/查询并集、分数取最大）。返回新增条数。"""
        added = 0
        for item in items:
            if not item.get("doc_id") or not item.get("article_no"):
                continue
            key = self._key(str(item["doc_id"]), str(item["article_no"]))
            if key in self._evidence:
                self._merge(key, item)
            else:
                self._evidence[key] = dict(item)
                added += 1
        return added

    def _merge(self, key: str, new: dict) -> None:
        old = self._evidence[key]
        if len(new.get("text", "")) > len(old.get("text", "")):
            old["text"] = new["text"]
            old["page"] = new.get("page", old.get("page"))
            old["title"] = new.get("title", old.get("title"))
        old["full_text"] = old.get("full_text") or new.get("full_text")
        for field in ("channels", "queries"):
            merged = list(dict.fromkeys([*(old.get(field) or []), *(new.get(field) or [])]))
            old[field] = merged
        try:
            old["score"] = max(float(old.get("score") or 0), float(new.get("score") or 0))
        except (TypeError, ValueError):
            pass

    # ---------------- 读取 ----------------

    def get_evidence(self) -> list[dict]:
        return list(self._evidence.values())

    def get_by_article(self, doc_id: str, article_no: str) -> Optional[dict]:
        return self._evidence.get(self._key(doc_id, article_no))

    def snapshot_text(self, max_chars: int = 6000) -> str:
        """证据拼接为上下文文本，按【E1】【E2】编号，供生成 Agent 引用。"""
        parts: list[str] = []
        used = 0
        for i, ev in enumerate(self._evidence.values(), 1):
            title = ev.get("title") or ev.get("doc_id") or "制度"
            article = ev.get("article_no") or "?"
            text = ev.get("text") or ""
            block = f"【E{i}】《{title}》第{article}条：{text}"
            if used and used + len(block) > max_chars:
                break
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts)

    def summary(self) -> str:
        """给 supervisor 的轻量证据摘要。"""
        titles = sorted({ev.get("title") or ev.get("doc_id") or "?" for ev in self._evidence.values()})
        return f"证据池 {len(self._evidence)} 条，涉及 {len(titles)} 份制度：{'、'.join(titles[:8])}"

    # ---------------- 观测 ----------------

    def add_trace(self, event: dict) -> None:
        event.setdefault("ts", int(time.time() * 1000))
        self.trace.append(event)

    def snapshot_trace(self) -> list[dict]:
        return list(self.trace)
