"""每请求黑板上下文（contextvars）。

专家节点执行期间把 DocStore 挂到 contextvar 上，工具（search/get_clause 等）通过
record_evidence()/emit() 写入。节点结束后包装器把黑板内容并入 LangGraph 状态。
"""
from __future__ import annotations

import contextvars
from typing import Optional

from app.store.docstore import DocStore

_BLACKBOARD: contextvars.ContextVar[Optional[DocStore]] = contextvars.ContextVar(
    "policymind_docstore", default=None
)


def get_blackboard() -> Optional[DocStore]:
    return _BLACKBOARD.get()


def set_blackboard(ds: DocStore) -> None:
    _BLACKBOARD.set(ds)


class blackboard_context:
    """with blackboard_context(DocStore()): ... 期间工具写入该黑板。"""

    def __init__(self, ds: DocStore) -> None:
        self.ds = ds
        self._token = None

    def __enter__(self) -> DocStore:
        self._token = _BLACKBOARD.set(self.ds)
        return self.ds

    def __exit__(self, *exc) -> bool:
        _BLACKBOARD.reset(self._token)
        return False


def emit(event: str, **data) -> None:
    """向当前黑板写一条 trace 事件（无黑板时静默）。"""
    ds = _BLACKBOARD.get()
    if ds is not None:
        ds.add_trace({"event": event, **data})


def record_evidence(items: list[dict]) -> None:
    """工具把检索命中的条款写入黑板证据池。

    items 元素：{doc_id, article_no, title, page, text, channel, score, query}
    """
    ds = _BLACKBOARD.get()
    if ds is not None:
        ds.add_evidence(items)
