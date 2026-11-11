"""进程内指标采集：请求量、错误数、SSE 会话数、Agent 调用次数与延迟样本。

不依赖 Prometheus 等外部组件，满足毕业项目的实验统计与调试需求。
所有读写均由 threading.Lock 保护；延迟仅保留最近 200 条。
"""
from __future__ import annotations

import threading
from collections import Counter, deque
from typing import Any, Iterable

_LATENCY_WINDOW = 200


class Metrics:
    """线程安全的进程内计数器。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total_requests = 0
        self._errors = 0
        self._sse_sessions = 0
        self._agent_calls: Counter[str] = Counter()
        self._latencies: deque[float] = deque(maxlen=_LATENCY_WINDOW)

    # ---------------- 写入 ----------------

    def record_request(
        self,
        latency_ms: float | None = None,
        agents_used: Iterable[str] | dict[str, int] | None = None,
    ) -> None:
        """记录一次问答请求：延迟样本 + 本次用到的 Agent（名称列表或 {name: count}）。"""
        with self._lock:
            self._total_requests += 1
            if latency_ms is not None:
                try:
                    self._latencies.append(max(0.0, float(latency_ms)))
                except (TypeError, ValueError):
                    pass

            if not agents_used:
                return
            if isinstance(agents_used, dict):
                pairs = agents_used.items()
            else:
                pairs = ((agent, 1) for agent in agents_used)
            for agent, count in pairs:
                if not agent:
                    continue
                try:
                    calls = int(count)
                except (TypeError, ValueError):
                    calls = 1
                if calls > 0:
                    self._agent_calls[str(agent)] += calls

    def record_error(self) -> None:
        with self._lock:
            self._errors += 1

    def record_sse_session(self) -> None:
        with self._lock:
            self._sse_sessions += 1

    # ---------------- 读取 ----------------

    def snapshot(self) -> dict[str, Any]:
        """返回当前指标快照，含最近 200 条延迟的平均值。"""
        with self._lock:
            latencies = list(self._latencies)
            agent_calls = dict(self._agent_calls)
            total_requests = self._total_requests
            errors = self._errors
            sse_sessions = self._sse_sessions

        avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0
        return {
            "total_requests": total_requests,
            "errors": errors,
            "sse_sessions": sse_sessions,
            "agent_calls": agent_calls,
            "avg_latency": avg_latency,
            "latency_samples": len(latencies),
        }


# 全局单例：业务模块 import metrics 后直接调用方法。
metrics = Metrics()


def record_request(
    latency_ms: float | None = None,
    agents_used: Iterable[str] | dict[str, int] | None = None,
) -> None:
    metrics.record_request(latency_ms=latency_ms, agents_used=agents_used)


def record_error() -> None:
    metrics.record_error()


def record_sse_session() -> None:
    metrics.record_sse_session()


def snapshot() -> dict[str, Any]:
    return metrics.snapshot()
