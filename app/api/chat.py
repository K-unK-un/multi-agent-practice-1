"""问答接口：SSE 流式问答、非流式调试问答、会话历史。"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.core.logger import get_logger
from app.core.metrics import metrics
from app.core.security import require_auth
from app.graph.supervisor_graph import run_answer_once, stream_answer
from app.store.mysql import get_mysql

logger = get_logger("api.chat")
router = APIRouter(prefix="/chat", tags=["chat"])


class ChatStreamRequest(BaseModel):
    session_id: str | None = Field(default=None, max_length=36)
    question: str = Field(min_length=1, max_length=4000)
    user_scope: list[str] | None = None

    @field_validator("question")
    @classmethod
    def _strip_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("问题不能为空")
        return value

    @field_validator("session_id")
    @classmethod
    def _strip_session_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("user_scope")
    @classmethod
    def _clean_scope(cls, value: list[str] | None) -> list[str] | None:
        if not value:
            return None
        cleaned = [str(item).strip() for item in value if item and str(item).strip()]
        return list(dict.fromkeys(cleaned)) or None


class ChatRequest(ChatStreamRequest):
    """非流式问答请求体（评测 / 调试用）。"""


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """兼容存储层返回 dict 或 ORM 对象。"""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _json_safe(value: Any) -> Any:
    """转成可写入 JSON 列的类型（datetime 等转字符串）。"""
    if value is None:
        return None
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return str(value)


def _as_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _new_session_id(user_scope: list[str] | None) -> str:
    """为新会话建一行 sessions 记录并返回其 ID；失败回退本地 UUID。

    ``MySQLStore.create_session(user_scope)`` 自行生成 session_id，因此只有
    客户端未携带 session_id 时才调用；客户端自带 ID 时直接沿用（messages 表
    无外键约束，不建会话行也能落库）。
    """
    try:
        session_id = get_mysql().create_session(list(user_scope or []))
        if session_id:
            return str(session_id)
    except Exception:
        logger.warning("会话创建失败，回退本地 UUID", exc_info=True)
    return str(uuid.uuid4())


def _agents_from_results(result: Any) -> list[str]:
    """从非流式结果 agent_results 中提取专家名（去重保序）。"""
    agents: list[str] = []
    if isinstance(result, dict):
        for item in result.get("agent_results") or []:
            if not isinstance(item, dict):
                continue
            agent = item.get("agent")
            if agent and str(agent) not in agents:
                agents.append(str(agent))
    return agents


def _save_messages(
    session_id: str,
    question: str,
    answer: str,
    *,
    citations: Any = None,
    agents_used: Any = None,
    token_usage: Any = None,
    latency_ms: int | None = None,
    user_id: int | None = None,
) -> None:
    """落库用户问题与助手回答；任何异常只记日志，不中断流/响应。"""
    try:
        store = get_mysql()
        store.append_message({"session_id": session_id, "role": "user", "content": question, "user_id": user_id})
        store.append_message(
            {
                "session_id": session_id,
                "role": "assistant",
                "content": answer,
                "citations": _json_safe(citations or []),
                "agents_used": _json_safe(agents_used or []),
                "token_usage": _json_safe(token_usage or {}),
                "latency_ms": _as_int(latency_ms),
                "user_id": user_id,
            }
        )
    except Exception:
        logger.warning("问答消息落库失败（不影响响应）session_id=%s", session_id, exc_info=True)


@router.post("/stream")
async def chat_stream(
    body: ChatStreamRequest,
    current_user: dict[str, Any] = Depends(require_auth),
) -> StreamingResponse:
    """SSE 流式问答：事件与 LangGraph 图事件一一对应。"""
    session_id = body.session_id or await asyncio.to_thread(
        _new_session_id, body.user_scope
    )
    question = body.question
    user_scope = body.user_scope
    user_level = int(current_user.get("level") or 1)
    user_id = _as_int(current_user.get("id"))

    async def event_generator():
        started = time.perf_counter()
        metrics.record_sse_session()

        token_parts: list[str] = []
        citation_events: list[dict[str, Any]] = []
        agents_seen: list[str] = []
        done_data: dict[str, Any] | None = None

        # 浏览器断线重连间隔（Retry 是注释行，前端解析时忽略）。
        yield "retry: 3000\n\n"

        try:
            async for item in stream_answer(
                question=question,
                user_scope=user_scope,
                session_id=session_id,
                user_level=user_level,
            ):
                event_name = str(item.get("event") or "message")
                data = item.get("data")
                if not isinstance(data, dict):
                    data = {"value": data}

                if event_name == "token":
                    text = data.get("text")
                    if isinstance(text, str):
                        token_parts.append(text)
                elif event_name == "citation":
                    citation_events.append(data)
                elif event_name == "agent_start":
                    agent = data.get("agent")
                    if agent and str(agent) not in agents_seen:
                        agents_seen.append(str(agent))
                elif event_name == "done":
                    done_data = data
                elif event_name == "error":
                    metrics.record_error()

                payload = json.dumps(data, ensure_ascii=False, default=str)
                yield f"event: {event_name}\ndata: {payload}\n\n"
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            logger.info("SSE 客户端断开 session_id=%s", session_id)
            raise
        except Exception as exc:
            metrics.record_error()
            logger.exception("SSE 流式问答失败 session_id=%s", session_id)
            payload = json.dumps(
                {"message": str(exc), "type": type(exc).__name__},
                ensure_ascii=False,
                default=str,
            )
            yield f"event: error\ndata: {payload}\n\n"
        finally:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            done = done_data or {}
            latency_ms = _as_int(done.get("latency_ms"), elapsed_ms)
            agents_used = done.get("agents_used") or agents_seen

            # 仅当链路正常发出 done 事件时落库；落库失败不影响已发送的流。
            if done_data is not None:
                final_answer = done.get("final_answer")
                if not isinstance(final_answer, str) or not final_answer.strip():
                    final_answer = "".join(token_parts)
                if final_answer and final_answer.strip():
                    final_citations = done.get("citations") or citation_events
                    _save_messages(
                        session_id=session_id,
                        question=question,
                        answer=final_answer,
                        citations=final_citations,
                        agents_used=agents_used,
                        token_usage=done.get("token_usage") or {},
                        latency_ms=latency_ms,
                        user_id=user_id,
                    )

            metrics.record_request(latency_ms, agents_used)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("")
def chat_once(
    body: ChatRequest,
    current_user: dict[str, Any] = Depends(require_auth),
) -> dict[str, Any]:
    """非流式问答（评测 / 调试）：同步执行，阻塞在 FastAPI 线程池中。"""
    session_id = body.session_id or _new_session_id(body.user_scope)
    started = time.perf_counter()
    try:
        result = run_answer_once(
            question=body.question,
            user_scope=body.user_scope,
            session_id=session_id,
            user_level=int(current_user.get("level") or 1),
        )
    except Exception as exc:
        metrics.record_error()
        logger.exception("非流式问答失败 session_id=%s", session_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"问答失败：{exc}",
        )

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    if not isinstance(result, dict):
        result = {"final_answer": str(result)}

    payload = dict(result)
    payload["session_id"] = session_id
    latency_ms = _as_int(payload.get("latency_ms"), elapsed_ms)
    payload["latency_ms"] = latency_ms

    trace = payload.get("trace")
    payload["trace"] = trace[:50] if isinstance(trace, list) else []

    answer = payload.get("final_answer")
    if answer is None:
        answer = ""
    elif not isinstance(answer, str):
        answer = str(answer)

    agents_used = payload.get("agents_used") or _agents_from_results(result)
    payload["agents_used"] = agents_used
    _save_messages(
        session_id=session_id,
        question=body.question,
        answer=answer,
        citations=payload.get("citations"),
        agents_used=agents_used,
        token_usage=payload.get("token_usage"),
        latency_ms=latency_ms,
        user_id=_as_int(current_user.get("id")),
    )
    metrics.record_request(latency_ms, agents_used)
    return payload


@router.get("/history/{session_id}")
def chat_history(
    session_id: str,
    current_user: dict[str, Any] = Depends(require_auth),
) -> list[dict[str, Any]]:
    """返回会话消息（只暴露 role/content/created_at）。普通用户只能读自己的会话，管理员可读任意。"""
    try:
        rows = get_mysql().list_messages(session_id) or []
    except Exception:
        logger.exception("读取会话历史失败 session_id=%s", session_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="读取会话历史失败",
        )

    # 会话归属校验：匿名（开发模式）不校验；管理员放行；普通用户只能看自己的
    owner_id = _as_int(_field(rows[0], "user_id")) if rows else None
    if (
        current_user.get("role") != "admin"
        and owner_id is not None
        and owner_id != _as_int(current_user.get("id"))
    ):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权查看他人会话")

    history: list[dict[str, Any]] = []
    for row in rows:
        history.append(
            {
                "role": _field(row, "role") or "",
                "content": _field(row, "content") or "",
                "created_at": _json_safe(_field(row, "created_at")),
            }
        )
    return history
