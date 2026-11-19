"""模型客户端：DeepSeek 分级（light/main/expert）+ Ollama embedding。进程内单例懒加载。"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any, Type, TypeVar

from langchain_core.messages import SystemMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.core.config import settings

T = TypeVar("T")

# 分级：light = 低温度 flash（supervisor/critic），main = 常规专家（flash），expert = 难例与跨版本对比（pro）
_TIERS = {
    "light": ("llm_model_main", 0.0),
    "main": ("llm_model_main", "llm_temperature"),
    "expert": ("llm_model_expert", "llm_temperature"),
}


@lru_cache(maxsize=4)
def get_llm(tier: str = "main") -> ChatOpenAI:
    if tier not in _TIERS:
        tier = "main"
    model_attr, temp_attr = _TIERS[tier]
    temperature = temp_attr if isinstance(temp_attr, float) else getattr(settings, temp_attr)
    return ChatOpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=getattr(settings, model_attr),
        temperature=temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
    )


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return "" if content is None else str(content)


def _strip_json_fence(text: str) -> str:
    value = (text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    return value.strip()


def _validate_structured(schema: Type[T], data: Any) -> T:
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema.model_validate(data)  # type: ignore[return-value]
    return data  # type: ignore[return-value]


def _parse_structured_message(message: Any, schema: Type[T], tool_name: str) -> T:
    """优先解析 tool_calls；模型未调用工具时回退解析 JSON 正文。"""
    tool_calls = getattr(message, "tool_calls", None) or []
    args: Any = None
    for call in tool_calls:
        if isinstance(call, dict) and call.get("name") == tool_name:
            args = call.get("args")
            break
    if args is None and tool_calls:
        first = tool_calls[0]
        if isinstance(first, dict):
            args = first.get("args")

    if args is not None:
        if isinstance(args, str):
            args = json.loads(_strip_json_fence(args))
        return _validate_structured(schema, args)

    text = _strip_json_fence(_content_text(getattr(message, "content", "")))
    if not text:
        raise ValueError("模型未返回工具调用或 JSON 内容")
    return _validate_structured(schema, json.loads(text))


def _prepend_instruction(payload: Any, instruction: str) -> Any:
    """在模型输入前补一句“必须调用工具”，提高 auto tool_choice 的命中率。"""
    if isinstance(payload, str):
        return f"{instruction}\n\n{payload}"
    if isinstance(payload, list):
        return [SystemMessage(content=instruction), *payload]
    to_messages = getattr(payload, "to_messages", None)
    if callable(to_messages):
        return [SystemMessage(content=instruction), *to_messages()]
    return f"{instruction}\n\n{str(payload)}"


def get_structured_llm(tier: str, schema: Type[T]) -> Runnable:
    """DeepSeek 兼容的结构化输出。

    DeepSeek 的 thinking 模型不支持“强制指定工具名”的 tool_choice；LangChain 默认的
    with_structured_output(method="function_calling") 会设置 tool_choice=工具名，导致
    400（Thinking mode does not support this tool_choice）。这里改用 auto，并在模型
    未调用工具时回退解析 JSON 正文。
    """
    llm = get_llm(tier)
    tool = convert_to_openai_tool(schema)
    tool_name = str(tool.get("function", {}).get("name") or "structured_output")
    bound = llm.bind_tools([tool], tool_choice="auto")
    instruction = (
        f"你必须调用名为 {tool_name} 的工具来输出结果，"
        "并严格按该工具参数的 JSON Schema 填写；不要输出额外解释。"
    )

    def _invoke(payload: Any) -> T:
        message = bound.invoke(_prepend_instruction(payload, instruction))
        return _parse_structured_message(message, schema, tool_name)

    return RunnableLambda(_invoke)


@lru_cache(maxsize=1)
def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model=settings.embedding_model,
    )


def apply_hf_offline_env() -> None:
    """加载本地 HF 模型前调用：设置 HF_HOME/HF_HUB_OFFLINE，避免联网校验。"""
    os.environ.setdefault("HF_HOME", settings.hf_home)
    if settings.hf_hub_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
