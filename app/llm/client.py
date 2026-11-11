"""模型客户端：DeepSeek 分级（light/main/expert）+ Ollama embedding。进程内单例懒加载。"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Type, TypeVar

from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings

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


def get_structured_llm(tier: str, schema: Type[T]) -> ChatOpenAI:
    """绑定结构化输出的模型。DeepSeek 走 function_calling 通道（json_schema 兼容性不稳）。"""
    return get_llm(tier).with_structured_output(schema, method="function_calling")


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
