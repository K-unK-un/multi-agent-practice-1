"""全局配置：从 .env 读取，pydantic-settings 管理。所有模块通过 settings 取值，禁止硬编码。"""
from __future__ import annotations

import json
import os

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---------------- 运行环境 ----------------
    app_env: str = "dev"
    log_level: str = "INFO"
    tz: str = "Asia/Shanghai"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ---------------- 大模型：DeepSeek（OpenAI 兼容）----------------
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    llm_model_main: str = "deepseek-flash"
    llm_model_expert: str = "deepseek-v4-pro"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 2048
    # synthesize 最终答案的输出上限，避免回答过长（0 表示不单独限制）
    synthesize_max_tokens: int = 800
    llm_timeout: int = 60
    llm_max_retries: int = 2

    # ---------------- Embedding：Ollama 本地 ----------------
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 768
    embedding_batch_size: int = 32
    embedding_timeout: int = 60

    # ---------------- Reranker / 语义核查：本地 HF 缓存 ----------------
    hf_home: str = "G:/AI"
    hf_hub_offline: bool = True
    reranker_model: str = "BAAI/bge-reranker-large"
    reranker_device: str = "cpu"
    reranker_batch_size: int = 8
    reranker_max_length: int = 512
    # GPU 下使用 fp16，降低显存占用并提升速度；CPU 会自动忽略
    reranker_fp16: bool = True
    rerank_top_k: int = 6
    # critic 句向量核查（默认关闭：先纯规则，G 组实验后再开）
    enable_semantic_check: bool = False
    semantic_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    semantic_threshold: float = 0.45

    # ---------------- 向量库：Chroma（Docker HTTP）----------------
    chroma_mode: str = "http"
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str = "policymind_clauses"
    chroma_tenant: str = "default_tenant"
    chroma_database: str = "default_database"

    # ---------------- 关系库：MySQL（Docker）----------------
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_database: str = "policymind"
    mysql_user: str = "policymind"
    mysql_password: str = "change-me"
    database_url: str = "mysql+pymysql://policymind:change-me@localhost:3306/policymind?charset=utf8mb4"

    # ---------------- 缓存 / Checkpointer：Redis ----------------
    redis_url: str = "redis://localhost:6379/0"
    redis_key_prefix: str = "policymind"
    # memory | redis（sqlite 需另装 langgraph-checkpoint-sqlite，默认不依赖）
    checkpoint_backend: str = "memory"
    checkpoint_ttl_seconds: int = 604800
    checkpoint_db_path: str = "data/checkpoints.sqlite"

    # ---------------- BM25 索引（本机内存）----------------
    bm25_index_path: str = "data/index/bm25_index.pkl"
    bm25_top_k: int = 10
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_tokenizer: str = "jieba"

    # ---------------- 检索与编排参数 ----------------
    retrieval_top_k: int = 10
    rrf_k: int = 60
    max_retrieval_loop: int = 2
    max_dispatch_round: int = 3
    max_review_round: int = 1
    grade_min_coverage: float = 0.6
    context_max_chars: int = 6000
    enable_query_rewrite: bool = True

    # ---------------- 安全与服务 ----------------
    jwt_secret: str = "dev-secret-change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440
    max_upload_mb: int = 50
    cors_origins: list[str] = ["http://localhost:8000"]
    auth_enabled: bool = True
    upload_dir: str = "data/uploads"
    # 直接运行 app/main.py 时是否自动打开浏览器（uvicorn 命令启动时不生效）
    auto_open_browser: bool = True
    # 启动时后台预热 reranker，避免第一个请求承担模型加载时间
    warmup_models: bool = True

    # ---------------- 日志与评估 ----------------
    log_dir: str = "logs"
    log_rotation_mb: int = 20
    eval_dataset_path: str = "eval/dataset.jsonl"
    eval_report_dir: str = "eval/reports"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, v):
        """兼容两种写法：JSON 数组 `["a","b"]` 或逗号分隔 `a,b`。"""
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("["):
                try:
                    return json.loads(v)
                except (json.JSONDecodeError, ValueError):
                    pass
            return [x.strip() for x in v.split(",") if x.strip()]
        return v


settings = Settings()

# 必须在 chromadb / sentence-transformers / huggingface_hub 等第三方库 import 之前
# 写入进程环境变量：huggingface_hub 会在 import 时缓存 HF_HUB_OFFLINE 的值，之后再
# 设置就来不及了，导致加载本地 reranker/句向量模型时仍尝试联网（WinError 10060）。
os.environ.setdefault("HF_HOME", settings.hf_home)
if settings.hf_hub_offline:
    os.environ["HF_HUB_OFFLINE"] = "1"
