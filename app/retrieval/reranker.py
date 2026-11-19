"""CrossEncoder 精排（bge-reranker-large）。

进程内单例懒加载；加载前先 apply_hf_offline_env() 走本地 HF 缓存（G:/AI）离线加载。
模型文件缺失 / 加载失败 / 打分异常时：warning + 原序降级返回，绝不向上抛。
"""
from __future__ import annotations

import threading
from typing import Any

from app.core.config import settings
from app.core.logger import get_logger
from app.llm.client import apply_hf_offline_env

logger = get_logger("retrieval.reranker")


def _to_float(value: Any) -> float:
    """兼容 numpy 标量 / shape=(n,1) 数组 / torch tensor。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            return float(value[0])
        except Exception:
            return 0.0


def _resolve_device(requested: str) -> str:
    """请求 cuda 但 CUDA 不可用时回退 cpu，避免整个精排器加载失败。"""
    device = str(requested or "cpu").strip() or "cpu"
    if not device.lower().startswith("cuda"):
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return device
        logger.warning("RERANKER_DEVICE=%s 但 CUDA 不可用，回退 CPU", requested)
    except Exception as exc:
        logger.warning("检测 CUDA 失败，回退 CPU：%s", exc)
    return "cpu"


class Reranker:
    """CrossEncoder 精排器。"""

    def __init__(self) -> None:
        self._model: Any = None
        self._load_failed = False
        self._lock = threading.Lock()

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        if self._load_failed:
            return None
        with self._lock:
            if self._model is not None:
                return self._model
            if self._load_failed:
                return None
            try:
                apply_hf_offline_env()
                from sentence_transformers import CrossEncoder

                device = _resolve_device(settings.reranker_device)
                model_kwargs = None
                if device.lower().startswith("cuda") and settings.reranker_fp16:
                    try:
                        import torch

                        model_kwargs = {"torch_dtype": torch.float16}
                    except Exception as exc:
                        logger.warning("启用 fp16 失败，使用默认精度：%s", exc)

                self._model = CrossEncoder(
                    settings.reranker_model,
                    cache_folder=settings.hf_home,
                    device=device,
                    max_length=settings.reranker_max_length,
                    model_kwargs=model_kwargs,
                )
                logger.info(
                    "Reranker 已加载：%s（device=%s, fp16=%s）",
                    settings.reranker_model,
                    device,
                    bool(model_kwargs),
                )
            except Exception as exc:
                # 只尝试一次：避免每条请求都重复一次昂贵的失败加载
                self._load_failed = True
                self._model = None
                logger.warning("Reranker 加载失败（精排降级为原序返回）：%s", exc)
            return self._model

    def warmup(self) -> bool:
        """启动预热：提前加载模型，避免第一个问答请求承担加载时间。"""
        return self._ensure_model() is not None

    def rerank(self, question: str, docs: list[dict]) -> list[dict]:
        """对候选条款打分并降序截断到 rerank_top_k。

        输入 docs 元素需含 text；输出写 rerank_score 并把 score 覆盖为 rerank_score。
        模型不可用时保持原序返回前 rerank_top_k 条。
        """
        candidates = [dict(doc) for doc in (docs or []) if isinstance(doc, dict)]
        if not candidates:
            return []

        model = self._ensure_model()
        if model is None:
            return candidates[: settings.rerank_top_k]

        pairs = [(question or "", doc.get("text") or "") for doc in candidates]
        try:
            scores = model.predict(
                pairs,
                batch_size=settings.reranker_batch_size,
                show_progress_bar=False,
            )
        except Exception as exc:
            logger.warning("Reranker 打分失败（降级为原序返回）：%s", exc)
            return candidates[: settings.rerank_top_k]

        score_list = [_to_float(score) for score in scores]
        for doc, score in zip(candidates, score_list):
            doc["rerank_score"] = score
            doc["score"] = score

        candidates.sort(
            key=lambda doc: float(doc.get("rerank_score") or 0.0), reverse=True
        )
        return candidates[: settings.rerank_top_k]


_reranker: Reranker | None = None
_reranker_lock = threading.Lock()


def get_reranker() -> Reranker:
    """进程内单例懒加载（模型本身在首次 rerank 时才真正加载）。"""
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                _reranker = Reranker()
    return _reranker
