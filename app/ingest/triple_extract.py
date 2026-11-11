"""P8 预留：Neo4j 图谱三元组抽取（主流程不调用）。

P8 启用时在此实现：
- 输入：父块条款列表（doc_id / article_no / content / title）
- 输出：[{"head": ..., "relation": ..., "tail": ..., "doc_id": ..., "article_no": ...}]
  例如 {"head": "年休假", "relation": "要求", "tail": "累计工作满1年", ...}
- 建图与召回接口按 README 第 5 节的"通道可插拔"设计注册，不影响现有双路检索。
"""
from __future__ import annotations

from app.core.logger import get_logger

logger = get_logger("ingest.triple_extract")


def extract_triples(clauses: list[dict]) -> list[dict]:
    """Neo4j 图谱通道占位：P8 启用时实现，主流程不调用，当前恒返回空列表。"""
    _ = clauses  # 显式保留参数语义，避免静态检查告警
    logger.debug("triple_extract 为 P8 占位实现，跳过（channels=%s）", len(clauses or []))
    return []
