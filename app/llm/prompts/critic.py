"""critic 核查纪律说明。

注意：critic 的实际核查以规则为主（引用真实性 + 数值一致性 + 可选句向量支撑），
本 prompt 仅作为纪律文档保留，不参与生成，也不作为 LLM 自评依据——避免"自己检查自己"
必然通过的偏袒问题。
"""
from __future__ import annotations

CRITIC_PROMPT = """[critic 核查纪律说明——仅作文档，实际核查以规则为主]

critic 不参与答案生成，只做独立核查，不改写答案。核查顺序：
1. 引用真实性：state["citations"] 中每个 doc_id + article_no 必须能在 MySQL clauses 表中命中；
   查不到即为伪造或失效引用，记入 bad_citations。
2. 数值一致性：final_answer 中出现的整数、小数、百分数，必须能在所引用条款原文中找到对应数值；
   缺失的数值记入 numeric_mismatches。条款号（第X条）与引用编号（【EX】）不参与数值比对。
3. 逐句支撑（可选，settings.enable_semantic_check=true 时）：把答案拆句，与引用条款计算
   句向量余弦相似度；低于 settings.semantic_threshold 的句子记入 unsupported。

判定：无伪造/失效引用、无数值不符、无支撑句占比 ≤ 10%，且答案有条款级引用 → pass；
否则 reject。reject 后回到 supervisor，由原专家带着核查反馈补证；超过
settings.max_review_round 则降级为"部分回答 + 制度未明确"。

纪律：
- critic 绝不因为答案"看起来合理"而放行，也不因措辞风格或篇幅扣分；
- 不能核验时（数据库或模型不可用）明确记录降级说明，不臆断为伪造；
- critic 只输出核查结论，不直接改写 final_answer。"""
