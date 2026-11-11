"""覆盖度评估 prompt + 结构化输出 schema。

检索子图 grade 节点调用：评估当前候选条款对原始问题的覆盖程度（0~1），
列出缺失信息点，并给出用于定向补查的新查询。只评估、不回答。
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

_GRADE_SYSTEM = """你是企业规章制度问答系统的检索质量评估器。给定用户的原始问题和当前召回候选条款（只提供 doc_id、条款号与正文前 60 字片段），请判断这些条款是否足以完整回答问题。

评估口径：
1. coverage（覆盖度，0~1 的小数）：
   - 1.0 表示候选条款已覆盖回答所需全部要点；
   - 0.6 以上通常可支撑作答；
   - 低于 0.6 表示存在关键信息缺口，需要继续检索。
2. missing：列出回答该问题仍然缺失的关键信息点，例如「病假工资比例」「工龄区间对应的年假天数」。没有缺口时输出空列表。
3. suggestions：针对每个缺口给出 1~3 个新的中文检索查询。补查时应更换关键词或补充同义词（如「疾病救济费」「年休假跨年度结转」）；覆盖已充分时输出空列表。
4. 只评估覆盖程度，不要回答问题，不要编造候选条款中不存在的制度内容。

注意：候选条款片段被截断为前 60 字，判断时以条款号、标题和可见片段为依据，不要假设未展示的内容。"""

GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _GRADE_SYSTEM),
        ("human", "原始问题：{question}\n\n候选条款：\n{candidates}"),
    ]
)


class GradeSchema(BaseModel):
    """覆盖度评估结果。"""

    coverage: float = Field(
        description="候选条款对原问题的覆盖度，0~1 的小数；低于 0.6 视为不足"
    )
    missing: list[str] = Field(
        default_factory=list,
        description="回答原问题仍缺失的关键信息点，没有缺口则为空列表",
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="用于补查的 1~3 个中文检索查询，覆盖充分时为空列表",
    )
