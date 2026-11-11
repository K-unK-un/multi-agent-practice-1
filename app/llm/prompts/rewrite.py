"""查询改写 prompt + 结构化输出 schema。

检索子图 rewrite 节点调用：把用户口语化问题改写成 1~3 个适合混合检索
（向量 + BM25）的中文子查询。约束：只改写、不回答；保留工龄/天数/岗位/版本等
限定条件；复合问题拆分为独立子查询。
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

_REWRITE_SYSTEM = """你是企业规章制度问答系统的检索查询改写器。请把用户的口语化问题改写成适合中文混合检索（向量召回 + BM25 关键词召回）的子查询。

改写要求：
1. 消解口语、省略和代词：把「这个」「那」「它」等指代替换为具体对象，把不完整句子补全。
2. 补书面同义词：把员工口语替换为制度文本常用表述。例如「扣钱」→「工资扣减 / 薪酬扣除」，「能攒到明年」→「跨年度结转 / 顺延」，「请病假拿多少」→「病假工资 / 疾病救济费标准」。
3. 拆分复合问题：一个问题包含多个事项时，拆成相互独立的子查询。例如「出差期间生病，报销和病假怎么算」拆为「出差费用报销标准」与「病假工资计算标准」。
4. 每个子查询必须自成完整语义，保留关键限定条件（工龄、天数、岗位、制度版本、生效时间等），不要丢失数字和条件。
5. 输出 1~3 个中文子查询；不要输出解释、推理过程或制度条款答案；不要编造制度名称。

只输出 queries 列表。"""

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _REWRITE_SYSTEM),
        ("human", "用户原始问题：{question}"),
    ]
)


class RewriteSchema(BaseModel):
    """改写后的检索子查询列表（1~3 条）。"""

    queries: list[str] = Field(
        default_factory=list,
        description="1~3 个中文检索子查询，已消解口语与代词、补充书面同义词、拆分复合问题",
    )
