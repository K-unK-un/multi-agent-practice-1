"""MySQL 表结构（SQLAlchemy 2.0）。MySQL 是元数据主库：文档/条款/规则表/会话/反馈/用户。"""
from __future__ import annotations

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Document(Base):
    """制度文档。同一制度多版本 = 多行，旧版 status=archived。"""

    __tablename__ = "documents"
    id = Column(String(64), primary_key=True)  # sha1(title+version+effective_date)
    title = Column(String(255), nullable=False, index=True)
    category = Column(String(64))
    version = Column(String(64))
    effective_date = Column(Date)  # 生效日期，元数据硬过滤依据
    status = Column(String(16), default="active", index=True)  # active | archived
    scope = Column(JSON, default=list)  # ["employee"] / ["manager"] 等
    source_path = Column(String(512))
    checksum = Column(String(64))  # 文件内容 hash，用于变更检测
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class Clause(Base):
    """条款（父块）：完整条款原文，生成阶段引用来源。"""

    __tablename__ = "clauses"
    __table_args__ = (Index("ix_clause_doc_article", "doc_id", "article_no", unique=True),)
    id = Column(String(64), primary_key=True)  # sha1(doc_id + article_no)
    doc_id = Column(String(64), ForeignKey("documents.id"), nullable=False, index=True)
    article_no = Column(String(32), nullable=False)  # "12" / "12.3"
    title = Column(String(255))
    chapter = Column(String(255))
    content = Column(Text, nullable=False)  # 父块全文
    content_hash = Column(String(40))
    page = Column(Integer)
    order_no = Column(Integer, default=0)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class RuleRow(Base):
    """规则表：阶梯/比例/上限/门槛类条款的结构化行，供 eligibility_agent 查表 + 计算。"""

    __tablename__ = "rule_table"
    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String(64), ForeignKey("documents.id"), index=True)
    article_no = Column(String(32))
    condition = Column(JSON)  # 条件区间，如 {"工龄": {"min": 10, "max": 20, "unit": "年"}}
    conclusion = Column(JSON)  # 结论，如 {"value": 10, "unit": "天", "text": "年休假10天"}
    value = Column(Float)
    unit = Column(String(16))
    basis_text = Column(Text)  # 依据条款原文
    needs_review = Column(Boolean, default=False, index=True)  # 数值校验不一致时置 True
    created_at = Column(DateTime, default=func.now())


class Session(Base):
    __tablename__ = "sessions"
    id = Column(String(36), primary_key=True)
    user_scope = Column(JSON, default=list)
    created_at = Column(DateTime, default=func.now())
    last_active = Column(DateTime, default=func.now(), onupdate=func.now())


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), index=True)
    user_id = Column(Integer, index=True)  # 提问用户（后续按用户隔离历史/审计；旧数据为 NULL）
    role = Column(String(16))
    content = Column(Text)
    citations = Column(JSON)
    agents_used = Column(JSON)
    token_usage = Column(JSON)
    latency_ms = Column(Integer)
    trace = Column(JSON)
    created_at = Column(DateTime, default=func.now())


class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36))
    message_id = Column(Integer)
    rating = Column(Integer)  # 1~5
    comment = Column(Text)
    correction = Column(Text)  # 用户给出的正确答案（纠错闭环）
    created_at = Column(DateTime, default=func.now())


class User(Base):
    """用户表：用户名 + 密码注册，等级制授权。

    - role: "admin"（唯一管理员，最高权限）| "user"（普通用户）
    - level: 1~4 信息层级。注册默认 1；管理员手动设置；等级越高可见文档/信息层级越高。
      4 级用户查看权限与管理员相同（无修改权限）；admin 恒为 4。
    - is_active: 预留禁用账号能力
    - scopes: 文档 scope 标签（与 level 正交，供检索层过滤，后续拓展）
    """

    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(16), default="user", index=True)  # admin | user
    level = Column(Integer, default=1, index=True)  # 1~4
    is_active = Column(Boolean, default=True, index=True)
    scopes = Column(JSON, default=list)  # ["employee"] / ["hr"] / ["manager"]
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    last_login_at = Column(DateTime)
