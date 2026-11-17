"""MySQL 数据访问层（同步 SQLAlchemy 2.0）。

MySQL 是元数据主库：文档、条款、规则表、会话、消息、反馈、用户。
本模块只做「行数据 dict 进出」的薄封装，不把 ORM 对象暴露给上层；上层
（ingest / retrieval / api）因此不需要关心 SQLAlchemy Session 生命周期。

约定：
- 连接串来自 ``settings.database_url``，禁止硬编码；
- 引擎与 Session 工厂进程内懒加载，``get_mysql()`` 返回单例；
- 所有时间戳统一使用 ``settings.tz`` 时区的 ``datetime.now(ZoneInfo(...))``；
- dict 输出时 ``date`` / ``datetime`` 统一转 ``"YYYY-MM-DD"`` 字符串；
- 连接类错误统一抛 ``RuntimeError("MySQL 连接失败: ...")``，避免把驱动异常
  直接泄漏到 API 层。
"""
from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

from sqlalchemy import create_engine, delete, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import InterfaceError, OperationalError
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.core.logger import get_logger
from app.store.schemas import (
    Base,
    Clause,
    Document,
    Feedback,
    Message,
    RuleRow,
    Session as SessionRow,
    User,
)

logger = get_logger("store.mysql")

# 允许写入的字段白名单：上层 dict 里混入多余键（如 doc_id / confidence）时忽略，
# 避免 ORM 构造器抛 TypeError。
_DOC_FIELDS = (
    "title",
    "category",
    "version",
    "effective_date",
    "status",
    "scope",
    "source_path",
    "checksum",
)
_MESSAGE_FIELDS = (
    "session_id",
    "role",
    "content",
    "citations",
    "agents_used",
    "token_usage",
    "latency_ms",
    "trace",
    "user_id",
)
_FEEDBACK_FIELDS = ("session_id", "message_id", "rating", "comment", "correction")


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------

def _now() -> datetime:
    """统一时区的当前时间（写入 MySQL 时由驱动按本地时间存储）。"""
    return datetime.now(ZoneInfo(settings.tz))


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _to_jsonable(value: Any) -> Any:
    """dict 输出前的归一化：date/datetime → "YYYY-MM-DD"，容器递归处理。"""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_jsonable(v) for v in value]
    return value


def _parse_date(value: Any) -> date | None:
    """把 date / datetime / "YYYY-MM-DD" 统一为 date；空值与零日期返回 None。"""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text or text.startswith("0000-00-00"):
        return None
    try:
        return date.fromisoformat(text[:10].replace("/", "-"))
    except ValueError:
        return None


def _scope_list(value: Any) -> list[str]:
    """scope 归一化为非空字符串列表；兼容 JSON 数组、JSON 字符串、逗号串。"""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                loaded = json.loads(text)
                if isinstance(loaded, list):
                    return [str(item).strip() for item in loaded if str(item).strip()]
            except (TypeError, ValueError):
                pass
        return [part.strip() for part in text.replace("，", ",").split(",") if part.strip()]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _row_to_dict(row: Any) -> dict:
    """ORM 行 → 普通 dict（列名取自 __table__.columns）。"""
    return {col.name: _to_jsonable(getattr(row, col.key)) for col in row.__table__.columns}


def _document_to_dict(document: Document) -> dict:
    data = _row_to_dict(document)
    data["scope"] = _scope_list(data.get("scope"))
    return data


def _merge_clause_doc(clause: Clause, document: Document) -> dict:
    """条款 + 所属文档元数据合并。

    ``title`` 取文档标题（引用展示需要），条款自身的小标题保留在 ``clause_title``。
    其余文档字段（version/effective_date/status/scope/...）直接平铺，供检索过滤
    与引用校验使用。
    """
    data = _row_to_dict(clause)
    doc = _document_to_dict(document)
    data["clause_title"] = data.get("title")
    data["title"] = doc.get("title")
    for key in (
        "category",
        "version",
        "effective_date",
        "status",
        "scope",
        "source_path",
        "checksum",
    ):
        data[key] = doc.get(key)
    return data


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# MySQLStore
# ---------------------------------------------------------------------------

class MySQLStore:
    """同步 SQLAlchemy 封装；线程安全（每次操作独立 Session）。"""

    def __init__(self) -> None:
        self._engine: Engine | None = None
        self._session_factory: sessionmaker[OrmSession] | None = None
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- 引擎

    def _ensure_engine(self) -> Engine:
        """懒加载引擎与 Session 工厂；不建立实际连接（create_engine 只加载驱动）。"""
        if self._engine is not None:
            return self._engine
        with self._lock:
            if self._engine is None:
                try:
                    engine = create_engine(
                        settings.database_url,
                        pool_pre_ping=True,
                        pool_recycle=3600,
                        echo=False,
                    )
                except Exception as exc:  # 驱动缺失 / URL 非法
                    raise RuntimeError(f"MySQL 连接失败: {exc}") from exc
                self._engine = engine
                self._session_factory = sessionmaker(
                    bind=engine, autoflush=False, expire_on_commit=False
                )
        return self._engine

    def _new_session(self) -> OrmSession:
        if self._session_factory is None:
            self._ensure_engine()
        assert self._session_factory is not None
        return self._session_factory()

    @staticmethod
    def _rollback_quietly(session: OrmSession) -> None:
        try:
            session.rollback()
        except Exception:  # 连接已断时 rollback 也可能失败，不要覆盖原始异常
            pass

    @contextmanager
    def _session_scope(self) -> Iterator[OrmSession]:
        """事务边界：正常提交，异常回滚；连接类异常转中文 RuntimeError。"""
        session = self._new_session()
        try:
            yield session
            session.commit()
        except (OperationalError, InterfaceError) as exc:
            self._rollback_quietly(session)
            raise RuntimeError(f"MySQL 连接失败: {exc}") from exc
        except Exception:
            self._rollback_quietly(session)
            raise
        finally:
            session.close()

    def init_db(self) -> None:
        """建表 + 轻量迁移（幂等）。应用启动时调用一次；重复调用不会破坏已有数据。"""
        engine = self._ensure_engine()
        try:
            Base.metadata.create_all(engine)
            self._migrate(engine)
        except Exception as exc:
            raise RuntimeError(f"MySQL 连接失败: {exc}") from exc

    @staticmethod
    def _migrate(engine) -> None:
        """补齐历史表缺失列（MySQL 8 不支持 ADD COLUMN IF NOT EXISTS，先查 information_schema）。

        仅做增量加列，不删列、不改类型，保证对已有数据零破坏。
        """
        from sqlalchemy import inspect as sa_inspect
        from sqlalchemy import text

        inspector = sa_inspect(engine)
        existing_tables = set(inspector.get_table_names())
        if not existing_tables:
            return

        def columns_of(table: str) -> set[str]:
            if table not in existing_tables:
                return set()
            return {col["name"] for col in inspector.get_columns(table)}

        additions: dict[str, dict[str, str]] = {
            "users": {
                "level": "INTEGER NOT NULL DEFAULT 1",
                "is_active": "BOOLEAN NOT NULL DEFAULT TRUE",
                "updated_at": "DATETIME NULL",
                "last_login_at": "DATETIME NULL",
            },
            "messages": {
                "user_id": "INTEGER NULL",
            },
        }
        for table, columns in additions.items():
            have = columns_of(table)
            for name, ddl in columns.items():
                if name in have:
                    continue
                with engine.begin() as conn:
                    conn.execute(text(f"ALTER TABLE `{table}` ADD COLUMN `{name}` {ddl}"))
                logger.info("迁移：%s.%s 新增列 %s", table, name, ddl)

    def close(self) -> None:
        """释放连接池；之后再次调用任一方法会自动重建引擎。"""
        with self._lock:
            if self._engine is not None:
                self._engine.dispose()
            self._engine = None
            self._session_factory = None

    # ---------------------------------------------------------------- 文档

    def upsert_document(self, doc: dict) -> str:
        """按主键 id 插入或更新文档元数据；返回 doc_id。"""
        doc_id = str(doc.get("id") or "").strip()
        if not doc_id:
            raise ValueError("upsert_document 缺少 id")
        now = _now()
        with self._session_scope() as session:
            entity = session.get(Document, doc_id)
            if entity is None:
                entity = Document(id=doc_id, created_at=now)
                session.add(entity)

            for key in _DOC_FIELDS:
                if key not in doc:
                    continue
                value = doc.get(key)
                if key == "effective_date":
                    value = _parse_date(value)
                elif key == "scope":
                    value = _scope_list(value)
                setattr(entity, key, value)

            if not entity.title:
                raise ValueError(f"upsert_document 缺少 title（doc_id={doc_id}）")
            if not entity.status:
                entity.status = "active"
            if entity.scope is None:
                entity.scope = []
            entity.updated_at = now
        return doc_id

    def archive_document(self, doc_id: str) -> None:
        """把文档标记为 archived（不物理删除，保留历史版本可查）。"""
        with self._session_scope() as session:
            entity = session.get(Document, doc_id)
            if entity is None:
                logger.warning("archive_document 未找到文档: %s", doc_id)
                return
            entity.status = "archived"
            entity.updated_at = _now()

    def list_documents(self, active_only: bool = True) -> list[dict]:
        """列出文档；active_only=False 时包含 archived。"""
        with self._session_scope() as session:
            stmt = select(Document).order_by(
                Document.effective_date.desc(),
                Document.title.asc(),
                Document.id.asc(),
            )
            if active_only:
                stmt = stmt.where(Document.status == "active")
            rows = session.execute(stmt).scalars().all()
            return [_document_to_dict(row) for row in rows]

    def get_document(self, doc_id: str) -> dict | None:
        with self._session_scope() as session:
            entity = session.get(Document, doc_id)
            return _document_to_dict(entity) if entity is not None else None

    def list_documents_with_counts(
        self, active_only: bool = False
    ) -> list[dict]:
        """文档列表 + 每份文档的条款数（入库管理面板用）。

        与 list_documents 分开查询再在 Python 侧合并：文档量级很小，避免
        ONLY_FULL_GROUP_BY 下按全部列 GROUP BY 的兼容性问题。
        """
        with self._session_scope() as session:
            stmt = select(Document).order_by(
                Document.effective_date.desc(),
                Document.title.asc(),
                Document.id.asc(),
            )
            if active_only:
                stmt = stmt.where(Document.status == "active")
            documents = [
                _document_to_dict(row)
                for row in session.execute(stmt).scalars().all()
            ]

            count_stmt = (
                select(Clause.doc_id, func.count(Clause.id))
                .group_by(Clause.doc_id)
            )
            counts = {
                str(doc_id): int(count)
                for doc_id, count in session.execute(count_stmt).all()
            }

        for document in documents:
            document["clause_count"] = counts.get(str(document.get("id")), 0)
        return documents

    # ---------------------------------------------------------------- 条款

    def replace_clauses_for_doc(self, doc_id: str, rows: list[dict]) -> int:
        """全量替换某文档的条款（先删后插，同一事务）。返回写入条数。

        ``rows`` 约定键：id/article_no/title/chapter/content/content_hash/page/order_no。
        为兼容上游切分器，缺少 ``id`` 时按 schema 约定用 sha1(doc_id+article_no) 补齐，
        缺少 ``content_hash`` 时用 sha1(content) 补齐。
        """
        doc_id = str(doc_id)
        now = _now()
        entities: list[Clause] = []
        for raw in rows or []:
            if not isinstance(raw, dict):
                continue
            article_no = str(raw.get("article_no") or "").strip()
            content = raw.get("content")
            if not article_no:
                raise ValueError(f"条款缺少 article_no（doc_id={doc_id}）")
            if content is None:
                raise ValueError(
                    f"条款缺少 content（doc_id={doc_id}, article_no={article_no}）"
                )
            clause_id = str(raw.get("id") or "").strip() or _sha1(
                f"{doc_id}{article_no}"
            )[:40]
            content_hash = raw.get("content_hash") or _sha1(str(content))[:40]
            page = raw.get("page")
            order_no = raw.get("order_no")
            entities.append(
                Clause(
                    id=clause_id,
                    doc_id=doc_id,
                    article_no=article_no,
                    title=raw.get("title"),
                    chapter=raw.get("chapter"),
                    content=str(content),
                    content_hash=content_hash,
                    page=int(page) if page not in (None, "") else None,
                    order_no=int(order_no) if order_no not in (None, "") else 0,
                    created_at=now,
                    updated_at=now,
                )
            )

        with self._session_scope() as session:
            session.execute(delete(Clause).where(Clause.doc_id == doc_id))
            if entities:
                session.add_all(entities)
        return len(entities)

    def get_clause(
        self, doc_id: str, article_no: str, include_archived: bool = False
    ) -> dict | None:
        """按 (doc_id, article_no) 取单条条款 + 文档元数据。"""
        with self._session_scope() as session:
            stmt = (
                select(Clause, Document)
                .join(Document, Clause.doc_id == Document.id)
                .where(
                    Clause.doc_id == str(doc_id),
                    Clause.article_no == str(article_no),
                )
            )
            if not include_archived:
                stmt = stmt.where(Document.status == "active")
            row = session.execute(stmt).first()
            if row is None:
                return None
            clause, document = row
            return _merge_clause_doc(clause, document)

    def get_clauses(
        self,
        doc_ids: list[str] | None = None,
        include_archived: bool = False,
        scopes: list[str] | None = None,
    ) -> list[dict]:
        """批量取条款 + 文档元数据。

        - ``doc_ids``：None 表示不过滤；空列表表示显式「无结果」。
        - ``include_archived``：False 时只返回 active 文档的条款。
        - ``scopes``：None 表示不做权限过滤；给定列表时保留「文档 scope 与用户
          scope 交集非空」或「文档 scope 为空（全员可见）」的条款。过滤在 Python
          侧完成，避免依赖 MySQL JSON 函数的版本差异。
        """
        if doc_ids is not None and not doc_ids:
            return []
        wanted = set(str(item) for item in scopes) if scopes is not None else None

        with self._session_scope() as session:
            stmt = select(Clause, Document).join(
                Document, Clause.doc_id == Document.id
            )
            if doc_ids is not None:
                stmt = stmt.where(
                    Clause.doc_id.in_([str(item) for item in doc_ids])
                )
            if not include_archived:
                stmt = stmt.where(Document.status == "active")
            stmt = stmt.order_by(
                Clause.doc_id.asc(),
                Clause.order_no.asc(),
                Clause.article_no.asc(),
            )
            rows = session.execute(stmt).all()

            result: list[dict] = []
            for clause, document in rows:
                if wanted is not None:
                    doc_scopes = set(_scope_list(document.scope))
                    if doc_scopes and not (doc_scopes & wanted):
                        continue
                result.append(_merge_clause_doc(clause, document))
            return result

    def list_versions(self, title_keyword: str) -> list[dict]:
        """按标题关键字列出版本（含 archived），按生效日期倒序。"""
        keyword = str(title_keyword or "").strip()
        with self._session_scope() as session:
            stmt = (
                select(Document)
                .where(Document.title.contains(keyword, autoescape=True))
                .order_by(
                    Document.effective_date.desc(),
                    Document.version.desc(),
                    Document.id.asc(),
                )
            )
            rows = session.execute(stmt).scalars().all()
            return [_document_to_dict(row) for row in rows]

    # ---------------------------------------------------------------- 规则表

    def upsert_rule_rows(self, rows: list[dict]) -> int:
        """按 (doc_id, article_no) 先删旧再插新；同一事务。返回写入条数。"""
        rows = [row for row in (rows or []) if isinstance(row, dict)]
        if not rows:
            return 0

        keys = list(
            dict.fromkeys(
                (
                    str(row.get("doc_id") or ""),
                    str(row.get("article_no") or ""),
                )
                for row in rows
            )
        )
        now = _now()
        entities: list[RuleRow] = []
        for row in rows:
            entities.append(
                RuleRow(
                    doc_id=row.get("doc_id"),
                    article_no=row.get("article_no"),
                    condition=row.get("condition"),
                    conclusion=row.get("conclusion"),
                    value=_coerce_float(row.get("value")),
                    unit=row.get("unit"),
                    basis_text=row.get("basis_text"),
                    needs_review=bool(row.get("needs_review", False)),
                    created_at=now,
                )
            )

        with self._session_scope() as session:
            for doc_id, article_no in keys:
                stmt = delete(RuleRow)
                stmt = (
                    stmt.where(RuleRow.doc_id == doc_id)
                    if doc_id
                    else stmt.where(RuleRow.doc_id.is_(None))
                )
                stmt = (
                    stmt.where(RuleRow.article_no == article_no)
                    if article_no
                    else stmt.where(RuleRow.article_no.is_(None))
                )
                session.execute(stmt)
            session.add_all(entities)
        return len(entities)

    def get_rule_rows(self, doc_id: str | None = None) -> list[dict]:
        with self._session_scope() as session:
            stmt = select(RuleRow).order_by(
                RuleRow.doc_id.asc(),
                RuleRow.article_no.asc(),
                RuleRow.id.asc(),
            )
            if doc_id is not None:
                stmt = stmt.where(RuleRow.doc_id == str(doc_id))
            rows = session.execute(stmt).scalars().all()
            return [_row_to_dict(row) for row in rows]

    # ---------------------------------------------------------------- 会话 / 消息

    def create_session(self, user_scope: list[str]) -> str:
        session_id = str(uuid4())
        now = _now()
        with self._session_scope() as session:
            session.add(
                SessionRow(
                    id=session_id,
                    user_scope=_scope_list(user_scope),
                    created_at=now,
                    last_active=now,
                )
            )
        return session_id

    def append_message(self, msg: dict) -> None:
        """追加一条消息；citations/agents_used/token_usage/trace 直接存 JSON。"""
        payload = {key: msg.get(key) for key in _MESSAGE_FIELDS if key in msg}
        payload["created_at"] = _now()
        with self._session_scope() as session:
            session.add(Message(**payload))

    def list_messages(self, session_id: str) -> list[dict]:
        with self._session_scope() as session:
            stmt = (
                select(Message)
                .where(Message.session_id == str(session_id))
                .order_by(Message.id.asc())
            )
            rows = session.execute(stmt).scalars().all()
            return [_row_to_dict(row) for row in rows]

    # ---------------------------------------------------------------- 反馈

    def insert_feedback(self, fb: dict) -> None:
        payload = {key: fb.get(key) for key in _FEEDBACK_FIELDS if key in fb}
        payload["created_at"] = _now()
        with self._session_scope() as session:
            session.add(Feedback(**payload))

    # ---------------------------------------------------------------- 用户

    def _user_dict(self, entity: User) -> dict:
        data = _row_to_dict(entity)
        data["scopes"] = _scope_list(data.get("scopes"))
        return data

    def get_user(self, username: str) -> dict | None:
        with self._session_scope() as session:
            stmt = select(User).where(User.username == str(username))
            entity = session.execute(stmt).scalars().first()
            return self._user_dict(entity) if entity is not None else None

    def get_user_by_id(self, user_id: int) -> dict | None:
        with self._session_scope() as session:
            entity = session.get(User, int(user_id))
            return self._user_dict(entity) if entity is not None else None

    def create_user(
        self,
        username: str,
        password_hash: str,
        *,
        role: str = "user",
        level: int = 1,
        scopes: list[str] | None = None,
    ) -> dict:
        """创建用户。用户名唯一性由 DB 唯一索引兜底（冲突抛 IntegrityError，上层转 409）。"""
        username = str(username or "").strip()
        if not username:
            raise ValueError("create_user 缺少 username")
        if role == "admin":
            level = 4  # 管理员恒为最高层级
        with self._session_scope() as session:
            entity = User(
                username=username,
                password_hash=password_hash,
                scopes=_scope_list(scopes),
                role=role,
                level=int(level),
                is_active=True,
                created_at=_now(),
            )
            session.add(entity)
            session.flush()
            return self._user_dict(entity)

    def list_users(self) -> list[dict]:
        """全部用户（含管理员），按创建时间倒序。"""
        with self._session_scope() as session:
            stmt = select(User).order_by(User.id.desc())
            return [self._user_dict(entity) for entity in session.execute(stmt).scalars()]

    def update_user(
        self,
        user_id: int,
        *,
        level: int | None = None,
        is_active: bool | None = None,
        password_hash: str | None = None,
    ) -> dict | None:
        """管理员更新用户：设等级 / 禁用启用 / 重置密码。返回更新后的用户，不存在返回 None。"""
        with self._session_scope() as session:
            entity = session.get(User, int(user_id))
            if entity is None:
                return None
            if entity.role != "admin":  # 管理员账号等级恒为 4，不通过此接口修改
                if level is not None:
                    entity.level = int(level)
            if is_active is not None:
                entity.is_active = bool(is_active)
            if password_hash:
                entity.password_hash = password_hash
            entity.updated_at = _now()
            session.flush()
            return self._user_dict(entity)

    def touch_login(self, username: str) -> None:
        """登录成功后刷新 last_login_at（失败静默，不影响登录流程）。"""
        try:
            with self._session_scope() as session:
                entity = session.execute(
                    select(User).where(User.username == str(username))
                ).scalars().first()
                if entity is not None:
                    entity.last_login_at = _now()
        except Exception:
            logger.warning("更新 last_login_at 失败 username=%s", username, exc_info=True)


# ---------------------------------------------------------------------------
# 模块级单例
# ---------------------------------------------------------------------------

_mysql_store: MySQLStore | None = None
_mysql_lock = threading.Lock()


def get_mysql() -> MySQLStore:
    """进程内 MySQLStore 单例（懒加载 + 双检锁）。

    首次调用同时校验引擎能否创建（驱动是否安装等），失败抛
    ``RuntimeError("MySQL 连接失败: ...")``，让上层可以在启动/建库阶段统一降级。
    """
    global _mysql_store
    if _mysql_store is None:
        with _mysql_lock:
            if _mysql_store is None:
                store = MySQLStore()
                store._ensure_engine()  # 只加载驱动，不建立网络连接
                _mysql_store = store
    return _mysql_store
