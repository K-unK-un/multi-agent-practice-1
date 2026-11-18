"""知识库构建主链路：解析 → 清洗 → 条款切分 → 规则抽取 → Chroma + BM25 双路索引。

设计要点：
- MySQL 是元数据主库：文档/条款/规则先落库，Chroma 只存子块文本 + 元数据；
- 版本管理：同 title 的旧 active 版本置为 archived，不物理删除；
- 所有 DB / Chroma / LLM 操作都单独 try/except，失败记入 warnings，不中断整批建库；
- 若某文档 MySQL 写入失败，则跳过它的向量索引，避免出现"有向量、无元数据"的孤儿数据；
- 全部文件处理完后统一重建 BM25（全量 active 条款），最后由调用方决定是否热加载。
"""
from __future__ import annotations

import hashlib
import re
import threading
import uuid
from datetime import date, datetime
from pathlib import Path

from app.core.config import settings
from app.core.logger import get_logger
from app.ingest.cleaner import clean_pages
from app.ingest.loader import ParsedDoc, extract_meta, load_document
from app.ingest.rule_extract import extract_rules
from app.ingest.splitter import split_clauses
from app.llm.client import get_embeddings
from app.retrieval.bm25 import BM25Index
from app.store.chroma import get_chroma
from app.store.mysql import get_mysql

logger = get_logger("ingest.pipeline")

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".md", ".txt"}
DEFAULT_SCOPE = ["employee"]

# 任务注册表：kb API 轮询用。所有读写都持锁，供 FastAPI 多线程/后台任务访问。
TASK_REGISTRY: dict[str, dict] = {}
_TASK_LOCK = threading.Lock()


# ---------------------------------------------------------------- 小工具

def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _file_checksum(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# kb API 上传落盘名：{uuid4().hex[:12]}_{原文件名}
_UPLOAD_PREFIX_RE = re.compile(r"^[0-9a-fA-F]{12}_")


def _source_name(file_path: str) -> str:
    """展示/审计用原始文件名：去掉上传时加的随机 uuid 前缀（截断到 255）。"""
    name = _UPLOAD_PREFIX_RE.sub("", Path(str(file_path or "")).name)
    return name[:255]


def _content_fingerprint(text: str) -> str:
    """正文内容指纹：去掉全部空白后做 SHA1。

    与文件 checksum 的区别：checksum 识别“字节完全相同的文件”；
    content_fingerprint 识别“重新导出/换格式但正文相同的文件”。
    """
    normalized = re.sub(r"\s+", "", text or "")
    return _sha1(normalized) if normalized else ""


def _doc_field(document, key: str):
    """兼容 list_documents() 返回 dict 或 ORM 对象。"""
    if isinstance(document, dict):
        return document.get(key)
    return getattr(document, key, None)


# ---------------------------------------------------------------- 任务注册表

def create_task(kind: str, params: dict) -> str:
    """创建异步任务记录，返回 task_id（uuid4）。状态初始为 pending。"""
    task_id = uuid.uuid4().hex
    with _TASK_LOCK:
        TASK_REGISTRY[task_id] = {
            "task_id": task_id,
            "kind": kind,
            "params": dict(params or {}),
            "status": "pending",
            "result": None,
            "error": None,
            "created_at": _now_iso(),
            "started_at": None,
            "finished_at": None,
        }
    return task_id


def update_task(task_id: str, **kwargs) -> dict | None:
    """线程安全地更新任务字段；running/done/failed 自动补时间戳。"""
    with _TASK_LOCK:
        task = TASK_REGISTRY.get(task_id)
        if task is None:
            return None
        status = kwargs.get("status")
        if status == "running" and not task.get("started_at"):
            task["started_at"] = _now_iso()
        if status in ("done", "failed"):
            task["finished_at"] = _now_iso()
        task.update(kwargs)
        return dict(task)


def get_task(task_id: str) -> dict | None:
    """返回任务副本；不存在返回 None。"""
    with _TASK_LOCK:
        task = TASK_REGISTRY.get(task_id)
        return dict(task) if task is not None else None


# ---------------------------------------------------------------- 文档元数据

def build_doc_meta(file_path: str, scope: list[str]) -> dict:
    """文件名 → 文档元数据（doc_id/version/effective_date 等）。

    doc_id = sha1(f"{title}|{version}|{effective_date or ''}")[:40]
    同一制度不同版本/生效日期会得到不同 doc_id，旧版本在新版入库时归档。
    """
    path = Path(file_path)
    info = extract_meta(str(path))
    # 入库用跨版本一致的制度名（去掉 V3/(2025) 等），ParsedDoc.title 仍保留原始 stem
    title = info.get("base_title") or info["title"]
    version = info.get("version")
    effective_date = info.get("effective_date")

    doc_id = _sha1(
        f"{title}|{version or ''}|{effective_date.isoformat() if effective_date else ''}"
    )[:40]

    try:
        checksum = _file_checksum(path)
    except Exception as exc:
        logger.warning("文件 checksum 计算失败 %s: %s", path, exc)
        checksum = None

    return {
        "id": doc_id,
        "doc_id": doc_id,
        "title": title,
        "category": None,
        "version": version,
        "effective_date": effective_date,
        "status": "active",
        "scope": list(scope),
        "source_path": str(path.resolve()),
        "source_name": _source_name(file_path),
        "checksum": checksum,
        "content_fingerprint": None,  # 清洗完成后由 run_ingest 填入
        "ingested_at": _now_iso(),
    }


def _as_date(value) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _version_key(version) -> tuple[float, ...]:
    numbers = re.findall(r"\d+(?:\.\d+)?", str(version or ""))
    return tuple(float(number) for number in numbers) or (0.0,)


def _is_newer(new_meta: dict, old_document) -> bool:
    """判断本次入库文档是否应覆盖旧行：先比生效日期，再比版本号。

    返回 False 表示旧行反而更新（例如误传历史版本），此时不应归档旧行。
    """
    new_date = _as_date(new_meta.get("effective_date"))
    old_date = _as_date(_doc_field(old_document, "effective_date"))
    if new_date and old_date:
        if new_date != old_date:
            return new_date > old_date
    elif new_date and not old_date:
        return True
    elif not new_date and old_date:
        return False

    new_version = new_meta.get("version")
    old_version = _doc_field(old_document, "version")
    if new_version and old_version:
        new_key, old_key = _version_key(new_version), _version_key(old_version)
        if new_key != old_key:
            return new_key > old_key
    return True


def _archive_old_versions(
    mysql, new_meta: dict, warnings: list[str]
) -> tuple[bool, list[str]]:
    """同 title 的 active 旧版本置 archived（只归档，不删除）。

    返回 ``(incoming_is_older, archived_ids)``：
    - incoming_is_older=True 表示库中已存在更新的同制度版本，本次上传属于历史版本，
      调用方应把新行直接标记为 archived，避免出现多个 active 版本；
    - archived_ids 是本次成功归档的旧版本文档 id，供入库台账记录动作。
    """
    title = new_meta["title"]
    new_doc_id = new_meta["doc_id"]
    archived_ids: list[str] = []
    try:
        documents = list(mysql.list_documents() or [])
    except Exception as exc:
        warnings.append(f"读取文档列表失败，跳过版本归档: {exc}")
        return False, archived_ids

    incoming_is_older = False
    for document in documents:
        old_id = _doc_field(document, "id")
        old_title = _doc_field(document, "title")
        old_status = _doc_field(document, "status") or "active"
        if old_title != title or old_status != "active":
            continue
        if str(old_id) == new_doc_id:
            continue
        if not _is_newer(new_meta, document):
            incoming_is_older = True
            warnings.append(
                f"检测到同制度更新版本（doc_id={old_id}），本次上传的 {new_meta.get('version') or '未知版本'}"
                " 将标记为 archived"
            )
            continue
        try:
            mysql.archive_document(old_id)
            archived_ids.append(str(old_id))
            warnings.append(f"同制度旧版本已归档: doc_id={old_id}")
            logger.info("归档旧版本: title=%s doc_id=%s", title, old_id)
            # 同步向量元数据，避免旧版条款仍被 status=active 的 where 过滤命中
            try:
                _update_doc_vectors_status(
                    get_chroma(), str(old_id), "archived", warnings
                )
            except Exception as exc:
                warnings.append(f"同步旧版本向量状态失败 doc_id={old_id}: {exc}")
        except Exception as exc:
            warnings.append(f"归档旧版本失败 doc_id={old_id}: {exc}")
    return incoming_is_older, archived_ids


# ---------------------------------------------------------------- 向量索引

def _update_doc_vectors_status(
    store, doc_id: str, status: str, warnings: list[str]
) -> None:
    """把某文档全部子块向量的 metadata.status 改为 archived/active。

    旧版本归档后若不同步 Chroma 元数据，向量路仍会按 status=active 命中旧版条款，
    造成"新版已生效、旧版仍被召回"的版本混检。
    """
    try:
        collection = store.get_or_create_collection()
    except Exception as exc:
        warnings.append(f"连接 Chroma 失败，未同步向量状态 doc_id={doc_id}: {exc}")
        return
    getter = getattr(collection, "get", None)
    updater = getattr(collection, "update", None)
    if not (callable(getter) and callable(updater)):
        return
    try:
        existing = getter(where={"doc_id": str(doc_id)}, include=["metadatas"])
        ids = (existing or {}).get("ids") or []
        metadatas = (existing or {}).get("metadatas") or []
        if not ids:
            return
        updated: list[dict] = []
        for index, _ in enumerate(ids):
            meta = dict(metadatas[index] or {}) if index < len(metadatas) else {}
            meta["doc_id"] = str(doc_id)
            meta["status"] = status
            updated.append(meta)
        updater(ids=ids, metadatas=updated)
        logger.info("同步向量状态 %s 条: doc_id=%s status=%s", len(ids), doc_id, status)
    except Exception as exc:
        warnings.append(f"同步向量状态失败 doc_id={doc_id}: {exc}")


def _clear_doc_vectors(store, doc_id: str, warnings: list[str]) -> None:
    """删除该文档的旧向量，保证同一 doc_id 重新上传时是覆盖而不是重复写入。

    ChromaStore 的公开接口只有 add；这里通过 get_or_create_collection() 返回的
    chromadb Collection 按 metadata doc_id 清理旧行（若实现不支持则静默跳过）。
    """
    try:
        collection = store.get_or_create_collection()
    except Exception:
        return
    getter = getattr(collection, "get", None)
    deleter = getattr(collection, "delete", None)
    if not (callable(getter) and callable(deleter)):
        return
    try:
        existing = getter(where={"doc_id": str(doc_id)}, include=["metadatas"])
        ids = (existing or {}).get("ids") or []
        if ids:
            deleter(ids=ids)
            logger.info("清理旧向量 %s 条: doc_id=%s", len(ids), doc_id)
    except Exception as exc:
        warnings.append(f"清理旧向量失败 doc_id={doc_id}: {exc}")


def _index_children(
    children: list[dict], warnings: list[str], doc_meta: dict | None = None
) -> int:
    """子块分批 embedding 后写入 Chroma；返回成功写入的条数。

    metadata 除契约要求的 doc_id/article_no/page/title 外，还写入 retrieval/filters.py
    依赖的 status/effective_date/scope：Chroma where 对缺失 key 一律不匹配，缺了这三个
    字段向量路会被元数据硬过滤全部误杀。scope 取单值（空列表写 "" = 全员可见）；
    多 scope 文档 Chroma 无法用单值表达，取首个 scope 并给出 warning。
    """
    if not children:
        return 0

    doc_meta = doc_meta or {}
    doc_id = str(doc_meta.get("doc_id") or "")
    status = str(doc_meta.get("status") or "active")
    effective_date = doc_meta.get("effective_date")
    if hasattr(effective_date, "isoformat"):
        effective_date_str = effective_date.isoformat()
    else:
        effective_date_str = str(effective_date or "")
    scopes = list(doc_meta.get("scope") or [])
    scope_value = str(scopes[0]) if scopes else ""
    if len(scopes) > 1:
        warnings.append(
            f"文档 {doc_id} 配置了多个 scope={scopes}，Chroma metadata 只支持单值，"
            f"已写入首个 scope={scope_value}"
        )

    try:
        embeddings_client = get_embeddings()
    except Exception as exc:
        warnings.append(f"Embedding 客户端初始化失败，跳过向量索引: {exc}")
        return 0

    store = None
    try:
        store = get_chroma()
        store.get_or_create_collection()
    except Exception as exc:
        warnings.append(f"Chroma collection 初始化失败，跳过向量索引: {exc}")
        return 0

    batch_size = max(1, int(settings.embedding_batch_size or 32))

    # 两阶段：先把该文档所有子块 embedding 完，再清理旧向量、批量写入。
    # 任一 embedding 批次失败则整篇放弃索引（保留旧向量），避免"旧向量被清空、
    # 新向量只写了一半"的更糟状态。
    prepared: list[tuple[list[dict], list[list[float]]]] = []
    for offset in range(0, len(children), batch_size):
        batch = children[offset: offset + batch_size]
        texts = [block["text"] for block in batch]
        try:
            vectors = embeddings_client.embed_documents(texts)
        except Exception as exc:
            warnings.append(
                f"Embedding 失败（offset={offset}, size={len(batch)}），整篇跳过索引: {exc}"
            )
            return 0
        if len(vectors) != len(batch):
            warnings.append(
                f"Embedding 返回条数不匹配（{len(vectors)} != {len(batch)}），整篇跳过索引"
            )
            return 0
        prepared.append((batch, vectors))

    if doc_id:
        _clear_doc_vectors(store, doc_id, warnings)

    indexed = 0
    for batch_index, (batch, vectors) in enumerate(prepared):
        offset = batch_index * batch_size
        texts = [block["text"] for block in batch]
        metadatas = [
            {
                "doc_id": str(block.get("doc_id") or ""),
                "article_no": str(block.get("article_no") or ""),
                "page": int(block.get("page") or 0),
                "title": str(block.get("title") or ""),
                # retrieval/filters.py 的 where 依赖，缺失会被误杀
                "status": status,
                "effective_date": effective_date_str,
                "scope": scope_value,
                # 入库审计信息（不影响检索过滤，仅供追溯与判重）
                "version": str(doc_meta.get("version") or ""),
                "source_name": str(doc_meta.get("source_name") or ""),
                "content_fingerprint": str(doc_meta.get("content_fingerprint") or ""),
                "ingested_at": str(doc_meta.get("ingested_at") or ""),
            }
            for block in batch
        ]
        try:
            store.add(
                ids=[block["id"] for block in batch],
                documents=texts,
                embeddings=vectors,
                metadatas=metadatas,
            )
            indexed += len(batch)
        except Exception as exc:
            warnings.append(
                f"Chroma 写入失败（offset={offset}, size={len(batch)}）: {exc}"
            )

    return indexed


# ---------------------------------------------------------------- BM25

def _rebuild_bm25(mysql, warnings: list[str]) -> None:
    """全量重建 BM25：MySQL 取全部 active 条款 → BM25Index.build_from_clauses()。"""
    try:
        clauses = list(mysql.get_clauses(include_archived=False) or [])
    except Exception as exc:
        warnings.append(f"读取 active 条款失败，跳过 BM25 重建: {exc}")
        return
    try:
        BM25Index.build_from_clauses(clauses)
        logger.info("BM25 重建完成：%s 条父块", len(clauses))
    except Exception as exc:
        warnings.append(f"BM25 重建失败: {exc}")


# ---------------------------------------------------------------- 入库台账

def _record_ingest(
    mysql,
    *,
    task_id: str | None,
    task_kind: str,
    operator: dict | None,
    file_path: str,
    meta: dict | None,
    action: str,
    ok: bool,
    document_status: str | None,
    warnings: list[str],
) -> None:
    """写入一条入库台账（失败不影响主流程）。"""
    if mysql is None:
        return
    operator = operator or {}
    record = {
        "task_id": task_id,
        "task_kind": task_kind or "cli",
        "doc_id": (meta or {}).get("doc_id"),
        "title": (meta or {}).get("title"),
        "version": (meta or {}).get("version"),
        "effective_date": (meta or {}).get("effective_date"),
        "action": action,
        "ok": bool(ok),
        "document_status": document_status,
        "source_name": (meta or {}).get("source_name") or _source_name(file_path),
        "source_path": str(Path(file_path).resolve()) if file_path else None,
        "checksum": (meta or {}).get("checksum"),
        "content_fingerprint": (meta or {}).get("content_fingerprint"),
        "operator_id": operator.get("id"),
        "operator_name": operator.get("username") or "cli",
        "warnings": list(warnings or []),
    }
    try:
        mysql.insert_ingest_record(record)
    except Exception:
        logger.warning(
            "写入入库台账失败 file=%s action=%s", file_path, action, exc_info=True
        )


# ---------------------------------------------------------------- 主入口

def run_ingest(
    file_paths: list[str],
    scope: list[str] | None = None,
    with_rules: bool = True,
    *,
    task_id: str | None = None,
    task_kind: str = "cli",
    operator: dict | None = None,
) -> dict:
    """批量建库。返回 {documents, clauses, rules, chroma_count, warnings}。"""
    warnings: list[str] = []
    scope_list = list(scope) if scope else list(DEFAULT_SCOPE)

    documents = 0          # 成功写入 MySQL 的文档数
    clause_count = 0       # 成功替换入库的父块数
    rule_count = 0         # 成功入库的规则数
    all_parents: list[dict] = []

    mysql = None
    try:
        mysql = get_mysql()
    except Exception as exc:
        warnings.append(f"MySQL 初始化失败，本次仅做解析但不入库: {exc}")

    for file_path in file_paths or []:
        name = Path(file_path).name
        warn_start = len(warnings)
        meta: dict | None = None

        # 1) 解析
        try:
            parsed = load_document(file_path)
        except Exception as exc:
            warnings.append(f"[{name}] 解析失败: {exc}")
            _record_ingest(
                mysql, task_id=task_id, task_kind=task_kind, operator=operator,
                file_path=file_path, meta=None, action="failed", ok=False,
                document_status=None, warnings=warnings[warn_start:],
            )
            continue

        # 2) 元数据（先于切分构建：用清洗后的制度名做文首标题去噪）
        try:
            meta = build_doc_meta(file_path, scope_list)
        except Exception as exc:
            warnings.append(f"[{name}] 元数据提取失败: {exc}")
            _record_ingest(
                mysql, task_id=task_id, task_kind=task_kind, operator=operator,
                file_path=file_path, meta=None, action="failed", ok=False,
                document_status=None, warnings=warnings[warn_start:],
            )
            continue

        # 3) 清洗（保留页码）
        try:
            cleaned_pages = clean_pages(parsed.pages)
            cleaned_full = "\n".join(
                page["text"] for page in cleaned_pages if page["text"]
            )
        except Exception as exc:
            warnings.append(f"[{name}] 清洗失败: {exc}")
            _record_ingest(
                mysql, task_id=task_id, task_kind=task_kind, operator=operator,
                file_path=file_path, meta=meta, action="failed", ok=False,
                document_status=None, warnings=warnings[warn_start:],
            )
            continue

        if not cleaned_full.strip():
            warnings.append(f"[{name}] 清洗后无有效文本，跳过")
            _record_ingest(
                mysql, task_id=task_id, task_kind=task_kind, operator=operator,
                file_path=file_path, meta=meta, action="failed", ok=False,
                document_status=None, warnings=warnings[warn_start:],
            )
            continue

        # 3.1) 正文内容指纹：用于识别“换文件名/重新导出但内容相同”的重复文档
        meta["content_fingerprint"] = _content_fingerprint(cleaned_full)

        parsed_clean = ParsedDoc(
            title=meta["title"],  # 去掉上传随机前缀/版本号后的制度名
            full_text=cleaned_full,
            pages=cleaned_pages,
            format=parsed.format,
        )

        # 4) 切分
        try:
            parents, children = split_clauses(parsed_clean, meta)
        except Exception as exc:
            warnings.append(f"[{name}] 条款切分失败: {exc}")
            _record_ingest(
                mysql, task_id=task_id, task_kind=task_kind, operator=operator,
                file_path=file_path, meta=meta, action="failed", ok=False,
                document_status=None, warnings=warnings[warn_start:],
            )
            continue
        if not parents:
            warnings.append(f"[{name}] 未切出任何条款，跳过")
            _record_ingest(
                mysql, task_id=task_id, task_kind=task_kind, operator=operator,
                file_path=file_path, meta=meta, action="failed", ok=False,
                document_status=None, warnings=warnings[warn_start:],
            )
            continue

        doc_ok = False
        if mysql is None:
            warnings.append(f"[{name}] MySQL 不可用，跳过入库与向量索引")
            continue

        # 4.1) 查重 / 冲突识别（只记录并告警，不阻断入库）
        existing = None
        try:
            existing = mysql.get_document(meta["doc_id"])
        except Exception as exc:
            warnings.append(f"[{name}] 查询既有文档失败: {exc}")

        if meta["content_fingerprint"]:
            try:
                duplicates = mysql.find_documents_by_fingerprint(
                    meta["content_fingerprint"], exclude_id=meta["doc_id"]
                ) or []
            except Exception as exc:
                warnings.append(f"[{name}] 内容指纹查重失败: {exc}")
                duplicates = []
            if duplicates:
                preview = "、".join(
                    f"{_doc_field(doc, 'title') or '未命名'}({_doc_field(doc, 'id')})"
                    for doc in duplicates[:3]
                )
                warnings.append(f"[{name}] 检测到相同内容的其他文档: {preview}")

        # 5) 版本管理 + 文档入库
        incoming_is_older, archived_ids = _archive_old_versions(mysql, meta, warnings)
        doc_row = {
            "id": meta["id"],
            "title": meta["title"],
            "category": meta["category"],
            "version": meta["version"],
            "effective_date": meta["effective_date"],
            "status": "archived" if incoming_is_older else "active",
            "scope": meta["scope"],
            "source_path": meta["source_path"],
            "source_name": meta["source_name"],
            "checksum": meta["checksum"],
            "content_fingerprint": meta["content_fingerprint"],
        }
        if incoming_is_older:
            meta["status"] = "archived"
        else:
            meta["status"] = "active"
        try:
            mysql.upsert_document(doc_row)
            doc_ok = True
            documents += 1
        except Exception as exc:
            warnings.append(f"[{name}] 文档元数据入库失败: {exc}")

        if not doc_ok:
            _record_ingest(
                mysql, task_id=task_id, task_kind=task_kind, operator=operator,
                file_path=file_path, meta=meta, action="failed", ok=False,
                document_status=None, warnings=warnings[warn_start:],
            )
            continue

        # 6) 条款入库
        try:
            mysql.replace_clauses_for_doc(meta["doc_id"], parents)
            clause_count += len(parents)
        except Exception as exc:
            warnings.append(f"[{name}] 条款入库失败，跳过向量索引: {exc}")
            _record_ingest(
                mysql, task_id=task_id, task_kind=task_kind, operator=operator,
                file_path=file_path, meta=meta, action="failed", ok=False,
                document_status=meta.get("status"), warnings=warnings[warn_start:],
            )
            continue

        # 7) 子块向量索引（先按 doc_id 清理旧向量，实现"重新上传即覆盖"）
        indexed = _index_children(children, warnings, meta)
        logger.info(
            "文档处理完成: %s (doc_id=%s, 父块=%s, 子块=%s, 已索引=%s)",
            name,
            meta["doc_id"],
            len(parents),
            len(children),
            indexed,
        )

        # 7.1) 判定动作并写入入库台账
        if existing is not None:
            same_content = (
                bool(existing.get("checksum"))
                and existing.get("checksum") == meta.get("checksum")
            ) or (
                bool(existing.get("content_fingerprint"))
                and existing.get("content_fingerprint")
                == meta.get("content_fingerprint")
            )
            action = "reingest_unchanged" if same_content else "overwrite"
        elif incoming_is_older:
            action = "older_archived"
        elif archived_ids:
            action = "upgrade"
        else:
            action = "create"

        _record_ingest(
            mysql, task_id=task_id, task_kind=task_kind, operator=operator,
            file_path=file_path, meta=meta, action=action, ok=True,
            document_status=meta.get("status"), warnings=warnings[warn_start:],
        )
        all_parents.extend(parents)

    # 8) BM25 全量重建（即使本次没有新文档，也保证索引与 MySQL 一致）
    if mysql is not None:
        _rebuild_bm25(mysql, warnings)

    # 9) 规则表抽取
    if with_rules:
        if mysql is None:
            warnings.append("MySQL 不可用，跳过规则表抽取")
        elif not all_parents:
            logger.info("本次没有新增父块，跳过规则抽取")
        else:
            try:
                rows = extract_rules(all_parents)
            except Exception as exc:  # extract_rules 内部已兜底，这里再防一层
                warnings.append(f"规则抽取异常: {exc}")
                rows = []
            if rows:
                try:
                    mysql.upsert_rule_rows(rows)
                    rule_count = len(rows)
                except Exception as exc:
                    warnings.append(f"规则表入库失败: {exc}")

    # 10) 向量库总量
    chroma_count = 0
    try:
        chroma_count = int(get_chroma().count())
    except Exception as exc:
        warnings.append(f"读取 Chroma 计数失败: {exc}")

    result = {
        "documents": documents,
        "clauses": clause_count,
        "rules": rule_count,
        "chroma_count": chroma_count,
        "warnings": warnings,
    }
    logger.info("建库完成: %s", result)
    return result


def rebuild_all(
    data_dir: str = "data/docs",
    *,
    task_id: str | None = None,
    task_kind: str = "reindex",
    operator: dict | None = None,
) -> dict:
    """遍历目录下所有受支持文档并全量重建索引。空目录返回提示。"""
    root = Path(data_dir)
    files: list[str] = []
    if root.is_dir():
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            if path.name.startswith("~$"):  # Word 临时锁文件
                continue
            files.append(str(path))

    if not files:
        logger.warning("知识库目录为空或不存在: %s", root)
        return {"documents": 0, "warnings": ["目录为空"]}

    return run_ingest(
        files, task_id=task_id, task_kind=task_kind, operator=operator
    )
