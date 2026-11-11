"""文档解析：PDF / DOCX / MD / TXT → 统一中间结构 ParsedDoc。

约定：
- DOCX 解析只用标准库 zipfile + lxml（项目未安装 python-docx，禁止引入）。
- 页码：PDF 按真实页；DOCX / MD / TXT 无页概念，统一记为第 1 页。
- 扫描件（PDF 全部页面提取为空）直接抛 RuntimeError，由上层记 warning 并跳过。
"""
from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from lxml import etree
from pypdf import PdfReader

from app.core.logger import get_logger

logger = get_logger("ingest.loader")

# WordprocessingML 主命名空间（transitional，Word 默认输出）
_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_W_P = f"{{{_W_NS}}}p"
_W_T = f"{{{_W_NS}}}t"
_W_TAB = f"{{{_W_NS}}}tab"
_W_BR = f"{{{_W_NS}}}br"

_SUPPORTED_FORMATS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".md": "md",
    ".txt": "txt",
}

# 文件名里的版本/年份线索，例如 "年假管理制度V3(2025)"、"差旅费管理办法（2025年修订）"
_RE_VERSION = re.compile(r"[vV]\s*(\d+(?:\.\d+)?)")
_RE_VERSION_CN = re.compile(r"第\s*(\d+)\s*版")
_RE_FULL_DATE = re.compile(
    r"(20\d{2})\s*[年./\-]\s*(\d{1,2})\s*[月./\-]\s*(\d{1,2})\s*日?"
)
_RE_YEAR = re.compile(r"(20\d{2})")


@dataclass
class ParsedDoc:
    """解析后的文档中间结构。pages 保留页码，供切分时把条款映射回页码。"""

    title: str
    full_text: str
    pages: list[dict] = field(default_factory=list)
    format: str = ""


# ---------------------------------------------------------------- PDF

def _load_pdf(path: Path) -> list[dict]:
    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # 加密/损坏 PDF
        raise RuntimeError(f"PDF 解析失败: {path} ({exc})") from exc

    pages: list[dict] = []
    for page_no, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # 单页失败不拖垮整篇
            logger.warning("PDF 第 %s 页文本提取失败: %s", page_no, exc)
            text = ""
        pages.append({"page_no": page_no, "text": text})

    if not any((p["text"] or "").strip() for p in pages):
        raise RuntimeError("疑似扫描件，需 OCR，暂不支持")
    return pages


# ---------------------------------------------------------------- DOCX

def _docx_paragraph_text(paragraph: etree._Element) -> str:
    """拼接一个 w:p 内的文本节点；tab/br 保留为制表符/换行。"""
    parts: list[str] = []
    for node in paragraph.iter():
        if node.tag == _W_T:
            parts.append(node.text or "")
        elif node.tag == _W_TAB:
            parts.append("\t")
        elif node.tag == _W_BR:
            parts.append("\n")
    return "".join(parts)


def _load_docx(path: Path) -> list[dict]:
    try:
        with zipfile.ZipFile(str(path)) as zf:
            try:
                xml_bytes = zf.read("word/document.xml")
            except KeyError as exc:
                raise RuntimeError(f"DOCX 缺少 word/document.xml: {path}") from exc
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"DOCX 文件损坏或不是合法 zip: {path}") from exc

    try:
        root = etree.fromstring(xml_bytes)
    except etree.XMLSyntaxError as exc:
        raise RuntimeError(f"DOCX document.xml 解析失败: {path} ({exc})") from exc

    paragraphs = [_docx_paragraph_text(p) for p in root.iter(_W_P)]
    text = "\n".join(paragraphs)
    if not text.strip():
        logger.warning("DOCX 未提取到文本: %s", path)
    # DOCX 无可靠页码信息，统一作为第 1 页，页内偏移由清洗/切分保持
    return [{"page_no": 1, "text": text}]


# ---------------------------------------------------------------- MD / TXT

def _load_plain_text(path: Path) -> list[dict]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("UTF-8 解码失败，回退 GBK: %s", path)
        text = path.read_text(encoding="gbk")
    # 去掉 UTF-8 BOM，避免污染首行标题
    text = text.lstrip("\ufeff")
    return [{"page_no": 1, "text": text}]


# ---------------------------------------------------------------- 元数据

_RE_PAREN_YEAR = re.compile(r"[（(]\s*20\d{2}[^）)]*[)）]")
_RE_TRAILING_DATE = re.compile(
    r"[\s_\-—·]*20\d{2}\s*年?(?:\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*日)?)?\s*$"
)
# kb API 上传落盘名：{uuid4().hex[:12]}_{原文件名}，建库前去掉随机前缀，
# 否则同一制度每次上传的 title 都不同，版本归档永远匹配不上。
_RE_UPLOAD_PREFIX = re.compile(r"^[0-9a-fA-F]{12}_")


def _base_title(stem: str) -> str:
    """去掉上传随机前缀与版本/年份标记，得到跨版本一致的制度名。

    例："年假管理制度V3(2025)" → "年假管理制度"；
        "差旅费管理办法（2025年3月1日修订）V2.1" → "差旅费管理办法"；
        "a1b2c3d4e5f6_年假管理制度V2(2025)" → "年假管理制度"。
    这是版本管理（同 title 旧版归档）能匹配上的关键；失败时回退 stem。
    """
    title = _RE_UPLOAD_PREFIX.sub("", stem)
    title = _RE_VERSION.sub("", title)
    title = _RE_VERSION_CN.sub("", title)
    title = _RE_PAREN_YEAR.sub("", title)
    title = _RE_TRAILING_DATE.sub("", title)
    title = title.strip(" _-—·\t")
    return title or stem


def extract_meta(file_path: str) -> dict:
    """从文件名提取 title / base_title / version / effective_year / effective_date。

    - title：Path(path).stem（原始文件名，契约字段）；
    - base_title：去掉版本/年份后的制度名，供 Document.title 与版本归档使用；
    - version：优先 "V3"/"v3.1"，其次 "第3版"；统一成大写 V 前缀；
    - effective_date：优先完整日期（2025年3月1日 / 2025-03-01 / 2025.3.1）；
      只有年份时取该年 1 月 1 日（建库过滤用，宁可保守）。
    - 全部失败时保留 stem 兜底，版本/日期为 None。
    """
    stem = Path(file_path).stem
    version: str | None = None
    match = _RE_VERSION.search(stem)
    if match:
        version = f"V{match.group(1)}"
    else:
        match = _RE_VERSION_CN.search(stem)
        if match:
            version = f"V{match.group(1)}"

    effective_date: date | None = None
    effective_year: int | None = None

    match = _RE_FULL_DATE.search(stem)
    if match:
        year, month, day = (int(x) for x in match.groups())
        try:
            effective_date = date(year, month, day)
            effective_year = year
        except ValueError:
            logger.warning("文件名日期非法，忽略: %s", stem)
    if effective_date is None:
        match = _RE_YEAR.search(stem)
        if match:
            effective_year = int(match.group(1))
            try:
                effective_date = date(effective_year, 1, 1)
            except ValueError:
                effective_date = None

    return {
        "title": stem,
        "base_title": _base_title(stem),
        "version": version,
        "effective_year": effective_year,
        "effective_date": effective_date,
    }


# ---------------------------------------------------------------- 入口

def load_document(path: str) -> ParsedDoc:
    """按扩展名解析文档，返回 ParsedDoc；不支持的格式抛 ValueError。"""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    suffix = file_path.suffix.lower()
    fmt = _SUPPORTED_FORMATS.get(suffix)
    if fmt is None:
        raise ValueError(f"不支持的文件格式: {suffix or '(无扩展名)'}，仅支持 pdf/docx/md/txt")

    if fmt == "pdf":
        pages = _load_pdf(file_path)
    elif fmt == "docx":
        pages = _load_docx(file_path)
    else:
        pages = _load_plain_text(file_path)

    full_text = "\n".join((p.get("text") or "") for p in pages)
    return ParsedDoc(title=file_path.stem, full_text=full_text, pages=pages, format=fmt)
