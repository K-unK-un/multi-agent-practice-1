"""条款切分：按"章/节 + 第X条"强边界切父块，再切 ~300 字子块（滑窗重叠 50 字）。

父块（parent_clauses）字段严格按契约：
    {doc_id, article_no, title, chapter, content, page, order_no}
子块（child_blocks）字段严格按契约：
    {id, doc_id, article_no, title, page, text, content_hash}

设计说明（伪条款）：
- 文档开头、第一个章节/条款之前的发布说明等文本 → 伪条款 article_no="前言"；
- 章节标题下、该章第一条之前的说明文字 → 并入该章第一条的 content（标题仍取条款首行），
  这样章节导语不会丢失，也不会产生难以引用的虚拟条款号；
- 某章通篇没有"第X条"（例如仅用"一、二、"分节）→ 独立伪条款 article_no="章{n}"。
- 重复出现的条款号（OCR/引用误识别）合并进首条，保证 MySQL (doc_id, article_no) 唯一索引可写入。
"""
from __future__ import annotations

import hashlib
import re

from app.core.logger import get_logger
from app.ingest.loader import ParsedDoc

logger = get_logger("ingest.splitter")

# 条款边界：第X条。负向断言避免把正文里的"第十二条规定/所称/中/的/至..."误判为新条款。
_RE_ARTICLE = re.compile(
    r"^第\s*([一二三四五六七八九十百零〇\d]+)\s*条"
    r"(?!\s*(?:所|规|称|中|的|款|至|到|和|及|与|、|，|,))"
)
# 章/节/篇标题
_RE_CHAPTER = re.compile(r"^第\s*([一二三四五六七八九十百零〇\d]+)\s*[章节篇]")
# "一、总则" 这类中文序号标题（仅作章节上下文，长度限制避免误伤正文枚举）
_RE_CHAPTER_CN = re.compile(r"^([一二三四五六七八九十]+)\s*、")
_RE_CHAPTER_CN_MAX_LEN = 40

_CN_DIGITS = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_CN_UNITS = {"十": 10, "百": 100}

CHILD_SIZE = 300      # 子块目标长度
CHILD_OVERLAP = 50    # 滑窗重叠
TITLE_MAX_LEN = 255   # 对齐 MySQL String(255)


# ---------------------------------------------------------------- 工具

def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _cn_to_int(value: str) -> int | None:
    """中文数字（含百）转 int；无法解析返回 None。"""
    if value.isdigit():
        return int(value)
    total = 0
    current = 0
    for char in value:
        if char in _CN_DIGITS:
            current = _CN_DIGITS[char]
        elif char in _CN_UNITS:
            total += (current or 1) * _CN_UNITS[char]
            current = 0
        else:
            return None
    return total + current


def _normalize_article_no(raw: str) -> str:
    number = _cn_to_int(raw.strip())
    return str(number) if number is not None else raw.strip()


def _iter_lines(parsed: ParsedDoc):
    """产出 (line, page_no)；页面为空时回退到 full_text 按第 1 页处理。"""
    pages = parsed.pages or [{"page_no": 1, "text": parsed.full_text}]
    for page in pages:
        page_no = int(page.get("page_no") or 1)
        for raw in (page.get("text") or "").splitlines():
            line = raw.strip()
            if line:
                yield line, page_no


def _is_chapter_heading(line: str) -> bool:
    if _RE_CHAPTER.match(line):
        return True
    return bool(_RE_CHAPTER_CN.match(line)) and len(line) <= _RE_CHAPTER_CN_MAX_LEN


def _render(lines: list[tuple[str, int]]) -> tuple[str, list[tuple[int, int]]]:
    """把 [(line, page)] 拼成 content，并返回 [(char_start, page)] 偏移表。"""
    parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for line, page_no in lines:
        offsets.append((cursor, page_no))
        parts.append(line)
        cursor += len(line) + 1  # +1 = "\n"
    return "\n".join(parts), offsets


def _page_at(offsets: list[tuple[int, int]], position: int) -> int:
    """字符偏移 → 页码（取最后一个起始位置 ≤ position 的行）。"""
    if not offsets:
        return 0
    page = offsets[0][1]
    for start, page_no in offsets:
        if start <= position:
            page = page_no
        else:
            break
    return page


def _sentence_spans(text: str) -> list[tuple[int, str]]:
    spans: list[tuple[int, str]] = []
    start = 0
    for match in re.finditer(r"[。！？；!?;\n]", text):
        end = match.end()
        if end > start:
            spans.append((start, text[start:end]))
        start = end
    if start < len(text):
        spans.append((start, text[start:]))
    return spans


def _sliding_windows(
    text: str, size: int = CHILD_SIZE, overlap: int = CHILD_OVERLAP
) -> list[tuple[int, str]]:
    """按句切滑窗，返回 [(start_offset, window_text)]，窗口 ≤ size，相邻重叠约 overlap。

    超长单句（>size）退化为定长硬切，仍保留 overlap。
    """
    spans = _sentence_spans(text)
    if not spans:
        return []
    windows: list[tuple[int, str]] = []
    index = 0
    total = len(spans)

    while index < total:
        window_start = spans[index][0]
        current_len = 0
        cursor = index
        while cursor < total:
            sentence_len = len(spans[cursor][1])
            if current_len and current_len + sentence_len > size:
                break
            if not current_len and sentence_len > size:
                break  # 单句超长，走硬切分支
            current_len += sentence_len
            cursor += 1

        if cursor == index:
            # 单个超长句：定长硬切
            sentence = spans[index][1]
            sentence_start = spans[index][0]
            sentence_end = sentence_start + len(sentence)
            position = sentence_start
            while position < sentence_end:
                end = min(position + size, sentence_end)
                windows.append((position, text[position:end]))
                if end >= sentence_end:
                    break
                position = end - overlap
            index += 1
            continue

        end = spans[cursor - 1][0] + len(spans[cursor - 1][1])
        windows.append((window_start, text[window_start:end]))
        if cursor >= total:
            break

        # 回退若干整句作为下一窗口的重叠部分（总长 ≤ overlap）
        tail_index = cursor - 1
        tail_len = 0
        while tail_index > index and tail_len + len(spans[tail_index][1]) <= overlap:
            tail_len += len(spans[tail_index][1])
            tail_index -= 1
        next_index = tail_index + 1
        if next_index <= index:
            next_index = index + 1
        index = next_index

    return windows


def _make_children(
    clause: dict, offsets: list[tuple[int, int]], doc_id: str
) -> list[dict]:
    content = clause["content"]
    if len(content) <= CHILD_SIZE:
        segments = [(0, content)]
    else:
        segments = _sliding_windows(content)

    children: list[dict] = []
    for index, (start, raw_text) in enumerate(segments):
        leading = len(raw_text) - len(raw_text.lstrip())
        text = raw_text.strip()
        if not text:
            continue
        child_id = _sha1(f"{doc_id}|{clause['article_no']}|{index}")[:40]
        children.append(
            {
                "id": child_id,
                "doc_id": doc_id,
                "article_no": clause["article_no"],
                "title": clause["title"],
                "page": _page_at(offsets, start + leading),
                "text": text,
                "content_hash": _sha1(text)[:40],
            }
        )
    return children


# ---------------------------------------------------------------- 主入口

def split_clauses(
    parsed: ParsedDoc, doc_meta: dict
) -> tuple[list[dict], list[dict]]:
    """返回 (parent_clauses, child_blocks)。父块按文档顺序编号 order_no（从 1 开始）。"""
    doc_id = str(doc_meta.get("doc_id") or _sha1(parsed.title)[:40])

    builders: list[dict] = []          # 待渲染的条款构造器
    seen_articles: dict[str, int] = {}  # article_no -> builders 下标（重复条款号合并）
    current: dict | None = None         # 当前条款构造器
    front_lines: list[tuple[str, int]] = []      # 首个章节/条款前的文本
    chapter_pre_lines: list[tuple[str, int]] = []  # 章标题后、本章首条前的文本
    chapter = ""
    chapter_heading = ""
    chapter_page = 0
    chapter_index = 0

    def emit(builder: dict) -> None:
        """加入构造器；重复条款号合并到已存在的条款（保持引用的唯一性）。"""
        article_no = builder["article_no"]
        if article_no in seen_articles:
            existing = builders[seen_articles[article_no]]
            existing["lines"].extend(builder["lines"])
            logger.warning(
                "文档 %s 中条款号 %s 重复，已合并到首条", doc_id, article_no
            )
            return
        seen_articles[article_no] = len(builders)
        builders.append(builder)

    def emit_pseudo(lines: list[tuple[str, int]], article_no: str, chapter_text: str) -> None:
        if not lines:
            return
        emit(
            {
                "article_no": article_no,
                "chapter": chapter_text,
                "lines": lines,
            }
        )

    def flush_current() -> None:
        nonlocal current
        if current is not None and current["lines"]:
            emit(current)
        current = None

    def flush_front() -> None:
        nonlocal front_lines
        if front_lines:
            # 首行常是文档标题（与文件名 stem 相同），不单独做成"前言"条款，
            # 否则会产生一条只有标题的噪声父块。
            title_key = parsed.title.strip()
            meaningful = [
                (line, page_no)
                for line, page_no in front_lines
                if line.strip() != title_key
            ]
            if meaningful:
                emit_pseudo(meaningful, "前言", "")
            front_lines = []

    def flush_chapter_pre() -> None:
        nonlocal chapter_pre_lines
        if chapter_pre_lines:
            lines = ([(chapter_heading, chapter_page)] if chapter_heading else [])
            lines = lines + chapter_pre_lines
            emit_pseudo(lines, f"章{chapter_index}", chapter)
            chapter_pre_lines = []

    for line, page_no in _iter_lines(parsed):
        if _is_chapter_heading(line):
            flush_current()
            flush_chapter_pre()   # 上一章只有散文、没有条款时的兜底
            flush_front()
            chapter_heading = line
            chapter = line
            chapter_page = page_no
            chapter_index += 1
            continue

        article_match = _RE_ARTICLE.match(line)
        if article_match:
            flush_front()
            article_no = _normalize_article_no(article_match.group(1))
            lines: list[tuple[str, int]] = [(line, page_no)]
            # 章导语并入本章第一条，标题仍取条款首行
            if chapter_pre_lines:
                lines.extend(chapter_pre_lines)
                chapter_pre_lines = []
            flush_current()
            current = {
                "article_no": article_no,
                "chapter": chapter,
                "lines": lines,
            }
            continue

        # 普通正文：优先归入当前条款，其次章导语，最后文首前言
        if current is not None:
            current["lines"].append((line, page_no))
        elif chapter:
            chapter_pre_lines.append((line, page_no))
        else:
            front_lines.append((line, page_no))

    flush_current()
    flush_chapter_pre()
    flush_front()

    parent_clauses: list[dict] = []
    child_blocks: list[dict] = []
    for order_no, builder in enumerate(builders, start=1):
        content, offsets = _render(builder["lines"])
        if not content.strip():
            continue
        first_line = builder["lines"][0][0] if builder["lines"] else ""
        parent = {
            "doc_id": doc_id,
            "article_no": builder["article_no"],
            "title": (first_line or builder.get("chapter") or builder["article_no"])[:TITLE_MAX_LEN],
            "chapter": (builder.get("chapter") or "")[:TITLE_MAX_LEN],
            "content": content,
            "page": _page_at(offsets, 0),
            "order_no": order_no,
        }
        parent_clauses.append(parent)
        child_blocks.extend(_make_children(parent, offsets, doc_id))

    logger.info(
        "切分完成 doc=%s: 父块 %s, 子块 %s", doc_id, len(parent_clauses), len(child_blocks)
    )
    return parent_clauses, child_blocks
