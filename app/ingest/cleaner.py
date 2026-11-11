"""文本清洗：纯确定性规则，不调用 LLM。

处理顺序：全半角归一 → 页眉/页脚（跨页重复短行）→ 页码行 → 目录行 → 空白归一。
clean_pages() 是页码感知版本，供 pipeline 在保留页码的前提下清洗；
clean_text() 是契约要求的单文本入口，内部复用同一套规则。
"""
from __future__ import annotations

import re
from collections import Counter

# ---------------------------------------------------------------- 全半角归一
# 只转换"半角化不改变中文语义"的字符：
#   - 全角数字/字母/空格
#   - 全角 ASCII 标点（% # & @ * + - / < = > $ \ ^ _ | ~ ` . 等）
# 中文句读（，。、；：？！“”‘’（）《》【】…—·）一律保留。
_FULLWIDTH_TRANS = str.maketrans(
    {
        **{chr(0xFF10 + i): str(i) for i in range(10)},          # ０-９
        **{chr(0xFF21 + i): chr(ord("A") + i) for i in range(26)},  # Ａ-Ｚ
        **{chr(0xFF41 + i): chr(ord("a") + i) for i in range(26)},  # ａ-ｚ
        "　": " ",   # 全角空格
        "％": "%",
        "＃": "#",
        "＆": "&",
        "＠": "@",
        "＊": "*",
        "＋": "+",
        "－": "-",
        "／": "/",
        "＜": "<",
        "＝": "=",
        "＞": ">",
        "＄": "$",
        "＼": "\\",
        "＾": "^",
        "＿": "_",
        "｜": "|",
        "～": "~",
        "｀": "`",
        "．": ".",
    }
)

# 页码行： "第 3 页" / "3页" / "-- 3 --" / "—·— 3 —·—"
_PAGE_LINE_PATTERNS = [
    re.compile(r"^第?\s*\d{1,4}\s*页\s*$"),
    re.compile(r"^[-—·]{2,}\s*\d{1,4}\s*[-—·]{2,}$"),
    re.compile(r"^第\s*\d{1,4}\s*页\s*[，,/]?\s*共\s*\d{1,4}\s*页\s*$"),
]
# 目录行特征：点线/省略号引导 + 行尾页码
_TOC_LEADER_RE = re.compile(r"(…|\.{4,}|·{4,})")
_LINE_END_NUMBER_RE = re.compile(r"\d{1,4}\s*$")
# 目录标题（整行仅这两个词时删）
_TOC_TITLE_RE = re.compile(r"^(目\s*录|contents)$", re.IGNORECASE)
# 条款/章节标题即使跨页重复也不能当页眉删掉（防止正文标题被误伤）
_STRUCT_HEAD_RE = re.compile(r"^第\s*[一二三四五六七八九十百零〇\d]+\s*[条章节篇]")

_REPEAT_MIN_COUNT = 3   # 全文重复 ≥3 次
_REPEAT_MAX_LEN = 40    # 且长度 ≤40 字，才认定为页眉/页脚


def _normalize_line(raw: str) -> str:
    """单行归一：全半角 + 多空白折叠 + 去首尾空白。"""
    line = raw.translate(_FULLWIDTH_TRANS)
    line = re.sub(r"[ \t　]+", " ", line)
    return line.strip()


def _is_noise_line(line: str) -> bool:
    """页码行 / 目录标题 / 目录条目。"""
    if any(pattern.match(line) for pattern in _PAGE_LINE_PATTERNS):
        return True
    if _TOC_TITLE_RE.match(line):
        return True
    if _TOC_LEADER_RE.search(line) and _LINE_END_NUMBER_RE.search(line):
        return True
    return False


def clean_pages(pages: list[dict]) -> list[dict]:
    """页码感知清洗。输入/输出均为 [{page_no, text}]，页序与页码保持不变。

    页眉/页脚判定在"全文"范围统计：同一短行（去空白后）出现 ≥3 次即删除，
    因此清洗必须一次拿到全部页面，不能逐页独立调用。
    """
    normalized: list[tuple[int, list[str]]] = []
    counter: Counter[str] = Counter()

    for page in pages or []:
        page_no = int(page.get("page_no") or 1)
        lines: list[str] = []
        for raw in (page.get("text") or "").splitlines():
            line = _normalize_line(raw)
            lines.append(line)
            if line:
                counter[line] += 1
        normalized.append((page_no, lines))

    repeated = {
        line
        for line, count in counter.items()
        if count >= _REPEAT_MIN_COUNT
        and len(line) <= _REPEAT_MAX_LEN
        and not _STRUCT_HEAD_RE.match(line)
    }

    result: list[dict] = []
    for page_no, lines in normalized:
        kept = [
            line
            for line in lines
            if line and line not in repeated and not _is_noise_line(line)
        ]
        result.append({"page_no": page_no, "text": "\n".join(kept)})
    return result


def clean_text(text: str) -> str:
    """单文本清洗入口（契约函数）。规则与 clean_pages 完全一致。"""
    cleaned = clean_pages([{"page_no": 1, "text": text or ""}])
    return "\n".join(p["text"] for p in cleaned if p["text"])
