# data_ops/text_extract.py
"""文档解析与父子块切片，用于喂入 RAG 知识库。

支持格式：
- .pdf  → 按页解析（需 pypdf；缺包时给出明确报错）
- .xlsx / .xls → 按 sheet+row 解析（需 openpyxl/pandas）
- .csv → pandas
- .txt / .md → 原始文本

输出统一结构：
{
  "source": "<file path>",
  "parents": [{"id": "p0", "text": "..."}],     # 大块，长上下文
  "children": [{"id": "c0", "parent_id": "p0", "text": "..."}],  # 小块，精确检索
}
"""
from __future__ import annotations
import os
import re
from typing import Dict, List, Optional


PARENT_CHARS = 1500       # 父块目标字符数
CHILD_CHARS = 350         # 子块目标字符数
CHILD_OVERLAP = 50        # 子块重叠字符数


def extract_text(path: str) -> str:
    """根据扩展名分流到合适的解析器，返回纯文本拼接结果"""
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _extract_pdf(path)
    if ext in (".xlsx", ".xls"):
        return _extract_excel(path)
    if ext == ".csv":
        return _extract_csv(path)
    if ext in (".txt", ".md", ".markdown", ""):
        return _read_plain(path)
    # 兜底：当成纯文本读
    return _read_plain(path)


def _read_plain(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _extract_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # 老命名兼容
        except ImportError:
            raise ImportError("需要 pypdf 或 PyPDF2 才能解析 PDF：pip install pypdf")
    reader = PdfReader(path)
    parts = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            parts.append(f"\n[Page {i + 1}]\n{text}")
    return "\n".join(parts)


def _extract_excel(path: str) -> str:
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("需要 pandas + openpyxl：pip install pandas openpyxl")
    try:
        sheets = pd.read_excel(path, sheet_name=None)
    except ImportError:
        raise ImportError("缺少 Excel 引擎，请安装 openpyxl：pip install openpyxl")
    parts = []
    for name, df in sheets.items():
        parts.append(f"\n[Sheet: {name}]\n{df.to_csv(index=False)}")
    return "\n".join(parts)


def _extract_csv(path: str) -> str:
    import pandas as pd
    df = pd.read_csv(path)
    return df.to_csv(index=False)


def _split_into_blocks(text: str, target_chars: int, overlap: int = 0) -> List[str]:
    """按段落贪心拼接成接近 target_chars 的块，长段会被硬切。

    overlap > 0 时相邻块尾首重叠 overlap 字符，提升检索召回。
    """
    text = (text or "").strip()
    if not text:
        return []

    # 优先按双换行/换行切段
    raw_paragraphs = re.split(r"\n\s*\n|\n", text)
    paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

    blocks: List[str] = []
    buf = ""
    for para in paragraphs:
        if len(para) > target_chars:
            # 长段硬切
            if buf:
                blocks.append(buf)
                buf = ""
            for i in range(0, len(para), target_chars):
                blocks.append(para[i:i + target_chars])
            continue

        if not buf:
            buf = para
        elif len(buf) + 1 + len(para) <= target_chars:
            buf = f"{buf}\n{para}"
        else:
            blocks.append(buf)
            buf = para

    if buf:
        blocks.append(buf)

    if overlap > 0 and len(blocks) > 1:
        with_overlap = [blocks[0]]
        for i in range(1, len(blocks)):
            prev_tail = blocks[i - 1][-overlap:]
            with_overlap.append(prev_tail + blocks[i])
        blocks = with_overlap

    return blocks


def chunk_parent_child(text: str,
                       parent_chars: int = PARENT_CHARS,
                       child_chars: int = CHILD_CHARS,
                       child_overlap: int = CHILD_OVERLAP) -> Dict[str, List[Dict]]:
    """父子块切片：先切大块（parents），每个大块内部再切小块（children）。

    Returns:
        {"parents": [...], "children": [...]} 见模块 docstring。
    """
    parent_blocks = _split_into_blocks(text, parent_chars, overlap=0)
    parents: List[Dict] = []
    children: List[Dict] = []
    child_counter = 0

    for pi, p_text in enumerate(parent_blocks):
        pid = f"p{pi}"
        parents.append({"id": pid, "text": p_text})
        for c_text in _split_into_blocks(p_text, child_chars, overlap=child_overlap):
            children.append({
                "id": f"c{child_counter}",
                "parent_id": pid,
                "text": c_text,
            })
            child_counter += 1

    return {"parents": parents, "children": children}


def extract_and_chunk(path: str,
                      parent_chars: int = PARENT_CHARS,
                      child_chars: int = CHILD_CHARS) -> Dict:
    """一站式：解析 + 父子块切片，输出可直接喂给 RAG 引擎的结构"""
    text = extract_text(path)
    chunks = chunk_parent_child(text, parent_chars=parent_chars, child_chars=child_chars)
    return {
        "source": path,
        "parents": chunks["parents"],
        "children": chunks["children"],
    }
