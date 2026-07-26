import os
import tempfile
import pytest

from data_ops.text_extract import (
    extract_text, chunk_parent_child, extract_and_chunk,
    PARENT_CHARS, CHILD_CHARS,
)


def _write_tmp(content: str, suffix: str = ".txt") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def test_extract_text_plain():
    path = _write_tmp("段落一。\n\n段落二。", ".txt")
    try:
        assert "段落一" in extract_text(path)
    finally:
        os.unlink(path)


def test_extract_text_missing_file():
    with pytest.raises(FileNotFoundError):
        extract_text("/nonexistent/file.txt")


def test_chunk_parent_child_empty():
    result = chunk_parent_child("")
    assert result == {"parents": [], "children": []}


def test_chunk_parent_child_short():
    """短文本 → 一个父块、一个或多个子块"""
    result = chunk_parent_child("这是一段研报摘要。", parent_chars=100, child_chars=50)
    assert len(result["parents"]) == 1
    assert len(result["children"]) >= 1
    # 子块都应能映射回唯一父块
    assert result["children"][0]["parent_id"] == "p0"


def test_chunk_parent_child_long_text_splits():
    """长文本 → 多个父块，子块数量大于父块数量"""
    text = ("第一段内容\n\n" + "正文" * 200 + "\n\n第二段内容\n\n" + "正文" * 200)
    result = chunk_parent_child(text, parent_chars=200, child_chars=80, child_overlap=10)

    assert len(result["parents"]) >= 2
    assert len(result["children"]) > len(result["parents"])

    # 每个子块都映射到存在的父块
    parent_ids = {p["id"] for p in result["parents"]}
    for c in result["children"]:
        assert c["parent_id"] in parent_ids


def test_chunk_overlap_present():
    """开启 overlap 后相邻子块应有共同尾首"""
    text = "abcdefghij" * 100
    result = chunk_parent_child(text, parent_chars=500, child_chars=100, child_overlap=20)
    children = result["children"]
    if len(children) >= 2:
        assert children[1]["text"].startswith(children[0]["text"][-20:])


def test_extract_and_chunk_round_trip():
    path = _write_tmp("研报标题\n\n正文段落 A\n\n正文段落 B 含数据 12.34%", ".txt")
    try:
        out = extract_and_chunk(path, parent_chars=200, child_chars=80)
        assert out["source"] == path
        assert len(out["parents"]) >= 1
        assert any("12.34%" in c["text"] for c in out["children"])
    finally:
        os.unlink(path)


def test_extract_csv():
    path = _write_tmp("col_a,col_b\n1,2\n3,4\n", ".csv")
    try:
        text = extract_text(path)
        assert "col_a" in text
        assert "1,2" in text
    finally:
        os.unlink(path)


def test_pdf_missing_lib_or_extracts(monkeypatch):
    """没装 pypdf 时给出明确 ImportError"""
    # 直接测试 _extract_pdf 路径：模拟 pypdf 不存在
    import data_ops.text_extract as te
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("pypdf", "PyPDF2"):
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    path = _write_tmp("fake pdf bytes", ".pdf")
    try:
        with pytest.raises(ImportError):
            te.extract_text(path)
    finally:
        os.unlink(path)
