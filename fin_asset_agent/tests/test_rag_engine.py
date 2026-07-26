import os
import tempfile
import pytest

from data_ops.rag_engine import RAGEngine, TfidfVectorizer, tokenize, cosine_topk
import numpy as np


def test_tokenize_mixed_chinese_english():
    toks = tokenize("Apple 苹果 2024Q1 利润")
    assert "apple" in toks
    assert "苹" in toks and "果" in toks
    assert "2024q1" in toks
    assert "利" in toks


def test_tfidf_fit_transform_shapes():
    docs = ["股票投资风险", "债券投资稳健", "股票市场波动"]
    vec = TfidfVectorizer()
    mat = vec.fit_transform(docs)
    assert mat.shape[0] == 3
    assert mat.shape[1] == len(vec.vocab)
    # 每行 L2 范数为 1（或 0 表示空）
    for row in mat:
        norm = np.linalg.norm(row)
        assert abs(norm - 1.0) < 1e-4 or norm == 0


def test_cosine_topk():
    docs = ["股票投资风险", "债券投资稳健", "黄金避险"]
    vec = TfidfVectorizer()
    mat = vec.fit_transform(docs)
    q = vec.transform(["股票投资"])[0]
    hits = cosine_topk(q, mat, k=2)
    assert len(hits) == 2
    # 第一条命中应该是股票相关
    assert hits[0][0] == 0


def test_rag_search_empty_engine():
    eng = RAGEngine()
    assert eng.search("任意 query") == []


def test_rag_add_raw_text_and_search():
    eng = RAGEngine()
    eng.add_raw_text("股票投资风险较高，需要分散配置。\n\n债券投资相对稳健，适合保守型用户。", source="A")
    eng.add_raw_text("贵金属黄金常被视为避险资产。", source="B")
    eng.build_index()

    hits = eng.search("黄金避险", top_k=2)
    assert len(hits) >= 1
    # 命中应该包含黄金避险信息
    assert any("黄金" in h["text"] or "避险" in h["text"] for h in hits)
    assert all("score" in h for h in hits)


def test_rag_return_children_mode():
    eng = RAGEngine()
    eng.add_raw_text("段落 A 关于股票。\n\n段落 B 关于债券。", source="X")
    eng.build_index()
    hits = eng.search("股票", top_k=2, return_parents=False)
    assert len(hits) >= 1
    # 子块模式应有 parent_id
    assert "parent_id" in hits[0]


def test_rag_parent_dedup():
    """同一父块下多个子块命中时，结果应去重到父块"""
    eng = RAGEngine()
    # 一个长段落 → 1 个父块 + 多个子块
    long_text = "股票" * 200 + "\n\n" + "债券" * 200
    eng.add_raw_text(long_text, source="big")
    eng.build_index()
    hits = eng.search("股票", top_k=3, return_parents=True)
    parent_ids = [h["id"] for h in hits]
    assert len(parent_ids) == len(set(parent_ids))  # 父块全去重


def test_rag_persistence_roundtrip():
    eng = RAGEngine()
    eng.add_raw_text("货币基金 流动性强 收益稳健。", source="seed")
    eng.build_index()

    fd, path = tempfile.mkstemp(suffix=".pkl")
    os.close(fd)
    try:
        eng.save(path)
        eng2 = RAGEngine.load(path)
        hits = eng2.search("货币基金", top_k=1)
        assert len(hits) == 1
        assert "货币基金" in hits[0]["text"]
    finally:
        os.unlink(path)


def test_rag_add_file_end_to_end():
    fd, path = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("MVO 是马科维茨现代资产配置理论的核心算法。\n\n夏普比率衡量风险调整后的收益。")
    try:
        eng = RAGEngine()
        eng.add_document(path)
        eng.build_index()
        hits = eng.search("夏普比率", top_k=1)
        assert len(hits) == 1
        assert "夏普" in hits[0]["text"]
    finally:
        os.unlink(path)
