# data_ops/rag_engine.py
"""非结构化知识库检索增强引擎（RAG）。

默认走纯 numpy 的 TF-IDF + 余弦相似度，零额外依赖；
当 sentence-transformers 可用时，可切换到稠密向量模式获得更好语义召回。

支持父子块检索 (Parent-Child Retriever)：
- 在子块上做匹配（更精准）
- 命中后返回对应父块的完整文本（更长上下文）
"""
from __future__ import annotations
import math
import os
import json
import pickle
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from data_ops.text_extract import extract_and_chunk


# ── 简单分词器：中英文混合，按字符 + 单词混合切 ──

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[一-鿿]")


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [tok.lower() for tok in _TOKEN_RE.findall(text)]


# ── TF-IDF 向量化器 ──


class TfidfVectorizer:
    """轻量 TF-IDF（基于 sublinear tf + IDF + L2 归一化）"""

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray = np.array([])
        self.fitted = False

    def fit_transform(self, docs: List[str]) -> np.ndarray:
        token_lists = [tokenize(d) for d in docs]
        vocab = {}
        df = Counter()
        for tokens in token_lists:
            for t in set(tokens):
                if t not in vocab:
                    vocab[t] = len(vocab)
                df[t] += 1
        self.vocab = vocab
        n_docs = max(len(docs), 1)
        self.idf = np.zeros(len(vocab))
        for t, idx in vocab.items():
            self.idf[idx] = math.log((1 + n_docs) / (1 + df[t])) + 1.0
        self.fitted = True
        return self._transform_token_lists(token_lists)

    def transform(self, docs: List[str]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("调用 transform 前必须先 fit_transform")
        return self._transform_token_lists([tokenize(d) for d in docs])

    def _transform_token_lists(self, token_lists: List[List[str]]) -> np.ndarray:
        n, dim = len(token_lists), len(self.vocab)
        mat = np.zeros((n, dim), dtype=np.float32)
        for i, tokens in enumerate(token_lists):
            tf = Counter(tokens)
            for tok, cnt in tf.items():
                idx = self.vocab.get(tok)
                if idx is None:
                    continue
                mat[i, idx] = (1.0 + math.log(cnt)) * self.idf[idx]
            norm = float(np.linalg.norm(mat[i]))
            if norm > 0:
                mat[i] /= norm
        return mat


def cosine_topk(query_vec: np.ndarray, mat: np.ndarray, k: int) -> List[Tuple[int, float]]:
    """对单个 query 向量与文档矩阵计算余弦相似度并取 top-k"""
    if mat.size == 0:
        return []
    sims = mat @ query_vec
    k = min(k, len(sims))
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return [(int(i), float(sims[i])) for i in idx]


# ── RAG 引擎主类 ──


class RAGEngine:
    """父子块检索 RAG 引擎，索引保存在内存，可序列化到磁盘"""

    def __init__(self):
        self.parents: List[Dict[str, Any]] = []   # [{id, text, source}]
        self.children: List[Dict[str, Any]] = []  # [{id, parent_id, text, source}]
        self.vectorizer = TfidfVectorizer()
        self.child_matrix: Optional[np.ndarray] = None

    # ---- 索引构建 ----

    def add_document(self, path: str, parent_chars: Optional[int] = None,
                     child_chars: Optional[int] = None):
        """解析单个文件并加入待索引集合（需调用 build_index 才能查询）"""
        kwargs = {}
        if parent_chars:
            kwargs["parent_chars"] = parent_chars
        if child_chars:
            kwargs["child_chars"] = child_chars
        doc = extract_and_chunk(path, **kwargs)
        for p in doc["parents"]:
            self.parents.append({
                "id": f"{path}::{p['id']}",
                "text": p["text"],
                "source": path,
            })
        parent_id_map = {p["id"]: f"{path}::{p['id']}" for p in doc["parents"]}
        for c in doc["children"]:
            self.children.append({
                "id": f"{path}::{c['id']}",
                "parent_id": parent_id_map[c["parent_id"]],
                "text": c["text"],
                "source": path,
            })

    def add_raw_text(self, text: str, source: str = "raw"):
        """直接索引一段已经准备好的文本（用于测试或动态注入）"""
        from data_ops.text_extract import chunk_parent_child
        doc = chunk_parent_child(text)
        for p in doc["parents"]:
            self.parents.append({
                "id": f"{source}::{p['id']}",
                "text": p["text"],
                "source": source,
            })
        parent_id_map = {p["id"]: f"{source}::{p['id']}" for p in doc["parents"]}
        for c in doc["children"]:
            self.children.append({
                "id": f"{source}::{c['id']}",
                "parent_id": parent_id_map[c["parent_id"]],
                "text": c["text"],
                "source": source,
            })

    def build_index(self):
        """对当前已加入的所有子块统一向量化"""
        texts = [c["text"] for c in self.children]
        if not texts:
            self.child_matrix = np.zeros((0, 0), dtype=np.float32)
            return
        self.child_matrix = self.vectorizer.fit_transform(texts)

    # ---- 查询 ----

    def search(self, query: str, top_k: int = 5,
               return_parents: bool = True) -> List[Dict[str, Any]]:
        """检索 query，返回 top_k 个父块（去重）+ 命中分数

        Args:
            return_parents: True 返回父块完整上下文；False 返回原始子块
        """
        if self.child_matrix is None or self.child_matrix.size == 0:
            return []
        q_vec = self.vectorizer.transform([query])[0]
        hits = cosine_topk(q_vec, self.child_matrix, k=top_k * 3 if return_parents else top_k)

        if not return_parents:
            return [
                {**self.children[i], "score": round(score, 4)}
                for i, score in hits[:top_k]
            ]

        # 按父块去重：同一父块取最高子块分
        parent_index = {p["id"]: p for p in self.parents}
        seen = {}
        for i, score in hits:
            pid = self.children[i]["parent_id"]
            if pid not in seen or score > seen[pid]["score"]:
                seen[pid] = {**parent_index[pid], "score": round(score, 4)}
            if len(seen) >= top_k:
                break

        return sorted(seen.values(), key=lambda x: -x["score"])[:top_k]

    # ---- 持久化 ----

    def save(self, path: str):
        data = {
            "parents": self.parents,
            "children": self.children,
            "vocab": self.vectorizer.vocab,
            "idf": self.vectorizer.idf.tolist() if self.vectorizer.fitted else [],
            "fitted": self.vectorizer.fitted,
            "child_matrix": self.child_matrix.tolist() if self.child_matrix is not None else None,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "RAGEngine":
        with open(path, "rb") as f:
            data = pickle.load(f)
        eng = cls()
        eng.parents = data["parents"]
        eng.children = data["children"]
        eng.vectorizer.vocab = data["vocab"]
        eng.vectorizer.idf = np.array(data["idf"])
        eng.vectorizer.fitted = data["fitted"]
        if data["child_matrix"] is not None:
            eng.child_matrix = np.array(data["child_matrix"], dtype=np.float32)
        return eng
