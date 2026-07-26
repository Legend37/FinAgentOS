# sandbox/visualizer.py
"""离线绘图工具：有效前沿面、配置饼图、净值曲线。

设计目的：为审计 PDF / 离线汇报生成静态 PNG 图。
依赖 matplotlib（懒加载，运行时缺包会给出明确报错）。
"""
from __future__ import annotations
import os
from typing import Dict, List, Optional, Tuple
import numpy as np


def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")  # 无头环境
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:
        raise ImportError("需要 matplotlib：pip install matplotlib") from e


def efficient_frontier(
    expected_returns: List[float],
    cov_matrix: List[List[float]],
    n_samples: int = 5000,
    risk_free_rate: float = 0.02,
    highlight_weights: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    seed: int = 42,
) -> str:
    """蒙特卡洛随机采样画有效前沿面（散点 + Sharpe 颜色 + 标记当前组合）。

    Returns:
        save_path（若提供）或临时文件路径
    """
    plt = _require_matplotlib()
    rng = np.random.default_rng(seed)
    mu = np.asarray(expected_returns, dtype=float)
    sigma = np.asarray(cov_matrix, dtype=float)
    n = len(mu)
    if n == 0 or sigma.shape != (n, n):
        raise ValueError("expected_returns 与 cov_matrix 维度不一致")

    weights = rng.dirichlet(np.ones(n), size=n_samples)
    rets = weights @ mu
    vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, sigma, weights))
    sharpes = (rets - risk_free_rate) / np.where(vols > 1e-9, vols, 1e-9)

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(vols, rets, c=sharpes, cmap="viridis", s=8, alpha=0.7)
    fig.colorbar(sc, label="Sharpe Ratio")
    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    ax.set_title("Efficient Frontier (Monte Carlo)")

    if highlight_weights is not None and len(highlight_weights) == n:
        w = np.asarray(highlight_weights, dtype=float)
        cur_ret = float(w @ mu)
        cur_vol = float(np.sqrt(w @ sigma @ w))
        ax.scatter([cur_vol], [cur_ret], color="red", marker="*", s=250,
                   edgecolors="black", linewidths=1.0, label="Current Portfolio", zorder=5)
        ax.legend(loc="lower right")

    return _save(fig, save_path, "efficient_frontier")


def allocation_pie(
    labels: List[str],
    weights: List[float],
    save_path: Optional[str] = None,
    title: str = "Portfolio Allocation",
) -> str:
    """配置饼图"""
    plt = _require_matplotlib()
    if len(labels) != len(weights):
        raise ValueError(f"labels({len(labels)}) 与 weights({len(weights)}) 维度不一致")
    weights = np.asarray(weights, dtype=float)
    if weights.sum() <= 0:
        raise ValueError("权重总和必须为正")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(weights, labels=labels, autopct="%1.1f%%", startangle=90,
           wedgeprops={"edgecolor": "white", "linewidth": 1})
    ax.set_title(title)
    ax.axis("equal")
    return _save(fig, save_path, "allocation_pie")


def nav_curve(
    nav_data: List[Dict],
    save_path: Optional[str] = None,
    title: str = "Portfolio NAV",
    benchmark: Optional[List[Dict]] = None,
) -> str:
    """净值曲线，支持叠加基准

    nav_data: [{date, value}, ...]，由 backtest_engine 输出。
    """
    plt = _require_matplotlib()
    if not nav_data:
        raise ValueError("nav_data 不能为空")
    dates = [d["date"] for d in nav_data]
    values = [d["value"] for d in nav_data]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, values, color="#0F172A", linewidth=1.5, label="Strategy")
    if benchmark:
        b_dates = [d["date"] for d in benchmark]
        b_values = [d["value"] for d in benchmark]
        ax.plot(b_dates, b_values, color="#6B7280", linewidth=1.0, linestyle="--", label="Benchmark")
        ax.legend(loc="upper left")

    # 横坐标稀疏化（避免日期太挤）
    step = max(1, len(dates) // 8)
    ax.set_xticks(range(0, len(dates), step))
    ax.set_xticklabels([dates[i] for i in range(0, len(dates), step)], rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel("NAV")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return _save(fig, save_path, "nav_curve")


def weights_comparison_bar(
    labels: List[str],
    base: List[float],
    adjusted: List[float],
    save_path: Optional[str] = None,
    title: str = "Base vs Adjusted Weights",
) -> str:
    """并列柱状图：base vs adjusted"""
    plt = _require_matplotlib()
    if not (len(labels) == len(base) == len(adjusted)):
        raise ValueError("labels / base / adjusted 维度必须一致")

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(labels)), 4))
    ax.bar(x - w / 2, base, w, label="Base (MVO)", color="#6B7280")
    ax.bar(x + w / 2, adjusted, w, label="Adjusted (LLM)", color="#0F172A")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Weight")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return _save(fig, save_path, "weights_bar")


def _save(fig, save_path: Optional[str], default_stem: str) -> str:
    plt = _require_matplotlib()
    if save_path is None:
        import tempfile
        fd, save_path = tempfile.mkstemp(prefix=f"{default_stem}_", suffix=".png")
        os.close(fd)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path
