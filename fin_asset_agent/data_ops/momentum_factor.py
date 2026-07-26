"""动量因子计算引擎：基于过去 1/3/6 月收益排名，生成动量得分。

动量是量化中被广泛验证有效的因子（Jegadeesh & Titman, 1993）：
过去表现好的资产，未来短期继续表现好的概率更高。

使用场景：Core-Satellite 架构中的 Satellite 选股。
"""
import datetime as dt
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from data_ops.market_data import MarketDataFetcher


def _parse_ticker_from_name(name: str) -> str:
    """从 '工商银行 (601398.SS)' 抽 ticker。"""
    import re
    m = re.search(r"\(([^)]+)\)\s*$", name.strip())
    return m.group(1).strip() if m else name.strip()


def compute_momentum_scores(
    asset_names: List[str],
    as_of: Optional[dt.date] = None,
    fetcher: Optional[object] = None,
) -> Tuple[Dict[str, float], Dict]:
    """计算各资产的综合动量得分。

    Returns:
        scores: {asset_name: score}，score 越高动量越强
        meta: 原始数据，用于展示
    """
    as_of = as_of or dt.date.today()
    fetcher = fetcher or MarketDataFetcher()

    tickers = [_parse_ticker_from_name(n) for n in asset_names]

    # 拉取过去 180 天价格（覆盖 6 个月）
    end = as_of
    start = end - dt.timedelta(days=200)
    start_str = start.isoformat()
    end_str = (end + dt.timedelta(days=1)).isoformat()

    prices = None
    try:
        prices = fetcher.fetch_price_window(tickers, start_str, end_str)
    except Exception:
        pass

    if prices is None or prices.empty:
        return {}, {"error": "无法获取价格数据", "as_of": as_of.isoformat()}

    scores = {}
    details = []

    for name, ticker in zip(asset_names, tickers):
        if ticker not in prices.columns:
            continue
        s = prices[ticker].dropna()
        if len(s) < 20:
            continue

        # 计算各窗口收益
        ret_1m = s.iloc[-1] / s.iloc[-min(22, len(s))] - 1.0 if len(s) >= 22 else 0
        ret_3m = s.iloc[-1] / s.iloc[-min(66, len(s))] - 1.0 if len(s) >= 66 else 0
        ret_6m = s.iloc[-1] / s.iloc[0] - 1.0

        # 综合动量得分：加权平均（6月40% + 3月35% + 1月25%）
        score = ret_6m * 0.4 + ret_3m * 0.35 + ret_1m * 0.25

        scores[name] = score
        details.append({
            "name": name,
            "ticker": ticker,
            "ret_1m": round(ret_1m, 4),
            "ret_3m": round(ret_3m, 4),
            "ret_6m": round(ret_6m, 4),
            "score": round(score, 4),
        })

    return scores, {"as_of": as_of.isoformat(), "details": details}


def select_momentum_satellites(
    asset_names: List[str],
    top_n: int = 3,
    max_weight_per_satellite: float = 0.15,
    as_of: Optional[dt.date] = None,
    fetcher: Optional[object] = None,
) -> Tuple[List[Dict], Dict]:
    """从资产池中选出动量最强的 N 个作为卫星持仓。

    Returns:
        satellites: [{name, ticker, weight, momentum_score}, ...]
        meta: 调试信息
    """
    scores, meta = compute_momentum_scores(asset_names, as_of, fetcher)
    if not scores:
        return [], meta

    # 按动量得分排序，取 top_n
    sorted_assets = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top = sorted_assets[:top_n]

    # 卫星总权重固定 30%，内部按动量得分比例分配
    total_score = sum(max(s, 0.01) for _, s in top)
    satellites = []
    for name, score in top:
        w = (max(score, 0.01) / total_score) * 0.30
        w = min(w, max_weight_per_satellite)
        satellites.append({
            "name": name,
            "ticker": _parse_ticker_from_name(name),
            "weight": round(w, 4),
            "momentum_score": round(score, 4),
        })

    # 归一化到 30%
    total = sum(s["weight"] for s in satellites)
    if total > 0:
        for s in satellites:
            s["weight"] = round(s["weight"] / total * 0.30, 4)

    meta["selected"] = satellites
    return satellites, meta
