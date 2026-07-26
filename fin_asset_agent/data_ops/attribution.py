# data_ops/attribution.py
"""归因复盘：给定一份历史方案（资产名 + 权重 + 建议日），回查真实行情，
算出 horizon 天后的真实收益 / 波动 / Sharpe 与各资产贡献度。

设计要点：
- snapshot 里存的是展示名（如 "工商银行 (601398.SS)"），需从括号里抽出真实 ticker。
- 建议日尚未满 1 天 → status=pending（前端显示"等待数据积累"）。
- 类别级持仓（Cash_/Fixed_/Equities/Alternative_）无法回查 → status=unavailable。
- 区间行情拉取失败 → status=unavailable，绝不抛异常打断接口。
"""
import re
import datetime as dt
from typing import List, Dict, Optional

import numpy as np

from data_ops.market_data import MarketDataFetcher

TRADING_DAYS = 252
RISK_FREE = 0.02

# 类别级占位标的前缀（来自 PORTFOLIO_REVIEW 的 holdings）
_CATEGORY_PREFIXES = ("Cash_", "Fixed_", "Equities", "Alternative_")


def extract_ticker(name: str) -> str:
    """从展示名抽真实 ticker：'工商银行 (601398.SS)' -> '601398.SS'；无括号则原样返回。"""
    if not name:
        return ""
    m = re.search(r"\(([^)]+)\)\s*$", name.strip())
    return (m.group(1).strip() if m else name.strip())


def _is_real_ticker(t: str) -> bool:
    return bool(t) and not t.startswith(_CATEGORY_PREFIXES)


def _as_date(d) -> dt.date:
    if isinstance(d, dt.datetime):
        return d.date()
    if isinstance(d, dt.date):
        return d
    if isinstance(d, str):
        return dt.date.fromisoformat(d[:10])
    return dt.date.today()


def compute_attribution(asset_names: List[str], weights: List[float], advice_date,
                        horizon_days: int = 7, as_of: Optional[dt.date] = None,
                        fetcher: Optional[MarketDataFetcher] = None) -> Dict:
    """计算一份方案的真实表现归因。返回 dict，永不抛异常。

    status: ok | pending | unavailable
    """
    as_of = as_of or dt.date.today()
    advice_date = _as_date(advice_date)
    end_target = advice_date + dt.timedelta(days=horizon_days)
    end = min(end_target, as_of)
    elapsed = (end - advice_date).days

    base = {
        "horizon_days": horizon_days,
        "advice_date": advice_date.isoformat(),
        "as_of": end.isoformat(),
        "elapsed_days": max(elapsed, 0),
    }

    if elapsed < 1:
        return {**base, "status": "pending",
                "reason": "建议尚未满 1 天，暂无可计算的真实表现，请稍后回看。"}

    pairs = [
        (n, extract_ticker(n), float(w))
        for n, w in zip(asset_names, weights or [])
    ]
    real = [(n, t, w) for (n, t, w) in pairs if _is_real_ticker(t)]
    if not real:
        return {**base, "status": "unavailable",
                "reason": "持仓为类别级标的（现金/固收/权益等），无法回查个券真实价格。"}

    fetcher = fetcher or MarketDataFetcher()
    start_str = advice_date.isoformat()
    end_str = (end + dt.timedelta(days=1)).isoformat()  # 包含 end 当天
    try:
        prices = fetcher.fetch_price_window([t for (_, t, _) in real], start_str, end_str)
    except Exception as e:
        return {**base, "status": "unavailable", "reason": f"行情拉取异常：{e}"}

    if prices is None or prices.empty or len(prices) < 2:
        return {**base, "status": "unavailable", "reason": "区间内有效行情不足 2 个交易日。"}

    # 逐资产区间收益 + 贡献度
    contributions: Dict[str, float] = {}
    used = []  # (name, ticker, weight, ret)
    for (n, t, w) in real:
        if t not in prices.columns:
            continue
        col = prices[t].dropna()
        if len(col) < 2 or float(col.iloc[0]) == 0:
            continue
        r = float(col.iloc[-1] / col.iloc[0] - 1.0)
        used.append((n, t, w, r))

    if not used:
        return {**base, "status": "unavailable", "reason": "无有效价格序列。"}

    # 只对有数据的资产归一化权重
    w_sum = sum(w for (_, _, w, _) in used) or 1.0
    realized_return = sum((w / w_sum) * r for (_, _, w, r) in used)
    for (n, _, w, r) in used:
        contributions[n] = round((w / w_sum) * r, 6)

    # 组合日收益序列 → 年化波动 / Sharpe
    cols = [t for (_, t, _, _) in used]
    sub = prices[cols].dropna()
    wvec = np.array([w for (_, _, w, _) in used], dtype=float)
    wvec = wvec / (wvec.sum() or 1.0)
    daily = sub.pct_change().dropna()
    n_days = len(daily)
    if n_days >= 1:
        port_daily = daily.values @ wvec
        vol = float(np.std(port_daily) * np.sqrt(TRADING_DAYS)) if n_days > 1 else 0.0
        ann_ret = realized_return * (TRADING_DAYS / max(n_days, 1))
        sharpe = float((ann_ret - RISK_FREE) / vol) if vol > 1e-9 else 0.0
    else:
        vol, sharpe = 0.0, 0.0

    best = max(contributions.items(), key=lambda kv: kv[1], default=(None, 0.0))
    worst = min(contributions.items(), key=lambda kv: kv[1], default=(None, 0.0))

    return {
        **base,
        "status": "ok",
        "realized_return": round(realized_return, 6),
        "realized_volatility": round(vol, 6),
        "realized_sharpe": round(sharpe, 4),
        "asset_contributions": contributions,
        "best_contributor": best[0],
        "worst_contributor": worst[0],
        "trading_days": n_days,
    }
