# data_ops/backtest_review.py
"""历史区间净值回测：给定一份方案（资产名 + 权重），拉「过去 window_days」真实行情，
跑固定权重回测，输出净值曲线 + 年化收益 / 波动 / 夏普 / 最大回撤。

与 attribution 的区别：
- attribution 看「建议日之后」horizon 天已发生的真实表现（前瞻验证、可缓存）。
- backtest_review 看「截至今天的过去」window 天，把当前权重套在历史行情上，
  估算大致年化收益；窗口随日期滑动，故不缓存。

离线兜底：真实行情拉不到时（断网 / 数据源被墙 / 系统时钟在未来导致窗口无数据），
对正确的 [today-window, today] 业务日窗口生成确定性模拟价格序列，标 simulated=True。
前端必须明确标注「无参考意义」——这只是为了让功能在无行情环境下也能演示净值曲线形态。

复用 attribution 的 ticker 抽取 / 类别判断逻辑，永不抛异常（status=unavailable）。
"""
import datetime as dt
import hashlib
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

from data_ops.market_data import MarketDataFetcher
from data_ops.attribution import extract_ticker, _is_real_ticker, TRADING_DAYS, RISK_FREE
from sandbox.backtest_engine import run_backtest


def _seed_for(ticker: str) -> int:
    """从 ticker 派生稳定种子（不依赖 PYTHONHASHSEED），保证模拟序列跨进程可复现。"""
    return int(hashlib.sha1(ticker.encode("utf-8")).hexdigest()[:8], 16)


def _synthetic_prices(tickers: List[str], start: dt.date, end: dt.date,
                      min_days: int = 5) -> pd.DataFrame:
    """对 [start, end] 业务日窗口生成确定性模拟收盘价（几何随机游走，无参考意义）。"""
    idx = pd.bdate_range(start=start, end=end)
    if len(idx) < 2:
        idx = pd.bdate_range(end=end, periods=max(min_days, 2))
    data = {}
    for t in tickers:
        rng = np.random.default_rng(_seed_for(t))
        rets = rng.normal(0.0004, 0.012, size=len(idx))   # 温和漂移 + 波动
        data[t] = 100.0 * np.cumprod(1.0 + rets)
    return pd.DataFrame(data, index=idx)


def _run_on_prices(base: Dict, real: List, prices: pd.DataFrame,
                   rebalance: str, initial_capital: float,
                   simulated: bool) -> Dict:
    """在给定价格矩阵上跑固定权重回测，组装结果。real=[(name, ticker, weight)]。"""
    used = [(n, t, w) for (n, t, w) in real if t in prices.columns]
    if not used:
        return {**base, "status": "unavailable", "reason": "无有效价格序列。"}
    cols = [t for (_, t, _) in used]
    sub = prices[cols].dropna()
    if sub.empty or len(sub) < 2:
        return {**base, "status": "unavailable", "reason": "区间内有效行情不足 2 个交易日。"}

    w_used = [w for (_, _, w) in used]  # run_backtest 内部会归一化
    try:
        bt = run_backtest(sub, w_used, initial_capital=initial_capital, rebalance=rebalance)
    except Exception as e:
        return {**base, "status": "unavailable", "reason": f"回测计算异常：{e}"}

    m = bt["metrics"]
    dropped = [n for (n, t, _) in real if t not in prices.columns]
    # 展示用的窗口取「实际有行情的首/末交易日」（如今天未收盘，末日自然落在上一交易日）
    actual_window = {}
    try:
        idx0, idx1 = sub.index[0], sub.index[-1]
        actual_window = {
            "start_date": idx0.strftime("%Y-%m-%d") if hasattr(idx0, "strftime") else str(idx0),
            "end_date": idx1.strftime("%Y-%m-%d") if hasattr(idx1, "strftime") else str(idx1),
        }
    except Exception:
        pass
    return {
        **base,
        **actual_window,
        "status": "ok",
        "simulated": simulated,
        "assets_used": [n for (n, _, _) in used],
        "dropped_assets": dropped,
        "annualized_return": m["annualized_return"],
        "total_return": m["total_return"],
        "annualized_volatility": m["annualized_volatility"],
        "sharpe_ratio": m["sharpe_ratio"],
        "max_drawdown": m["max_drawdown"],
        "calmar_ratio": m["calmar_ratio"],
        "trading_days": m["n_days"],
        "initial_capital": m["initial_capital"],
        "final_capital": m["final_capital"],
        "nav": bt["nav"],
    }


# 风险偏好 → 经典终身组合基准（展示名, ticker, 权重）
# 参考：Harry Browne 永久组合 / Rick Ferri 核心四 / 等权分散
BENCHMARK_PORTFOLIOS = {
    "保守型": {
        "name": "永久投资组合（25/25/25/25）",
        "assets": [
            ("沪深300ETF", "510300.SS", 0.25),
            ("30年国债ETF", "511090.SS", 0.25),
            ("黄金ETF", "518880.SS", 0.25),
            ("国债ETF", "511010.SS", 0.25),
        ],
    },
    "稳健型": {
        "name": "核心四组合（40/20/20/20）",
        "assets": [
            ("沪深300ETF", "510300.SS", 0.40),
            ("纳斯达克100", "513100.SS", 0.20),
            ("国债ETF", "511010.SS", 0.20),
            ("黄金ETF", "518880.SS", 0.20),
        ],
    },
    "平衡型": {
        "name": "股债金三等分（34/33/33）",
        "assets": [
            ("沪深300ETF", "510300.SS", 0.34),
            ("国债ETF", "511010.SS", 0.33),
            ("黄金ETF", "518880.SS", 0.33),
        ],
    },
    "进取型": {
        "name": "全球权益（60/20/20）",
        "assets": [
            ("沪深300ETF", "510300.SS", 0.60),
            ("纳斯达克100", "513100.SS", 0.20),
            ("创业板ETF", "159915.SZ", 0.20),
        ],
    },
    "激进型": {
        "name": "100% 权益",
        "assets": [
            ("沪深300ETF", "510300.SS", 1.0),
        ],
    },
}


def _resolve_risk_level(risk_level: Optional[str]) -> str:
    """把各种 risk_tolerance_level 写法映射到标准键。"""
    if not risk_level:
        return "平衡型"
    rl = risk_level.strip().lower()
    mapping = {
        "保守": "保守型", "稳健": "稳健型", "平衡": "平衡型",
        "进取": "进取型", "激进": "激进型",
    }
    for key, val in mapping.items():
        if key in rl:
            return val
    return "平衡型"


def _fetch_benchmark(window_days: int, as_of: dt.date,
                     fetcher: MarketDataFetcher,
                     risk_level: Optional[str] = None) -> Optional[Dict]:
    """拉取与用户风险偏好匹配的终身组合基准在同一窗口的表现。"""
    level = _resolve_risk_level(risk_level)
    bench_def = BENCHMARK_PORTFOLIOS.get(level, BENCHMARK_PORTFOLIOS["平衡型"])
    assets = bench_def["assets"]  # [(name, ticker, weight), ...]

    start = as_of - dt.timedelta(days=window_days)
    start_str = start.isoformat()
    end_str = (as_of + dt.timedelta(days=1)).isoformat()

    # 拉取所有基准资产的价格
    tickers = [t for (_, t, _) in assets]
    prices = None
    try:
        prices = fetcher.fetch_price_window(tickers, start_str, end_str)
    except Exception:
        pass

    # fallback：对缺失的标的单独用 akshare 补
    missing = []
    if prices is not None and not prices.empty:
        missing = [t for t in tickers if t not in prices.columns]
    else:
        missing = tickers[:]

    for t in missing:
        try:
            import akshare as ak
            if t == "510300.SS":
                df = ak.stock_zh_index_daily(symbol="sh000300")
            elif t == "511010.SS":
                # 国债ETF 用 fund_etf_hist_em
                df = ak.fund_etf_hist_em(symbol="511010", period="daily",
                                          start_date=start.strftime("%Y%m%d"),
                                          end_date=(as_of + dt.timedelta(days=1)).strftime("%Y%m%d"))
            else:
                continue
            if df is None or getattr(df, "empty", True):
                continue
            if "date" in df.columns and "close" in df.columns:
                d = df[["date", "close"]].copy()
                d["date"] = pd.to_datetime(d["date"])
                d = d[(d["date"] >= pd.to_datetime(start)) &
                      (d["date"] <= pd.to_datetime(as_of + dt.timedelta(days=1)))]
                if not d.empty:
                    s = d.set_index("date")["close"].astype(float).sort_index()
                    if prices is None:
                        prices = pd.DataFrame({t: s})
                    else:
                        prices[t] = s
        except Exception:
            pass

    if prices is None or prices.empty or len(prices) < 2:
        return None

    # 计算加权收益
    valid = [(n, t, w) for (n, t, w) in assets if t in prices.columns]
    if not valid:
        return None

    w_sum = sum(w for (_, _, w) in valid) or 1.0
    sub = prices[[t for (_, t, _) in valid]].dropna()
    if sub.empty or len(sub) < 2:
        return None

    # 加权净值
    wvec = np.array([w / w_sum for (_, _, w) in valid], dtype=float)
    weighted = sub.values @ wvec
    n_days = len(weighted)

    total_ret = float(weighted[-1] / weighted[0] - 1.0)
    ann_ret = total_ret * (TRADING_DAYS / max(n_days, 1))
    daily = pd.Series(weighted).pct_change().dropna()
    vol = float(np.std(daily) * np.sqrt(TRADING_DAYS)) if len(daily) > 1 else 0.0
    sharpe = float((ann_ret - RISK_FREE) / vol) if vol > 1e-9 else 0.0

    nav_series = pd.Series(weighted)
    cummax = nav_series.cummax()
    dd = (nav_series - cummax) / cummax
    mdd = float(dd.min())

    return {
        "name": bench_def["name"],
        "risk_level": level,
        "assets": [{"name": n, "ticker": t, "weight": round(w, 2)} for (n, t, w) in valid],
        "total_return": total_ret,
        "annualized_return": ann_ret,
        "annualized_volatility": vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": mdd,
        "trading_days": n_days,
    }


def _attach_benchmark(res: Dict, window_days: int, as_of: dt.date,
                      fetcher: MarketDataFetcher,
                      risk_level: Optional[str] = None) -> None:
    """给回测结果附加基准对比与超额收益（原地修改 dict）。"""
    benchmark = _fetch_benchmark(window_days, as_of, fetcher, risk_level)
    if benchmark:
        res["benchmark"] = benchmark
        res["excess_return"] = res.get("total_return", 0) - benchmark["total_return"]
        res["excess_annualized_return"] = res.get("annualized_return", 0) - benchmark["annualized_return"]


def compute_backtest(asset_names: List[str], weights: List[float],
                     window_days: int = 30, rebalance: str = "none",
                     as_of: Optional[dt.date] = None,
                     initial_capital: float = 1_000_000.0,
                     allow_simulated: bool = True,
                     fetcher: Optional[MarketDataFetcher] = None,
                     risk_level: Optional[str] = None) -> Dict:
    """对一份方案做「过去 window_days」固定权重净值回测。返回 dict，永不抛异常。

    status: ok | unavailable；ok 时 simulated 标明是否为离线模拟数据。
    日期窗口：[as_of - window_days, as_of]，as_of 默认今天（如 today=6-07、window=7 → 5-31~6-07）。
    """
    as_of = as_of or dt.date.today()
    start = as_of - dt.timedelta(days=window_days)
    base = {
        "window_days": window_days,
        "rebalance": rebalance,
        "start_date": start.isoformat(),
        "end_date": as_of.isoformat(),
    }

    # 抽真实 ticker，过滤类别级占位标的（Cash_/Fixed_/Equities/Alternative_）
    pairs = [(n, extract_ticker(n), float(w)) for n, w in zip(asset_names, weights or [])]
    real = [(n, t, w) for (n, t, w) in pairs if _is_real_ticker(t)]
    if not real:
        return {**base, "status": "unavailable",
                "reason": "持仓为类别级标的（现金/固收/权益等），无法回查个券真实价格。"}

    # 1) 优先真实行情
    fetcher = fetcher or MarketDataFetcher()
    start_str = start.isoformat()
    end_str = (as_of + dt.timedelta(days=1)).isoformat()  # 包含 as_of 当天
    real_prices = None
    try:
        prices = fetcher.fetch_price_window([t for (_, t, _) in real], start_str, end_str)
        if prices is not None and not prices.empty and len(prices) >= 2:
            real_prices = prices
    except Exception as e:
        print(f"[Backtest] 真实行情拉取异常，转离线模拟: {e}")

    if real_prices is not None:
        res = _run_on_prices(base, real, real_prices, rebalance, initial_capital, simulated=False)
        if res.get("status") == "ok":
            _attach_benchmark(res, window_days, as_of, fetcher, risk_level)
            return res
        # 真实数据有但对齐后不足 → 若允许，落到模拟
        if not allow_simulated:
            return res

    # 2) 离线模拟兜底（无参考意义）
    if not allow_simulated:
        return {**base, "status": "unavailable", "reason": "区间内有效行情不足 2 个交易日。"}

    syn = _synthetic_prices([t for (_, t, _) in real], start, as_of)
    res = _run_on_prices(base, real, syn, rebalance, initial_capital, simulated=True)
    if res.get("status") == "ok":
        res["reason"] = "离线模拟数据（无法获取真实行情）：仅演示净值曲线形态，无任何参考意义。"
        _attach_benchmark(res, window_days, as_of, fetcher, risk_level)
    return res
