# sandbox/backtest_engine.py
"""轻量历史回测引擎：基于固定权重 + 定期再平衡，纯 numpy/pandas 实现。

不引入 Backtrader 等重依赖，便于嵌入沙箱与单元测试。
输入：日收盘价矩阵（DataFrame）+ 权重向量；
输出：净值曲线、年化收益率、年化波动率、夏普比率、最大回撤。
"""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from config import mvo as mvo_config

TRADING_DAYS = 252


def _normalize_weights(weights: List[float]) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        raise ValueError("权重向量不能为空")
    total = w.sum()
    if total <= 0:
        raise ValueError(f"权重之和必须为正，当前 {total}")
    return w / total


def _rebalance_indices(n_days: int, freq: str) -> List[int]:
    """生成再平衡日的 index 列表（含第 0 日）"""
    if freq == "none":
        return [0]
    if freq == "daily":
        return list(range(n_days))
    step = {"weekly": 5, "monthly": 21, "quarterly": 63, "yearly": 252}.get(freq, 21)
    return list(range(0, n_days, step))


def run_backtest(
    prices: pd.DataFrame,
    weights: List[float],
    initial_capital: float = 1_000_000.0,
    rebalance: str = "monthly",
    risk_free_rate: Optional[float] = None,
) -> Dict:
    """对固定目标权重组合做历史回测。

    prices: index 为日期，columns 为 tickers，值为收盘价。
    weights: 与 prices.columns 一一对应。
    rebalance: none / daily / weekly / monthly / quarterly / yearly
    返回结构: metrics + nav 时间序列（list of {date, nav}）
    """
    if prices is None or prices.empty:
        raise ValueError("prices 不能为空")
    if len(weights) != prices.shape[1]:
        raise ValueError(f"权重维度 {len(weights)} 与资产数 {prices.shape[1]} 不一致")

    rf = risk_free_rate if risk_free_rate is not None else mvo_config.risk_free_rate
    w_target = _normalize_weights(weights)
    prices = prices.ffill().bfill().dropna(how="all")
    n_days, n_assets = prices.shape

    rebal_set = set(_rebalance_indices(n_days, rebalance))

    nav = np.zeros(n_days)
    nav[0] = initial_capital
    # 初始按目标权重买入：每个资产持有的份额 = (NAV * w) / price
    shares = (initial_capital * w_target) / prices.iloc[0].values

    for t in range(1, n_days):
        portfolio_value = float(np.dot(shares, prices.iloc[t].values))
        nav[t] = portfolio_value
        if t in rebal_set:
            shares = (portfolio_value * w_target) / prices.iloc[t].values

    nav_series = pd.Series(nav, index=prices.index)
    daily_returns = nav_series.pct_change().dropna()

    metrics = _compute_metrics(nav_series, daily_returns, rf)
    metrics["initial_capital"] = float(initial_capital)
    metrics["final_capital"] = float(nav_series.iloc[-1])
    metrics["n_days"] = int(n_days)
    metrics["rebalance"] = rebalance

    return {
        "metrics": metrics,
        "nav": [
            {"date": idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx),
             "value": round(float(v), 4)}
            for idx, v in nav_series.items()
        ],
    }


def _compute_metrics(nav: pd.Series, daily_returns: pd.Series, rf: float) -> Dict[str, float]:
    if len(nav) < 2 or daily_returns.empty:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
        }

    total_return = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    years = len(daily_returns) / TRADING_DAYS
    if years <= 0:
        ann_return = 0.0
    else:
        ann_return = float((nav.iloc[-1] / nav.iloc[0]) ** (1.0 / years) - 1.0)

    daily_std = float(daily_returns.std())
    ann_vol = daily_std * np.sqrt(TRADING_DAYS)
    sharpe = (ann_return - rf) / ann_vol if ann_vol > 1e-9 else 0.0

    # 最大回撤
    running_max = nav.cummax()
    drawdowns = nav / running_max - 1.0
    max_dd = float(drawdowns.min())

    calmar = ann_return / abs(max_dd) if max_dd < 0 else 0.0

    return {
        "total_return": round(total_return, 4),
        "annualized_return": round(ann_return, 4),
        "annualized_volatility": round(ann_vol, 4),
        "sharpe_ratio": round(float(sharpe), 4),
        "max_drawdown": round(max_dd, 4),
        "calmar_ratio": round(float(calmar), 4),
    }


def compare_strategies(
    prices: pd.DataFrame,
    strategies: Dict[str, List[float]],
    initial_capital: float = 1_000_000.0,
    rebalance: str = "monthly",
) -> Dict[str, Dict]:
    """同一行情下并列回测多套权重（如 base vs adjusted），便于审计报告对照"""
    out = {}
    for name, w in strategies.items():
        out[name] = run_backtest(prices, w, initial_capital, rebalance)
    return out


# =====================================================================
# Monte Carlo 前向风险模拟
# =====================================================================


def monte_carlo_paths(
    expected_returns: List[float],
    cov_matrix: List[List[float]],
    weights: List[float],
    horizon_days: int = 252,
    n_paths: int = 10_000,
    initial_capital: float = 1_000_000.0,
    seed: int = 42,
) -> Dict:
    """基于多元正态假设的蒙特卡洛前向模拟。

    与 run_backtest 的区别：回测看历史已发生路径；MC 看未来可能的路径分布。
    输出可用于计算 VaR / CVaR / 终值分布。

    Returns:
        {
            "terminal_values": np.ndarray (n_paths,),   # 各路径终值
            "terminal_returns": np.ndarray (n_paths,),  # 各路径累计收益率
            "n_paths": int,
            "horizon_days": int,
            "summary": {mean, std, p5, p50, p95}
        }
    """
    mu = np.asarray(expected_returns, dtype=float)
    sigma = np.asarray(cov_matrix, dtype=float)
    w = _normalize_weights(weights)
    n_assets = len(mu)
    if sigma.shape != (n_assets, n_assets):
        raise ValueError(f"协方差矩阵 {sigma.shape} 与资产数 {n_assets} 不一致")

    rng = np.random.default_rng(seed)
    daily_mu = mu / TRADING_DAYS
    daily_cov = sigma / TRADING_DAYS

    # 多元正态抽样：n_paths × horizon_days × n_assets
    samples = rng.multivariate_normal(daily_mu, daily_cov, size=(n_paths, horizon_days))
    # 组合每日收益：(n_paths, horizon_days)
    port_daily = samples @ w
    # 累计净值：每条路径终值
    cum_log_ret = np.sum(np.log1p(port_daily), axis=1)
    terminal_returns = np.expm1(cum_log_ret)
    terminal_values = initial_capital * (1.0 + terminal_returns)

    summary = {
        "mean": float(terminal_values.mean()),
        "std": float(terminal_values.std()),
        "p5": float(np.percentile(terminal_values, 5)),
        "p50": float(np.percentile(terminal_values, 50)),
        "p95": float(np.percentile(terminal_values, 95)),
        "mean_return": float(terminal_returns.mean()),
        "prob_loss": float((terminal_returns < 0).mean()),
    }

    return {
        "terminal_values": terminal_values,
        "terminal_returns": terminal_returns,
        "n_paths": n_paths,
        "horizon_days": horizon_days,
        "initial_capital": float(initial_capital),
        "summary": summary,
    }


# =====================================================================
# VaR / CVaR
# =====================================================================


def value_at_risk(
    returns: np.ndarray,
    confidence: float = 0.95,
    initial_capital: float = 1_000_000.0,
) -> Dict:
    """从收益率分布算 VaR + CVaR。

    Args:
        returns: 一维收益率数组（来自 monte_carlo_paths 的 terminal_returns 或历史日收益）
        confidence: 0.95 表示 95% VaR（5% 最差路径）

    Returns:
        {
            "confidence": 0.95,
            "var_return": 0.123 (亏损为正数表示),
            "var_amount": 123000.0 (人民币),
            "cvar_return": 0.18,
            "cvar_amount": 180000.0,
        }
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence 必须 ∈ (0, 1)，得到 {confidence}")
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        raise ValueError("returns 为空")

    alpha = 1 - confidence
    var_threshold = float(np.percentile(arr, alpha * 100))
    # VaR 通常表示为正数（亏损额）
    var_loss = -var_threshold
    # CVaR：超过 VaR 阈值的尾部期望
    tail = arr[arr <= var_threshold]
    cvar_loss = -float(tail.mean()) if tail.size > 0 else var_loss

    return {
        "confidence": confidence,
        "var_return": round(var_loss, 4),
        "var_amount": round(var_loss * initial_capital, 2),
        "cvar_return": round(cvar_loss, 4),
        "cvar_amount": round(cvar_loss * initial_capital, 2),
        "sample_size": int(arr.size),
    }


# =====================================================================
# 压力测试场景库
# =====================================================================


# 预设情景：(资产类别 → 单日 shock 收益)
# 这些数字基于历史事件的代表性 shock 估算，仅作教学演示
STRESS_SCENARIOS = {
    "2008_financial_crisis": {
        "label": "2008 金融危机（雷曼破产前后 3 个月权益类暴跌）",
        "shocks": {"equity": -0.42, "bond": -0.05, "commodity": -0.30, "cash": 0.0, "default": -0.35},
    },
    "2020_covid_crash": {
        "label": "2020 新冠疫情冲击（3 月份全球流动性危机）",
        "shocks": {"equity": -0.30, "bond": -0.03, "commodity": -0.50, "cash": 0.0, "default": -0.25},
    },
    "2022_rate_hike": {
        "label": "2022 美联储加息（成长股大幅回撤、债券负收益）",
        "shocks": {"equity": -0.20, "bond": -0.12, "commodity": 0.10, "cash": 0.02, "default": -0.15},
    },
    "china_2015_bubble_burst": {
        "label": "2015 A 股股灾（杠杆出清，3 周指数 -45%）",
        "shocks": {"equity": -0.45, "bond": 0.02, "commodity": -0.10, "cash": 0.0, "default": -0.40},
    },
}


def _classify_asset(ticker_or_name: str) -> str:
    """简单启发式分类资产到 equity/bond/commodity/cash"""
    s = (ticker_or_name or "").lower()
    if any(k in s for k in ["国债", "债", "bond", "511010", "fixed_income", "金融"]):
        return "bond"
    if any(k in s for k in ["货币", "cash", "511880", "现金", "cash_equivalents"]):
        return "cash"
    if any(k in s for k in ["黄金", "原油", "商品", "gold", "oil", "commodity", "alternative"]):
        return "commodity"
    if any(k in s for k in ["股", "etf", "茅", "银行", "电力", "比亚迪", "宁德", "芯片", "证券", "equity", ".ss", ".sz", "aapl", "msft"]):
        return "equity"
    return "default"


def stress_test(
    weights: List[float],
    asset_labels: List[str],
    scenarios: Optional[List[str]] = None,
    initial_capital: float = 1_000_000.0,
) -> Dict:
    """对当前权重组合跑全部（或指定）历史情景压力测试。

    Returns:
        {
            "scenarios": {
                "2008_financial_crisis": {
                    "label": "...",
                    "portfolio_return": -0.32,
                    "portfolio_pnl": -320000,
                    "asset_contributions": [...]
                },
                ...
            },
            "worst_scenario": "...",
            "worst_pnl": ...
        }
    """
    if len(weights) != len(asset_labels):
        raise ValueError(f"weights({len(weights)}) 与 asset_labels({len(asset_labels)}) 维度不一致")

    w = _normalize_weights(weights)
    classes = [_classify_asset(lbl) for lbl in asset_labels]

    if scenarios is None:
        scenarios = list(STRESS_SCENARIOS.keys())

    results = {}
    for s_key in scenarios:
        if s_key not in STRESS_SCENARIOS:
            continue
        scen = STRESS_SCENARIOS[s_key]
        asset_shocks = np.array([
            scen["shocks"].get(cls, scen["shocks"]["default"]) for cls in classes
        ])
        contribs = w * asset_shocks
        port_ret = float(contribs.sum())
        results[s_key] = {
            "label": scen["label"],
            "portfolio_return": round(port_ret, 4),
            "portfolio_pnl": round(port_ret * initial_capital, 2),
            "asset_contributions": [
                {"asset": lbl, "class": cls, "weight": round(float(wi), 4),
                 "shock": round(float(sh), 4), "contribution": round(float(c), 4)}
                for lbl, cls, wi, sh, c in zip(asset_labels, classes, w, asset_shocks, contribs)
            ],
        }

    if results:
        worst = min(results.items(), key=lambda kv: kv[1]["portfolio_return"])
        worst_key, worst_data = worst
    else:
        worst_key, worst_data = None, None

    return {
        "scenarios": results,
        "worst_scenario": worst_key,
        "worst_return": worst_data["portfolio_return"] if worst_data else None,
        "worst_pnl": worst_data["portfolio_pnl"] if worst_data else None,
    }


def comprehensive_risk_report(
    expected_returns: List[float],
    cov_matrix: List[List[float]],
    weights: List[float],
    asset_labels: List[str],
    horizon_days: int = 252,
    n_paths: int = 5_000,
    initial_capital: float = 1_000_000.0,
    confidence: float = 0.95,
) -> Dict:
    """一站式风险报告：MC 终值分布 + VaR/CVaR + 压力测试

    用于 T_t 微调和 R_t 风控审查的输入。
    """
    mc = monte_carlo_paths(
        expected_returns, cov_matrix, weights,
        horizon_days=horizon_days, n_paths=n_paths,
        initial_capital=initial_capital,
    )
    var = value_at_risk(mc["terminal_returns"], confidence=confidence, initial_capital=initial_capital)
    stress = stress_test(weights, asset_labels, initial_capital=initial_capital)

    return {
        "monte_carlo": {
            "horizon_days": mc["horizon_days"],
            "n_paths": mc["n_paths"],
            "summary": mc["summary"],
        },
        "var_cvar": var,
        "stress_test": stress,
    }
