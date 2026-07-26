import numpy as np
import pandas as pd
import pytest

from sandbox.backtest_engine import run_backtest, compare_strategies


def _make_prices(n_days=252, n_assets=3, seed=42):
    """生成确定性的合成价格序列：随机游走，年化漂移 8%/12%/15%"""
    rng = np.random.default_rng(seed)
    drifts = [0.08, 0.12, 0.15][:n_assets]
    sigmas = [0.15, 0.20, 0.25][:n_assets]
    daily_drifts = np.array(drifts) / 252
    daily_sigmas = np.array(sigmas) / np.sqrt(252)
    rets = rng.normal(daily_drifts, daily_sigmas, size=(n_days, n_assets))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    idx = pd.date_range("2024-01-01", periods=n_days, freq="B")
    return pd.DataFrame(prices, index=idx, columns=[f"A{i}" for i in range(n_assets)])


def test_backtest_basic_metrics():
    prices = _make_prices()
    result = run_backtest(prices, weights=[0.4, 0.3, 0.3])
    m = result["metrics"]

    assert m["initial_capital"] == 1_000_000.0
    assert m["n_days"] == len(prices)
    assert m["final_capital"] > 0
    # 各项指标都应该是有限数
    assert all(np.isfinite([m["annualized_return"], m["annualized_volatility"], m["sharpe_ratio"], m["max_drawdown"]]))
    # NAV 时间序列长度匹配
    assert len(result["nav"]) == len(prices)


def test_backtest_weights_normalized():
    """非归一化权重应被内部归一化，不报错"""
    prices = _make_prices()
    r1 = run_backtest(prices, weights=[1.0, 1.0, 1.0])
    r2 = run_backtest(prices, weights=[1/3, 1/3, 1/3])
    assert abs(r1["metrics"]["final_capital"] - r2["metrics"]["final_capital"]) < 1.0


def test_backtest_max_drawdown_non_positive():
    """最大回撤应该 ≤ 0"""
    prices = _make_prices()
    result = run_backtest(prices, weights=[0.5, 0.3, 0.2])
    assert result["metrics"]["max_drawdown"] <= 0.0


def test_backtest_constant_prices_zero_return():
    """常数行情 → 总收益为 0，波动率为 0"""
    idx = pd.date_range("2024-01-01", periods=100, freq="B")
    prices = pd.DataFrame(np.full((100, 2), 50.0), index=idx, columns=["A", "B"])
    result = run_backtest(prices, weights=[0.5, 0.5])

    assert abs(result["metrics"]["total_return"]) < 1e-6
    assert result["metrics"]["annualized_volatility"] < 1e-6
    assert result["metrics"]["max_drawdown"] >= -1e-6


def test_backtest_dimension_mismatch():
    prices = _make_prices(n_assets=3)
    with pytest.raises(ValueError):
        run_backtest(prices, weights=[0.5, 0.5])


def test_backtest_empty_prices():
    with pytest.raises(ValueError):
        run_backtest(pd.DataFrame(), weights=[1.0])


def test_compare_strategies():
    prices = _make_prices()
    out = compare_strategies(
        prices,
        {"base": [1/3, 1/3, 1/3], "concentrated": [0.7, 0.2, 0.1]},
    )
    assert set(out.keys()) == {"base", "concentrated"}
    assert "metrics" in out["base"]
    assert "metrics" in out["concentrated"]


def test_backtest_rebalance_freq_no_crash():
    prices = _make_prices()
    for freq in ["none", "daily", "weekly", "monthly", "quarterly", "yearly"]:
        r = run_backtest(prices, weights=[0.4, 0.3, 0.3], rebalance=freq)
        assert r["metrics"]["rebalance"] == freq
