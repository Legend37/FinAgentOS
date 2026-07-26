"""测试 #5 风险沙箱扩展：Monte Carlo / VaR / 压力测试"""
import numpy as np
import pytest

from sandbox.backtest_engine import (
    monte_carlo_paths, value_at_risk, stress_test,
    comprehensive_risk_report, STRESS_SCENARIOS,
)


# ── Monte Carlo ──


def test_mc_basic_shape():
    mc = monte_carlo_paths(
        expected_returns=[0.10, 0.08, 0.15],
        cov_matrix=[[0.04, 0.01, 0.02], [0.01, 0.03, 0.01], [0.02, 0.01, 0.05]],
        weights=[0.4, 0.3, 0.3],
        horizon_days=252, n_paths=1000,
    )
    assert mc["terminal_values"].shape == (1000,)
    assert mc["terminal_returns"].shape == (1000,)
    assert mc["summary"]["p5"] < mc["summary"]["p50"] < mc["summary"]["p95"]


def test_mc_higher_variance_widens_distribution():
    """高方差资产 → 终值分布更宽"""
    low_vol = monte_carlo_paths(
        expected_returns=[0.05], cov_matrix=[[0.001]],
        weights=[1.0], horizon_days=252, n_paths=2000, seed=1,
    )
    high_vol = monte_carlo_paths(
        expected_returns=[0.05], cov_matrix=[[0.10]],
        weights=[1.0], horizon_days=252, n_paths=2000, seed=1,
    )
    assert high_vol["summary"]["std"] > low_vol["summary"]["std"] * 5


def test_mc_dimension_error():
    with pytest.raises(ValueError):
        monte_carlo_paths([0.1, 0.2], [[0.04]], [0.5, 0.5])


def test_mc_summary_includes_loss_prob():
    mc = monte_carlo_paths(
        expected_returns=[0.05], cov_matrix=[[0.04]],
        weights=[1.0], horizon_days=252, n_paths=2000,
    )
    assert 0.0 <= mc["summary"]["prob_loss"] <= 1.0


# ── VaR / CVaR ──


def test_var_basic():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.05, 0.20, 10_000)
    v = value_at_risk(returns, confidence=0.95, initial_capital=1_000_000)

    assert v["confidence"] == 0.95
    # 95% VaR 大致接近正态 5% 分位
    assert 0.20 < v["var_return"] < 0.40
    # CVaR 严格 ≥ VaR（条件期望更糟）
    assert v["cvar_return"] >= v["var_return"]
    # 金额 ≈ 比例 × 本金（容忍 round 引入的偏差）
    assert abs(v["var_amount"] - v["var_return"] * 1_000_000) < 1000


def test_var_99_stricter_than_95():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.05, 0.20, 10_000)
    v95 = value_at_risk(returns, confidence=0.95)
    v99 = value_at_risk(returns, confidence=0.99)
    assert v99["var_return"] > v95["var_return"]


def test_var_invalid_confidence():
    with pytest.raises(ValueError):
        value_at_risk(np.array([0.1, 0.2]), confidence=1.5)


def test_var_empty_returns():
    with pytest.raises(ValueError):
        value_at_risk(np.array([]))


# ── 压力测试 ──


def test_stress_all_scenarios():
    result = stress_test(
        weights=[0.5, 0.3, 0.2],
        asset_labels=["贵州茅台 (股票)", "国债ETF (固收)", "黄金 (商品)"],
        initial_capital=1_000_000,
    )
    # 4 个内置情景全应出现
    assert set(result["scenarios"].keys()) == set(STRESS_SCENARIOS.keys())
    # 在所有股灾情景里组合都应该亏损
    for key, data in result["scenarios"].items():
        if "crisis" in key or "crash" in key or "burst" in key:
            assert data["portfolio_return"] < 0


def test_stress_worst_scenario_identified():
    result = stress_test(
        weights=[1.0],
        asset_labels=["茅台 股票"],
        initial_capital=1_000_000,
    )
    assert result["worst_scenario"] is not None
    assert result["worst_return"] < 0


def test_stress_bond_heavy_less_loss():
    """债券为主组合在股灾情景下的损失应该明显小于全股票组合"""
    bond_heavy = stress_test(
        weights=[0.1, 0.9], asset_labels=["茅台 股票", "国债ETF"],
        initial_capital=1_000_000,
    )
    equity_heavy = stress_test(
        weights=[0.9, 0.1], asset_labels=["茅台 股票", "国债ETF"],
        initial_capital=1_000_000,
    )
    # 2008 情景下债券为主应该损失更小
    assert bond_heavy["scenarios"]["2008_financial_crisis"]["portfolio_return"] > \
           equity_heavy["scenarios"]["2008_financial_crisis"]["portfolio_return"]


def test_stress_dimension_mismatch():
    with pytest.raises(ValueError):
        stress_test([0.5, 0.5], ["only_one"])


def test_stress_explicit_scenarios():
    """只跑指定情景"""
    result = stress_test(
        weights=[1.0], asset_labels=["茅台"],
        scenarios=["2008_financial_crisis"],
        initial_capital=1_000_000,
    )
    assert list(result["scenarios"].keys()) == ["2008_financial_crisis"]


def test_stress_asset_contribution_breakdown():
    result = stress_test(
        weights=[0.5, 0.5],
        asset_labels=["A 股票", "国债"],
    )
    s2008 = result["scenarios"]["2008_financial_crisis"]
    assert len(s2008["asset_contributions"]) == 2
    # 第一个分类为 equity，shock 应该是 -0.42
    assert s2008["asset_contributions"][0]["class"] == "equity"
    assert s2008["asset_contributions"][0]["shock"] == pytest.approx(-0.42)


# ── 综合报告 ──


def test_comprehensive_report_combines_all():
    report = comprehensive_risk_report(
        expected_returns=[0.10, 0.05, 0.15],
        cov_matrix=[[0.04, 0.01, 0.02], [0.01, 0.01, 0.005], [0.02, 0.005, 0.06]],
        weights=[0.4, 0.4, 0.2],
        asset_labels=["茅台 (股票)", "国债ETF (固收)", "黄金 (商品)"],
        horizon_days=252, n_paths=500,
    )
    assert "monte_carlo" in report
    assert "var_cvar" in report
    assert "stress_test" in report
    assert report["monte_carlo"]["n_paths"] == 500
    assert report["var_cvar"]["confidence"] == 0.95
