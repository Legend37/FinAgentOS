"""区间净值回测（backtest_review）测试：复用 attribution 的 ticker 抽取逻辑，
把方案权重套在「过去 window_days」真实行情上，跑固定权重买入持有回测。"""
import os
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from memory import db_models, state_manager
from data_ops import backtest_review as BR


@pytest.fixture(autouse=True)
def _inmem_db(tmp_path):
    db_models.reset_engine(f"sqlite:///{tmp_path / 'bt.db'}")
    yield
    db_models.reset_engine()
    os.environ.pop("FINAGENT_DB_URL", None)


# ---------- compute_backtest ----------

def test_backtest_unavailable_for_category_holdings():
    r = BR.compute_backtest(["Cash_Equivalents", "Fixed_Income"], [0.5, 0.5], window_days=30)
    assert r["status"] == "unavailable"


def test_backtest_unavailable_when_no_prices_strict(mocker):
    # allow_simulated=False：真实行情拿不到就老实 unavailable
    fetcher = mocker.Mock()
    fetcher.fetch_price_window.return_value = pd.DataFrame()
    r = BR.compute_backtest(["工商银行 (601398.SS)"], [1.0],
                            window_days=30, allow_simulated=False, fetcher=fetcher)
    assert r["status"] == "unavailable"


def test_backtest_simulated_fallback_when_no_prices(mocker):
    # 默认 allow_simulated=True：真实行情拿不到 → 离线模拟，status=ok 且 simulated=True
    fetcher = mocker.Mock()
    fetcher.fetch_price_window.return_value = pd.DataFrame()
    r = BR.compute_backtest(["工商银行 (601398.SS)", "黄金ETF (518880.SS)"], [0.6, 0.4],
                            window_days=30, fetcher=fetcher)
    assert r["status"] == "ok"
    assert r["simulated"] is True
    assert "无任何参考意义" in r["reason"]
    assert len(r["nav"]) >= 2
    assert r["trading_days"] >= 2


def test_backtest_simulated_is_deterministic(mocker):
    fetcher = mocker.Mock()
    fetcher.fetch_price_window.return_value = pd.DataFrame()
    kw = dict(window_days=30, as_of=dt.date(2025, 3, 1), fetcher=fetcher)
    r1 = BR.compute_backtest(["工商银行 (601398.SS)"], [1.0], **kw)
    r2 = BR.compute_backtest(["工商银行 (601398.SS)"], [1.0], **kw)
    assert r1["total_return"] == r2["total_return"]   # 同 ticker+窗口 → 可复现


def test_backtest_ok_with_mocked_prices(mocker):
    # 22 个交易日，两资产分别 +10% / -2%（买入持有）
    prices = pd.DataFrame({
        "601398.SS": np.linspace(10, 11, 22),     # +10%
        "518880.SS": np.linspace(5, 4.9, 22),      # -2%
    }, index=pd.bdate_range(end=dt.date.today(), periods=22))
    fetcher = mocker.Mock()
    fetcher.fetch_price_window.return_value = prices

    r = BR.compute_backtest(
        ["工商银行 (601398.SS)", "黄金ETF (518880.SS)"], [0.6, 0.4],
        window_days=30, rebalance="none", fetcher=fetcher,
    )
    assert r["status"] == "ok"
    assert r["simulated"] is False   # 有真实行情就不该走模拟
    # 买入持有：份额固定，终值收益 ≈ 0.6*10% + 0.4*(-2%) 的份额加权（非严格线性，量级吻合）
    assert 0.04 < r["total_return"] < 0.06
    assert r["trading_days"] == 22
    assert len(r["nav"]) == 22
    assert r["nav"][0]["value"] == pytest.approx(r["initial_capital"], rel=1e-6)


def test_backtest_drops_assets_without_prices(mocker):
    # 只返回其中一个标的的行情，另一个应被剔除
    prices = pd.DataFrame({
        "601398.SS": np.linspace(10, 11, 10),
    }, index=pd.bdate_range(end=dt.date.today(), periods=10))
    fetcher = mocker.Mock()
    fetcher.fetch_price_window.return_value = prices

    r = BR.compute_backtest(
        ["工商银行 (601398.SS)", "纳指ETF (513100.SS)"], [0.5, 0.5],
        window_days=30, fetcher=fetcher,
    )
    assert r["status"] == "ok"
    assert r["assets_used"] == ["工商银行 (601398.SS)"]
    assert r["dropped_assets"] == ["纳指ETF (513100.SS)"]


# ---------- state_manager 编排层 ----------

def test_get_snapshot_backtest_missing_snapshot():
    r = state_manager.get_snapshot_backtest(999999, window_days=30)
    assert r["status"] == "unavailable"
    assert "快照不存在" in r["reason"]


def test_get_snapshot_backtest_uses_total_wealth(mocker):
    ident = state_manager.init_user_identity(profile={"name": "张三"})
    sid = state_manager.record_snapshot(
        user_id=ident["user_id"], intent="ASSET_ALLOCATION",
        tickers=["工商银行 (601398.SS)"], base_weights=[1.0], final_weights=[1.0],
        total_wealth=500000.0,
    )
    prices = pd.DataFrame(
        {"601398.SS": np.linspace(10, 11, 22)},
        index=pd.bdate_range(end=dt.date.today(), periods=22),
    )
    mocker.patch(
        "data_ops.backtest_review.MarketDataFetcher",
        return_value=mocker.Mock(fetch_price_window=mocker.Mock(return_value=prices)),
    )
    r = state_manager.get_snapshot_backtest(sid, window_days=30)
    assert r["status"] == "ok"
    assert r["initial_capital"] == pytest.approx(500000.0)
