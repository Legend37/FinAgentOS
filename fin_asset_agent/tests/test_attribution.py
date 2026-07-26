"""6-3 测试：历史方案时间线 + 归因复盘 + follow 标记。"""
import os
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from memory import db_models, state_manager
from data_ops import attribution as A


@pytest.fixture(autouse=True)
def _inmem_db(tmp_path):
    # 用文件型临时库（而非 :memory:），TestClient 工作线程也能共享同一份数据
    db_models.reset_engine(f"sqlite:///{tmp_path / 'attr.db'}")
    yield
    db_models.reset_engine()
    os.environ.pop("FINAGENT_DB_URL", None)


def _fake_prices(periods=6, start=10.0, end=11.0, t="601398.SS"):
    idx = pd.bdate_range(end=dt.date.today(), periods=periods)
    return pd.DataFrame({t: np.linspace(start, end, periods)}, index=idx)


# ---------- compute_attribution ----------

def test_extract_ticker():
    assert A.extract_ticker("工商银行 (601398.SS)") == "601398.SS"
    assert A.extract_ticker("BTC-USD") == "BTC-USD"
    assert A.extract_ticker("") == ""


def test_attribution_pending_when_fresh():
    r = A.compute_attribution(["工商银行 (601398.SS)"], [1.0],
                              advice_date=dt.date.today(), horizon_days=7)
    assert r["status"] == "pending"
    assert r["elapsed_days"] == 0


def test_attribution_unavailable_for_category_holdings():
    r = A.compute_attribution(["Cash_Equivalents", "Fixed_Income"], [0.5, 0.5],
                              advice_date=dt.date.today() - dt.timedelta(days=8), horizon_days=7)
    assert r["status"] == "unavailable"


def test_attribution_ok_with_mocked_prices(mocker):
    prices = pd.DataFrame({
        "601398.SS": np.linspace(10, 11, 6),     # +10%
        "518880.SS": np.linspace(5, 4.9, 6),      # -2%
    }, index=pd.bdate_range(end=dt.date.today(), periods=6))
    fetcher = mocker.Mock()
    fetcher.fetch_price_window.return_value = prices

    r = A.compute_attribution(
        ["工商银行 (601398.SS)", "黄金ETF (518880.SS)"], [0.6, 0.4],
        advice_date=dt.date.today() - dt.timedelta(days=8), horizon_days=7,
        fetcher=fetcher,
    )
    assert r["status"] == "ok"
    # 0.6*0.10 + 0.4*(-0.02) = 0.052
    assert r["realized_return"] == pytest.approx(0.052, abs=1e-3)
    assert r["best_contributor"] == "工商银行 (601398.SS)"
    assert r["worst_contributor"] == "黄金ETF (518880.SS)"


def test_attribution_unavailable_when_no_prices(mocker):
    fetcher = mocker.Mock()
    fetcher.fetch_price_window.return_value = pd.DataFrame()
    r = A.compute_attribution(["工商银行 (601398.SS)"], [1.0],
                              advice_date=dt.date.today() - dt.timedelta(days=8),
                              horizon_days=7, fetcher=fetcher)
    assert r["status"] == "unavailable"


# ---------- state_manager helpers ----------

def _make_snapshot():
    ident = state_manager.init_user_identity(profile={"name": "张三"})
    sid = state_manager.record_snapshot(
        ident["user_id"], "ASSET_ALLOCATION",
        tickers=["工商银行 (601398.SS)"], base_weights=[1.0], final_weights=[1.0],
        total_wealth=500000, timing_reason="t",
    )
    return ident, sid


def test_snapshot_has_advice_date_and_default_unfollowed():
    ident, sid = _make_snapshot()
    snaps = state_manager.list_user_snapshots(ident["user_id"])
    assert len(snaps) == 1
    assert snaps[0]["advice_date"] == dt.date.today().isoformat()
    assert snaps[0]["is_followed"] is None


def test_mark_snapshot_followed_toggle():
    ident, sid = _make_snapshot()
    assert state_manager.mark_snapshot_followed(sid, True) is True
    assert state_manager.get_snapshot(sid)["is_followed"] is True
    assert state_manager.mark_snapshot_followed(sid, False) is True
    assert state_manager.get_snapshot(sid)["is_followed"] is False
    assert state_manager.mark_snapshot_followed(999999, True) is False


def test_get_user_id_by_uuid():
    ident, _ = _make_snapshot()
    assert state_manager.get_user_id_by_uuid(ident["user_uuid"]) == ident["user_id"]
    assert state_manager.get_user_id_by_uuid("nope") is None


def test_get_or_compute_attribution_caches_ok(mocker):
    ident, sid = _make_snapshot()
    ok = {"status": "ok", "horizon_days": 7, "realized_return": 0.05,
          "realized_volatility": 0.1, "realized_sharpe": 0.3,
          "asset_contributions": {"工商银行 (601398.SS)": 0.05}}
    spy = mocker.patch("data_ops.attribution.compute_attribution", return_value=ok)

    r1 = state_manager.get_or_compute_attribution(sid, horizon_days=7)
    assert r1["status"] == "ok"
    # 第二次应命中缓存，不再调用 compute
    r2 = state_manager.get_or_compute_attribution(sid, horizon_days=7)
    assert r2["realized_return"] == 0.05
    assert spy.call_count == 1


def test_get_or_compute_attribution_does_not_cache_pending(mocker):
    ident, sid = _make_snapshot()
    pending = {"status": "pending", "horizon_days": 7, "reason": "fresh"}
    spy = mocker.patch("data_ops.attribution.compute_attribution", return_value=pending)
    state_manager.get_or_compute_attribution(sid, horizon_days=7)
    state_manager.get_or_compute_attribution(sid, horizon_days=7)
    # pending 不缓存 → 两次都会调用 compute
    assert spy.call_count == 2


# ---------- API 端点 ----------

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    import main_api
    return TestClient(main_api.app)


def test_api_list_snapshots(client):
    ident, sid = _make_snapshot()
    resp = client.get(f"/api/sessions/{ident['user_uuid']}/snapshots")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["snapshots"]) == 1
    assert body["snapshots"][0]["id"] == sid


def test_api_list_snapshots_unknown_user(client):
    resp = client.get("/api/sessions/not-a-real-uuid/snapshots")
    assert resp.status_code == 404


def test_api_follow_snapshot(client):
    ident, sid = _make_snapshot()
    resp = client.post(f"/api/sessions/{ident['user_uuid']}/snapshots/{sid}/follow",
                       json={"is_followed": True})
    assert resp.status_code == 200
    assert resp.json()["is_followed"] is True
    assert state_manager.get_snapshot(sid)["is_followed"] is True


def test_api_attribution_pending(client):
    ident, sid = _make_snapshot()  # 今日建议 → pending
    resp = client.get(f"/api/sessions/{ident['user_uuid']}/attribution/{sid}?horizon=7")
    assert resp.status_code == 200
    assert resp.json()["status"] == "pending"
