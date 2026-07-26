"""6-3 测试：Telegram 主动推送（notifier / advice / review_job / 端点）。全程 mock，不触网。"""
import os
import datetime as dt

import pytest

from memory import db_models, state_manager
from data_ops import notifier, advice, review_job


@pytest.fixture(autouse=True)
def _db(tmp_path):
    db_models.reset_engine(f"sqlite:///{tmp_path / 'tg.db'}")
    yield
    db_models.reset_engine()
    os.environ.pop("FINAGENT_DB_URL", None)


def _mk_user_and_snapshot(advice_days_ago=8):
    ident = state_manager.init_user_identity(profile={"name": "张三"})
    sid = state_manager.record_snapshot(
        ident["user_id"], "ASSET_ALLOCATION",
        tickers=["工商银行 (601398.SS)"], base_weights=[1.0], final_weights=[1.0],
        total_wealth=500000, timing_reason="t",
    )
    # 把 advice_date 改到 N 天前，模拟"到期"
    s = db_models.get_session()
    try:
        snap = s.query(db_models.PortfolioSnapshot).filter_by(id=sid).one()
        snap.advice_date = dt.date.today() - dt.timedelta(days=advice_days_ago)
        s.commit()
    finally:
        s.close()
    return ident, sid


# ---------- notifier ----------

def _mock_session(mocker, *, post_resp=None, post_exc=None):
    """notifier 现在走 net_proxy.build_session()，mock 出一个带 .post/.get 的假 session。"""
    session = mocker.Mock()
    if post_exc is not None:
        session.post.side_effect = post_exc
    elif post_resp is not None:
        session.post.return_value = post_resp
    mocker.patch("data_ops.notifier.net_proxy.build_session", return_value=session)
    return session


def test_send_telegram_ok(mocker):
    resp = mocker.Mock()
    resp.json.return_value = {"ok": True, "result": {"message_id": 42}}
    _mock_session(mocker, post_resp=resp)
    r = notifier.send_telegram("hi", chat_id="123", token="t")
    assert r["ok"] is True and r["message_id"] == 42


def test_send_telegram_skips_without_chat(mocker):
    # 置空 config 默认 chat_id，才能测"无目标"跳过分支（否则会回退到 config.yaml 的默认）
    mocker.patch.object(notifier.tg_config, "default_chat_id", "")
    _mock_session(mocker, post_resp=mocker.Mock())
    r = notifier.send_telegram("hi", chat_id="", token="t")
    assert r["ok"] is False and r["skipped"] is True


def test_send_telegram_api_error(mocker):
    resp = mocker.Mock()
    resp.json.return_value = {"ok": False, "description": "chat not found"}
    _mock_session(mocker, post_resp=resp)
    r = notifier.send_telegram("hi", chat_id="123", token="t")
    assert r["ok"] is False and "chat not found" in r["error"]


def test_send_telegram_network_exception(mocker):
    _mock_session(mocker, post_exc=Exception("boom"))
    r = notifier.send_telegram("hi", chat_id="123", token="t")
    assert r["ok"] is False and "boom" in r["error"]


# ---------- advice ----------

def test_generate_advice_ok_gain():
    snap = {"id": 1, "intent": "ASSET_ALLOCATION", "advice_date": "2026-05-20"}
    attr = {"status": "ok", "realized_return": 0.08, "realized_volatility": 0.12,
            "realized_sharpe": 0.5, "best_contributor": "A", "worst_contributor": "B",
            "elapsed_days": 7}
    out = advice.generate_review_advice(snap, attr)
    assert "表现良好" in out["reason"]
    assert "+8.00%" in out["text"] and "止盈" in out["text"]


def test_generate_advice_ok_loss():
    snap = {"id": 2, "intent": "ASSET_ALLOCATION", "advice_date": "2026-05-20"}
    attr = {"status": "ok", "realized_return": -0.07, "realized_volatility": 0.2,
            "realized_sharpe": -0.3, "best_contributor": "A", "worst_contributor": "B",
            "elapsed_days": 7}
    out = advice.generate_review_advice(snap, attr)
    assert "回撤" in out["reason"] and "止损" in out["text"]


def test_generate_advice_pending():
    out = advice.generate_review_advice({"id": 3, "advice_date": "2026-06-03"},
                                        {"status": "pending"})
    assert "等待数据" in out["reason"]


# ---------- state_manager telegram + pending ----------

def test_set_get_telegram():
    ident, _ = _mk_user_and_snapshot()
    assert state_manager.set_user_telegram(ident["user_id"], "123456789") is True
    assert state_manager.get_user_telegram(ident["user_id"]) == "123456789"
    state_manager.set_user_telegram(ident["user_id"], "")
    assert state_manager.get_user_telegram(ident["user_id"]) is None


def test_pending_advice_crud():
    ident, sid = _mk_user_and_snapshot()
    pid = state_manager.record_pending_advice(ident["user_id"], sid, "复盘到期", "建议正文")
    pend = state_manager.list_pending_advice(ident["user_id"])
    assert len(pend) == 1 and pend[0]["id"] == pid
    assert state_manager.mark_pending_read(pid) is True
    assert state_manager.list_pending_advice(ident["user_id"], unread_only=True) == []


def test_snapshots_due_for_review_excludes_advised():
    ident, sid = _mk_user_and_snapshot(advice_days_ago=8)
    due = state_manager.snapshots_due_for_review(min_age_days=7)
    assert any(s["id"] == sid for s in due)
    # 生成 pending 后，不再算"到期未处理"
    state_manager.record_pending_advice(ident["user_id"], sid, "r", "t")
    due2 = state_manager.snapshots_due_for_review(min_age_days=7)
    assert all(s["id"] != sid for s in due2)


def test_snapshots_due_respects_age():
    _mk_user_and_snapshot(advice_days_ago=2)  # 才 2 天，未到 7 天
    assert state_manager.snapshots_due_for_review(min_age_days=7) == []


# ---------- review_job ----------

def test_weekly_review_job_pushes(mocker):
    ident, sid = _mk_user_and_snapshot(advice_days_ago=8)
    state_manager.set_user_telegram(ident["user_id"], "123")
    mocker.patch("data_ops.review_job.state_manager.get_or_compute_attribution",
                 return_value={"status": "ok", "realized_return": 0.03, "realized_volatility": 0.1,
                               "realized_sharpe": 0.2, "best_contributor": "A",
                               "worst_contributor": "B", "elapsed_days": 7, "horizon_days": 7})
    send = mocker.patch("data_ops.review_job.send_telegram", return_value={"ok": True, "message_id": 1})
    res = review_job.weekly_review_job(min_age_days=7)
    assert res["processed"] == 1
    assert res["results"][0]["notify_ok"] is True
    send.assert_called_once()
    # PendingAdvice 落库 + 推送状态 sent
    pend = state_manager.list_pending_advice(ident["user_id"])
    assert len(pend) == 1 and pend[0]["notify_status"] == "sent"


def test_push_snapshot_review_manual(mocker):
    ident, sid = _mk_user_and_snapshot(advice_days_ago=0)  # 今天的也能手动推
    state_manager.set_user_telegram(ident["user_id"], "123")
    mocker.patch("data_ops.review_job.send_telegram", return_value={"ok": True, "message_id": 9})
    r = review_job.push_snapshot_review(sid)
    assert r["ok"] is True and r["advice_id"] > 0


# ---------- API ----------

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    import main_api
    return TestClient(main_api.app)


def test_api_bind_telegram(client, mocker):
    ident, _ = _mk_user_and_snapshot()
    mocker.patch("main_api.send_telegram", return_value={"ok": True, "message_id": 1})
    r = client.post(f"/api/sessions/{ident['user_uuid']}/telegram",
                    json={"chat_id": "123456789", "send_test": True})
    assert r.status_code == 200
    assert r.json()["test"]["ok"] is True
    assert state_manager.get_user_telegram(ident["user_id"]) == "123456789"


def test_api_notify_snapshot(client, mocker):
    ident, sid = _mk_user_and_snapshot(advice_days_ago=0)
    state_manager.set_user_telegram(ident["user_id"], "123")
    mocker.patch("data_ops.review_job.send_telegram", return_value={"ok": True, "message_id": 7})
    r = client.post(f"/api/sessions/{ident['user_uuid']}/snapshots/{sid}/notify", json={"horizon": 7})
    assert r.status_code == 200 and r.json()["ok"] is True


def test_api_pending_and_ack(client):
    ident, sid = _mk_user_and_snapshot()
    pid = state_manager.record_pending_advice(ident["user_id"], sid, "r", "t")
    lst = client.get(f"/api/sessions/{ident['user_uuid']}/pending").json()["pending"]
    assert len(lst) == 1
    ack = client.post(f"/api/sessions/{ident['user_uuid']}/pending/{pid}/ack")
    assert ack.status_code == 200 and ack.json()["read"] is True


def test_api_cron_weekly_review(client, mocker):
    ident, sid = _mk_user_and_snapshot(advice_days_ago=8)
    state_manager.set_user_telegram(ident["user_id"], "123")
    mocker.patch("data_ops.review_job.send_telegram", return_value={"ok": True, "message_id": 1})
    r = client.post("/api/cron/weekly-review", json={"min_age_days": 7, "push": True})
    assert r.status_code == 200 and r.json()["processed"] >= 1
