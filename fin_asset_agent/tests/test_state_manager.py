import os
import pytest

from memory import db_models, state_manager


@pytest.fixture(autouse=True)
def _inmem_db():
    db_models.reset_engine("sqlite:///:memory:")
    yield
    db_models.reset_engine()
    os.environ.pop("FINAGENT_DB_URL", None)


def test_get_or_create_user_upsert():
    profile = {"name": "Alice", "age": 30, "risk_tolerance_level": "平衡型"}
    uid1 = state_manager.get_or_create_user(profile)

    # 同名再调一次：应是 upsert，不重复创建
    profile["age"] = 31
    uid2 = state_manager.get_or_create_user(profile)
    assert uid1 == uid2

    s = db_models.get_session()
    users = s.query(db_models.User).filter_by(name="Alice").all()
    assert len(users) == 1
    assert users[0].age == 31
    s.close()


def test_init_user_identity_reuses_uuid():
    first = state_manager.init_user_identity(profile={"name": "Alice"})
    second = state_manager.init_user_identity(user_uuid=first["user_uuid"], profile={"name": "Alice 2"})

    assert first["user_id"] == second["user_id"]
    assert first["user_uuid"] == second["user_uuid"]
    assert second["session_id"] == f"user-{first['user_uuid']}"

    s = db_models.get_session()
    users = s.query(db_models.User).all()
    assert len(users) == 1
    assert users[0].name == "Alice 2"
    assert users[0].last_active_at is not None
    s.close()


def test_record_and_recall_turns():
    uid = state_manager.get_or_create_user({"name": "Bob"})
    state_manager.record_turn("sess-1", "user", "我有 50 万", user_id=uid, intent="ASSET_ALLOCATION")
    state_manager.record_turn("sess-1", "assistant", "推荐配置...", user_id=uid)
    state_manager.record_turn("sess-1", "user", "再保守一点", user_id=uid)

    turns = state_manager.recall_recent_turns("sess-1", limit=5)
    assert len(turns) == 3
    assert turns[0]["content"] == "我有 50 万"
    assert turns[-1]["content"] == "再保守一点"


def test_recall_recent_turns_limits():
    for i in range(15):
        state_manager.record_turn("sess-2", "user", f"msg{i}")
    turns = state_manager.recall_recent_turns("sess-2", limit=5)
    assert len(turns) == 5
    # 应该是最后 5 条且正序
    assert turns[0]["content"] == "msg10"
    assert turns[-1]["content"] == "msg14"


def test_snapshot_and_risk_link():
    uid = state_manager.get_or_create_user({"name": "Carol"})
    snap_id = state_manager.record_snapshot(
        user_id=uid,
        intent="ASSET_ALLOCATION",
        tickers=["600519.SS"],
        base_weights=[1.0],
        final_weights=[1.0],
        total_wealth=500000,
        timing_reason="单一资产测试",
    )
    risk_id = state_manager.record_risk_assessment(
        user_id=uid, snapshot_id=snap_id,
        risk_status="PASS", risk_score=60, risk_report="OK",
    )
    assert snap_id > 0 and risk_id > 0

    latest = state_manager.get_latest_snapshot(uid)
    assert latest["tickers"] == ["600519.SS"]
    assert latest["intent"] == "ASSET_ALLOCATION"


def test_turn_content_truncation():
    huge = "x" * 5000
    state_manager.record_turn("sess-3", "user", huge)
    turns = state_manager.recall_recent_turns("sess-3", limit=1)
    assert len(turns[0]["content"]) <= state_manager.MAX_TURN_CHARS


def test_purge_old_turns():
    for i in range(20):
        state_manager.record_turn("sess-4", "user", f"m{i}")
    deleted = state_manager.purge_old_turns("sess-4", keep_last=5)
    assert deleted == 15

    turns = state_manager.recall_recent_turns("sess-4", limit=100)
    assert len(turns) == 5
    assert turns[0]["content"] == "m15"
