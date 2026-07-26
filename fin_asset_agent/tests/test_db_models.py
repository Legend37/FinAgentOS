import os
import pytest

from memory import db_models
from memory.db_models import (
    User, PortfolioSnapshot, RiskAssessment, ConversationTurn,
    get_session, reset_engine,
)


@pytest.fixture(autouse=True)
def _inmem_db():
    """每个测试独立内存 SQLite，避免污染"""
    reset_engine("sqlite:///:memory:")
    yield
    reset_engine()
    os.environ.pop("FINAGENT_DB_URL", None)


def test_create_user_and_query():
    s = get_session()
    s.add(User(name="Alice", age=30, risk_tolerance_level="平衡型"))
    s.commit()

    u = s.query(User).filter_by(name="Alice").one()
    assert u.uuid
    assert u.age == 30
    assert u.risk_tolerance_level == "平衡型"
    s.close()


def test_snapshot_json_roundtrip():
    s = get_session()
    u = User(name="Bob")
    s.add(u)
    s.flush()

    snap = PortfolioSnapshot(
        user_id=u.id,
        intent="ASSET_ALLOCATION",
        total_wealth=500000.0,
    )
    snap.tickers = ["600519.SS", "601398.SS"]
    snap.base_weights = [0.6, 0.4]
    snap.final_weights = [0.55, 0.45]
    s.add(snap)
    s.commit()

    loaded = s.query(PortfolioSnapshot).filter_by(user_id=u.id).one()
    assert loaded.tickers == ["600519.SS", "601398.SS"]
    assert loaded.base_weights == [0.6, 0.4]
    assert loaded.final_weights == [0.55, 0.45]
    s.close()


def test_risk_assessment_link():
    s = get_session()
    u = User(name="Carol")
    s.add(u)
    s.flush()

    r = RiskAssessment(user_id=u.id, risk_status="PASS", risk_score=72, risk_report="OK")
    s.add(r)
    s.commit()

    fetched = s.query(RiskAssessment).filter_by(user_id=u.id).one()
    assert fetched.risk_status == "PASS"
    assert fetched.risk_score == 72
    s.close()


def test_conversation_turn_persistence():
    s = get_session()
    s.add(ConversationTurn(session_id="sess-1", role="user", content="你好"))
    s.add(ConversationTurn(session_id="sess-1", role="assistant", content="你好！"))
    s.commit()

    turns = s.query(ConversationTurn).filter_by(session_id="sess-1").order_by(ConversationTurn.id).all()
    assert [t.role for t in turns] == ["user", "assistant"]
    assert turns[1].content == "你好！"
    s.close()
