import pytest
from unittest.mock import patch
from memory import db_models

# LangGraph 图在 import 时就编译，必须在导入链上提前 mock
# 先 mock workflow 层的 run_fin_agent_pipeline，再导入 app
with patch(
    "core_brain.workflow.run_fin_agent_pipeline",
    return_value={
        "assets": ["贵州茅台 (酒类消费)", "工商银行 (大型金融)", "比亚迪 (新能源车)"],
        "base_weights": [0.2951, 0.5204, 0.1846],
        "final_weights": [0.30, 0.50, 0.20],
        "risk_status": "PASS",
        "timing_reason": "用户偏好稳健，微调降低比亚迪敞口",
        "risk_report": "审核通过，方案合规，权重分散合理。",
        "user_profile": {
            "name": "张伟",
            "age": 35,
            "occupation": "软件工程师",
            "risk_tolerance_level": "平衡型",
            "risk_score": 60,
            "investment_horizon": "中长期",
            "financial_goals": "资产稳健增值",
        },
        "asset_snapshot": {
            "total_wealth": 800000,
            "current_allocation": {
                "Cash_Equivalents": 200000,
                "Fixed_Income": 200000,
                "Equities": 300000,
                "Alternative_Assets": 100000,
            },
        },
    },
):
    from main_api import app


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    return TestClient(app)


@pytest.fixture(autouse=True)
def _test_db(tmp_path):
    db_models.reset_engine(f"sqlite:///{tmp_path / 'api.db'}")
    yield
    db_models.reset_engine()


class TestAllocateAPI:
    """POST /api/allocate 接口测试"""

    def test_health(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "service": "finagent-os"}

    def test_auth_init_returns_stable_identity(self, client):
        first = client.post("/api/auth/init", json={"name": "Alice"})
        assert first.status_code == 200
        body = first.json()
        assert body["user_uuid"]
        assert body["session_id"] == f"user-{body['user_uuid']}"

        second = client.post(
            "/api/auth/init",
            json={"user_uuid": body["user_uuid"], "name": "Alice 2"},
        )
        assert second.status_code == 200
        assert second.json()["user_id"] == body["user_id"]
        assert second.json()["user_uuid"] == body["user_uuid"]

    def test_missing_api_key_returns_400(self, client):
        resp = client.post("/api/allocate", json={"query": "测试", "api_key": ""})
        assert resp.status_code == 400
        assert "不能为空" in resp.json()["detail"]

    def test_valid_request_returns_200(self, client):
        resp = client.post(
            "/api/allocate",
            json={"query": "帮我配置稳健型组合", "api_key": "sk-test"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "assets" in body
        assert "base_weights" in body
        assert "final_weights" in body
        assert "risk_status" in body
        assert "timing_reason" in body
        assert "risk_report" in body
        assert "user_profile" in body
        assert "asset_snapshot" in body

    def test_chat_records_turns_and_snapshot(self, client):
        resp = client.post(
            "/api/chat",
            json={
                "api_key": "sk-test",
                "message": "帮我做一个稳健配置",
                "name": "Alice",
                "total_wealth": 800000,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["user_uuid"]
        assert body["session_id"] == f"user-{body['user_uuid']}"
        assert body["assistant_message"]
        assert body["snapshot_id"] is not None

        s = db_models.get_session()
        turns = s.query(db_models.ConversationTurn).order_by(db_models.ConversationTurn.id).all()
        snaps = s.query(db_models.PortfolioSnapshot).all()
        assert [t.role for t in turns] == ["user", "assistant"]
        assert len(snaps) == 1
        assert snaps[0].total_wealth == 800000
        s.close()

    def test_response_schema_types(self, client):
        resp = client.post(
            "/api/allocate",
            json={"query": "测试", "api_key": "sk-test"},
        )
        body = resp.json()

        assert isinstance(body["assets"], list)
        assert isinstance(body["base_weights"], list)
        assert isinstance(body["final_weights"], list)
        assert body["risk_status"] in ("PASS", "FAILED")
        assert isinstance(body["user_profile"], dict)
        assert "total_wealth" in body["asset_snapshot"]

    def test_workflow_exception_returns_500(self, client):
        """工作流内部异常 → 500"""
        import main_api as api_module
        with patch.object(
            api_module, "run_fin_agent_pipeline",
            side_effect=RuntimeError("计算节点崩溃"),
        ):
            resp = client.post(
                "/api/allocate",
                json={"query": "测试", "api_key": "sk-test"},
            )
        assert resp.status_code == 500
        assert "计算节点崩溃" in resp.json()["detail"]
