"""测试 #4 multi-agent critic 闭环 + skill self-evolve 记忆"""
import os
import pytest

from core_brain.agents.critic import critic_node
from core_brain.workflow import route_after_critic, GraphState
from memory import db_models, state_manager


def make_state(**overrides):
    defaults = {
        "user_query": "测试",
        "api_key": "sk-test",
        "user_profile": {"risk_tolerance_level": "平衡型", "risk_score": 60},
        "asset_snapshot": {},
        "available_assets": ["茅台", "工行", "比亚迪"],
        "expected_returns": [0.12, 0.08, 0.15],
        "cov_matrix": [[0.04, 0.01, 0.02], [0.01, 0.03, 0.01], [0.02, 0.01, 0.05]],
        "base_weights": [0.33, 0.33, 0.34],
        "adjusted_weights": [0.45, 0.25, 0.30],
        "final_weights": [],
        "risk_status": "",
        "timing_reason": "",
        "risk_report": "",
        "risk_simulation": {
            "monte_carlo": {"summary": {"prob_loss": 0.25, "mean_return": 0.10}},
            "var_cvar": {"var_return": 0.18, "cvar_return": 0.25},
            "stress_test": {"worst_scenario": "2008_financial_crisis", "worst_return": -0.32},
        },
        "news_summary": "市场情绪稳定，无重大利空",
        "critic_retries": 0,
    }
    defaults.update(overrides)
    return defaults


# ── Critic 节点 ──


def test_critic_approve(mocker):
    mocker.patch(
        "core_brain.agents.critic.call_deepseek",
        return_value={
            "critic_status": "APPROVE",
            "critic_score": 85,
            "critic_feedback": "方案与风险等级匹配，舆情同向，通过。",
            "lesson": "平衡型用户在 VaR ≤ 18% 时可接受 +5% 单一资产调整",
        },
    )
    result = critic_node(make_state())
    assert result["critic_status"] == "APPROVE"
    assert result["critic_score"] == 85
    assert result["critic_retries"] == 0
    # 新增：返回 adjusted_risk_simulation（critic 自己跑的沙箱）
    assert "adjusted_risk_simulation" in result
    assert "monte_carlo" in result["adjusted_risk_simulation"]
    # 新增：返回 lesson
    assert "平衡型" in result["critic_lesson"]
    # 新增：APPROVE 时应把 lesson 沉淀到 SkillRecord
    assert result["critic_skill_id"] is not None


def test_critic_reruns_sandbox_on_adjusted_weights(mocker):
    """关键：Critic 应该用 adjusted_weights 重新跑沙箱，不是用 base 的过期数字"""
    captured = {}

    def fake_report(expected_returns, cov_matrix, weights, asset_labels, **kw):
        captured["weights"] = list(weights)
        return {
            "monte_carlo": {"summary": {"prob_loss": 0.15, "mean_return": 0.12}},
            "var_cvar": {"var_return": 0.14, "cvar_return": 0.19},
            "stress_test": {"worst_scenario": "2022_rate_hike", "worst_return": -0.20},
        }

    mocker.patch("core_brain.agents.critic.comprehensive_risk_report", side_effect=fake_report)
    mocker.patch(
        "core_brain.agents.critic.call_deepseek",
        return_value={"critic_status": "APPROVE", "critic_score": 80,
                      "critic_feedback": "ok", "lesson": "经验"},
    )

    state = make_state(
        base_weights=[0.33, 0.33, 0.34],
        adjusted_weights=[0.50, 0.30, 0.20],  # 与 base 不同
    )
    result = critic_node(state)

    # Critic 应该用 ADJUSTED 权重跑沙箱，不是 base
    assert captured["weights"] == [0.50, 0.30, 0.20]
    # 沙箱结果应回写到 state（供前端展示对比）
    assert result["adjusted_risk_simulation"]["var_cvar"]["var_return"] == 0.14


def test_critic_revise_does_not_persist_lesson(mocker):
    """REVISE 时不写 SkillRecord（教训应只从成功案例提炼）"""
    mocker.patch(
        "core_brain.agents.critic.call_deepseek",
        return_value={
            "critic_status": "REVISE", "critic_score": 40,
            "critic_feedback": "VaR 太高", "lesson": "（不应保存）",
        },
    )
    result = critic_node(make_state(critic_retries=0))
    assert result["critic_status"] == "REVISE"
    assert result["critic_skill_id"] is None  # REVISE 不持久化


def test_critic_revise_increments_retry(mocker):
    mocker.patch(
        "core_brain.agents.critic.call_deepseek",
        return_value={
            "critic_status": "REVISE",
            "critic_score": 45,
            "critic_feedback": "VaR 18% 超出保守型用户承受范围，建议减少茅台权重至 30% 以下。",
            "lesson": "",
        },
    )
    result = critic_node(make_state(critic_retries=0))
    assert result["critic_status"] == "REVISE"
    assert result["critic_retries"] == 1
    assert "VaR" in result["critic_feedback"]


def test_critic_forces_approve_after_max_retries(mocker):
    """重试 ≥ 2 次后即使 LLM 说 REVISE 也应强制 APPROVE，防死循环"""
    mocker.patch(
        "core_brain.agents.critic.call_deepseek",
        return_value={
            "critic_status": "REVISE",
            "critic_score": 30,
            "critic_feedback": "还是有问题",
        },
    )
    result = critic_node(make_state(critic_retries=2))
    assert result["critic_status"] == "APPROVE"  # 强制放行
    assert "[重试上限]" in result["critic_feedback"]


def test_critic_falls_back_on_llm_error(mocker):
    mocker.patch(
        "core_brain.agents.critic.call_deepseek",
        side_effect=Exception("timeout"),
    )
    result = critic_node(make_state())
    assert result["critic_status"] == "APPROVE"
    assert "降级" in result["critic_feedback"]


# ── Critic → 路由 ──


def test_route_after_critic_approve_to_risk():
    state = make_state(critic_status="APPROVE", critic_retries=0)
    assert route_after_critic(state) == "Risk_R_t"


def test_route_after_critic_revise_loops_back():
    state = make_state(critic_status="REVISE", critic_retries=1)
    assert route_after_critic(state) == "Timing_T_t"


def test_route_after_critic_revise_but_max_retries_proceeds():
    """REVISE 但重试已达上限 → 强制 Risk_R_t（安全阀）"""
    state = make_state(critic_status="REVISE", critic_retries=3)
    assert route_after_critic(state) == "Risk_R_t"


# ── SkillRecord 自我进化记忆 ──


@pytest.fixture(autouse=True)
def _inmem_db():
    db_models.reset_engine("sqlite:///:memory:")
    yield
    db_models.reset_engine()
    os.environ.pop("FINAGENT_DB_URL", None)


def test_profile_signature_stable():
    """同样的画像应产生相同签名"""
    p1 = {"risk_tolerance_level": "平衡型", "age": 32, "investment_horizon": "中长期"}
    p2 = {"risk_tolerance_level": "平衡型", "age": 35, "investment_horizon": "中长期"}  # 同年龄段
    p3 = {"risk_tolerance_level": "保守型", "age": 32, "investment_horizon": "中长期"}

    assert state_manager.make_profile_signature(p1) == state_manager.make_profile_signature(p2)
    assert state_manager.make_profile_signature(p1) != state_manager.make_profile_signature(p3)


def test_record_and_recall_skill():
    profile = {"risk_tolerance_level": "平衡型", "age": 32, "investment_horizon": "中长期"}
    sid = state_manager.record_skill(
        profile, "ASSET_ALLOCATION",
        critic_feedback="VaR 偏高，建议减仓",
        critic_score=72,
        revision_summary="茅台从 45% 降至 30%",
    )
    assert sid > 0

    skills = state_manager.recall_similar_skills(profile, limit=5)
    assert len(skills) == 1
    assert skills[0]["critic_score"] == 72
    assert "茅台" in skills[0]["revision_summary"]
    # reuse_count 应递增
    assert skills[0]["reuse_count"] == 1


def test_recall_orders_by_score():
    profile = {"risk_tolerance_level": "成长型", "age": 28}
    state_manager.record_skill(profile, "ASSET_ALLOCATION", "f1", 50, "r1")
    state_manager.record_skill(profile, "ASSET_ALLOCATION", "f2", 85, "r2")
    state_manager.record_skill(profile, "ASSET_ALLOCATION", "f3", 70, "r3")

    skills = state_manager.recall_similar_skills(profile, limit=10)
    scores = [s["critic_score"] for s in skills]
    assert scores == sorted(scores, reverse=True)


def test_recall_filters_by_profile():
    """不同画像不能误召"""
    p1 = {"risk_tolerance_level": "保守型", "age": 65}
    p2 = {"risk_tolerance_level": "进取型", "age": 25}
    state_manager.record_skill(p1, "ASSET_ALLOCATION", "f1", 80, "r1")
    state_manager.record_skill(p2, "ASSET_ALLOCATION", "f2", 90, "r2")

    skills_for_p1 = state_manager.recall_similar_skills(p1)
    assert len(skills_for_p1) == 1
    assert skills_for_p1[0]["revision_summary"] == "r1"


# ── T_t 召回历史教训 → 注入 prompt ──


def test_timing_node_recalls_past_lessons(mocker):
    """T_t 应该调用 recall_similar_skills 并把结果注入 prompt"""
    from core_brain.agents.portfolio_mgr import timing_adjustment_node

    # 先预先种 1 条历史教训
    profile = {"risk_tolerance_level": "平衡型", "age": 32, "investment_horizon": "中长期"}
    state_manager.record_skill(
        profile, "ASSET_ALLOCATION",
        critic_feedback="VaR 偏高", critic_score=85,
        revision_summary="平衡型用户单一资产权重不应超过 35%",
    )

    captured = {}
    def fake_llm(api_key, prompt, **kw):
        captured["prompt"] = prompt
        return {"adjusted_weights": [0.4, 0.3, 0.3], "timing_reason": "ok"}

    mocker.patch("core_brain.agents.portfolio_mgr.call_deepseek", side_effect=fake_llm)

    state = {
        "user_query": "测试", "api_key": "sk-test",
        "user_profile": profile, "asset_snapshot": {},
        "available_assets": ["A", "B", "C"],
        "expected_returns": [0.1, 0.08, 0.12],
        "cov_matrix": [[0.04, 0.01, 0.02], [0.01, 0.03, 0.01], [0.02, 0.01, 0.05]],
        "base_weights": [0.33, 0.33, 0.34],
        "adjusted_weights": [], "final_weights": [],
        "risk_status": "", "timing_reason": "", "risk_report": "",
        "risk_simulation": {}, "news_bundle": {}, "news_summary": "无新闻",
        "critic_status": "", "critic_score": 0, "critic_feedback": "",
        "critic_retries": 0,
    }
    timing_adjustment_node(state)

    # 历史教训应被注入 prompt
    assert "历史评分" in captured["prompt"]
    assert "35%" in captured["prompt"]
