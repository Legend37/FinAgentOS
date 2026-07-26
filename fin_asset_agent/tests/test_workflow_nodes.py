import pytest
import json
from unittest.mock import patch, MagicMock

from core_brain.agents import (
    profile_generation_node,
    asset_screening_node,
    timing_adjustment_node,
    risk_compliance_node,
    portfolio_review_node,
    news_collection_node,
)
from core_brain.workflow import (
    sandbox_allocation_node, risk_simulation_node,
    route_by_intent, route_after_profile, general_response_node, GraphState,
)


def make_state(**overrides):
    """构建测试用的最小合法 state"""
    defaults = {
        "user_query": "测试诉求",
        "api_key": "sk-test",
        "user_profile": {},
        "asset_snapshot": {},
        "available_assets": [],
        "expected_returns": [],
        "cov_matrix": [],
        "base_weights": [],
        "adjusted_weights": [],
        "final_weights": [],
        "risk_status": "",
        "timing_reason": "",
        "risk_report": "",
    }
    defaults.update(overrides)
    return defaults


# ── 档案生成节点 (P_t) ──


def test_profile_node_uses_user_data():
    """用户画像来自 state 中的真实数据，不再调用 LLM"""
    state = make_state(user_profile={
        "age": 35,
        "total_wealth": 800000,
        "risk_tolerance_level": "成长型",
        "investment_horizon": "长期 (5年以上)",
        "financial_goals": "资产快速增值",
    })

    result = profile_generation_node(state)

    assert result["user_profile"]["age"] == 35
    assert result["user_profile"]["risk_tolerance_level"] == "成长型"
    assert result["user_profile"]["risk_score"] == 80  # 成长型 → 80
    assert result["asset_snapshot"]["total_wealth"] == 800000


def test_profile_node_derives_risk_score():
    """根据 risk_tolerance_level 自动推导 risk_score"""
    for level, expected_score in [
        ("保守型", 20),
        ("稳健型", 40),
        ("平衡型", 60),
        ("成长型", 80),
        ("进取型", 95),
    ]:
        state = make_state(user_profile={"risk_tolerance_level": level, "total_wealth": 500000})
        result = profile_generation_node(state)
        assert result["user_profile"]["risk_score"] == expected_score


def test_profile_node_defaults_on_empty_input():
    """空 state → 使用默认平衡型档案"""
    state = make_state(user_profile={})

    result = profile_generation_node(state)

    assert result["user_profile"]["name"] == "用户"
    assert result["user_profile"]["risk_tolerance_level"] == "平衡型"
    assert result["user_profile"]["risk_score"] == 60
    assert result["asset_snapshot"]["total_wealth"] == 500000


# ── 资产筛选节点 (S_t) ──


def test_asset_screening_falls_back_to_preset_pool(mocker):
    """无 custom + LLM 不可用 → 降级到风险等级预设池"""
    mock_returns = [0.12, 0.08, 0.15]
    mock_cov = [[0.04, 0.01, 0.02], [0.01, 0.03, 0.01], [0.02, 0.01, 0.05]]

    mocker.patch(
        "core_brain.agents.analyst.MarketDataFetcher",
        return_value=mocker.MagicMock(
            fetch_and_calculate=mocker.MagicMock(
                return_value=(mock_returns, mock_cov)
            )
        ),
    )
    # LLM 选品失败 → 降级
    mocker.patch(
        "core_brain.agents.analyst.llm_pick_assets",
        side_effect=Exception("LLM unavailable"),
    )

    result = asset_screening_node(make_state())

    # 平衡型预设池
    assert len(result["available_assets"]) == 3
    assert "贵州茅台" in result["available_assets"][0]
    assert "工商银行" in result["available_assets"][1]
    assert "比亚迪" in result["available_assets"][2] or "五粮液" in result["available_assets"][2]
    assert "降级" in result["selection_rationale"]


def test_asset_screening_uses_llm_picker(mocker):
    """LLM 可用 → 用 universe 选品"""
    mocker.patch(
        "core_brain.agents.analyst.MarketDataFetcher",
        return_value=mocker.MagicMock(
            fetch_and_calculate=mocker.MagicMock(
                return_value=([0.10]*5, [[0.04]*5]*5)
            )
        ),
    )
    mocker.patch(
        "core_brain.agents.analyst.llm_pick_assets",
        return_value={
            "tickers": ["518880.SS", "600048.SS", "BTC-USD", "GLD", "159995.SZ"],
            "names": {
                "518880.SS": "黄金ETF (518880.SS)",
                "600048.SS": "保利发展 (600048.SS)",
                "BTC-USD": "比特币 (BTC-USD)",
                "GLD": "SPDR 黄金 (GLD)",
                "159995.SZ": "芯片ETF (159995.SZ)",
            },
            "rationale": "用户偏好房地产+黄金+虚拟币，5 只覆盖 3 类",
        },
    )

    state = make_state(user_profile={
        "risk_tolerance_level": "成长型",
        "preferred_categories": ["reit", "commodity", "crypto"],
    }, user_query="想买些金子和币")

    result = asset_screening_node(state)

    assert len(result["available_assets"]) == 5
    assert "黄金" in result["available_assets"][0]
    assert "保利" in result["available_assets"][1]
    assert "比特币" in result["available_assets"][2]
    assert "用户偏好" in result["selection_rationale"]


def test_asset_screening_uses_custom_tickers(mocker):
    """传入 custom_tickers → 覆盖风险等级预设池"""
    captured = {}

    def fake_fetch(tickers):
        captured["tickers"] = list(tickers)
        return ([0.1] * len(tickers), [[0.04] * len(tickers)] * len(tickers))

    mocker.patch(
        "core_brain.agents.analyst.MarketDataFetcher",
        return_value=mocker.MagicMock(
            fetch_and_calculate=mocker.MagicMock(side_effect=fake_fetch)
        ),
    )

    state = make_state(user_profile={
        "risk_tolerance_level": "平衡型",
        "custom_tickers": ["AAPL", "MSFT", "600519.SS"],
    })
    result = asset_screening_node(state)

    assert captured["tickers"] == ["AAPL", "MSFT", "600519.SS"]
    assert len(result["available_assets"]) == 3
    # 茅台命中预设池名字
    assert "贵州茅台" in result["available_assets"][2]


def test_asset_screening_falls_back_when_custom_empty(mocker):
    """custom_tickers 为空列表 → 回退到风险等级预设池"""
    captured = {}

    def fake_fetch(tickers):
        captured["tickers"] = list(tickers)
        return ([0.1] * len(tickers), [[0.04] * len(tickers)] * len(tickers))

    mocker.patch(
        "core_brain.agents.analyst.MarketDataFetcher",
        return_value=mocker.MagicMock(
            fetch_and_calculate=mocker.MagicMock(side_effect=fake_fetch)
        ),
    )

    state = make_state(user_profile={
        "risk_tolerance_level": "保守型",
        "custom_tickers": [],
    })
    asset_screening_node(state)

    # 应使用保守型预设池
    assert "511010.SS" in captured["tickers"]


# ── 沙箱计算节点 (A_t) ──


def test_sandbox_node_computes_valid_weights():
    """正常收益/协方差 → 输出合法权重"""
    state = make_state(
        expected_returns=[0.12, 0.08, 0.15],
        cov_matrix=[
            [0.04, 0.01, 0.02],
            [0.01, 0.03, 0.01],
            [0.02, 0.01, 0.05],
        ],
        available_assets=["茅台 酒类消费", "工行 大型金融", "比亚迪 新能源车"],
    )
    result = sandbox_allocation_node(state)
    w = result["base_weights"]

    assert abs(sum(w) - 1.0) < 1e-4
    assert all(0.0 <= wi <= 1.0 for wi in w)
    assert len(w) == 3


# ── 权重微调节点 (T_t) ──


def test_timing_node_applies_adjustment(mocker):
    """LLM 返回调整后权重 → 写入 adjusted_weights 和 timing_reason"""
    mocker.patch(
        "core_brain.agents.portfolio_mgr.call_deepseek",
        return_value={
            "adjusted_weights": [0.35, 0.25, 0.40],
            "timing_reason": "用户偏好新能源，适当超配比亚迪",
        },
    )

    state = make_state(
        base_weights=[0.33, 0.33, 0.34],
        available_assets=["茅台", "工行", "比亚迪"],
    )
    result = timing_adjustment_node(state)

    assert result["adjusted_weights"] == [0.35, 0.25, 0.40]
    assert "比亚迪" in result["timing_reason"]


def test_timing_node_falls_back_on_llm_error(mocker):
    """LLM 故障 → 降级为沙箱原始权重"""
    mocker.patch("core_brain.agents.portfolio_mgr.call_deepseek", side_effect=Exception("超时"))

    state = make_state(base_weights=[0.33, 0.33, 0.34])
    result = timing_adjustment_node(state)

    assert result["adjusted_weights"] == [0.33, 0.33, 0.34]
    assert "失败" in result["timing_reason"]


# ── 风控合规节点 (R_t) ──


def test_risk_node_passes_valid_weights(mocker):
    """权重和为 1、无违规词 → PASS"""
    mocker.patch(
        "core_brain.agents.risk_officer.call_deepseek",
        return_value={
            "risk_status": "PASS",
            "risk_report": "审核通过，方案合规。",
        },
    )

    state = make_state(
        adjusted_weights=[0.35, 0.25, 0.40],
        available_assets=["茅台", "工行", "比亚迪"],
    )
    result = risk_compliance_node(state)

    assert result["risk_status"] == "PASS"
    assert result["final_weights"] == [0.35, 0.25, 0.40]


def test_risk_node_fails_when_sum_not_one(mocker):
    """权重和偏离 1.0 → 硬风控 FAILED（不依赖 LLM 判断）"""
    mocker.patch(
        "core_brain.agents.risk_officer.call_deepseek",
        return_value={
            "risk_status": "PASS",  # LLM 错误地给了 PASS
            "risk_report": "...",
        },
    )

    state = make_state(
        adjusted_weights=[0.5, 0.2, 0.1],  # 和 = 0.8
        available_assets=["A", "B", "C"],
    )
    result = risk_compliance_node(state)

    assert result["risk_status"] == "FAILED"
    assert result["final_weights"] == []  # FAILED 时清空


def test_risk_node_falls_back_on_llm_error(mocker):
    """LLM 故障 → 兜底 PASS + 保留原权重"""
    mocker.patch("core_brain.agents.risk_officer.call_deepseek", side_effect=Exception("超时"))

    state = make_state(adjusted_weights=[0.4, 0.3, 0.3])
    result = risk_compliance_node(state)

    assert result["risk_status"] == "PASS"
    assert result["final_weights"] == [0.4, 0.3, 0.3]


def test_risk_node_passes_on_edge_sum(mocker):
    """权重和 0.995（边界值）→ 仍然 PASS"""
    mocker.patch(
        "core_brain.agents.risk_officer.call_deepseek",
        return_value={
            "risk_status": "PASS",
            "risk_report": "审核通过。",
        },
    )

    state = make_state(
        adjusted_weights=[0.333, 0.333, 0.329],  # 和 = 0.995
        available_assets=["A", "B", "C"],
        user_profile={"risk_tolerance_level": "平衡型"},
    )
    result = risk_compliance_node(state)

    assert result["risk_status"] == "PASS"


def test_risk_node_hard_gate_single_asset_cap(mocker):
    """保守型用户单一资产 > 40% → 软上限裁切后 PASS（不再硬拒）"""
    spy = mocker.patch("core_brain.agents.risk_officer.call_deepseek",
                       return_value={"risk_status": "PASS", "risk_report": "OK"})

    state = make_state(
        adjusted_weights=[0.7, 0.2, 0.1],  # 0.7 超过保守型 0.4 上限
        available_assets=["A", "B", "C"],
        user_profile={"risk_tolerance_level": "保守型"},
    )
    result = risk_compliance_node(state)

    # 软上限裁切：0.7→0.4，多余份额按比例分给其余资产
    assert result["risk_status"] == "PASS"
    assert result["final_weights"] == pytest.approx([0.4, 0.4, 0.2])
    spy.assert_called()  # 硬门禁通过后才调 LLM


def test_risk_node_hard_gate_negative_weight(mocker):
    """出现负权重 → 硬门禁 FAILED"""
    spy = mocker.patch("core_brain.agents.risk_officer.call_deepseek")

    state = make_state(
        adjusted_weights=[0.6, -0.1, 0.5],
        available_assets=["A", "B", "C"],
        user_profile={"risk_tolerance_level": "平衡型"},
    )
    result = risk_compliance_node(state)

    assert result["risk_status"] == "FAILED"
    assert "负权重" in result["risk_report"]
    spy.assert_not_called()


def test_risk_node_hard_gate_progressive_caps(mocker):
    """进取型 90% 直接通过；保守型 90% 被软上限裁切到 40% 后仍然 PASS"""
    mocker.patch(
        "core_brain.agents.risk_officer.call_deepseek",
        return_value={"risk_status": "PASS", "risk_report": "OK"},
    )

    weights = [0.9, 0.05, 0.05]

    state_aggressive = make_state(
        adjusted_weights=weights,
        available_assets=["A", "B", "C"],
        user_profile={"risk_tolerance_level": "进取型"},
    )
    result_agg = risk_compliance_node(state_aggressive)
    assert result_agg["risk_status"] == "PASS"
    # 进取型上限 0.95，0.9 未超，权重不变
    assert result_agg["final_weights"] == pytest.approx([0.9, 0.05, 0.05])

    state_conservative = make_state(
        adjusted_weights=weights,
        available_assets=["A", "B", "C"],
        user_profile={"risk_tolerance_level": "保守型"},
    )
    result_cons = risk_compliance_node(state_conservative)
    # 保守型上限 0.4，0.9→0.4 裁切后仍 PASS（不再硬拒）
    assert result_cons["risk_status"] == "PASS"
    assert result_cons["final_weights"] == pytest.approx([0.4, 0.3, 0.3])


# ── 意图分支路由 ──


def test_route_chit_chat_to_general_response():
    """CHIT_CHAT 意图 → 兜底应答，绕过 MVO"""
    state = make_state(intent="CHIT_CHAT")
    assert route_by_intent(state) == ["general_response"]


def test_route_qa_rag_to_general_response():
    """QA_RAG 意图 → 兜底应答，绕过 MVO"""
    state = make_state(intent="QA_RAG")
    assert route_by_intent(state) == ["general_response"]


def test_route_asset_allocation_parallel_fanout():
    """ASSET_ALLOCATION → 并行 fan-out (profile + screen)"""
    state = make_state(intent="ASSET_ALLOCATION")
    result = route_by_intent(state)
    assert set(result) == {"profile", "screen"}


def test_route_portfolio_review_to_review_branch():
    """PORTFOLIO_REVIEW 意图 → 仅 profile 起始，后续走 Review"""
    state = make_state(intent="PORTFOLIO_REVIEW")
    assert route_by_intent(state) == ["profile"]


def test_general_response_node_sets_chat_status(mocker):
    """CHIT_CHAT → LLM 文本回复写入 risk_report/timing_reason，CHAT 状态"""
    mocker.patch(
        "core_brain.workflow.call_deepseek_text",
        return_value="你好！欢迎使用 FinAgent。",
    )
    state = make_state(intent="CHIT_CHAT", user_query="嗨")
    result = general_response_node(state)

    assert result["risk_status"] == "CHAT"
    assert result["final_weights"] == []
    assert "FinAgent" in result["risk_report"]


def test_general_response_node_falls_back_on_llm_error(mocker):
    """LLM 故障 → 静态兜底文本，CHAT 状态不变"""
    mocker.patch(
        "core_brain.workflow.call_deepseek_text",
        side_effect=Exception("API down"),
    )
    state = make_state(intent="CHIT_CHAT")
    result = general_response_node(state)

    assert result["risk_status"] == "CHAT"
    assert "FinAgent" in result["risk_report"]


def test_general_response_node_bypasses_mvo_chain(mocker):
    """兜底应答返回后不包含 assets/returns/cov — 证明未进入 S_t/A_t"""
    mocker.patch("core_brain.workflow.call_deepseek_text", return_value="问答结果")
    state = make_state(intent="QA_RAG")
    result = general_response_node(state)

    # general_response_node 不输出 MVO 管线的中间字段
    assert "available_assets" not in result
    assert "expected_returns" not in result
    assert "cov_matrix" not in result
    assert "base_weights" not in result


# ── PORTFOLIO_REVIEW 分支 ──


def test_route_after_profile_review_branch():
    """P_t 之后：PORTFOLIO_REVIEW → Review 节点"""
    state = make_state(intent="PORTFOLIO_REVIEW")
    assert route_after_profile(state) == "Review"


def test_route_after_profile_allocation_branch():
    """P_t 之后：ASSET_ALLOCATION → 直接 Sandbox_A_t（与并行 S_t 汇合）"""
    state = make_state(intent="ASSET_ALLOCATION")
    assert route_after_profile(state) == "Sandbox_A_t"


def test_review_node_uses_holdings(mocker):
    """传入显式 holdings → 直接使用，不跑 S_t 资产池"""
    mocker.patch(
        "core_brain.agents.reviewer.MarketDataFetcher",
        return_value=mocker.MagicMock(
            fetch_and_calculate=mocker.MagicMock(
                return_value=([0.1, 0.08], [[0.04, 0.01], [0.01, 0.03]])
            )
        ),
    )
    state = make_state(
        intent="PORTFOLIO_REVIEW",
        asset_snapshot={
            "total_wealth": 500000,
            "holdings": [
                {"ticker": "600519.SS", "name": "茅台", "weight": 0.6},
                {"ticker": "601398.SS", "name": "工行", "weight": 0.4},
            ],
        },
    )
    result = portfolio_review_node(state)

    assert result["available_assets"] == ["茅台", "工行"]
    assert result["base_weights"] == [0.6, 0.4]
    assert len(result["expected_returns"]) == 2


def test_review_node_falls_back_to_allocation_buckets():
    """无 holdings → 从 current_allocation 推导，权重和归一化"""
    state = make_state(
        intent="PORTFOLIO_REVIEW",
        asset_snapshot={
            "total_wealth": 500000,
            "current_allocation": {
                "Cash_Equivalents": 300000,
                "Equities": 200000,
            },
        },
    )
    result = portfolio_review_node(state)

    assert result["base_weights"] == [0.6, 0.4]
    assert result["expected_returns"] == []  # 桶式描述无法拉行情


def test_review_node_empty_snapshot():
    """空 snapshot → 安全返回空字段，不抛"""
    state = make_state(intent="PORTFOLIO_REVIEW", asset_snapshot={})
    result = portfolio_review_node(state)

    assert result["available_assets"] == []
    assert result["base_weights"] == []


# ── 风险模拟节点 (RiskSim) ──


def test_risk_sim_node_skips_when_empty():
    """缺少 base_weights → 返回空 dict，不报错"""
    state = make_state(base_weights=[], expected_returns=[], cov_matrix=[])
    result = risk_simulation_node(state)
    assert result == {"risk_simulation": {}}


def test_risk_sim_node_produces_full_report():
    """正常输入 → 返回 MC + VaR + 压力测试三段"""
    state = make_state(
        base_weights=[0.5, 0.5],
        expected_returns=[0.10, 0.08],
        cov_matrix=[[0.04, 0.01], [0.01, 0.03]],
        available_assets=["茅台 股票", "国债"],
    )
    result = risk_simulation_node(state)

    assert "risk_simulation" in result
    sim = result["risk_simulation"]
    assert "monte_carlo" in sim
    assert "var_cvar" in sim
    assert "stress_test" in sim
    assert sim["stress_test"]["worst_scenario"] is not None


# ── 新闻节点 (N_t) ──


def test_news_node_uses_custom_tickers(mocker):
    mocker.patch(
        "core_brain.agents.news_agent.news_fetcher.fetch_news_bundle",
        return_value={
            "by_ticker": {"600519.SS": [{"title": "茅台业绩超预期", "date": "2026-05-28"}]},
            "market_headlines": [],
            "financial_news": [],
            "total_count": 1,
        },
    )
    mocker.patch(
        "core_brain.agents.news_agent.news_fetcher.summarize_for_prompt",
        return_value="[600519.SS | 2026-05-28] 茅台业绩超预期",
    )

    state = make_state(user_profile={"custom_tickers": ["600519.SS"]})
    result = news_collection_node(state)

    assert "news_summary" in result
    assert "茅台" in result["news_summary"]
    assert result["news_bundle"]["total_count"] == 1


def test_news_node_falls_back_on_error(mocker):
    """新闻服务异常 → 降级为空摘要，不抛"""
    mocker.patch(
        "core_brain.agents.news_agent.news_fetcher.fetch_news_bundle",
        side_effect=Exception("akshare down"),
    )

    state = make_state(user_profile={})
    result = news_collection_node(state)

    assert result["news_bundle"]["total_count"] == 0
    assert "暂不可用" in result["news_summary"]


def test_news_node_no_tickers_still_fetches_macro(mocker):
    """没自选 ticker → 仍拉宏观/财经"""
    spy = mocker.patch(
        "core_brain.agents.news_agent.news_fetcher.fetch_news_bundle",
        return_value={"by_ticker": {}, "market_headlines": [], "financial_news": [], "total_count": 0},
    )
    mocker.patch(
        "core_brain.agents.news_agent.news_fetcher.summarize_for_prompt",
        return_value="（暂无相关新闻）",
    )

    state = make_state(user_profile={})
    news_collection_node(state)

    # fetch_news_bundle 应被调用，且 tickers 参数为 None
    call_kwargs = spy.call_args.kwargs
    assert call_kwargs.get("tickers") is None
