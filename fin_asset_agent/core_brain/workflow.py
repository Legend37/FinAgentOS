# core_brain/workflow.py
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List

from sandbox.mvo_solver import MVOSolver
from sandbox.backtest_engine import comprehensive_risk_report
from core_brain.agents import (
    profile_generation_node,
    asset_screening_node,
    timing_adjustment_node,
    risk_compliance_node,
    portfolio_review_node,
    news_collection_node,
    critic_node,
)
from core_brain.agents.llm_client import call_deepseek_text
from core_brain.router import CoreRouter


class GraphState(TypedDict, total=False):
    user_query: str
    api_key: str
    base_url: str           # 自定义接口地址（留空回退 config 默认）
    model_router: str       # 解析后的 router 模型（None=回退 config）
    model_chat: str         # 解析后的 chat 模型
    model_primary: str      # 解析后的 primary 模型（T_t/Critic/R_t/选品）
    intent: str
    user_profile: dict
    asset_snapshot: dict
    available_assets: List[str]
    expected_returns: List[float]
    cov_matrix: List[List[float]]
    base_weights: List[float]
    adjusted_weights: List[float]
    final_weights: List[float]
    risk_status: str
    timing_reason: str
    risk_report: str
    history: list
    past_snapshots: list
    # 新增字段（#5 沙箱 + #1 新闻 + #4 Critic 闭环）
    risk_simulation: dict     # MC + VaR + 压力测试
    news_bundle: dict          # 原始新闻包
    news_summary: str          # 喂 LLM 的新闻短摘要
    critic_status: str         # APPROVE / REVISE
    critic_score: int          # 0-100
    critic_feedback: str       # 修订建议
    critic_retries: int        # 重试计数
    critic_lesson: str         # 反思提炼的教训
    critic_skill_id: int       # 沉淀到 SkillRecord 的记录 id
    adjusted_risk_simulation: dict  # Critic 在 adjusted_weights 上重算的沙箱
    selection_rationale: str   # AI 选品理由（来自 S_t）


def _format_history_for_prompt(history: list, limit: int = 6) -> str:
    if not history:
        return ""
    lines = []
    for turn in history[-limit:]:
        role = turn.get("role", "user")
        content = (turn.get("content") or "").replace("\n", " ")
        if len(content) > 240:
            content = content[:237] + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def route_by_intent(state: GraphState):
    """三分支路由（支持并行 fan-out）：
    - CHIT_CHAT / QA_RAG → general_response
    - PORTFOLIO_REVIEW → 仅 profile（之后串行 → Review）
    - ASSET_ALLOCATION → profile + screen 同时并行执行，在 Sandbox_A_t 汇合
    """
    intent = state.get("intent", "ASSET_ALLOCATION")
    if intent in ("CHIT_CHAT", "QA_RAG"):
        return ["general_response"]
    if intent == "PORTFOLIO_REVIEW":
        return ["profile"]
    return ["profile", "screen"]  # 并行 fan-out


def route_after_profile(state: GraphState) -> str:
    """P_t 之后：
    - PORTFOLIO_REVIEW → Review
    - ASSET_ALLOCATION → Sandbox_A_t（与并行运行的 Screening_S_t 在此汇合）
    """
    intent = state.get("intent", "ASSET_ALLOCATION")
    if intent == "PORTFOLIO_REVIEW":
        return "Review"
    return "Sandbox_A_t"


def route_after_critic(state: GraphState) -> str:
    """Critic 之后：
    - APPROVE → 进 Risk_R_t 终审
    - REVISE  → 回 Timing_T_t 重做（critic_node 已强制 ≤2 次安全阀）
    """
    status = state.get("critic_status", "APPROVE")
    retries = state.get("critic_retries", 0)
    if status == "REVISE" and retries <= 2:
        return "Timing_T_t"
    return "Risk_R_t"


def general_response_node(state: GraphState):
    """兜底应答 — 闲聊/问答调用 LLM 生成自然语言回复，不触发 MVO 管线"""
    intent = state.get("intent", "CHIT_CHAT")
    history_text = _format_history_for_prompt(state.get("history") or [])
    query = state.get("user_query") or "你好"

    if intent == "QA_RAG":
        system = (
            "你是一名严谨的金融顾问助理，用中文回答用户的金融知识问题。"
            "回答要简洁专业（200 字以内），不构造未经证实的数据；"
            "如果问题涉及具体投资建议，提醒用户切换到资产配置模式。"
        )
    else:
        system = (
            "你是一名亲切的金融助手。用中文与用户进行轻松对话（100 字以内），"
            "在合适的时候引导用户使用资产配置功能。不要编造具体数据。"
        )

    try:
        user_prompt = query
        if history_text:
            user_prompt = f"最近历史对话：\n{history_text}\n\n当前用户问题：{query}"
        reply = call_deepseek_text(
            state.get("api_key", ""), system, user_prompt,
            model=state.get("model_chat"), base_url=state.get("base_url"),
        )
    except Exception as e:
        print(f"[GeneralResponse] LLM 调用失败，降级为静态兜底: {e}")
        reply = "你好，我是 FinAgent 助手。如需资产配置，请在上方填写画像信息并点击执行。"

    return {
        "risk_status": "CHAT",
        "timing_reason": reply,
        "risk_report": reply,
        "final_weights": [],
    }


def sandbox_allocation_node(state: GraphState):
    """A_t: 沙箱调度 - 宏观择时 + MVO/RiskParity 二次规划"""
    risk_score = state.get("user_profile", {}).get("risk_score", 60)
    assets = state.get("available_assets", [])

    # 1) 获取资产类别，用于宏观映射
    from config import asset_universe
    categories = []
    for ticker in assets:
        info = asset_universe.get_by_ticker(ticker)
        categories.append(info.get("category", "stock_a") if info else "stock_a")

    # 2) 宏观择时：检测周期并调整预期收益
    from data_ops.macro_tactical import MacroTacticalAllocator
    allocator = MacroTacticalAllocator()
    base_returns = state["expected_returns"]
    adjusted_returns, macro_meta = allocator.adjust_expected_returns(
        base_returns, categories, strength=0.7
    )

    print(f"-> [A_t] Sandbox: 宏观周期={macro_meta['cycle_cn']}, 运行 risk_parity...")
    solver = MVOSolver(risk_score=risk_score, method="risk_parity")
    result = solver.optimize_portfolio(adjusted_returns, state["cov_matrix"])
    base_w = [round(w, 4) for w in result["optimal_weights"]]

    return {
        "base_weights": base_w,
        "macro_cycle": macro_meta.get("cycle"),
        "macro_cycle_cn": macro_meta.get("cycle_cn"),
        "macro_adjustments": macro_meta.get("adjustments"),
    }


def risk_simulation_node(state: GraphState):
    """RiskSim: 纯数值风险沙箱 (MC + VaR + 压力测试)，喂给 T_t 和 R_t

    只在 ASSET_ALLOCATION / PORTFOLIO_REVIEW 分支跑；CHIT_CHAT 不进。
    """
    base_weights = state.get("base_weights") or []
    exp_ret = state.get("expected_returns") or []
    cov = state.get("cov_matrix") or []
    assets = state.get("available_assets") or []

    if not base_weights or not exp_ret or not cov or not assets:
        print("-> [RiskSim] 缺少基础权重/行情，跳过")
        return {"risk_simulation": {}}

    try:
        report = comprehensive_risk_report(
            expected_returns=exp_ret,
            cov_matrix=cov,
            weights=base_weights,
            asset_labels=assets,
            horizon_days=252,
            n_paths=3_000,  # 控成本：3000 路径足够 VaR 稳定
        )
        print(f"-> [RiskSim] MC + VaR + 压力测试完成，worst={report['stress_test']['worst_scenario']}")
        return {"risk_simulation": report}
    except Exception as e:
        print(f"[RiskSim] 模拟失败: {e}")
        return {"risk_simulation": {}}


# --- 构建状态机图谱 ---
workflow = StateGraph(GraphState)
workflow.add_node("GeneralResponse", general_response_node)
workflow.add_node("Profile_P_t", profile_generation_node)
workflow.add_node("Screening_S_t", asset_screening_node)
workflow.add_node("Sandbox_A_t", sandbox_allocation_node)
workflow.add_node("Review", portfolio_review_node)
workflow.add_node("RiskSim", risk_simulation_node)
workflow.add_node("News_N_t", news_collection_node)
workflow.add_node("Timing_T_t", timing_adjustment_node)
workflow.add_node("Critic_C_t", critic_node)
workflow.add_node("Risk_R_t", risk_compliance_node)

workflow.add_conditional_edges(START, route_by_intent, {
    "general_response": "GeneralResponse",
    "profile": "Profile_P_t",
    "screen": "Screening_S_t",
})
workflow.add_edge("GeneralResponse", END)
workflow.add_conditional_edges("Profile_P_t", route_after_profile, {
    "Sandbox_A_t": "Sandbox_A_t",
    "Review": "Review",
})
workflow.add_edge("Screening_S_t", "Sandbox_A_t")
# A_t/Review 完成后并行跑 RiskSim（纯数值）和 News（IO），在 T_t 汇合
workflow.add_edge("Sandbox_A_t", "RiskSim")
workflow.add_edge("Sandbox_A_t", "News_N_t")
workflow.add_edge("Review", "RiskSim")
workflow.add_edge("Review", "News_N_t")
workflow.add_edge("RiskSim", "Timing_T_t")
workflow.add_edge("News_N_t", "Timing_T_t")
# T_t → Critic → 条件分支：APPROVE → R_t ; REVISE → 回 T_t（最多 2 次）
workflow.add_edge("Timing_T_t", "Critic_C_t")
workflow.add_conditional_edges("Critic_C_t", route_after_critic, {
    "Timing_T_t": "Timing_T_t",
    "Risk_R_t": "Risk_R_t",
})
workflow.add_edge("Risk_R_t", END)

fin_agent_app = workflow.compile()


def _summarize_news_bundle(bundle: dict) -> dict:
    """从 news_bundle 提取展示用元数据：按来源条数 + 最新日期 + 当周占比"""
    if not bundle:
        return {"by_source": {}, "newest_date": "", "in_last_week": 0, "total": 0}
    from collections import Counter
    import datetime as _dt
    by_source = Counter()
    dates = []
    for cat in ("market_headlines", "financial_news"):
        for it in bundle.get(cat, []):
            by_source[it.get("source", "?")] += 1
            if it.get("date"):
                dates.append(it["date"])
    for _t, lst in bundle.get("by_ticker", {}).items():
        for it in lst:
            by_source[it.get("source", "?")] += 1
            if it.get("date"):
                dates.append(it["date"])

    week_ago = (_dt.date.today() - _dt.timedelta(days=7)).isoformat()
    in_week = sum(1 for d in dates if d >= week_ago)
    return {
        "by_source": dict(by_source),
        "newest_date": max(dates) if dates else "",
        "oldest_date": min(dates) if dates else "",
        "in_last_week": in_week,
        "total": len(dates),
    }


def run_fin_agent_pipeline(query: str, api_key: str, user_profile: dict = None,
                           history: list = None, past_snapshots: list = None,
                           base_url: str = None, model: str = None,
                           router_model: str = None, chat_model: str = None,
                           primary_model: str = None) -> dict:
    """封装胶水层：衔接外部接口Payload与内部图状态机的转换

    自定义 LLM 接入（均可留空，留空回退 config.py 默认）：
        base_url:       自定义 OpenAI 兼容接口地址
        model:          单模型 ID，覆盖全部三档
        router/chat/primary_model: 分档覆盖，优先级高于 model
    """
    if user_profile is None:
        user_profile = {}

    # 模型解析：分档覆盖 > 单模型 > None（None 由 call 层回退 config 默认）
    model_router = router_model or model
    model_chat = chat_model or model
    model_primary = primary_model or model

    # 意图路由（含降级：LLM 不可用时默认走资产配置流程）
    try:
        router = CoreRouter(api_key, base_url=base_url, router_model=model_router)
        intent = router.route_query(query or user_profile.get("financial_goals", "资产配置"))
    except Exception as e:
        print(f"[Router] 意图识别失败，降级为 ASSET_ALLOCATION: {e}")
        intent = type("Intent", (), {"intent": "ASSET_ALLOCATION"})()

    initial_state = {
        "user_query": query,
        "api_key": api_key,
        "base_url": base_url,
        "model_router": model_router,
        "model_chat": model_chat,
        "model_primary": model_primary,
        "intent": intent.intent,
        "user_profile": user_profile,
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
        "history": history or [],
        "past_snapshots": past_snapshots or [],
        "risk_simulation": {},
        "news_bundle": {},
        "news_summary": "",
        "critic_status": "",
        "critic_score": 0,
        "critic_feedback": "",
        "critic_retries": 0,
        "critic_lesson": "",
        "critic_skill_id": None,
        "adjusted_risk_simulation": {},
        "selection_rationale": "",
    }
    final_state = fin_agent_app.invoke(initial_state)

    # 注入 cost 统计
    from core_brain.agents.llm_client import get_usage_summary
    usage = get_usage_summary()

    return {
        "assets": final_state.get("available_assets", []),
        "base_weights": final_state.get("base_weights", []),
        "adjusted_weights": final_state.get("adjusted_weights", []),  # T_t/Critic 微调后的提案，REJECTED 时仍可展示
        "final_weights": final_state.get("final_weights", []),
        "risk_status": final_state.get("risk_status", "PASS"),
        "timing_reason": final_state.get("timing_reason", "无调整"),
        "risk_report": final_state.get("risk_report", "未执行风控检查"),
        "user_profile": final_state.get("user_profile", {}),
        "asset_snapshot": final_state.get("asset_snapshot", {}),
        "intent": intent.intent,
        "risk_simulation": final_state.get("risk_simulation", {}),
        "news_summary": final_state.get("news_summary", ""),
        "news_meta": _summarize_news_bundle(final_state.get("news_bundle", {})),
        "critic_status": final_state.get("critic_status", ""),
        "critic_score": final_state.get("critic_score", 0),
        "critic_feedback": final_state.get("critic_feedback", ""),
        "critic_retries": final_state.get("critic_retries", 0),
        "critic_lesson": final_state.get("critic_lesson", ""),
        "critic_skill_id": final_state.get("critic_skill_id"),
        "adjusted_risk_simulation": final_state.get("adjusted_risk_simulation", {}),
        "selection_rationale": final_state.get("selection_rationale", ""),
        "macro_cycle": final_state.get("macro_cycle"),
        "macro_cycle_cn": final_state.get("macro_cycle_cn"),
        "macro_adjustments": final_state.get("macro_adjustments", []),
        "llm_usage": usage,
    }
