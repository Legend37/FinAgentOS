# core_brain/agents/profile_agent.py

RISK_SCORE_MAP = {
    "保守型": 20,
    "稳健型": 40,
    "平衡型": 60,
    "成长型": 80,
    "进取型": 95,
}


def profile_generation_node(state: dict) -> dict:
    """P_t: 档案专家 Agent - 基于用户真实输入整理画像，不再调 LLM 编造"""
    profile = state.get("user_profile", {})
    if not profile:
        profile = {}

    risk_level = profile.get("risk_tolerance_level", "平衡型")
    risk_score = profile.get("risk_score") or RISK_SCORE_MAP.get(risk_level, 60)
    total_wealth = profile.get("total_wealth", 500000)

    # 保留 PORTFOLIO_REVIEW 上游传入的 holdings 不覆盖
    existing_snapshot = state.get("asset_snapshot") or {}
    if existing_snapshot.get("holdings"):
        asset_snapshot = {
            "total_wealth": existing_snapshot.get("total_wealth", total_wealth),
            "holdings": existing_snapshot["holdings"],
            "current_allocation": existing_snapshot.get("current_allocation", {}),
        }
    else:
        asset_snapshot = {
            "total_wealth": total_wealth,
            "current_allocation": {
                "Cash_Equivalents": int(total_wealth * 0.6),
                "Fixed_Income": int(total_wealth * 0.2),
                "Equities": int(total_wealth * 0.15),
                "Alternative_Assets": int(total_wealth * 0.05),
            },
        }

    return {
        "user_profile": {
            "user_uuid": profile.get("user_uuid"),
            "email": profile.get("email"),
            "name": profile.get("name", "用户"),
            "age": profile.get("age", 30),
            "occupation": profile.get("occupation", "未填写"),
            "total_wealth": total_wealth,
            "risk_tolerance_level": risk_level,
            "risk_score": risk_score,
            "investment_horizon": profile.get("investment_horizon", "中长期"),
            "financial_goals": profile.get("financial_goals", "资产稳健增值"),
            "custom_tickers": profile.get("custom_tickers"),
            "preferred_categories": profile.get("preferred_categories"),
        },
        "asset_snapshot": asset_snapshot,
    }
