# core_brain/agents/portfolio_mgr.py
from .llm_client import call_deepseek


def _recall_past_lessons(profile: dict) -> str:
    """Self-evolve 召回：从 SkillRecord 表里取相似画像的历史教训作为种子"""
    try:
        from memory.state_manager import recall_similar_skills
        skills = recall_similar_skills(profile, limit=3)
        if not skills:
            return ""
        lines = []
        for s in skills:
            score = s.get("critic_score", 0)
            lesson = (s.get("revision_summary") or "").strip()
            if lesson:
                lines.append(f"  • (历史评分 {score}/100) {lesson}")
        if not lines:
            return ""
        return "📚 你之前对类似画像积累的教训（请参考但不必盲从）：\n" + "\n".join(lines)
    except Exception as e:
        print(f"[T_t] skill 召回失败（已忽略）: {e}")
        return ""


def _format_memory_context(history: list, snapshots: list) -> str:
    sections = []
    if history:
        lines = []
        for turn in history[-8:]:
            role = turn.get("role", "user")
            content = (turn.get("content") or "").replace("\n", " ")
            if len(content) > 220:
                content = content[:217] + "..."
            lines.append(f"- {role}: {content}")
        sections.append("最近对话记忆：\n" + "\n".join(lines))

    if snapshots:
        lines = []
        for snap in snapshots[:3]:
            tickers = snap.get("tickers") or []
            weights = snap.get("final_weights") or snap.get("base_weights") or []
            lines.append(
                f"- #{snap.get('id')} {snap.get('created_at')}: "
                f"assets={tickers}, weights={weights}, reason={snap.get('timing_reason') or ''}"
            )
        sections.append("最近方案快照：\n" + "\n".join(lines))

    if not sections:
        return ""
    return "\n\n历史上下文（只作为辅助记忆，当前用户请求优先）：\n" + "\n\n".join(sections)


def timing_adjustment_node(state: dict) -> dict:
    """T_t: 配置经理 Agent - 基于用户画像 + 新闻舆情 + 风险报告 + 历史教训 微调数学权重

    若 state 中有 critic_feedback（来自上一轮 Critic 评判），优先消化反馈做修订。
    若 SkillRecord 中有相似画像的历史教训，作为 prompt 种子注入。
    """
    profile = state.get("user_profile", {})
    news_summary = state.get("news_summary") or "（无可用新闻）"
    risk_report = state.get("risk_simulation") or {}
    critic_feedback = state.get("critic_feedback") or ""
    retry_count = state.get("critic_retries", 0)
    past_lessons = _recall_past_lessons(profile)
    memory_context = _format_memory_context(
        state.get("history") or [],
        state.get("past_snapshots") or [],
    )

    # 把风险报告压缩成 LLM 可读的短文本（控 token）
    risk_section = ""
    if risk_report:
        mc = risk_report.get("monte_carlo", {}).get("summary", {})
        var = risk_report.get("var_cvar", {})
        stress = risk_report.get("stress_test", {})
        risk_section = (
            f"\n  - 蒙特卡洛终值中位数收益: {mc.get('mean_return', 0):.1%}, 亏损概率: {mc.get('prob_loss', 0):.1%}\n"
            f"  - 95% VaR: {var.get('var_return', 0):.1%}, CVaR: {var.get('cvar_return', 0):.1%}\n"
            f"  - 最差压力情景: {stress.get('worst_scenario', 'N/A')} → 收益 {stress.get('worst_return', 0):.1%}\n"
        )

    # 评审反馈优先：如果是从 Critic 回炉，强调"修订上一轮方案"
    revision_section = ""
    if critic_feedback and retry_count > 0:
        prev_adj = state.get("adjusted_weights") or []
        revision_section = f"""
    ⚠️ 这是第 {retry_count} 次修订，上一轮你的方案被独立评审驳回：
    上一轮权重: {prev_adj}
    评审反馈: {critic_feedback}
    请认真消化反馈，做出针对性的调整。
    """

    prompt = f"""
    用户画像：
    - 年龄: {profile.get('age', '未知')}
    - 风险等级: {profile.get('risk_tolerance_level', '平衡型')} (评分 {profile.get('risk_score', 60)}/100)
    - 投资期限: {profile.get('investment_horizon', '中长期')}
    - 理财目标: {profile.get('financial_goals', '资产增值')}
    - 可投资资产: {profile.get('total_wealth', 500000):,} 元

    附加诉求: "{state['user_query']}"

    MVO沙箱计算出的基准权重为: {state['base_weights']}
    对应的资产列表为: {state['available_assets']}

    沙箱前向风险报告：{risk_section}

    📰 最新相关新闻摘要（已按时间倒序，🔥=当日 / 🆕=本周 / 📅=本月）：
    {news_summary}
    {past_lessons}
    {memory_context}
    {revision_section}
    任务：作为配置经理 Agent，结合"用户画像 + 沙箱风险报告 + 实时舆情 + 历史教训"对沙箱数学权重进行微调。
    要求：
    1. 微调后权重总和必须严格等于 1.0
    2. 如果舆情明显利空某一标的，应适度下调其权重并在 timing_reason 中说明
    3. 如果 VaR / 最差压力情景超出用户承受能力，应整体降低高波动资产权重

    请严格输出 JSON：
    "adjusted_weights": [浮点数列表, 与资产列表一一对应]
    "timing_reason": "一段中文说明，结合画像/新闻/风险报告解释调整逻辑"
    """
    try:
        res = call_deepseek(state["api_key"], prompt, role="primary",
                            model=state.get("model_primary"), base_url=state.get("base_url"))
        adjusted = [round(w, 4) for w in res.get("adjusted_weights", state["base_weights"])]
        return {"adjusted_weights": adjusted, "timing_reason": res.get("timing_reason", "模型无调整")}
    except Exception as e:
        print(f"Timing Node LLM Error: {e}")
        return {"adjusted_weights": state["base_weights"], "timing_reason": "微调调用失败，采用沙箱默认原权重。"}
