# core_brain/agents/critic.py
"""C_t: Critic Agent — 评判 T_t 的微调方案 + 反思 + 经验沉淀

工作流程：
  1. 拿 T_t 的 adjusted_weights 再跑一次确定性沙箱（MC + VaR + 压力测试）
  2. 对比"调整前"vs"调整后"的风险指标（VaR Δ、最差情景 Δ、亏损概率 Δ）
  3. LLM 基于"前后对比 + 画像 + 新闻"做反思评判：APPROVE / REVISE
  4. APPROVE 时让 LLM 提炼一条"教训"（lesson）→ 写入 SkillRecord 学习库
  5. 学习库通过画像签名索引，T_t 下次遇到相似画像时可召回作为种子

设计要点：
- Critic 跑的沙箱 ≠ RiskSim 跑的沙箱（后者在 base_weights 上，前者在 adjusted_weights 上）
- 双层安全阀：重试 ≥ 2 次强制 APPROVE 防死循环
"""
from .llm_client import call_deepseek
from sandbox.backtest_engine import comprehensive_risk_report


def _rerun_sandbox(state: dict, weights: list) -> dict:
    """Critic 私有沙箱：在 adjusted_weights 上重新跑 MC + VaR + 压力测试

    无行情数据时（如 PORTFOLIO_REVIEW 的类别级持仓 Cash_/Equities 等没有 cov_matrix）
    直接跳过，不进维度校验 —— 否则空协方差 (0,) 会撞 backtest_engine 的 shape 断言。
    """
    exp_ret = state.get("expected_returns") or []
    cov = state.get("cov_matrix") or []
    assets = state.get("available_assets") or []
    if not exp_ret or not cov or not assets:
        return {}
    try:
        return comprehensive_risk_report(
            expected_returns=exp_ret,
            cov_matrix=cov,
            weights=weights,
            asset_labels=assets,
            horizon_days=252,
            n_paths=2_000,  # Critic 用少一点路径加速
        )
    except Exception as e:
        print(f"[C_t] 沙箱重算失败: {e}")
        return {}


def _summarize_metrics(sim: dict) -> dict:
    """从沙箱报告里提取 4 个核心指标，便于前后对比"""
    if not sim:
        return {"prob_loss": None, "var": None, "cvar": None, "worst_return": None}
    mc = sim.get("monte_carlo", {}).get("summary", {})
    var = sim.get("var_cvar", {})
    stress = sim.get("stress_test", {})
    return {
        "prob_loss": mc.get("prob_loss"),
        "var": var.get("var_return"),
        "cvar": var.get("cvar_return"),
        "worst_return": stress.get("worst_return"),
        "worst_scenario": stress.get("worst_scenario"),
    }


def _fmt_delta(before: dict, after: dict) -> str:
    """前后对比的人类可读字符串"""
    def _pct(v):
        return f"{v*100:+.2f}%" if v is not None else "N/A"

    def _delta_pct(b, a):
        if b is None or a is None:
            return "N/A"
        d = a - b
        arrow = "↓ 改善" if d < 0 else ("↑ 恶化" if d > 0 else "持平")
        return f"{_pct(b)} → {_pct(a)}  ({arrow})"

    return (
        f"  亏损概率: {_delta_pct(before['prob_loss'], after['prob_loss'])}\n"
        f"  95% VaR:  {_delta_pct(before['var'], after['var'])}\n"
        f"  95% CVaR: {_delta_pct(before['cvar'], after['cvar'])}\n"
        f"  最差情景({after.get('worst_scenario') or 'N/A'}): "
        f"{_delta_pct(before['worst_return'], after['worst_return'])}"
    )


def _persist_lesson(state: dict, status: str, score: int, feedback: str, lesson: str):
    """APPROVE 后把 lesson 写入 SkillRecord 学习库（带画像签名索引）"""
    if status != "APPROVE" or not lesson or len(lesson.strip()) < 10:
        return None
    try:
        from memory.state_manager import record_skill
        sid = record_skill(
            profile=state.get("user_profile", {}),
            intent=state.get("intent", "ASSET_ALLOCATION"),
            critic_feedback=feedback,
            critic_score=score,
            revision_summary=lesson,
        )
        print(f"-> [C_t] 教训已沉淀到 SkillRecord (id={sid}): {lesson[:60]}...")
        return sid
    except Exception as e:
        print(f"[C_t] skill 持久化失败（已忽略）: {e}")
        return None


def critic_node(state: dict) -> dict:
    """Critic 评判 + 反思 + 教训沉淀"""
    profile = state.get("user_profile", {})
    base_w = state.get("base_weights") or []
    adj_w = state.get("adjusted_weights") or []
    assets = state.get("available_assets") or []
    base_sim = state.get("risk_simulation") or {}        # 在 base_weights 上算的
    news = state.get("news_summary") or "（无新闻）"
    retry_count = state.get("critic_retries", 0)

    # === 1. Critic 拿 adjusted_weights 重新跑沙箱 ===
    adjusted_sim = _rerun_sandbox(state, adj_w) if adj_w else {}

    before = _summarize_metrics(base_sim)
    after = _summarize_metrics(adjusted_sim)
    delta_text = _fmt_delta(before, after)

    # 权重变化幅度
    deltas = []
    for i, (b, a) in enumerate(zip(base_w, adj_w)):
        if abs(a - b) > 1e-4:
            asset = assets[i] if i < len(assets) else f"#{i}"
            deltas.append(f"{asset}: {b:.2%} → {a:.2%} (Δ{(a-b):+.2%})")
    weight_delta_text = "\n  ".join(deltas) if deltas else "无调整"

    # === 2. LLM 反思评判 prompt ===
    prompt = f"""
你是一名独立的策略评审委员（Critic）。配置经理（T_t）刚给出了一份权重微调方案，
你需要基于 **调整前后沙箱重算的对比** 做评判，并提炼一条可复用的教训。

用户画像：
- 风险等级: {profile.get('risk_tolerance_level', '平衡型')} ({profile.get('risk_score', 60)}/100)
- 投资期限: {profile.get('investment_horizon', '中长期')}
- 财务目标: {profile.get('financial_goals', '资产增值')}

权重微调（base → adjusted）：
  {weight_delta_text}

🔬 **沙箱前后对比**（同一资产、同一协方差，只换权重重新模拟）：
{delta_text}

📰 实时新闻摘要（🔥=当日 / 🆕=本周 / 📅=本月，应优先采信带 🔥🆕 标记的内容）：
{news[:500]}

评审准则：
1. 调整后 VaR / CVaR / 亏损概率是否朝改善方向走（↓ 是好的）
2. 调整方向是否与新闻舆情一致
3. 画像红线（仅在严重偏离时标记）：
   - 保守型: VaR ≤ 15%, 压力情景 ≥ -25%
   - 稳健型: VaR ≤ 20%, 压力情景 ≥ -35%
   - 平衡型: VaR ≤ 30%, 压力情景 ≥ -45%
   - 成长型/进取型: 不设红线
4. 综合打分 < 50 → REVISE，≥ 50 → APPROVE
（低分仅在你确信组合有害时给出；不确定时应给 60 以上放行）

当前重试: {retry_count}（≥ 2 即使有瑕疵也应 APPROVE 防死循环）

请严格输出 JSON：
{{
  "critic_status": "APPROVE" 或 "REVISE",
  "critic_score": 0-100 整数,
  "critic_feedback": "≤150 字中文评审依据，如 REVISE 须给出具体修订方向",
  "lesson": "≤80 字的一句话教训，提炼"对于这类画像，什么调整规则是有效的/无效的"。如无明显教训返回空字符串"
}}
"""
    try:
        res = call_deepseek(state["api_key"], prompt, role="critic",
                            model=state.get("model_primary"), base_url=state.get("base_url"))
        status = res.get("critic_status", "APPROVE")
        score = int(res.get("critic_score", 60))
        feedback = res.get("critic_feedback", "评审通过")
        lesson = (res.get("lesson") or "").strip()

        # === 3. 安全阀：≥2 次重试强制 APPROVE ===
        #   无量化沙箱基础（如类别级持仓体检）时，回炉 T_t 也无法改善 →
        #   首轮即放行，避免空转 REVISE 浪费 LLM 调用
        has_quant = bool(base_sim) or bool(adjusted_sim)
        if status != "APPROVE" and (retry_count >= 2 or not has_quant):
            reason = "重试上限" if retry_count >= 2 else "无沙箱数据，无法量化复核"
            status = "APPROVE"
            feedback = f"[{reason}] 原评审意见: {feedback}"

        # === 4. APPROVE 时把 lesson 沉淀到 SkillRecord ===
        skill_id = _persist_lesson(state, status, score, feedback, lesson)

        print(f"-> [C_t] Critic: {status} (score={score}, retry={retry_count}, "
              f"lesson={'✓' if skill_id else '—'})")
        return {
            "critic_status": status,
            "critic_score": score,
            "critic_feedback": feedback,
            "critic_lesson": lesson,
            "critic_skill_id": skill_id,
            "adjusted_risk_simulation": adjusted_sim,   # 给前端展示前后对比
            "critic_retries": retry_count + 1 if status == "REVISE" else retry_count,
        }
    except Exception as e:
        print(f"[C_t] Critic LLM Error: {e}")
        return {
            "critic_status": "APPROVE",
            "critic_score": 60,
            "critic_feedback": "评审降级：LLM 不可用，默认放行。",
            "critic_lesson": "",
            "critic_skill_id": None,
            "adjusted_risk_simulation": adjusted_sim,
            "critic_retries": retry_count,
        }
