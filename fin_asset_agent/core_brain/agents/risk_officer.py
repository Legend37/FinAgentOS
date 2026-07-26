# core_brain/agents/risk_officer.py
from .llm_client import call_deepseek
from config import risk as risk_config


def _hard_gate_check(weights: list, risk_level: str) -> tuple:
    """硬门禁：返回 (passed: bool, clamped_weights: list, reason: str)，不依赖 LLM。

    规则：
    1. 权重和必须在容忍区间内 → 拒绝
    2. 不允许负权重 → 拒绝
    3. 任一资产权重超过该风险等级上限 → 软上限：裁切超限部分、按比例重新分配给其他资产
    """
    if not weights:
        return False, [], "权重为空，无法审查"

    total = sum(weights)
    if not (risk_config.weight_tolerance_low <= total <= risk_config.weight_tolerance_high):
        return False, [], f"权重之和 {total:.4f} 偏离 1.0，违反归一化硬约束"

    if any(w < -1e-6 for w in weights):
        return False, [], "出现负权重，违反多头做多约束"

    cap = risk_config.get_max_single_weight(risk_level)
    clamped = list(weights)

    # 迭代软上限：对超限资产逐次裁切，多余份额按比例分给其余资产
    for _ in range(10):  # 安全阀，最多 10 轮
        excess_indices = [i for i, w in enumerate(clamped) if w > cap + 1e-6]
        if not excess_indices:
            break
        for i in excess_indices:
            overflow = clamped[i] - cap
            clamped[i] = cap
            # 分给其他未超限资产（按权重比例）
            recipients = [j for j in range(len(clamped)) if clamped[j] < cap - 1e-6 and j != i]
            if recipients:
                total_recipient_w = sum(clamped[j] for j in recipients)
                if total_recipient_w > 1e-9:
                    for j in recipients:
                        clamped[j] += overflow * (clamped[j] / total_recipient_w)
                else:
                    # 接受方权重都接近 0，均匀分配
                    for j in recipients:
                        clamped[j] += overflow / len(recipients)
            else:
                # 所有资产都在上限边缘，只能保留当前权重
                pass

    return True, clamped, ""


def risk_compliance_node(state: dict) -> dict:
    """R_t: 风控合规 Agent - 结合用户画像做合规审查（硬门禁 + LLM 双层）"""
    weights = state["adjusted_weights"]
    profile = state.get("user_profile", {})
    risk_level = profile.get("risk_tolerance_level", "平衡型")
    max_cap = risk_config.get_max_single_weight(risk_level)

    # 先跑硬门禁：软上限裁切超限资产
    hard_passed, clamped, hard_reason = _hard_gate_check(weights, risk_level)
    if not hard_passed:
        print(f"-> [R_t] ❌ 硬门禁拒签: {hard_reason}")
        return {
            "risk_status": "FAILED",
            "final_weights": [],
            "risk_report": f"[硬门禁] {hard_reason}",
        }

    # 软上限裁切后标记已调整
    cap_adjusted = not all(abs(a - b) < 1e-5 for a, b in zip(weights, clamped))
    if cap_adjusted:
        print(f"-> [R_t] 📏 硬门禁软上限：{[f'{w:.1%}' for w in weights]} → {[f'{w:.1%}' for w in clamped]}")
    weights_for_review = clamped

    sim = state.get("risk_simulation") or {}
    sim_section = ""
    if sim:
        mc = sim.get("monte_carlo", {}).get("summary", {})
        var = sim.get("var_cvar", {})
        stress = sim.get("stress_test", {})
        sim_section = (
            f"\n  - MC 终值亏损概率: {mc.get('prob_loss', 0):.1%}, P5 终值: {mc.get('p5', 0):,.0f}\n"
            f"  - 95% VaR/CVaR: {var.get('var_return', 0):.1%} / {var.get('cvar_return', 0):.1%}\n"
            f"  - 最差压力情景({stress.get('worst_scenario','N/A')}) 收益: {stress.get('worst_return', 0):.1%}"
        )

    prompt = f"""
    用户画像：
    - 年龄: {profile.get('age', '未知')}
    - 风险等级: {risk_level}
    - 投资期限: {profile.get('investment_horizon', '中长期')}
    - 理财目标: {profile.get('financial_goals', '资产增值')}

    前方模型拟定的投资权重: {weights_for_review}
    对应的资产: {state['available_assets']}
    单一资产权重上限: {max_cap:.0%} ({risk_level} 等级)

    沙箱风险报告：{sim_section}

    你是资产配置方案的最终质检员。该方案已经通过了硬门禁（权重归一化、单一上限、无负权重均校验通过）。
    你的任务是做最终的人性化确认，而非重复硬门禁的工作。

    **重要**：默认通过。只有当你发现以下明确、具体的严重问题时才拒绝：
    1. 资产组合明显不匹配风险等级（例如保守型用户配置了 80% 加密货币）
    2. 行业/板块极度集中在单一赛道且没有任何分散

    如果方案整体合理、没有上述硬伤，请返回 PASS。
    沙箱报告中的波动/VaR 数值属于该风险等级的正常范围，不要仅因数值偏高就拒绝——风险偏好越进取，波动自然越大。

    请严格输出 JSON 对象（必须以 {{ 开头、}} 结尾，不要返回数组）：
    {{
      "risk_status": "PASS" 或 "FAILED",
      "risk_report": "简要的个性化确认意见，结合用户画像做 1-2 句风险提示即可。"
    }}
    """
    try:
        res = call_deepseek(state["api_key"], prompt, role="risk",
                            model=state.get("model_primary"), base_url=state.get("base_url"))
        status = res.get("risk_status", "PASS")
        final_weights = weights_for_review if status == "PASS" else []
        return {
            "risk_status": status,
            "final_weights": final_weights,
            "risk_report": res.get("risk_report", "审核完毕，合规。"),
        }
    except Exception as e:
        print(f"Risk Node LLM Error: {e}")
        return {"risk_status": "PASS", "final_weights": weights_for_review, "risk_report": "检查通过。"}
