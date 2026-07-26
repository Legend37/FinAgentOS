# data_ops/advice.py
"""把一份方案 + 归因结果，组织成一条人类可读的"主动复盘建议"文案。

纯确定性模板（不调 LLM，便于测试与零成本推送）；建议方向由简单规则给出。
"""
from typing import Dict

INTENT_LABEL = {
    "ASSET_ALLOCATION": "资产配置",
    "PORTFOLIO_REVIEW": "持仓体检",
    "QA_RAG": "知识问答",
    "CHIT_CHAT": "闲聊",
}


def _pct(v) -> str:
    return f"{v*100:+.2f}%" if isinstance(v, (int, float)) else "N/A"


def generate_review_advice(snapshot: Dict, attribution: Dict) -> Dict:
    """根据快照 + 归因结果生成 {reason, text}。"""
    sid = snapshot.get("id")
    adate = snapshot.get("advice_date") or (snapshot.get("created_at") or "")[:10]
    intent = INTENT_LABEL.get(snapshot.get("intent"), snapshot.get("intent") or "方案")
    status = attribution.get("status")

    head = f"📊 FinAgent 复盘 · 方案 #{sid}（{intent}）\n建议日：{adate}"

    if status == "ok":
        ret = attribution.get("realized_return", 0.0)
        best = attribution.get("best_contributor") or "—"
        worst = attribution.get("worst_contributor") or "—"
        days = attribution.get("elapsed_days", attribution.get("horizon_days"))
        if ret <= -0.05:
            reason = "复盘到期 · 回撤偏大"
            suggestion = "回撤较大，建议检视重仓资产、考虑再平衡或对冲，必要时部分止损。"
        elif ret >= 0.05:
            reason = "复盘到期 · 表现良好"
            suggestion = "表现良好，可考虑部分止盈锁定收益，或维持配置让利润奔跑。"
        else:
            reason = "复盘到期 · 波动温和"
            suggestion = "波动温和，维持现有配置即可，关注后续宏观/政策变化。"
        text = (
            f"{head}\n"
            f"近 {days} 天真实表现：收益 {_pct(ret)}，年化波动 {_pct(attribution.get('realized_volatility'))}，"
            f"Sharpe {attribution.get('realized_sharpe', 0):.2f}\n"
            f"贡献最大：{best}；拖累最大：{worst}\n"
            f"💡 {suggestion}"
        )
        return {"reason": reason, "text": text}

    if status == "pending":
        return {
            "reason": "复盘到期 · 等待数据",
            "text": f"{head}\n该方案尚未积累足够交易日，真实表现待数据补齐后再复盘。",
        }

    # unavailable
    return {
        "reason": "复盘到期 · 无个券行情",
        "text": f"{head}\n{attribution.get('reason', '暂无法回查真实价格')}。建议下次配置时使用可回查的标的。",
    }
