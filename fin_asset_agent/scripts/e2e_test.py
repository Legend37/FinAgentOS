"""端到端真实 LLM 链路测试 — 跑通全部 5 项新能力

用法：
  cd fin_asset_agent
  DEEPSEEK_API_KEY=sk-xxxx python3 scripts/e2e_test.py
"""
import os
import sys
import json

# 让脚本能在 fin_asset_agent/ 下直接 import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_brain.workflow import run_fin_agent_pipeline
from core_brain.agents.llm_client import reset_usage_log, get_usage_summary


def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("❌ 请先设置 DEEPSEEK_API_KEY 环境变量")
        sys.exit(1)

    reset_usage_log()

    user_profile = {
        "name": "测试用户",
        "age": 32,
        "occupation": "工程师",
        "total_wealth": 500_000,
        "risk_tolerance_level": "平衡型",
        "investment_horizon": "中长期",
        "financial_goals": "教育金储备 + 抗通胀",
        "custom_tickers": ["600519.SS", "601398.SS", "000858.SZ"],
    }

    query = "近期市场震荡，帮我配一个平衡型组合，注意控制最大回撤"

    print("=" * 70)
    print(f"📝 用户诉求: {query}")
    print(f"👤 画像: {user_profile['risk_tolerance_level']} / {user_profile['investment_horizon']}")
    print("=" * 70)

    result = run_fin_agent_pipeline(query, api_key, user_profile)

    print("\n" + "=" * 70)
    print(f"🎯 意图: {result['intent']}")
    print(f"📊 资产: {result['assets']}")
    print(f"⚖️  基准权重 (MVO):  {result['base_weights']}")
    print(f"🎨 最终权重:        {result['final_weights']}")
    print(f"✅ 风控状态: {result['risk_status']}")
    print(f"🧐 Critic: {result.get('critic_status')} (score={result.get('critic_score')}, "
          f"重试={result.get('critic_retries')})")
    print("=" * 70)

    sim = result.get("risk_simulation", {})
    if sim:
        mc = sim.get("monte_carlo", {}).get("summary", {})
        var = sim.get("var_cvar", {})
        stress = sim.get("stress_test", {})
        print(f"\n📈 沙箱前向风险报告:")
        print(f"   MC 终值亏损概率: {mc.get('prob_loss', 0):.1%}")
        print(f"   95% VaR: {var.get('var_return', 0):.1%}  CVaR: {var.get('cvar_return', 0):.1%}")
        print(f"   最差压力情景: {stress.get('worst_scenario')} → {stress.get('worst_return', 0):.1%}")

    news = result.get("news_summary", "")
    if news and "暂无" not in news:
        print(f"\n📰 注入的新闻摘要 (前 300 字):")
        print(f"   {news[:300]}")

    print(f"\n💬 配置经理理由: {result['timing_reason'][:200]}")
    print(f"\n🧐 Critic 反馈: {result.get('critic_feedback', '')[:200]}")
    print(f"\n🛡️  风控报告: {result['risk_report'][:200]}")

    # 成本汇总
    usage = result.get("llm_usage", {})
    print("\n" + "=" * 70)
    print(f"💰 LLM 成本汇总:")
    print(f"   总调用: {usage.get('total_calls', 0)} 次")
    print(f"   输入 token: {usage.get('total_prompt_tokens', 0):,}")
    print(f"   输出 token: {usage.get('total_completion_tokens', 0):,}")
    print(f"   总费用: ¥{usage.get('total_cost_rmb', 0):.4f}")
    print(f"   按模型拆分:")
    for model, stats in usage.get("by_model", {}).items():
        print(f"     - {model}: {stats['calls']} 次 / {stats['tokens']:,} tokens / ¥{stats['cost_rmb']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
