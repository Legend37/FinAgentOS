# core_brain/agents/news_agent.py
"""N_t: 新闻舆情节点 — 拉个股新闻 + 财经头条，喂给 T_t 做舆情驱动的微调"""
from data_ops import news_fetcher


def news_collection_node(state: dict) -> dict:
    """N_t: 抓取与组合相关的新闻，summary 写入 state，T_t 读取"""
    profile = state.get("user_profile", {})

    # 优先用自选 ticker，回退到资产池的 raw ticker（asset_screening 已把 ticker 标准化）
    tickers = profile.get("custom_tickers")
    if not tickers:
        # 通过 asset_snapshot.holdings 反推 ticker（PORTFOLIO_REVIEW 分支）
        holdings = state.get("asset_snapshot", {}).get("holdings", [])
        if holdings:
            tickers = [h.get("ticker") for h in holdings if h.get("ticker")]

    try:
        bundle = news_fetcher.fetch_news_bundle(
            tickers=tickers[:5] if tickers else None,
            per_ticker=3, headlines=4,
        )
        summary = news_fetcher.summarize_for_prompt(bundle, max_items=12)
        print(f"-> [N_t] News Agent: 抓取 {bundle['total_count']} 条相关新闻")
    except Exception as e:
        print(f"[N_t] 新闻抓取失败，降级为空: {e}")
        bundle = {"by_ticker": {}, "market_headlines": [], "financial_news": [], "total_count": 0}
        summary = "（新闻服务暂不可用）"

    return {
        "news_bundle": bundle,
        "news_summary": summary,
    }
