# core_brain/agents/reviewer.py
"""PORTFOLIO_REVIEW 分支专用：基于用户既有持仓做风险解析，跳过 S_t/A_t 重新选池。"""
from data_ops.market_data import MarketDataFetcher
from config import data as data_config


def portfolio_review_node(state: dict) -> dict:
    """Review: 用户已有持仓 → 直接读取 asset_snapshot 中的 holdings，
    拉取行情用于后续 T_t 微调与 R_t 审查，但不重选资产池、不跑 MVO。
    """
    snapshot = state.get("asset_snapshot", {})
    holdings = snapshot.get("holdings", [])

    if not holdings:
        current = snapshot.get("current_allocation", {})
        total = snapshot.get("total_wealth", 1) or 1
        holdings = [
            {"ticker": key, "name": key, "weight": (val / total)}
            for key, val in current.items()
            if val > 0
        ]

    if not holdings:
        return {
            "available_assets": [],
            "expected_returns": [],
            "cov_matrix": [],
            "base_weights": [],
        }

    tickers = [h.get("ticker") for h in holdings]
    names = [h.get("name", h.get("ticker", "")) for h in holdings]
    base_weights = [round(float(h.get("weight", 0.0)), 4) for h in holdings]

    print(f"-> [Review] 已有持仓审查: {tickers} (跳过 S_t/A_t)")

    real_tickers = [t for t in tickers if t and not t.startswith(("Cash_", "Fixed_", "Equities", "Alternative_"))]
    if real_tickers and len(real_tickers) == len(tickers):
        try:
            fetcher = MarketDataFetcher(lookback_years=data_config.lookback_years)
            expected_returns, cov_matrix = fetcher.fetch_and_calculate(tickers)
        except Exception as e:
            print(f"[Review] 行情拉取失败，降级为空矩阵: {e}")
            expected_returns, cov_matrix = [], []
    else:
        expected_returns, cov_matrix = [], []

    return {
        "available_assets": names,
        "expected_returns": expected_returns,
        "cov_matrix": cov_matrix,
        "base_weights": base_weights,
    }
