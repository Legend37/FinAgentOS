# core_brain/agents/analyst.py
from data_ops.market_data import MarketDataFetcher
from config import asset_pool, asset_universe, data as data_config
from .asset_selector import llm_pick_assets, fallback_preset_pool


def asset_screening_node(state: dict) -> dict:
    """S_t: 分析师 Agent - 三级选品策略：

    优先级 1：用户自选 tickers → 直接用
    优先级 2：LLM 智能选品（从 50+ 资产宇宙中按画像/诉求/类别偏好挑）
    优先级 3：风险等级预设池（LLM 不可用时兜底）
    """
    profile = state.get("user_profile", {})
    risk_level = profile.get("risk_tolerance_level", "平衡型")

    custom = profile.get("custom_tickers") or []
    custom = [t.strip() for t in custom if t and t.strip()]

    selection_rationale = ""

    if custom:
        # 优先级 1: 用户显式自选
        tickers = custom
        # 名字从 universe + 旧预设池里都查一遍
        all_names = {}
        for p in asset_pool.pools.values():
            all_names.update(p.get("names", {}))
        for a in asset_universe.get_all():
            all_names[a["ticker"]] = f"{a['name']} ({a['ticker']})"
        names = [all_names.get(t, f"{t} (自选)") for t in tickers]
        selection_rationale = "用户显式指定标的，未触发 AI 选品。"
        print(f"-> [S_t] Analyst: 用户自选 tickers={tickers}")
    else:
        # 优先级 2: LLM 从 universe 选
        api_key = state.get("api_key", "")
        query = state.get("user_query", "")
        categories = profile.get("preferred_categories") or []

        try:
            pick = llm_pick_assets(api_key, profile, query, categories, n=6,
                                   model=state.get("model_primary"),
                                   base_url=state.get("base_url"))
            tickers = pick["tickers"]
            names = [pick["names"][t] for t in tickers]
            selection_rationale = pick["rationale"]
            print(f"-> [S_t] Analyst: LLM 从宇宙挑了 {len(tickers)} 只 — {tickers}")
            if categories:
                print(f"   用户偏好类别: {categories}")
        except Exception as e:
            # 优先级 3: 降级到预设池
            print(f"-> [S_t] Analyst: LLM 选品失败 ({e})，降级到风险等级预设池")
            fb = fallback_preset_pool(risk_level)
            tickers = fb["tickers"]
            names = [fb["names"].get(t, f"{t} (兜底)") for t in tickers]
            selection_rationale = fb["rationale"]

    fetcher = MarketDataFetcher(lookback_years=data_config.lookback_years)
    expected_returns, cov_matrix = fetcher.fetch_and_calculate(tickers)

    return {
        "available_assets": names,
        "expected_returns": expected_returns,
        "cov_matrix": cov_matrix,
        "selection_rationale": selection_rationale,
    }
