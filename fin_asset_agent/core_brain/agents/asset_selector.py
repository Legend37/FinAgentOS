# core_brain/agents/asset_selector.py
"""LLM 智能选品 — 从全资产宇宙挑出适合用户的 N 个标的

调用时机：S_t 分析师节点内，当用户没提供 custom_tickers 时触发。
模型：用 reasoner（primary_model），因为需要多维权衡。
降级：LLM 不可用 → 回退到 risk_level 预设池（保留向后兼容）。
"""
from typing import List, Dict, Optional
from .llm_client import call_deepseek
from config import asset_universe, asset_pool


# 风险等级 → 允许的最大 risk_band（用于先粗过滤宇宙）
RISK_BAND_CAP = {
    "保守型": 2,
    "稳健型": 3,
    "平衡型": 4,
    "成长型": 5,
    "进取型": 5,
}


def _resolve_ticker(raw: str) -> Optional[Dict]:
    """容错 ticker 解析 — 处理 LLM 常见的输出瑕疵：

    1. 精确匹配：'600519.SS' → 命中
    2. 大小写不敏感：'btc-usd' → 'BTC-USD'
    3. 后缀缺失：'513100' → '513100.SS'（LLM 经常丢 .SS/.SZ）
    4. 错挂后缀：'600519.SH' → '600519.SS'（.SH/.SZ 互换）

    返回 universe 中的标准 meta，找不到返回 None。
    """
    if not raw:
        return None
    raw = raw.strip()

    # ① 精确匹配
    meta = asset_universe.get_by_ticker(raw)
    if meta:
        return meta

    universe = asset_universe.get_all()
    raw_upper = raw.upper()

    # ② 大小写不敏感
    for a in universe:
        if a["ticker"].upper() == raw_upper:
            return a

    # ③ 后缀缺失（LLM 只返回 '513100' 这种）
    clean = raw_upper.split(".")[0].split("-")[0].strip()
    if clean:
        for a in universe:
            uniq_prefix = a["ticker"].upper().split(".")[0].split("-")[0]
            if uniq_prefix == clean:
                return a

    return None


def _format_universe_for_prompt(allowed: List[Dict]) -> str:
    """把可选标的列表格式化成 LLM 友好的短文本"""
    by_cat = {}
    for a in allowed:
        by_cat.setdefault(a["category"], []).append(a)

    lines = []
    for cat, items in by_cat.items():
        label = asset_universe.category_labels.get(cat, cat)
        lines.append(f"\n【{label}】")
        for a in items:
            lines.append(f"  - {a['ticker']:12s} {a['name']:8s} (波动等级 {a['risk_band']}) — {a['desc']}")
    return "\n".join(lines)


def llm_pick_assets(api_key: str, profile: Dict, query: str,
                    categories: Optional[List[str]] = None,
                    n: int = 6, model: Optional[str] = None,
                    base_url: Optional[str] = None) -> Dict:
    """LLM 从全宇宙挑 n 个最匹配的标的。

    Args:
        profile: user_profile 字典
        query: 用户自然语言诉求
        categories: 用户偏好的类别列表（前端勾选），空 = 全开
        n: 目标挑选数量

    Returns:
        {"tickers": [...], "names": {ticker: name}, "rationale": "..."}
        失败时抛异常，由调用方降级
    """
    risk_level = profile.get("risk_tolerance_level", "平衡型")
    max_band = RISK_BAND_CAP.get(risk_level, 4)

    # 先按风险等级粗过滤（保守型用户不让 LLM 看到高 risk_band 的标的）
    candidates = asset_universe.filter_by_risk_band(max_band)
    if categories:
        candidates = [a for a in candidates if a["category"] in set(categories)]

    if not candidates:
        raise ValueError(f"无候选标的（风险={risk_level}, 类别={categories}）")

    universe_text = _format_universe_for_prompt(candidates)

    prompt = f"""
你是一名资产配置研究员，从给定的资产宇宙中挑选最匹配用户的 {n} 个标的。

用户画像：
- 风险等级: {risk_level} (评分 {profile.get('risk_score', 60)}/100)
- 投资期限: {profile.get('investment_horizon', '中长期')}
- 财务目标: {profile.get('financial_goals', '资产增值')}

用户诉求："{query or '无特殊偏好'}"

用户勾选的偏好类别: {[asset_universe.category_labels.get(c, c) for c in (categories or [])] or '全开'}

可选资产宇宙（已按风险等级粗筛过；波动等级 1=最低，5=最高）：
{universe_text}

挑选准则（按优先级）：
1. 必须落实用户诉求里提到的方向（如"多配房地产"→ 至少 1-2 只 REIT/地产）
2. 分散度：不同类别尽量各取 1-2 只，避免全押一个赛道
3. 波动等级与用户风险匹配：保守型偏 1-2，平衡型偏 2-4，进取型偏 3-5
4. 优先选 ETF（一篮子分散）而非单个股票，除非用户明确点名
5. 严格只从上面列表中选，不要编造不存在的 ticker
6. **ticker 必须原样照抄，包括后缀**（如必须返回 "513100.SS" 而非 "513100"，加密类原文带连字符不要拆）

请严格输出 JSON：
{{
  "tickers": ["ticker1", "ticker2", ...],
  "rationale": "≤150 字中文，说明选这几个标的的理由 + 如何呼应用户诉求"
}}
"""

    res = call_deepseek(api_key, prompt, role="asset_selector", model=model, base_url=base_url)
    tickers = res.get("tickers") or []
    rationale = res.get("rationale", "")

    if not isinstance(tickers, list) or not tickers:
        raise ValueError(f"LLM 返回空 ticker 列表: {res}")

    # 容错校验：LLM 经常丢后缀（513100 ↔ 513100.SS），用 _resolve_ticker 自动补齐
    valid_tickers = []
    names = {}
    dropped = []
    for t in tickers:
        meta = _resolve_ticker(t)
        if meta is not None:
            canonical = meta["ticker"]
            if canonical not in valid_tickers:  # 去重
                valid_tickers.append(canonical)
                names[canonical] = f"{meta['name']} ({canonical})"
        else:
            dropped.append(t)

    if dropped:
        print(f"[asset_selector] 丢弃无法解析的 ticker: {dropped}")
    if not valid_tickers:
        raise ValueError(f"LLM 全编造 ticker，无一在 universe: {tickers}")

    return {
        "tickers": valid_tickers,
        "names": names,
        "rationale": rationale,
    }


def fallback_preset_pool(risk_level: str) -> Dict:
    """LLM 不可用时降级到风险等级预设池（保留 5-28 行为）"""
    pool = asset_pool.get_pool(risk_level)
    return {
        "tickers": list(pool["tickers"]),
        "names": dict(pool["names"]),
        "rationale": f"LLM 选品不可用，已降级为 {risk_level} 预设池。",
    }
