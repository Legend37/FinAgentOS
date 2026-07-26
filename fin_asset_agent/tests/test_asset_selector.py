"""LLM 智能选品测试"""
import pytest
from unittest.mock import patch
from core_brain.agents.asset_selector import (
    llm_pick_assets, fallback_preset_pool, RISK_BAND_CAP,
)
from config import asset_universe


def test_universe_has_diverse_categories():
    """宇宙覆盖至少 7 大类，每类至少 3 个标的"""
    universe = asset_universe.get_all()
    cats = {a["category"] for a in universe}
    assert {"stock_a", "stock_us", "etf_broad", "etf_sector",
            "bond_money", "commodity", "crypto"}.issubset(cats)
    counts = {}
    for a in universe:
        counts[a["category"]] = counts.get(a["category"], 0) + 1
    for cat, c in counts.items():
        assert c >= 3, f"{cat} 只有 {c} 个标的"


def test_universe_size_substantial():
    """资产宇宙不应小于 40 个"""
    assert len(asset_universe.get_all()) >= 40


def test_universe_includes_crypto_proxies():
    """虚拟币类应有 BTC/ETH/ETF 代理"""
    crypto = asset_universe.filter_by_categories(["crypto"])
    tickers = [a["ticker"] for a in crypto]
    assert "BTC-USD" in tickers
    assert any(t in tickers for t in ("IBIT", "MSTR", "COIN"))


def test_universe_filter_by_risk_band():
    """保守型（max_band=2）应过滤掉所有 band 3-5 资产"""
    safe = asset_universe.filter_by_risk_band(2)
    assert all(a["risk_band"] <= 2 for a in safe)
    # 黄金 (band 3) 不应出现在保守名单
    assert all(a["ticker"] != "518880.SS" for a in safe)


def test_llm_pick_assets_returns_valid_tickers(mocker):
    """LLM 返回的 ticker 必须在 universe 中"""
    mocker.patch(
        "core_brain.agents.asset_selector.call_deepseek",
        return_value={
            "tickers": ["600519.SS", "BTC-USD", "GLD"],
            "rationale": "示例理由",
        },
    )
    profile = {"risk_tolerance_level": "进取型", "risk_score": 95}
    result = llm_pick_assets("sk-test", profile, "随意配", n=3)
    assert result["tickers"] == ["600519.SS", "BTC-USD", "GLD"]
    assert "贵州茅台" in result["names"]["600519.SS"]
    assert "比特币" in result["names"]["BTC-USD"]


def test_llm_pick_filters_hallucinated_tickers(mocker):
    """LLM 编造的 ticker 应被过滤，只保留真实存在的"""
    mocker.patch(
        "core_brain.agents.asset_selector.call_deepseek",
        return_value={
            "tickers": ["600519.SS", "FAKE.XX", "INVALID999"],
            "rationale": "测试",
        },
    )
    profile = {"risk_tolerance_level": "平衡型"}
    result = llm_pick_assets("sk-test", profile, "")
    assert result["tickers"] == ["600519.SS"]
    assert "FAKE.XX" not in result["names"]


def test_llm_pick_recovers_dropped_suffix(mocker):
    """关键修复：LLM 经常丢 .SS/.SZ 后缀，应自动补齐而不是误判为编造"""
    mocker.patch(
        "core_brain.agents.asset_selector.call_deepseek",
        return_value={
            # LLM 返回了纯数字 ticker（无后缀），这是真实在 universe 里的
            "tickers": ["513100", "510300", "511010", "518880", "515100", "159920"],
            "rationale": "宽基 + 黄金 + 国债 + 红利 + 恒生",
        },
    )
    profile = {"risk_tolerance_level": "平衡型", "risk_score": 60}
    result = llm_pick_assets("sk-test", profile, "我想多元配置")

    # 应全部解析成功，并补齐成 universe 里的标准格式
    assert "513100.SS" in result["tickers"]
    assert "510300.SS" in result["tickers"]
    assert "159920.SZ" in result["tickers"]
    assert len(result["tickers"]) == 6


def test_llm_pick_case_insensitive_crypto(mocker):
    """LLM 返回 'btc-usd' 小写应解析成 'BTC-USD'"""
    mocker.patch(
        "core_brain.agents.asset_selector.call_deepseek",
        return_value={
            "tickers": ["btc-usd", "eth-usd"],
            "rationale": "进取型加密配置",
        },
    )
    result = llm_pick_assets("sk-test", {"risk_tolerance_level": "进取型"}, "")
    assert "BTC-USD" in result["tickers"]
    assert "ETH-USD" in result["tickers"]


def test_llm_pick_dedupes_when_suffix_collision(mocker):
    """LLM 同时返回 '600519' 和 '600519.SS' 应去重"""
    mocker.patch(
        "core_brain.agents.asset_selector.call_deepseek",
        return_value={
            "tickers": ["600519", "600519.SS", "601398.SS"],
            "rationale": "测试去重",
        },
    )
    result = llm_pick_assets("sk-test", {"risk_tolerance_level": "平衡型"}, "")
    assert result["tickers"].count("600519.SS") == 1
    assert len(result["tickers"]) == 2


def test_resolve_ticker_handles_various_formats():
    """_resolve_ticker 独立单测"""
    from core_brain.agents.asset_selector import _resolve_ticker

    # 精确
    assert _resolve_ticker("600519.SS")["ticker"] == "600519.SS"
    # 后缀缺失
    assert _resolve_ticker("600519")["ticker"] == "600519.SS"
    # 大小写
    assert _resolve_ticker("aapl")["ticker"] == "AAPL"
    # 虚拟币
    assert _resolve_ticker("BTC-USD")["ticker"] == "BTC-USD"
    assert _resolve_ticker("btc-usd")["ticker"] == "BTC-USD"
    # 真编造
    assert _resolve_ticker("FAKE999") is None
    assert _resolve_ticker("") is None
    assert _resolve_ticker(None) is None


def test_llm_pick_raises_on_all_hallucinated(mocker):
    """LLM 全部编造 → 抛异常让 analyst 降级"""
    mocker.patch(
        "core_brain.agents.asset_selector.call_deepseek",
        return_value={"tickers": ["FAKE1", "FAKE2"], "rationale": "x"},
    )
    with pytest.raises(ValueError):
        llm_pick_assets("sk-test", {"risk_tolerance_level": "平衡型"}, "")


def test_llm_pick_respects_categories(mocker):
    """用户勾选房地产 → 提示词里只展示 reit 类资产"""
    captured = {}

    def fake_llm(api_key, prompt, **kw):
        captured["prompt"] = prompt
        return {"tickers": ["600048.SS"], "rationale": "选保利"}

    mocker.patch("core_brain.agents.asset_selector.call_deepseek", side_effect=fake_llm)
    llm_pick_assets("sk-test", {"risk_tolerance_level": "成长型"}, "",
                    categories=["reit"], n=3)
    # 茅台 (stock_a) 不应出现在 prompt
    assert "茅台" not in captured["prompt"]
    # 保利 (reit) 应出现
    assert "保利" in captured["prompt"]


def test_risk_band_cap_for_conservative(mocker):
    """保守型用户的 prompt 不应包含高 risk_band 的虚拟币"""
    captured = {}

    def fake_llm(api_key, prompt, **kw):
        captured["prompt"] = prompt
        return {"tickers": ["511010.SS"], "rationale": "保守"}

    mocker.patch("core_brain.agents.asset_selector.call_deepseek", side_effect=fake_llm)
    llm_pick_assets("sk-test", {"risk_tolerance_level": "保守型"}, "")

    # band 5 的虚拟币 / 创业板 / 芯片 ETF 都不应出现
    assert "BTC-USD" not in captured["prompt"]
    assert "比特币" not in captured["prompt"]
    assert "创业板" not in captured["prompt"]


def test_fallback_preset_pool():
    """降级路径：取回旧的风险等级预设池"""
    fb = fallback_preset_pool("进取型")
    assert len(fb["tickers"]) == 3
    assert "降级" in fb["rationale"]


def test_risk_band_cap_mapping():
    """5 档风险等级都应有 max_band 映射"""
    for level in ["保守型", "稳健型", "平衡型", "成长型", "进取型"]:
        assert level in RISK_BAND_CAP
        assert 1 <= RISK_BAND_CAP[level] <= 5
