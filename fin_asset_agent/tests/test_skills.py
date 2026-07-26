"""Skill 系统测试 — base + 4 个具体 skill，全部 mock akshare"""
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from skills.base import BaseSkill, SkillRegistry, SkillResult
from skills import registry as _registry_singleton  # 触发自动注册并取出全局 registry 实例


# ── 基础协议测试 ──


def test_skill_result_ok_flag():
    ok = SkillResult("test", [], "")
    err = SkillResult("test", [], "", error="boom")
    assert ok.ok and not err.ok


def test_skill_result_to_rag_documents():
    r = SkillResult(
        "macro",
        items=[
            {"title": "CPI 2.3%", "content": "同比上涨", "date": "2026-05-28", "source": "AkShare"},
            {"title": "", "content": "", "date": "", "source": ""},  # 空记录应被过滤
        ],
        summary="",
    )
    docs = r.to_rag_documents()
    assert len(docs) == 1
    assert "CPI" in docs[0]["text"]
    assert "macro::" in docs[0]["source"]


def test_skill_registry_register_and_get():
    reg = SkillRegistry()

    class DummySkill(BaseSkill):
        name = "dummy"
        category = "test"

        def fetch(self, **p):
            return SkillResult(self.name, [], "ok")

    reg.register(DummySkill())
    assert reg.get("dummy") is not None
    assert reg.get("nonexistent") is None


def test_skill_registry_rejects_duplicate():
    reg = SkillRegistry()

    class S(BaseSkill):
        name = "x"
        def fetch(self, **p): return SkillResult(self.name, [], "")

    reg.register(S())
    with pytest.raises(ValueError):
        reg.register(S())


def test_skill_registry_by_category():
    reg = SkillRegistry()

    class A(BaseSkill):
        name = "a"; category = "macro"
        def fetch(self, **p): return SkillResult(self.name, [], "")

    class B(BaseSkill):
        name = "b"; category = "news"
        def fetch(self, **p): return SkillResult(self.name, [], "")

    reg.register(A()); reg.register(B())
    assert {s.name for s in reg.list_by_category("macro")} == {"a"}
    assert {s.name for s in reg.list_by_category("news")} == {"b"}


def test_skill_registry_fetch_many_handles_errors():
    reg = SkillRegistry()

    class Good(BaseSkill):
        name = "good"
        def fetch(self, **p): return SkillResult(self.name, [{"title": "ok"}], "")

    class Bad(BaseSkill):
        name = "bad"
        def fetch(self, **p): raise RuntimeError("oops")

    reg.register(Good()); reg.register(Bad())
    results = reg.fetch_many(["good", "bad", "missing"])
    assert results["good"].ok
    assert results["bad"].error == "oops"
    assert "未注册" in results["missing"].error


# ── 4 个真实 skill 注册 ──


def test_all_skills_registered():
    """4 个 skill 应在 import 时自动注册到全局 registry"""
    names = {s.name for s in _registry_singleton.list_all()}
    assert {"macro", "policy", "filing", "news"}.issubset(names)


def test_macro_skill_fetch(mocker):
    fake_df = pd.DataFrame({"月份": ["2026-04", "2026-05"], "同比": [2.1, 2.3]})
    with patch("skills.macro_skill.ak") as ak:
        # 给所有 macro_china_* 方法都 mock 一个 df
        ak.macro_china_cpi = MagicMock(return_value=fake_df)
        ak.macro_china_ppi = MagicMock(return_value=fake_df)
        ak.macro_china_pmi = MagicMock(return_value=fake_df)
        ak.macro_china_gdp = MagicMock(return_value=fake_df)
        ak.macro_china_lpr = MagicMock(return_value=fake_df)

        sk = _registry_singleton.get("macro")
        result = sk.fetch(indicators=["CPI", "PPI"], recent_n=2)

    assert result.ok
    assert len(result.items) >= 2  # 至少 CPI + PPI 各 1 条
    assert "CPI" in result.summary or "PPI" in result.summary


def test_policy_skill_keyword_filter():
    """policy_skill 应只保留含政策关键词的新闻"""
    fake_df = pd.DataFrame({
        "标题": ["央行降准 0.5%", "茅台业绩超预期", "证监会发布新规"],
        "发布时间": ["2026-05-28"] * 3,
    })
    with patch("skills.policy_skill.ak") as ak:
        ak.news_economic_baidu = MagicMock(return_value=fake_df)
        ak.news_cctv = MagicMock(return_value=pd.DataFrame())

        sk = _registry_singleton.get("policy")
        result = sk.fetch(limit=10)

    titles = [it["title"] for it in result.items]
    assert "央行降准 0.5%" in titles
    assert "证监会发布新规" in titles
    assert "茅台业绩超预期" not in titles  # 过滤掉


def test_filing_skill_requires_tickers():
    with patch("skills.filing_skill.ak"):
        sk = _registry_singleton.get("filing")
        result = sk.fetch(tickers=None)
    assert result.ok
    assert "未指定 ticker" in result.summary


def test_filing_skill_with_tickers(mocker):
    fake_df = pd.DataFrame({
        "公告标题": ["茅台一季报"],
        "公告日期": ["2026-04-28"],
        "公告链接": ["http://cninfo.cn/x"],
    })
    with patch("skills.filing_skill.ak") as ak:
        ak.stock_zh_a_disclosure_report_cninfo = MagicMock(return_value=fake_df)

        sk = _registry_singleton.get("filing")
        result = sk.fetch(tickers=["600519.SS"], limit_per_ticker=3)

    assert result.ok
    assert len(result.items) == 1
    assert result.items[0]["ticker"] == "600519.SS"


def test_news_skill_wraps_fetcher(mocker):
    mocker.patch(
        "skills.news_skill.news_fetcher.fetch_news_bundle",
        return_value={
            "by_ticker": {"600519.SS": [{"title": "T1", "date": "2026-05-28", "source": "EM", "url": "", "content": ""}]},
            "market_headlines": [{"title": "M1", "date": "2026-05-28", "source": "CCTV", "url": "", "content": ""}],
            "financial_news": [],
            "total_count": 2,
        },
    )
    mocker.patch(
        "skills.news_skill.news_fetcher.summarize_for_prompt",
        return_value="[600519.SS] T1\n[宏观] M1",
    )

    sk = _registry_singleton.get("news")
    result = sk.fetch(tickers=["600519.SS"])

    assert result.ok
    assert len(result.items) == 2
    assert "T1" in result.summary


def test_skill_relevance_filter():
    """is_relevant: CHIT_CHAT 时 filing/news 应过滤掉"""
    filing = _registry_singleton.get("filing")
    news = _registry_singleton.get("news")
    macro = _registry_singleton.get("macro")

    assert filing.is_relevant("ASSET_ALLOCATION", {}) is True
    assert filing.is_relevant("CHIT_CHAT", {}) is False
    assert news.is_relevant("CHIT_CHAT", {}) is False
    assert macro.is_relevant("CHIT_CHAT", {}) is True  # macro 默认全相关
