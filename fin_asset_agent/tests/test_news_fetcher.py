"""新闻抓取层测试 — 全部 mock akshare，零网络"""
import pandas as pd
import pytest
from unittest.mock import patch

from data_ops import news_fetcher as nf


@pytest.fixture
def mock_ak_news():
    df = pd.DataFrame({
        "新闻标题": ["业绩超预期", "新厂投产"],
        "新闻内容": ["公司一季度业绩...", "新能源厂投产..."],
        "发布时间": ["2026-05-28 10:00:00", "2026-05-27 09:00:00"],
        "新闻链接": ["http://eastmoney.com/n1", "http://eastmoney.com/n2"],
    })
    with patch("data_ops.news_fetcher.ak") as ak:
        ak.stock_news_em.return_value = df
        yield ak


def test_fetch_stock_news_basic(mock_ak_news):
    news = nf.fetch_stock_news("600519.SS", limit=5)
    assert len(news) == 2
    assert news[0]["title"] == "业绩超预期"
    assert news[0]["source"] == "东方财富"
    assert news[0]["date"] == "2026-05-28"
    assert "eastmoney" in news[0]["url"]


def test_fetch_stock_news_strips_ticker_suffix(mock_ak_news):
    nf.fetch_stock_news("600519.SS")
    # akshare 应该收到 clean 后的 ticker
    mock_ak_news.stock_news_em.assert_called_with(symbol="600519")


def test_fetch_stock_news_handles_failure():
    with patch("data_ops.news_fetcher.ak") as ak:
        ak.stock_news_em.side_effect = Exception("network down")
        news = nf.fetch_stock_news("600519.SS")
        assert news == []


def test_fetch_market_headlines():
    df = pd.DataFrame({
        "标题": ["东方财富财经早餐 5月29日周五", "东方财富财经早餐 5月28日周四"],
        "摘要": ["...", "..."],
        "发布时间": ["2026-05-29 06:00:32", "2026-05-28 06:00:36"],
        "链接": ["url1", "url2"],
    })
    with patch("data_ops.news_fetcher.ak") as ak:
        ak.stock_info_cjzc_em.return_value = df
        news = nf.fetch_market_headlines(limit=10)
    assert len(news) == 2
    assert news[0]["source"] == "东方财富·财经早餐"
    assert news[0]["date"] == "2026-05-29"


def test_fetch_news_bundle_combines_sources():
    stock_df = pd.DataFrame({
        "新闻标题": ["业绩超预期"], "新闻内容": ["..."],
        "发布时间": ["2026-05-28"], "新闻链接": ["url"],
    })
    macro_df = pd.DataFrame({
        "标题": ["东方财富财经早餐 5月29日"], "摘要": ["..."],
        "发布时间": ["2026-05-28"], "链接": ["url"],
    })
    fin_df = pd.DataFrame({
        "tag": ["主流"], "summary": ["资金面宽松"], "url": ["http://example.com/20260528/article.html"],
    })

    with patch("data_ops.news_fetcher.ak") as ak:
        ak.stock_news_em.return_value = stock_df
        ak.stock_info_cjzc_em.return_value = macro_df
        ak.stock_news_main_cx.return_value = fin_df

        bundle = nf.fetch_news_bundle(tickers=["600519.SS"], per_ticker=3, headlines=3)

    assert "600519.SS" in bundle["by_ticker"]
    assert len(bundle["market_headlines"]) == 1
    assert len(bundle["financial_news"]) == 1
    assert bundle["total_count"] == 3


def test_summarize_for_prompt():
    bundle = {
        "by_ticker": {"600519.SS": [
            {"title": "业绩超预期", "date": "2026-05-28", "content": "...", "source": "东方财富", "url": ""},
        ]},
        "market_headlines": [
            {"title": "央行降准", "date": "2026-05-28", "content": "", "source": "央视", "url": ""},
        ],
        "financial_news": [],
    }
    summary = nf.summarize_for_prompt(bundle, max_items=10)
    assert "业绩超预期" in summary
    assert "央行降准" in summary
    assert "600519.SS" in summary


def test_summarize_for_prompt_respects_max_items():
    bundle = {
        "by_ticker": {},
        "market_headlines": [{"title": f"news{i}", "date": "2026-05-28"} for i in range(20)],
        "financial_news": [],
    }
    s = nf.summarize_for_prompt(bundle, max_items=5)
    assert s.count("news") == 5


def test_summarize_empty_bundle():
    s = nf.summarize_for_prompt({"by_ticker": {}, "market_headlines": [], "financial_news": []})
    assert "暂无" in s


def test_normalize_date_formats():
    assert nf._normalize_date("2026-05-28 10:00:00") == "2026-05-28"
    assert nf._normalize_date("20260528") == "2026-05-28"
    assert nf._normalize_date("2026/05/28") == "2026-05-28"
    assert nf._normalize_date(None) == ""
    assert nf._normalize_date("") == ""


def test_fetch_news_bundle_no_tickers():
    """没指定 tickers 时只拉宏观/财经"""
    macro_df = pd.DataFrame({
        "标题": ["东方财富财经早餐 5月29日"], "摘要": ["..."],
        "发布时间": ["2026-05-28"], "链接": ["url"],
    })
    with patch("data_ops.news_fetcher.ak") as ak:
        ak.stock_info_cjzc_em.return_value = macro_df
        ak.stock_news_main_cx.return_value = pd.DataFrame()

        bundle = nf.fetch_news_bundle(tickers=None)

    assert bundle["by_ticker"] == {}
    assert len(bundle["market_headlines"]) == 1


# ── 新闻新近度过滤（2026 优先）──


def test_filter_drops_old_years():
    """min_year=2026 时 2024 / 2025 应被过滤掉"""
    items = [
        {"title": "新", "date": "2026-05-28"},
        {"title": "旧", "date": "2024-05-28"},
        {"title": "更旧", "date": "2023-01-01"},
    ]
    out = nf._filter_and_sort_by_recency(items, min_year=2026, max_age_days=None)
    assert len(out) == 1
    assert out[0]["title"] == "新"


def test_filter_sorts_recent_first():
    """日期降序：最近的在前面"""
    items = [
        {"title": "5月", "date": "2026-05-01"},
        {"title": "5月下", "date": "2026-05-28"},
        {"title": "4月", "date": "2026-04-15"},
    ]
    out = nf._filter_and_sort_by_recency(items, min_year=2026, max_age_days=None)
    assert [it["title"] for it in out] == ["5月下", "5月", "4月"]


def test_filter_max_age_days():
    """max_age_days=7 应过滤 1 个月前的"""
    today = pd.Timestamp.today().normalize()
    items = [
        {"title": "今日", "date": today.strftime("%Y-%m-%d")},
        {"title": "3 天前", "date": (today - pd.Timedelta(days=3)).strftime("%Y-%m-%d")},
        {"title": "30 天前", "date": (today - pd.Timedelta(days=30)).strftime("%Y-%m-%d")},
    ]
    out = nf._filter_and_sort_by_recency(items, min_year=2020, max_age_days=7)
    titles = [it["title"] for it in out]
    assert "今日" in titles
    assert "3 天前" in titles
    assert "30 天前" not in titles


def test_filter_drops_empty_dates():
    """空日期/无效日期的条目直接丢，不带进 prompt"""
    items = [
        {"title": "正常", "date": "2026-05-28"},
        {"title": "空", "date": ""},
        {"title": "脏", "date": "not-a-date"},
    ]
    out = nf._filter_and_sort_by_recency(items, min_year=2026, max_age_days=None)
    assert len(out) == 1
    assert out[0]["title"] == "正常"


def test_summary_adds_freshness_tags():
    """summarize_for_prompt 给当日/本周/本月的新闻打标记"""
    today = pd.Timestamp.today().normalize()
    bundle = {
        "by_ticker": {},
        "market_headlines": [
            {"title": "今日", "date": today.strftime("%Y-%m-%d")},
            {"title": "本周内", "date": (today - pd.Timedelta(days=5)).strftime("%Y-%m-%d")},
            {"title": "本月内", "date": (today - pd.Timedelta(days=20)).strftime("%Y-%m-%d")},
        ],
        "financial_news": [],
    }
    summary = nf.summarize_for_prompt(bundle, max_items=10)
    assert "🔥" in summary
    assert "🆕" in summary
    assert "📅" in summary


def test_fetch_bundle_filters_old_data():
    """fetch_news_bundle 默认 min_year=今年，混入老新闻应被过滤"""
    old_and_new = pd.DataFrame({
        "新闻标题": ["新业绩", "陈年公告"],
        "新闻内容": ["...", "..."],
        "发布时间": ["2026-05-28", "2023-01-15"],
        "新闻链接": ["url1", "url2"],
    })
    with patch("data_ops.news_fetcher.ak") as ak:
        ak.stock_news_em.return_value = old_and_new
        ak.news_cctv.return_value = pd.DataFrame()
        ak.stock_news_main_cx.return_value = pd.DataFrame()

        bundle = nf.fetch_news_bundle(tickers=["600519.SS"], per_ticker=5)

    titles = [n["title"] for n in bundle["by_ticker"].get("600519.SS", [])]
    assert "新业绩" in titles
    assert "陈年公告" not in titles


# ── Cascade 降级（数据源返回旧数据时） ──


def test_cascade_falls_back_when_strict_empty():
    """当年没新闻 → 应自动降级到去年，不返回空"""
    # 全是 2024 的新闻（早于 min_year=2026 严格过滤）
    items = [
        {"title": "去年 1", "date": "2024-05-28"},
        {"title": "去年 2", "date": "2024-06-15"},
        {"title": "去年 3", "date": "2024-07-01"},
    ]
    out = nf._cascade_filter(items, min_year=2026, max_age_days=90, limit=5)
    # Tier 1 严格过滤会空，Tier 2 放宽到 2025 也空，Tier 3 拿最新的
    assert len(out) == 3  # 都该被拿到
    # 排序后最新的在前
    assert out[0]["date"] == "2024-07-01"


def test_cascade_strict_layer_wins_when_fresh_available():
    """当年有新闻 → 严格层就够用，不触发降级"""
    items = [
        {"title": "新", "date": "2026-05-28"},
        {"title": "老", "date": "2024-05-28"},
    ]
    out = nf._cascade_filter(items, min_year=2026, max_age_days=365, limit=5)
    assert len(out) == 1
    assert out[0]["title"] == "新"


def test_cascade_empty_raw():
    """raw 完全为空 → 直接返回空"""
    out = nf._cascade_filter([], min_year=2026, max_age_days=90, limit=5)
    assert out == []


# ── 财经早餐主源（替换死掉的 news_cctv） ──


def test_fetch_eastmoney_morning_real_schema():
    """真实 stock_info_cjzc_em 列名: 标题/摘要/发布时间/链接"""
    df = pd.DataFrame({
        "标题": ["东方财富财经早餐 5月29日周五", "东方财富财经早餐 5月28日周四"],
        "摘要": ["1、美伊谈判 2、国务院印发...", "1、李强讲话 2、长鑫科技..."],
        "发布时间": ["2026-05-29 06:00:32", "2026-05-28 06:00:36"],
        "链接": ["http://finance.eastmoney.com/a/202605283753113416.html",
                 "http://finance.eastmoney.com/a/202605273751636449.html"],
    })
    with patch("data_ops.news_fetcher.ak") as ak:
        ak.stock_info_cjzc_em.return_value = df
        items = nf.fetch_eastmoney_morning(limit=5)

    assert len(items) == 2
    assert items[0]["date"] == "2026-05-29"
    assert items[0]["source"] == "东方财富·财经早餐"
    assert "美伊" in items[0]["content"]


def test_market_headlines_falls_back_to_baidu():
    """财经早餐源空 → 降级到百度经济热点"""
    baidu_df = pd.DataFrame({
        "标题": ["央行降准"], "摘要": ["..."], "发布时间": ["2025-11-26"],
    })
    with patch("data_ops.news_fetcher.ak") as ak:
        ak.stock_info_cjzc_em.return_value = pd.DataFrame()  # 主源空
        ak.news_economic_baidu.return_value = baidu_df

        items = nf.fetch_market_headlines(limit=5)

    assert len(items) == 1
    assert items[0]["source"] == "百度财经"


def test_infer_date_from_url():
    """财联社 URL 嵌的 20260528 → '2026-05-28'"""
    assert nf._infer_date_from_url("http://eastmoney.com/a/202605283751.html") == "2026-05-28"
    assert nf._infer_date_from_url("http://example.com/2026/05/28/article") == "2026-05-28"
    assert nf._infer_date_from_url("") == ""
    assert nf._infer_date_from_url("no-date-in-url") == ""


def test_financial_news_infers_date_from_url():
    """财联社主源无日期列 → 从 url 推断"""
    df = pd.DataFrame({
        "tag": ["主流"], "summary": ["资金面宽松"],
        "url": ["http://example.com/20260528/article.html"],
    })
    with patch("data_ops.news_fetcher.ak") as ak:
        ak.stock_news_main_cx.return_value = df
        items = nf.fetch_financial_news(limit=5)

    assert len(items) == 1
    assert items[0]["date"] == "2026-05-28"


def test_bundle_uses_cascade_for_macro():
    """实战：宏观新闻全是 2024 → cascade 应放行，bundle 不为 0 条"""
    old_macro = pd.DataFrame({
        "标题": [f"2024 头条 {i}" for i in range(5)],
        "摘要": ["..."] * 5,
        "发布时间": ["2024-04-24"] * 5,
        "链接": ["url"] * 5,
    })
    with patch("data_ops.news_fetcher.ak") as ak:
        ak.stock_info_cjzc_em.return_value = old_macro
        ak.news_economic_baidu.return_value = pd.DataFrame()
        ak.stock_news_main_cx.return_value = pd.DataFrame()

        bundle = nf.fetch_news_bundle(tickers=None, headlines=3)

    assert len(bundle["market_headlines"]) >= 1  # cascade 兜底，不为空
    assert bundle["total_count"] >= 1
