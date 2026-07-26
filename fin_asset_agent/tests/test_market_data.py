import pytest
import numpy as np
import pandas as pd
from data_ops.market_data import MarketDataFetcher


class TestExtractCloseSeries:
    """收盘价抽取 — 应对新版 yfinance MultiIndex 列，统一压成一维 Series。"""

    def test_multiindex_single_ticker_squeezed_to_1d(self):
        # 新版 yfinance 单标的也返回 ('字段','ticker') 双层列
        idx = pd.date_range("2025-01-02", periods=3)
        cols = pd.MultiIndex.from_tuples([("Close", "AAPL"), ("High", "AAPL"), ("Low", "AAPL")])
        df = pd.DataFrame([[10, 11, 9], [11, 12, 10], [12, 13, 11]], index=idx, columns=cols)
        s = MarketDataFetcher._extract_close_series(df)
        assert isinstance(s, pd.Series) and s.ndim == 1
        assert list(s.values) == [10, 11, 12]

    def test_multiindex_prefers_adj_close(self):
        idx = pd.date_range("2025-01-02", periods=2)
        cols = pd.MultiIndex.from_tuples([("Close", "KO"), ("Adj Close", "KO")])
        df = pd.DataFrame([[60, 59], [61, 60]], index=idx, columns=cols)
        s = MarketDataFetcher._extract_close_series(df)
        assert list(s.values) == [59, 60]   # 取 Adj Close 列

    def test_flat_columns_close(self):
        idx = pd.date_range("2025-01-02", periods=2)
        df = pd.DataFrame({"Open": [1, 2], "Close": [10, 11]}, index=idx)
        s = MarketDataFetcher._extract_close_series(df)
        assert isinstance(s, pd.Series) and list(s.values) == [10, 11]


class TestChinaStockDetection:
    """标的识别 — 纯本地逻辑，不触发网络"""

    def test_a_share_shanghai(self):
        fetcher = MarketDataFetcher()
        assert fetcher._is_china_stock("601899.SS") is True

    def test_a_share_shenzhen(self):
        fetcher = MarketDataFetcher()
        assert fetcher._is_china_stock("000858.SZ") is True

    def test_a_share_pure_digit(self):
        fetcher = MarketDataFetcher()
        assert fetcher._is_china_stock("600519") is True

    def test_us_stock(self):
        fetcher = MarketDataFetcher()
        assert fetcher._is_china_stock("AAPL") is False

    def test_us_stock_with_exchange(self):
        fetcher = MarketDataFetcher()
        assert fetcher._is_china_stock("TSLA.OQ") is False


class TestChinaETFDetection:
    """ETF/基金 vs 个股 — 决定走 fund_etf_hist_em 还是 stock_zh_a_hist。"""

    @pytest.mark.parametrize("code", ["510300", "512800", "518880", "511010", "515100", "513500"])
    def test_shanghai_etf(self, code):
        assert MarketDataFetcher._is_china_etf(code) is True

    @pytest.mark.parametrize("code", ["159915", "159995", "169101"])
    def test_shenzhen_etf(self, code):
        assert MarketDataFetcher._is_china_etf(code) is True

    @pytest.mark.parametrize("code", ["600519", "601398", "688981", "000858", "300750", "002594"])
    def test_individual_stocks_not_etf(self, code):
        assert MarketDataFetcher._is_china_etf(code) is False


class TestNormalizeChinaClose:
    """国内源（东财日期/收盘、新浪 date/close）规整成区间内日期索引收盘 Series。"""

    def test_eastmoney_columns_filtered_by_range(self):
        df = pd.DataFrame({
            "日期": ["2025-01-01", "2025-01-15", "2025-03-01"],
            "收盘": [10.0, 11.0, 12.0],
        })
        s = MarketDataFetcher._normalize_china_close(
            df, "日期", "收盘", pd.to_datetime("20250101"), pd.to_datetime("20250201"))
        assert list(s.values) == [10.0, 11.0]   # 3 月那条被区间过滤掉

    def test_sina_columns(self):
        df = pd.DataFrame({"date": ["2025-01-10"], "close": [3.9]})
        s = MarketDataFetcher._normalize_china_close(
            df, "date", "close", pd.to_datetime("20250101"), pd.to_datetime("20250201"))
        assert isinstance(s, pd.Series) and float(s.iloc[0]) == 3.9

    def test_missing_columns_returns_none(self):
        df = pd.DataFrame({"foo": [1]})
        assert MarketDataFetcher._normalize_china_close(
            df, "日期", "收盘", pd.to_datetime("20250101"), pd.to_datetime("20250201")) is None

    def test_empty_returns_none(self):
        assert MarketDataFetcher._normalize_china_close(
            pd.DataFrame(), "日期", "收盘", pd.to_datetime("20250101"), pd.to_datetime("20250201")) is None


class TestChinaDualSourceFallback:
    """东财失败 → 自动落到新浪源。"""

    def test_falls_back_to_sina_when_eastmoney_fails(self, mocker):
        f = MarketDataFetcher()
        # 让「走代理执行」直接执行被包的函数（跳过真实网络上下文）
        mocker.patch.object(MarketDataFetcher, "_run_via_proxy", staticmethod(lambda fn: fn()))
        mocker.patch("data_ops.market_data.ak.fund_etf_hist_em", side_effect=Exception("ProxyError"))
        sina_df = pd.DataFrame({"date": ["2025-01-10", "2025-01-11"], "close": [3.9, 3.95]})
        mocker.patch("data_ops.market_data.ak.fund_etf_hist_sina", return_value=sina_df)

        s = f._fetch_china_close("510300", True, "20250101", "20250201", "510300.SS")
        assert isinstance(s, pd.Series) and len(s) == 2 and float(s.iloc[-1]) == 3.95


class TestFallbackChannel:
    """逃生通道 — 断网时保证状态机能走完"""

    def test_fallback_returns_count_matches_tickers(self):
        fetcher = MarketDataFetcher()
        tickers = ["601899.SS", "000858.SZ", "601398.SS"]
        returns, cov = fetcher._get_fallback_data(tickers)
        assert len(returns) == 3

    def test_fallback_cov_dimensions(self):
        fetcher = MarketDataFetcher()
        tickers = ["A", "B"]
        returns, cov = fetcher._get_fallback_data(tickers)
        assert len(cov) == 2
        assert all(len(row) == 2 for row in cov)

    def test_fallback_cov_is_symmetric(self):
        fetcher = MarketDataFetcher()
        _, cov = fetcher._get_fallback_data(["X", "Y", "Z"])
        arr = np.array(cov)
        assert np.allclose(arr, arr.T)

    def test_fallback_empty_tickers(self):
        fetcher = MarketDataFetcher()
        returns, cov = fetcher._get_fallback_data([])
        assert returns == []
        assert cov == []


class TestCachePath:
    """缓存路径逻辑"""

    def test_cache_dir_creation(self, tmp_path):
        cache_dir = tmp_path / "test_cache"
        MarketDataFetcher(cache_dir=str(cache_dir))
        assert cache_dir.exists()
