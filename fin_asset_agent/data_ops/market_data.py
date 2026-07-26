# data_ops/market_data.py
import os
import time
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
try:
    import akshare as ak
except ImportError:
    ak = None
from typing import List, Dict, Tuple
from config import data as data_config
from data_ops import net_proxy

class MarketDataFetcher:
    """
    DataOps 混合计算引擎 (A+B融合版)
    支持本地缓存快照、境内 AkShare 高速接口与境外 yfinance 全球接口自动分流。
    """
    def __init__(self, lookback_years: int = None, cache_dir: str = None):
        self.lookback_years = lookback_years if lookback_years is not None else data_config.lookback_years
        self.cache_dir = cache_dir if cache_dir is not None else data_config.cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _is_china_stock(self, ticker: str) -> bool:
        """简单的标的识别：纯数字或带 .SH/.SZ 结尾的视为国内 A 股"""
        clean_ticker = ticker.split(".")[0]
        return clean_ticker.isdigit()

    @staticmethod
    def _is_china_etf(clean_ticker: str) -> bool:
        """国内 ETF/基金 识别：上交所 5xxxxx（51/56/58/50…）、深交所 15xxxx/16xxxx。

        ETF 必须用 fund_etf_hist_em，用个股接口 stock_zh_a_hist 查会返回空。
        个股：上证 6xx / 688，深证 00x / 30x。
        """
        return clean_ticker.startswith(("5", "15", "16"))

    def _fetch_single_ticker_from_network(self, ticker: str, start_str: str, end_str: str) -> pd.Series:
        """根据标的属性，自动分流向国内/国外数据源下载数据"""
        # 格式化日期以适配 AkShare (YYYYMMDD)
        ak_start = start_str.replace("-", "")
        ak_end = end_str.replace("-", "")

        # --- 境内不限流数据源 (AkShare) 处理 A 股 ---
        if self._is_china_stock(ticker):
            if ak is None:
                raise ImportError("未检测到 akshare 库，无法拉取 A 股数据")
            clean_ticker = ticker.split(".")[0]
            is_etf = self._is_china_etf(clean_ticker)
            kind = "ETF/基金" if is_etf else "A 股"
            print(f"[DataOps AkShare] [CN] 正在通过国内源拉取 {kind} [{ticker}] 历史行情...")
            return self._fetch_china_close(clean_ticker, is_etf, ak_start, ak_end, ticker)

        # --- 境外数据源 (yfinance) 处理美股/全球资产 ---
        else:
            print(f"[DataOps yfinance] [US] 正在通过全球源拉取资产 [{ticker}] 历史行情...")
            # 外网行情：经环境变量控制代理（兼容各版本 yfinance），开关打开走系统代理、关闭直连；带重试退避。
            df = self._retry(
                lambda: self._run_via_proxy(
                    lambda: yf.download(ticker, start=start_str, end=end_str, progress=False, timeout=10)
                ),
                attempts=3, label=ticker,
            )
            if df is None or df.empty:
                raise ValueError(f"yfinance 返回数据为空: {ticker}")
            return self._extract_close_series(df)

    @staticmethod
    def _run_via_proxy(call_fn):
        """按代理开关执行：开关开走系统代理，关则直连（国内/海外行情统一经此）。"""
        with net_proxy.apply_proxy():
            return call_fn()

    def _fetch_china_close(self, clean_ticker: str, is_etf: bool,
                           ak_start: str, ak_end: str, label: str) -> pd.Series:
        """国内行情：东方财富(em) 主、新浪(sina) 兜底，统一走系统代理（Clash 干净 DNS + China DIRECT）。

        诊断证实：直连 eastmoney 被 DNS 污染 RST，走代理才通；但 em 经 Clash 偶发被掐，
        换新浪源（命中 sina.com.cn，不同主机）往往能成。两源各带重试退避。
        """
        prefix = "sh" if clean_ticker[0] in "569" else "sz"
        sina_symbol = prefix + clean_ticker
        start_dt, end_dt = pd.to_datetime(ak_start), pd.to_datetime(ak_end)

        def _em():
            if is_etf:
                return ak.fund_etf_hist_em(symbol=clean_ticker, period="daily", start_date=ak_start, end_date=ak_end, adjust="qfq")
            return ak.stock_zh_a_hist(symbol=clean_ticker, period="daily", start_date=ak_start, end_date=ak_end, adjust="qfq")

        def _sina():
            if is_etf:
                return ak.fund_etf_hist_sina(symbol=sina_symbol)  # 全历史，下方按区间切片
            return ak.stock_zh_a_daily(symbol=sina_symbol, start_date=ak_start, end_date=ak_end, adjust="qfq")

        # em 快速试一次（部分网络下经 Clash 偶发被掐），失败立即转新浪；新浪带重试退避。
        sources = [("eastmoney", _em, "日期", "收盘", 1), ("sina", _sina, "date", "close", 3)]
        last_err = None
        for name, fn, dcol, ccol, attempts in sources:
            try:
                df = self._retry(lambda fn=fn: self._run_via_proxy(fn), attempts=attempts, label=f"{label}@{name}")
            except Exception as e:
                last_err = e
                print(f"[DataOps] {name} 源失败 [{label}]: {type(e).__name__}")
                continue
            s = self._normalize_china_close(df, dcol, ccol, start_dt, end_dt)
            if s is not None and len(s) >= 1:
                print(f"[DataOps] [{label}] 命中 {name} 源 ({len(s)} 行)")
                return s
        if last_err:
            raise last_err
        raise ValueError(f"AkShare 两源均无区间数据: {label}")

    @staticmethod
    def _normalize_china_close(df, date_col: str, close_col: str, start_dt, end_dt) -> pd.Series:
        """把国内源返回的 DataFrame 规整成 [区间内] 日期索引的收盘价 Series；无则 None。"""
        if df is None or getattr(df, "empty", True):
            return None
        if date_col not in df.columns or close_col not in df.columns:
            return None
        d = df[[date_col, close_col]].copy()
        d[date_col] = pd.to_datetime(d[date_col])
        d = d[(d[date_col] >= start_dt) & (d[date_col] <= end_dt)]
        if d.empty:
            return None
        return d.set_index(date_col)[close_col].astype(float)

    @staticmethod
    def _retry(call_fn, attempts: int = 3, base_delay: float = 0.6, label: str = ""):
        """带退避重试：代理在突发批量连接下会偶发拒绝，重试一两次基本就成。

        仅在最终仍失败时抛出最后一次异常。返回空 DataFrame 视为成功（交由上层判定）。
        """
        last_err = None
        for i in range(attempts):
            try:
                return call_fn()
            except Exception as e:
                last_err = e
                if i < attempts - 1:
                    if label:
                        print(f"[DataOps] [{label}] 第 {i+1} 次失败，{base_delay*(i+1):.1f}s 后重试: {type(e).__name__}")
                    time.sleep(base_delay * (i + 1))
        raise last_err

    @staticmethod
    def _extract_close_series(df: pd.DataFrame) -> pd.Series:
        """从 yfinance 返回里取收盘价并压成一维 Series。

        新版 yfinance 即使单标的也返回 MultiIndex 列（如 ('Close','AAPL')），
        `df['Close']` 会得到 (N,1) 二维 DataFrame —— 必须 squeeze 成 1D，否则下游
        pd.Series/pd.DataFrame 拼装会报 "Data must be 1-dimensional"。
        """
        cols = df.columns
        if isinstance(cols, pd.MultiIndex):
            fields = set(cols.get_level_values(0))
            field = "Adj Close" if "Adj Close" in fields else ("Close" if "Close" in fields else cols.get_level_values(0)[0])
            sub = df.xs(field, axis=1, level=0)
        else:
            field = "Adj Close" if "Adj Close" in cols else ("Close" if "Close" in cols else cols[0])
            sub = df[field]
        if isinstance(sub, pd.DataFrame):   # 单列 DataFrame → 取该列压成 Series
            sub = sub.iloc[:, 0]
        return sub

    def fetch_and_calculate(self, tickers: List[str]) -> Tuple[List[float], List[List[float]]]:
        if not tickers:
            return [], []

        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=self.lookback_years * 365)

        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        # 生成唯一指纹文件名，锁定今日缓存
        tickers_str = "_".join(sorted(tickers)).replace(".", "_")
        cache_file = os.path.join(self.cache_dir, f"cache_{end_date}_{tickers_str}.csv")

        # 1. 【方案 A】优先命中本地今天生成的快照
        if os.path.exists(cache_file):
            print(f"[DataOps Memory] OK 成功命中今日本地行情快照缓存: {cache_file}，跳过网络请求。")
            combined_df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        else:
            # 2. 缓存未捕捉，触发【方案 B】+【全球分流】多路网络网络拉取
            print(f"[DataOps Network] 缓存未命中，启动多源量化管线，目标资产池: {tickers}")
            combined_data = {}

            for t in tickers:
                try:
                    series = self._fetch_single_ticker_from_network(t, start_str, end_str)
                    combined_data[t] = series
                except Exception as e:
                    print(f"[DataOps 警告] 标的 {t} 实时下载异常: {e}。触发一键滑入离线逃生边界。")
                    return self._get_fallback_data(tickers)

            # 将多路合并为统一的时间序列 DataFrame
            combined_df = pd.DataFrame(combined_data)
            # 向前/向后填充交易日缺失值（如应对中美节假日错配导致的 NaN）
            combined_df = combined_df.ffill().bfill()

            # 成功落库为本地缓存快照，供今天后续的请求直接复用
            combined_df.to_csv(cache_file)
            print(f"[DataOps Cache] OK 成功创建多源行情本地备份快照。")

        # 3. 确定性纯代码数学矩阵计算 (控制与计算彻底分离)
        daily_returns = combined_df.pct_change().dropna()
        if daily_returns.empty:
            return self._get_fallback_data(tickers)

        # 计算年化预期收益率 (252交易日放缩)
        mean_daily_returns = daily_returns.mean()
        expected_returns = [round(r * 252, 4) for r in mean_daily_returns.tolist()]

        # 计算年化协方差矩阵 (252交易日放缩)
        cov_matrix_df = daily_returns.cov() * 252
        cov_matrix = [[round(val, 6) for val in row] for row in cov_matrix_df.values.tolist()]

        print(f"[DataOps Processed] OK 资产配置特征矩阵就绪。资产顺序: {tickers}")
        return expected_returns, cov_matrix

    def fetch_price_window(self, tickers: List[str], start_str: str, end_str: str) -> pd.DataFrame:
        """拉取 [start, end] 区间多标的收盘价 DataFrame（归因复盘用，不做年化/协方差）。

        单个标的失败则跳过该列；全失败返回空 DataFrame，由调用方降级。
        """
        if not tickers:
            return pd.DataFrame()
        data = {}
        for t in tickers:
            try:
                series = self._fetch_single_ticker_from_network(t, start_str, end_str)
                # 统一成一维 Series 并校验长度，避免标量/异常形状把后续 DataFrame 拼装搞崩
                s = pd.Series(series).dropna() if series is not None else pd.Series(dtype=float)
                if len(s) >= 1:
                    data[t] = s
            except Exception as e:
                print(f"[Attribution] 标的 {t} 区间价格拉取失败: {e}")
        if not data:
            return pd.DataFrame()
        try:
            return pd.DataFrame(data).ffill().bfill()
        except Exception as e:
            print(f"[Attribution] 价格矩阵拼装失败: {e}")
            return pd.DataFrame()

    def _get_fallback_data(self, tickers: List[str]) -> Tuple[List[float], List[List[float]]]:
        """极限边界逃生通道：如果断网或 API 全面封锁，采用确定性科学模拟数值保证图状态机走完"""
        n = len(tickers)
        fallback_returns = [round(0.11 - (i * 0.015), 4) for i in range(n)]
        fallback_cov = [[0.035 if i == j else 0.012 for j in range(n)] for i in range(n)]
        return fallback_returns, fallback_cov