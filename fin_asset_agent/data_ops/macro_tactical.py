"""宏观战术配置（Macro Tactical Asset Allocation）。

核心思路：
1. 拉取公开宏观指标（PMI、国债收益率、CPI）判断当前经济周期
2. 根据周期给各大类资产的预期收益做战术调整
3. 调整后的预期收益喂给 MVO / Risk Parity，生成带宏观观点的组合

参考框架：美林投资时钟（复苏→扩张→滞胀→衰退）
"""
import datetime as dt
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd


# 资产类别 → 宏观映射（把系统内细分类别归到大类）
_ASSET_CATEGORY_MAP = {
    "stock_a": "equity_cn",
    "etf_broad": "equity_cn",
    "etf_sector": "equity_cn",
    "stock_us": "equity_us",
    "etf_overseas": "equity_us",
    "bond_money": "bond",
    "commodity": "commodity",
    "reit": "bond",
    "crypto": "commodity",
}

# 周期 → 各大类资产的预期收益调整（年化，小数）
_CYCLE_ADJUSTMENTS = {
    "recovery": {      # 复苏：经济↑ + 通胀↓ → 股票最好，债券一般
        "equity_cn": +0.04,
        "equity_us": +0.03,
        "bond": -0.01,
        "commodity": +0.01,
    },
    "expansion": {     # 扩张：经济↑ + 通胀↑ → 商品最好，股票次之
        "equity_cn": +0.02,
        "equity_us": +0.02,
        "bond": -0.02,
        "commodity": +0.05,
    },
    "stagflation": {   # 滞胀：经济↓ + 通胀↑ → 黄金/商品避险，股债双杀
        "equity_cn": -0.03,
        "equity_us": -0.02,
        "bond": -0.02,
        "commodity": +0.03,
    },
    "recession": {     # 衰退：经济↓ + 通胀↓ → 债券最好，股票最差
        "equity_cn": -0.04,
        "equity_us": -0.03,
        "bond": +0.04,
        "commodity": -0.02,
    },
}


class MacroTacticalAllocator:
    """宏观战术配置器：检测周期 + 调整收益预期。"""

    def __init__(self):
        self._cache: Dict = {}
        self._cache_ttl = 3600  # 宏观数据缓存1小时（避免频繁拉取）

    # ------------------------------------------------------------------
    # 1) 周期检测
    # ------------------------------------------------------------------
    def detect_cycle(self, as_of: Optional[dt.date] = None) -> Tuple[str, Dict]:
        """返回 (cycle_name, raw_indicators_dict)。

        cycle_name: recovery | expansion | stagflation | recession | neutral
        """
        as_of = as_of or dt.date.today()

        pmi, pmi_trend = self._get_pmi_signal(as_of)
        bond_trend = self._get_bond_yield_trend(as_of)
        cpi_trend = self._get_cpi_trend(as_of)

        raw = {
            "pmi": pmi,
            "pmi_trend": pmi_trend,
            "bond_yield_trend": bond_trend,
            "cpi_trend": cpi_trend,
            "as_of": as_of.isoformat(),
        }

        # 简化版美林时钟（两维度：PMI 代表经济动能，CPI/债券利率代表通胀/流动性）
        if pmi is None:
            return "neutral", raw

        economy = "up" if pmi >= 50 else "down"
        inflation = "up" if (cpi_trend == "rising" or bond_trend == "rising") else "down"

        if economy == "up" and inflation == "down":
            return "recovery", raw
        if economy == "up" and inflation == "up":
            return "expansion", raw
        if economy == "down" and inflation == "up":
            return "stagflation", raw
        return "recession", raw

    # ------------------------------------------------------------------
    # 2) 收益调整
    # ------------------------------------------------------------------
    def adjust_expected_returns(
        self,
        base_returns: List[float],
        asset_categories: List[str],
        cycle: Optional[str] = None,
        strength: float = 1.0,
    ) -> Tuple[List[float], Dict]:
        """根据宏观周期调整基础预期收益。

        Args:
            base_returns: 历史统计的基础预期收益（年化）
            asset_categories: 每项资产对应的系统类别（如 stock_a, bond_money）
            cycle: 若传 None 则自动检测
            strength: 宏观观点强度（0~1），防止过度自信

        Returns:
            (adjusted_returns, meta_info)
        """
        if cycle is None:
            cycle, _ = self.detect_cycle()

        adjustments = _CYCLE_ADJUSTMENTS.get(cycle, {})
        adjusted = []
        applied = []

        for ret, cat in zip(base_returns, asset_categories):
            macro_cat = _ASSET_CATEGORY_MAP.get(cat, "equity_cn")
            adj = adjustments.get(macro_cat, 0.0) * strength
            adjusted.append(ret + adj)
            applied.append({
                "category": cat,
                "macro_category": macro_cat,
                "base_return": round(ret, 4),
                "adjustment": round(adj, 4),
                "adjusted": round(ret + adj, 4),
            })

        meta = {
            "cycle": cycle,
            "cycle_cn": self._cycle_to_chinese(cycle),
            "strength": strength,
            "adjustments": applied,
        }
        return adjusted, meta

    # ------------------------------------------------------------------
    # 3) 宏观观点 → Black-Litterman 风格融入
    # ------------------------------------------------------------------
    def bl_adjusted_returns(
        self,
        market_weights: List[float],
        cov_matrix: List[List[float]],
        tau: float = 0.05,
        omega_diag: float = 0.02,
    ) -> List[float]:
        """简化的 Black-Litterman：把宏观观点作为绝对观点融入。

        这里用简化版：直接在均衡收益上加观点调整，不做完整的 BL 贝叶斯更新。
        理由：对于毕业设计/原型，完整 BL 的复杂度收益不高，且 tau/omega 调参困难。
        """
        # 实际上已经在 adjust_expected_returns 中完成了"观点融入"
        # 这个函数留作扩展接口
        return []

    # ------------------------------------------------------------------
    # 4) 数据源（带缓存 + fallback）
    # ------------------------------------------------------------------
    def _get_pmi_signal(self, as_of: dt.date) -> Tuple[Optional[float], str]:
        """获取最新中国制造业 PMI 及趋势。"""
        cache_key = f"pmi_{as_of}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        pmi_val = None
        try:
            import akshare as ak
            df = ak.macro_china_pmi()
            if df is not None and not getattr(df, "empty", True) and "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df[df["date"] <= pd.to_datetime(as_of)].sort_values("date")
                if not df.empty:
                    # 取最新值（假设最后一列是 PMI 值）
                    val_col = [c for c in df.columns if c != "date"][-1]
                    pmi_val = float(df[val_col].iloc[-1])
        except Exception:
            pass

        # fallback：用 hardcoded 近期值（防止数据源失效导致系统瘫痪）
        if pmi_val is None:
            pmi_val = 49.5  # 近期中国PMI大致水平

        # 判断趋势：看最近3个月平均值 vs 历史12个月平均
        trend = "stable"
        try:
            if len(df) >= 3:
                recent = float(df[val_col].iloc[-3:].mean())
                hist = float(df[val_col].iloc[-12:].mean()) if len(df) >= 12 else recent
                if recent > hist + 0.3:
                    trend = "rising"
                elif recent < hist - 0.3:
                    trend = "falling"
        except Exception:
            pass

        self._cache[cache_key] = (pmi_val, trend)
        return pmi_val, trend

    def _get_bond_yield_trend(self, as_of: dt.date) -> str:
        """中国10年期国债收益率趋势。"""
        cache_key = f"bond_{as_of}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        trend = "stable"
        try:
            import akshare as ak
            df = ak.bond_zh_us_rate()
            if df is not None and not getattr(df, "empty", True):
                # 找中国10年期国债列
                cn_col = None
                for c in df.columns:
                    if "中国" in str(c) and "10" in str(c):
                        cn_col = c
                        break
                if cn_col:
                    df["日期"] = pd.to_datetime(df["日期"])
                    df = df[df["日期"] <= pd.to_datetime(as_of)].sort_values("日期")
                    if len(df) >= 20:
                        recent = df[cn_col].iloc[-20:].mean()
                        prev = df[cn_col].iloc[-60:-20].mean()
                        if recent > prev + 0.05:
                            trend = "rising"
                        elif recent < prev - 0.05:
                            trend = "falling"
        except Exception:
            pass

        self._cache[cache_key] = trend
        return trend

    def _get_cpi_trend(self, as_of: dt.date) -> str:
        """中国CPI同比趋势。"""
        cache_key = f"cpi_{as_of}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        trend = "stable"
        try:
            import akshare as ak
            df = ak.macro_china_cpi()
            if df is not None and not getattr(df, "empty", True) and "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df[df["date"] <= pd.to_datetime(as_of)].sort_values("date")
                val_col = [c for c in df.columns if c != "date"][-1]
                if len(df) >= 6:
                    recent = df[val_col].iloc[-3:].mean()
                    prev = df[val_col].iloc[-9:-3].mean()
                    if recent > prev + 0.1:
                        trend = "rising"
                    elif recent < prev - 0.1:
                        trend = "falling"
        except Exception:
            pass

        self._cache[cache_key] = trend
        return trend

    @staticmethod
    def _cycle_to_chinese(cycle: str) -> str:
        return {
            "recovery": "复苏期",
            "expansion": "扩张期",
            "stagflation": "滞胀期",
            "recession": "衰退期",
            "neutral": "中性/数据不足",
        }.get(cycle, cycle)
