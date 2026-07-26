# skills/macro_skill.py
"""宏观经济数据 skill — CPI / PPI / PMI / GDP / LPR 利率

数据源：AkShare 公开宏观接口，零授权
"""
from __future__ import annotations
from .base import BaseSkill, SkillResult
from .registry import registry

try:
    import akshare as ak
except ImportError:
    ak = None


_MACRO_INTERFACES = {
    "CPI": ("macro_china_cpi", "全国居民消费价格指数"),
    "PPI": ("macro_china_ppi", "工业生产者出厂价格指数"),
    "PMI": ("macro_china_pmi", "中国制造业 PMI"),
    "GDP": ("macro_china_gdp", "中国 GDP 同比"),
    "LPR": ("macro_china_lpr", "贷款市场报价利率 LPR"),
}


class MacroSkill(BaseSkill):
    name = "macro"
    description = "中国宏观经济指标（CPI/PPI/PMI/GDP/LPR）最新数据"
    category = "macro"
    key_required = False

    def fetch(self, indicators=None, recent_n: int = 3, **_) -> SkillResult:
        """抓取宏观指标。

        Args:
            indicators: list[str]，指定要拉的指标名（CPI/PPI/...），默认全部
            recent_n: 每个指标返回最近 N 条
        """
        if ak is None:
            return SkillResult(self.name, [], "", error="akshare 未安装")

        wanted = indicators or list(_MACRO_INTERFACES.keys())
        items = []
        summary_lines = []

        for key in wanted:
            iface, label = _MACRO_INTERFACES.get(key, (None, None))
            if iface is None:
                continue
            try:
                df = getattr(ak, iface)()
                if df is None or df.empty:
                    continue
                # 取最新 N 条（不同接口列名不一样，统一处理）
                tail = df.tail(recent_n)
                for _, row in tail.iterrows():
                    record = {col: str(row[col]) for col in df.columns}
                    record["title"] = f"{label} 最新数据"
                    record["content"] = " | ".join(f"{k}={v}" for k, v in record.items() if k not in ("title", "content"))
                    record["date"] = record.get("月份") or record.get("日期") or record.get("时间", "")
                    record["source"] = "AkShare 宏观"
                    record["indicator"] = key
                    items.append(record)
                # 摘要：取最后一条
                last = tail.iloc[-1]
                preview = " | ".join(f"{c}={last[c]}" for c in df.columns[:3])
                summary_lines.append(f"[{key}] {preview}")
            except Exception as e:
                summary_lines.append(f"[{key}] 抓取失败: {e}")

        return SkillResult(
            skill_name=self.name,
            items=items,
            summary="\n".join(summary_lines) if summary_lines else "（无宏观数据）",
            metadata={"indicators_requested": wanted, "items_total": len(items)},
        )


registry.register(MacroSkill())
