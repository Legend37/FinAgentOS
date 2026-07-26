# skills/filing_skill.py
"""公司公告 skill — 巨潮咨询的上市公司披露报告

数据源：AkShare 的 stock_zh_a_disclosure_report_cninfo（按 ticker 抓近期公告）
"""
from __future__ import annotations
from .base import BaseSkill, SkillResult
from .registry import registry
from data_ops.news_fetcher import _normalize_date, _safe_str

try:
    import akshare as ak
except ImportError:
    ak = None


class FilingSkill(BaseSkill):
    name = "filing"
    description = "上市公司公告披露（财报、重大事件、增减持等）— 巨潮咨询数据源"
    category = "filing"
    key_required = False

    def is_relevant(self, intent: str, profile) -> bool:
        # 闲聊和宏观问答不需要公司公告
        return intent in ("ASSET_ALLOCATION", "PORTFOLIO_REVIEW", "QA_RAG")

    def fetch(self, tickers=None, limit_per_ticker: int = 5, **_) -> SkillResult:
        """抓取指定 ticker 列表的最近公告"""
        if ak is None:
            return SkillResult(self.name, [], "", error="akshare 未安装")
        if not tickers:
            return SkillResult(self.name, [], "（未指定 ticker，跳过公司公告）",
                              metadata={"tickers": []})

        items = []
        for t in tickers[:10]:  # 限制并发量
            clean = t.split(".")[0]
            try:
                df = ak.stock_zh_a_disclosure_report_cninfo(
                    symbol=clean, market="沪深京", category="",
                    start_date="20260101", end_date="20261231",
                )
                if df is None or df.empty:
                    continue
                for _, row in df.head(limit_per_ticker).iterrows():
                    items.append({
                        "title": _safe_str(row.get("公告标题") or row.get("title") or ""),
                        "content": _safe_str(row.get("公告内容") or row.get("摘要") or ""),
                        "date": _normalize_date(row.get("公告日期") or row.get("date") or ""),
                        "source": f"巨潮咨询·{t}",
                        "url": _safe_str(row.get("公告链接") or ""),
                        "ticker": t,
                    })
            except Exception as e:
                print(f"[filing_skill] {t} 抓取失败: {e}")

        summary_lines = [f"[{it['ticker']} | {it['date']}] {it['title'][:60]}" for it in items[:8]]
        return SkillResult(
            skill_name=self.name,
            items=items,
            summary="\n".join(summary_lines) if summary_lines else "（无公司公告）",
            metadata={"tickers_processed": tickers},
        )


registry.register(FilingSkill())
