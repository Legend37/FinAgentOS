# skills/news_skill.py
"""新闻 skill — 把 data_ops/news_fetcher 封装进 skill 协议，便于与其他 skill 统一调度"""
from __future__ import annotations
from .base import BaseSkill, SkillResult
from .registry import registry
from data_ops import news_fetcher


class NewsSkill(BaseSkill):
    name = "news"
    description = "个股新闻 + 财经头条（东方财富/财联社/央视）"
    category = "news"
    key_required = False

    def is_relevant(self, intent: str, profile) -> bool:
        return intent in ("ASSET_ALLOCATION", "PORTFOLIO_REVIEW", "QA_RAG")

    def fetch(self, tickers=None, per_ticker: int = 3, headlines: int = 4,
              min_year: int = None, max_age_days: int = 90, **_) -> SkillResult:
        """抓取近期新闻（默认只取当年 + 90 天内）"""
        try:
            bundle = news_fetcher.fetch_news_bundle(
                tickers=tickers, per_ticker=per_ticker, headlines=headlines,
                min_year=min_year, max_age_days=max_age_days,
            )
        except Exception as e:
            return SkillResult(self.name, [], "（新闻服务暂不可用）", error=str(e))

        items = []
        for ticker, news_list in bundle.get("by_ticker", {}).items():
            for n in news_list:
                items.append({**n, "ticker": ticker})
        for n in bundle.get("financial_news", []):
            items.append({**n, "ticker": None})
        for n in bundle.get("market_headlines", []):
            items.append({**n, "ticker": None})

        summary = news_fetcher.summarize_for_prompt(bundle, max_items=12)
        return SkillResult(
            skill_name=self.name,
            items=items,
            summary=summary,
            metadata={"total": bundle.get("total_count", 0)},
        )


registry.register(NewsSkill())
