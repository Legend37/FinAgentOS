# skills/policy_skill.py
"""政策公告 skill — 央行 / 证监会 / 银保监会 等监管公告

数据源：AkShare 的 news_economic_baidu（财经热点，覆盖政策新闻）+ news_cctv（含政策报道）
"""
from __future__ import annotations
from .base import BaseSkill, SkillResult
from .registry import registry
from data_ops.news_fetcher import _normalize_date, _safe_str

try:
    import akshare as ak
except ImportError:
    ak = None


_POLICY_KEYWORDS = ("央行", "证监会", "银保监", "政策", "降准", "降息", "加息", "新规", "监管")


class PolicySkill(BaseSkill):
    name = "policy"
    description = "央行/证监会/银保监会等监管政策公告与新规"
    category = "policy"
    key_required = False

    def fetch(self, limit: int = 8, **_) -> SkillResult:
        if ak is None:
            return SkillResult(self.name, [], "", error="akshare 未安装")

        items = []
        # 财经热点（百度）— 政策相关性强
        try:
            df = ak.news_economic_baidu()
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    title = _safe_str(row.get("标题") or row.get("title") or "")
                    if any(k in title for k in _POLICY_KEYWORDS):
                        items.append({
                            "title": title,
                            "content": _safe_str(row.get("内容") or row.get("摘要") or ""),
                            "date": _normalize_date(row.get("发布时间") or row.get("时间") or ""),
                            "source": "百度财经",
                            "url": _safe_str(row.get("链接") or ""),
                        })
        except Exception as e:
            print(f"[policy_skill] news_economic_baidu 失败: {e}")

        # CCTV 新闻补充政策类
        try:
            df2 = ak.news_cctv()
            if df2 is not None and not df2.empty:
                for _, row in df2.iterrows():
                    title = _safe_str(row.get("title") or row.get("标题") or "")
                    if any(k in title for k in _POLICY_KEYWORDS):
                        items.append({
                            "title": title,
                            "content": _safe_str(row.get("content") or ""),
                            "date": _normalize_date(row.get("date") or row.get("日期") or ""),
                            "source": "央视新闻",
                            "url": "",
                        })
        except Exception as e:
            print(f"[policy_skill] news_cctv 失败: {e}")

        items = items[:limit]
        summary_lines = [f"[{it['source']} | {it['date']}] {it['title'][:60]}" for it in items]

        return SkillResult(
            skill_name=self.name,
            items=items,
            summary="\n".join(summary_lines) if summary_lines else "（暂无政策相关公告）",
            metadata={"matched_keywords": list(_POLICY_KEYWORDS)},
        )


registry.register(PolicySkill())
