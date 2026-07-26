# data_ops/news_fetcher.py
"""新闻抓取层 — 把 AkShare 的多源新闻接口统一封装成 {title, content, source, date, url} 结构。

零授权（AkShare 内置公开数据源），多源融合，优先级 = 数据新近度：
- 东方财富财经早餐 (stock_info_cjzc_em)   — ✅ 主源，每日一份，覆盖到当日（2026-05-29 实测）
- 百度经济热点    (news_economic_baidu)   — ✅ 政策与宏观，约近半年
- 财联社主流      (stock_news_main_cx)    — ✅ 实时财经，但无日期，URL 推断
- 个股新闻        (stock_news_em)         — ⚠️ AkShare 库有 regex bug，常失败
- ❌ 弃用 news_cctv：数据冻在 2024-04-24

所有调用都做了网络降级保护：网络异常 → 返回空 list，不让上游 workflow 崩。
"""
from __future__ import annotations
import datetime as dt
from typing import List, Dict, Optional

try:
    import akshare as ak
except ImportError:
    ak = None

import pandas as pd


def _safe_str(v) -> str:
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


def _normalize_date(v) -> str:
    """统一日期为 YYYY-MM-DD"""
    if v is None or v == "":
        return ""
    if isinstance(v, (dt.datetime, dt.date)):
        return v.strftime("%Y-%m-%d")
    s = str(v).strip()
    # 常见格式：'2026-05-28 10:00:00' / '2026-05-28' / '20260528' / '2026/05/28'
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d", "%Y%m%d"):
        try:
            return dt.datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    # 截断后再试一次（处理结尾有多余字符的情况）
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return dt.datetime.strptime(s[:10], fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    return s[:10]


def fetch_stock_news(ticker: str, limit: int = 10) -> List[Dict]:
    """抓个股新闻（东方财富数据源）"""
    if ak is None:
        return []
    clean = ticker.split(".")[0]  # 600519.SS → 600519
    try:
        df = ak.stock_news_em(symbol=clean)
    except Exception as e:
        print(f"[news_fetcher] stock_news_em({clean}) 失败: {e}")
        return []
    return _df_to_news(df, source="东方财富", default_url_col="新闻链接")[:limit]


def fetch_eastmoney_morning(limit: int = 10) -> List[Dict]:
    """✅ 主源：东方财富财经早餐，每日 06:00 一份，2026 实时有数据"""
    if ak is None:
        return []
    try:
        df = ak.stock_info_cjzc_em()
    except Exception as e:
        print(f"[news_fetcher] stock_info_cjzc_em 失败: {e}")
        return []
    return _df_to_news(df, source="东方财富·财经早餐", default_url_col="链接")[:limit]


def fetch_baidu_economic(limit: int = 10) -> List[Dict]:
    """✅ 政策/宏观源：百度经济热点"""
    if ak is None:
        return []
    try:
        df = ak.news_economic_baidu()
    except Exception as e:
        print(f"[news_fetcher] news_economic_baidu 失败: {e}")
        return []
    return _df_to_news(df, source="百度财经", default_url_col=None)[:limit]


def fetch_market_headlines(limit: int = 10) -> List[Dict]:
    """宏观财经头条 — 优先用财经早餐（活源），降级到百度经济热点

    旧的 news_cctv 已弃用（数据冻在 2024-04-24）
    """
    items = fetch_eastmoney_morning(limit=limit)
    if items:
        return items
    print(f"[news_fetcher] 财经早餐源空，回退到百度经济热点")
    return fetch_baidu_economic(limit=limit)


def _infer_date_from_url(url: str) -> str:
    """从 URL 推断日期（财联社/东财链接里嵌的 YYYYMMDD）"""
    import re
    if not url:
        return ""
    m = re.search(r'(20\d{2})[/\-_]?(\d{2})[/\-_]?(\d{2})', url)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return ""


def fetch_financial_news(limit: int = 10) -> List[Dict]:
    """抓财联社主流财经新闻 — 该源列: tag/summary/url，无日期，summary 兼标题"""
    if ak is None:
        return []
    try:
        df = ak.stock_news_main_cx()
    except Exception as e:
        print(f"[news_fetcher] stock_news_main_cx 失败: {e}")
        return []
    if df is None or df.empty:
        return []

    cols = list(df.columns)
    summary_col = _pick(cols, ["summary", "摘要", "内容", "content"])
    url_col = _pick(cols, ["url", "新闻链接", "链接"])
    tag_col = _pick(cols, ["tag", "标签", "category"])

    items = []
    for _, row in df.iterrows():
        title = _safe_str(row.get(summary_col, "")) if summary_col else ""
        if not title.strip():
            continue
        url = _safe_str(row.get(url_col, "")) if url_col else ""
        items.append({
            "title": title[:120],  # summary 通常较短可直接当 title
            "content": title,
            "source": "财联社",
            "date": _infer_date_from_url(url),
            "url": url,
            "tag": _safe_str(row.get(tag_col, "")) if tag_col else "",
        })
        if len(items) >= limit:
            break
    return items


def _df_to_news(df: pd.DataFrame, source: str, default_url_col: Optional[str]) -> List[Dict]:
    """把 AkShare DataFrame 转成统一 schema"""
    if df is None or df.empty:
        return []
    cols = list(df.columns)
    # 智能匹配列名（AkShare 中英文列名有差异）
    title_col = _pick(cols, ["新闻标题", "标题", "title"])
    content_col = _pick(cols, ["新闻内容", "内容", "摘要", "content"])
    date_col = _pick(cols, ["发布时间", "时间", "日期", "date", "datetime"])
    url_col = default_url_col if default_url_col in cols else _pick(cols, ["新闻链接", "链接", "url"])

    items = []
    for _, row in df.iterrows():
        title = _safe_str(row.get(title_col, "")) if title_col else ""
        if not title.strip():
            continue  # 跳过空标题（部分源会返回脏行）
        items.append({
            "title": title,
            "content": _safe_str(row.get(content_col, "")) if content_col else "",
            "source": source,
            "date": _normalize_date(row.get(date_col, "")) if date_col else "",
            "url": _safe_str(row.get(url_col, "")) if url_col else "",
        })
    return items


def _pick(cols: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    # 模糊匹配
    for col in cols:
        for c in candidates:
            if c in col:
                return col
    return None


def _date_score(date_str: str) -> int:
    """把日期字符串转成可排序的整数（YYYYMMDD），无效日期 → 0（沉到底）"""
    if not date_str:
        return 0
    try:
        return int(date_str.replace("-", "").replace("/", "")[:8])
    except (ValueError, TypeError):
        return 0


def _filter_and_sort_by_recency(items: List[Dict],
                                 min_year: int = None,
                                 max_age_days: int = None,
                                 limit: int = None) -> List[Dict]:
    """过滤过老的新闻 + 按日期降序，最新的在前。

    Args:
        min_year: 只保留该年（含）之后的新闻，默认 = 当前年
        max_age_days: 只保留 N 天内的新闻；None 不限
        limit: 排序后截取前 N 条
    """
    today = dt.date.today()
    if min_year is None:
        min_year = today.year

    cutoff_score = min_year * 10_000  # YYYY0000
    age_cutoff = None
    if max_age_days is not None:
        age_cutoff_date = today - dt.timedelta(days=max_age_days)
        age_cutoff = _date_score(age_cutoff_date.isoformat())

    def _keep(item):
        score = _date_score(item.get("date", ""))
        if score == 0:
            return False  # 无日期的丢掉，保证 prompt 不被脏数据污染
        if score < cutoff_score:
            return False
        if age_cutoff is not None and score < age_cutoff:
            return False
        return True

    fresh = [it for it in items if _keep(it)]
    fresh.sort(key=lambda it: _date_score(it.get("date", "")), reverse=True)
    return fresh[:limit] if limit else fresh


def _cascade_filter(raw: List[Dict], min_year: int, max_age_days: int, limit: int) -> List[Dict]:
    """优雅降级过滤：先严格 → 放宽到去年 → 拿任何可用的最新条目。

    解决数据源（如 AkShare news_cctv）实际返回旧日期时被全部过滤的问题。
    """
    if not raw:
        return []

    # Tier 1：严格 — 当年 + N 天内
    out = _filter_and_sort_by_recency(raw, min_year, max_age_days, limit=limit)
    if out:
        return out

    # Tier 2：放宽 — 近 2 年 + 一年内
    out = _filter_and_sort_by_recency(raw, min_year - 1, max_age_days * 4, limit=limit)
    if out:
        print(f"[news_fetcher] ⚠️ 当年新闻不足，已放宽到近 2 年（来源数据较旧）")
        return out

    # Tier 3：彻底放弃过滤，按日期降序取最新的 N 条
    sorted_raw = sorted(raw, key=lambda it: _date_score(it.get("date", "")), reverse=True)
    out = sorted_raw[:limit]
    if out:
        print(f"[news_fetcher] ⚠️ 数据源仅返回历史新闻，跳过新近度过滤")
    return out


def fetch_news_bundle(tickers: Optional[List[str]] = None,
                     per_ticker: int = 5,
                     headlines: int = 5,
                     min_year: int = None,
                     max_age_days: int = 90) -> Dict:
    """一站式新闻包：每个 ticker 拉 N 条个股新闻 + M 条宏观财经新闻。

    过滤策略（三级降级）：
      Tier 1：当年（如 2026）+ max_age_days 内
      Tier 2：上一年起 + max_age_days × 4 内（数据源较旧时）
      Tier 3：完全跳过过滤，取最新 N 条（数据源只有历史数据时）

    Args:
        min_year: 严格层只保留该年（含）之后的新闻，默认当年
        max_age_days: 严格层只保留 N 天内的新闻
    """
    if min_year is None:
        min_year = dt.date.today().year

    bundle = {
        "by_ticker": {},
        "market_headlines": [],
        "financial_news": [],
        "total_count": 0,
        "fetched_at": dt.datetime.utcnow().isoformat(),
        "freshness_filter": {"min_year": min_year, "max_age_days": max_age_days},
    }

    # 多拉一些上层再过滤，保证过滤后还能凑足
    fetch_multiplier = 4

    if tickers:
        for t in tickers:
            raw = fetch_stock_news(t, limit=per_ticker * fetch_multiplier)
            fresh = _cascade_filter(raw, min_year, max_age_days, limit=per_ticker)
            if fresh:
                bundle["by_ticker"][t] = fresh
                bundle["total_count"] += len(fresh)

    raw_macro = fetch_market_headlines(limit=headlines * fetch_multiplier)
    bundle["market_headlines"] = _cascade_filter(raw_macro, min_year, max_age_days, limit=headlines)
    bundle["total_count"] += len(bundle["market_headlines"])

    raw_fin = fetch_financial_news(limit=headlines * fetch_multiplier)
    bundle["financial_news"] = _cascade_filter(raw_fin, min_year, max_age_days, limit=headlines)
    bundle["total_count"] += len(bundle["financial_news"])

    return bundle


def summarize_for_prompt(bundle: Dict, max_chars_per_item: int = 200,
                         max_items: int = 12) -> str:
    """把新闻包压缩成 LLM prompt 友好的短文本（控制 token）。

    所有候选条目已经按日期降序排好，这里只负责按来源分桶、加 emoji 标记新近度。
    """
    lines = []
    count = 0

    # 标题加新近度标记，便于 LLM 优先采信
    today_date = dt.date.today()

    def _freshness_tag(date_str: str) -> str:
        if not date_str:
            return ""
        try:
            d = dt.date.fromisoformat(date_str[:10])
            days_diff = (today_date - d).days
        except (ValueError, TypeError):
            return ""
        if days_diff <= 1:
            return "🔥"   # 今/昨日
        if days_diff <= 7:
            return "🆕"   # 本周
        if days_diff <= 30:
            return "📅"   # 本月
        return ""

    for ticker, news_list in bundle.get("by_ticker", {}).items():
        for n in news_list:
            if count >= max_items:
                break
            title = n["title"][:80]
            date = n["date"]
            tag = _freshness_tag(date)
            lines.append(f"{tag}[{ticker} | {date}] {title}".strip())
            count += 1

    for n in bundle.get("financial_news", []):
        if count >= max_items:
            break
        title = n["title"][:80]
        date = n["date"]
        tag = _freshness_tag(date)
        lines.append(f"{tag}[财联社 | {date}] {title}".strip())
        count += 1

    for n in bundle.get("market_headlines", []):
        if count >= max_items:
            break
        title = n["title"][:80]
        date = n["date"]
        tag = _freshness_tag(date)
        lines.append(f"{tag}[宏观 | {date}] {title}".strip())
        count += 1

    return "\n".join(lines) if lines else "（暂无相关新闻）"
