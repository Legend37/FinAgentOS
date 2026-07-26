"""每周主动复盘 cron 入口。

用法（在 fin_asset_agent 目录下，PYTHONPATH=.）：
    python scripts/cron_weekly_review.py              # 处理建议日满 7 天的方案并推送
    python scripts/cron_weekly_review.py --force      # 不限到期，处理全部未推过的（测试用）
    python scripts/cron_weekly_review.py --no-push    # 只生成 PendingAdvice，不发 Telegram

可挂到系统计划任务 / Windows 任务计划程序 / Linux crontab 每天跑一次。
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_ops.review_job import weekly_review_job  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-age-days", type=int, default=7)
    ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument("--force", action="store_true", help="忽略到期天数，处理全部未推过的方案")
    ap.add_argument("--no-push", action="store_true", help="不发 Telegram，只落 PendingAdvice")
    args = ap.parse_args()

    res = weekly_review_job(
        min_age_days=args.min_age_days,
        horizon_days=args.horizon,
        push=not args.no_push,
        force=args.force,
    )
    print(f"[weekly_review] 处理 {res['processed']} 份方案")
    for r in res["results"]:
        flag = "✅" if r.get("notify_ok") else ("· " + str(r.get("notify_error") or "未推送"))
        print(f"  方案 #{r['snapshot_id']} | {r['reason']} | 归因={r['attribution_status']} | 推送 {flag}")


if __name__ == "__main__":
    main()
