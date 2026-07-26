# data_ops/review_job.py
"""每周主动复盘任务：找到期方案 → 算归因 → 生成建议 → 落 PendingAdvice → Telegram 推送。

- weekly_review_job(): 批量，给 cron / 手动触发用。
- push_snapshot_review(): 单份方案立即复盘 + 推送（前端"推送复盘到 Telegram"按钮）。
两者都依赖 state_manager（DB）、attribution（行情）、advice（文案）、notifier（Telegram）。
"""
import datetime as dt
from typing import Optional, Dict

from memory import state_manager
from data_ops.advice import generate_review_advice
from data_ops.notifier import send_telegram


def weekly_review_job(min_age_days: int = 7, horizon_days: int = 7,
                      as_of: Optional[dt.date] = None, push: bool = True,
                      force: bool = False) -> Dict:
    """批量复盘到期方案。force=True 时把 min_age_days 视为 0（处理全部未推过的）。"""
    if force:
        min_age_days = 0
    due = state_manager.snapshots_due_for_review(min_age_days, as_of)
    results = []
    for snap in due:
        attr = state_manager.get_or_compute_attribution(snap["id"], horizon_days)
        advice = generate_review_advice(snap, attr)
        pa_id = state_manager.record_pending_advice(
            snap["user_id"], snap["id"], advice["reason"], advice["text"]
        )
        notify = {"ok": False, "skipped": True, "error": "push disabled"}
        if push:
            chat = state_manager.get_user_telegram(snap["user_id"])
            notify = send_telegram(advice["text"], chat_id=chat)
            state_manager.update_pending_notify(pa_id, "telegram", notify)
        results.append({
            "snapshot_id": snap["id"], "advice_id": pa_id,
            "reason": advice["reason"], "attribution_status": attr.get("status"),
            "notify_ok": notify.get("ok"), "notify_error": notify.get("error"),
        })
    return {"processed": len(due), "results": results}


def push_snapshot_review(snapshot_id: int, horizon_days: int = 7,
                         chat_id_override: Optional[str] = None,
                         push: bool = True) -> Dict:
    """对单份方案立即生成复盘建议并推送（不受到期/去重限制，供手动触发）。"""
    snap = state_manager.get_snapshot(snapshot_id)
    if snap is None:
        return {"ok": False, "error": "快照不存在"}

    attr = state_manager.get_or_compute_attribution(snapshot_id, horizon_days)
    advice = generate_review_advice(snap, attr)
    pa_id = state_manager.record_pending_advice(
        snap["user_id"], snapshot_id, advice["reason"], advice["text"]
    )

    notify = {"ok": False, "skipped": True}
    if push:
        chat = chat_id_override or state_manager.get_user_telegram(snap["user_id"])
        notify = send_telegram(advice["text"], chat_id=chat)
        state_manager.update_pending_notify(pa_id, "telegram", notify)

    return {
        "ok": bool(notify.get("ok")),
        "advice_id": pa_id,
        "reason": advice["reason"],
        "text": advice["text"],
        "attribution_status": attr.get("status"),
        "notify": notify,
    }
