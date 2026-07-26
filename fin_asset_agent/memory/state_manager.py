# memory/state_manager.py
"""多轮会话 Context 切片管理与跨会话记忆读写。

设计目标：
- record_turn / record_snapshot / record_risk_assessment：把工作流副产物落库
- recall_recent_turns：取最近 N 轮做 LLM 上下文，避免 Token 爆量
- get_or_create_user：按 name 简单 upsert，生产环境应替换为账号体系
"""
from typing import List, Optional, Dict, Any
import datetime as dt

import hashlib
import uuid as uuid_lib
from memory.db_models import (
    User, PortfolioSnapshot, RiskAssessment, ConversationTurn, SkillRecord,
    PerformanceAttribution, PendingAdvice, get_session,
)


# 截断阈值：每条对话最多保留多少字符（防 Token 爆量）
MAX_TURN_CHARS = 800
# 默认召回轮数
DEFAULT_RECALL_TURNS = 10


def _truncate(text: str, limit: int = MAX_TURN_CHARS) -> str:
    if not text:
        return ""
    return text if len(text) <= limit else text[: limit - 3] + "..."


def get_or_create_user(profile: Dict[str, Any]) -> int:
    """按 name 简单 upsert，返回 user_id"""
    name = (profile.get("name") or "用户").strip() or "用户"
    user_uuid = (profile.get("user_uuid") or profile.get("uuid") or "").strip()
    email = (profile.get("email") or "").strip() or None
    s = get_session()
    try:
        user = None
        if user_uuid:
            user = s.query(User).filter_by(uuid=user_uuid).first()
        if user is None and email:
            user = s.query(User).filter_by(email=email).first()
        if user is None and not user_uuid and not email:
            user = s.query(User).filter_by(name=name).order_by(User.id.desc()).first()
        if user is None:
            user = User(uuid=user_uuid or str(uuid_lib.uuid4()), name=name, email=email)
            s.add(user)
        elif email and not user.email:
            user.email = email

        # 总是用最新画像覆盖
        user.name = name
        user.age = profile.get("age", user.age)
        user.occupation = profile.get("occupation", user.occupation)
        user.risk_tolerance_level = profile.get("risk_tolerance_level", user.risk_tolerance_level)
        user.investment_horizon = profile.get("investment_horizon", user.investment_horizon)
        user.financial_goals = profile.get("financial_goals", user.financial_goals)
        user.last_active_at = dt.datetime.utcnow()
        s.commit()
        return user.id
    finally:
        s.close()


def init_user_identity(user_uuid: Optional[str] = None, email: Optional[str] = None,
                       profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create or refresh a browser identity and return stable user/session data."""
    profile = dict(profile or {})
    if user_uuid:
        profile["user_uuid"] = user_uuid
    if email:
        profile["email"] = email
    user_id = get_or_create_user(profile)

    s = get_session()
    try:
        user = s.query(User).filter_by(id=user_id).one()
        return {
            "user_id": user.id,
            "user_uuid": user.uuid,
            "email": user.email,
            "name": user.name,
            "session_id": f"user-{user.uuid}",
            "last_active_at": user.last_active_at.isoformat() if user.last_active_at else None,
        }
    finally:
        s.close()


def record_turn(session_id: str, role: str, content: str,
                user_id: Optional[int] = None, intent: Optional[str] = None) -> int:
    """落一条对话记录"""
    s = get_session()
    try:
        turn = ConversationTurn(
            session_id=session_id,
            user_id=user_id,
            role=role,
            intent=intent,
            content=_truncate(content),
        )
        s.add(turn)
        s.commit()
        return turn.id
    finally:
        s.close()


def recall_recent_turns(session_id: str, limit: int = DEFAULT_RECALL_TURNS) -> List[Dict[str, Any]]:
    """召回某 session 最近 N 轮，按时间正序返回，供 LLM messages 用"""
    s = get_session()
    try:
        rows = (
            s.query(ConversationTurn)
            .filter_by(session_id=session_id)
            .order_by(ConversationTurn.id.desc())
            .limit(limit)
            .all()
        )
        rows.reverse()
        return [
            {"role": r.role, "content": r.content, "intent": r.intent, "ts": r.created_at.isoformat()}
            for r in rows
        ]
    finally:
        s.close()


def record_snapshot(user_id: int, intent: str, tickers: list, base_weights: list,
                    final_weights: list, total_wealth: Optional[float] = None,
                    timing_reason: Optional[str] = None) -> int:
    """落一条投资组合快照"""
    s = get_session()
    try:
        snap = PortfolioSnapshot(
            user_id=user_id,
            intent=intent,
            total_wealth=total_wealth,
            timing_reason=timing_reason,
            advice_date=dt.date.today(),
        )
        snap.tickers = tickers
        snap.base_weights = base_weights
        snap.final_weights = final_weights
        s.add(snap)
        s.commit()
        return snap.id
    finally:
        s.close()


def record_risk_assessment(user_id: int, snapshot_id: Optional[int],
                           risk_status: str, risk_score: Optional[int],
                           risk_report: str) -> int:
    """落一条风控评估"""
    s = get_session()
    try:
        r = RiskAssessment(
            user_id=user_id,
            snapshot_id=snapshot_id,
            risk_status=risk_status,
            risk_score=risk_score,
            risk_report=_truncate(risk_report, limit=4000),
        )
        s.add(r)
        s.commit()
        return r.id
    finally:
        s.close()


def get_latest_snapshot(user_id: int) -> Optional[Dict[str, Any]]:
    """取用户最近一次组合快照，供 PORTFOLIO_REVIEW 分支预填 holdings"""
    s = get_session()
    try:
        snap = (
            s.query(PortfolioSnapshot)
            .filter_by(user_id=user_id)
            .order_by(PortfolioSnapshot.id.desc())
            .first()
        )
        if snap is None:
            return None
        return {
            "id": snap.id,
            "intent": snap.intent,
            "tickers": snap.tickers,
            "base_weights": snap.base_weights,
            "final_weights": snap.final_weights,
            "total_wealth": snap.total_wealth,
            "timing_reason": snap.timing_reason,
            "created_at": snap.created_at.isoformat(),
        }
    finally:
        s.close()


def recall_user_snapshots(user_id: int, limit: int = 3) -> List[Dict[str, Any]]:
    """Return recent portfolio snapshots for context injection."""
    s = get_session()
    try:
        rows = (
            s.query(PortfolioSnapshot)
            .filter_by(user_id=user_id)
            .order_by(PortfolioSnapshot.id.desc())
            .limit(limit)
            .all()
        )
        return [{
            "id": snap.id,
            "intent": snap.intent,
            "tickers": snap.tickers,
            "base_weights": snap.base_weights,
            "final_weights": snap.final_weights,
            "total_wealth": snap.total_wealth,
            "timing_reason": snap.timing_reason,
            "created_at": snap.created_at.isoformat(),
        } for snap in rows]
    finally:
        s.close()


def make_profile_signature(profile: Dict[str, Any]) -> str:
    """生成稳定的画像签名（哈希），用于 SkillRecord 召回匹配"""
    parts = [
        str(profile.get("risk_tolerance_level", "")),
        str(profile.get("investment_horizon", "")),
        # 年龄段（10 年一档）
        str((profile.get("age", 30) // 10) * 10),
        ",".join(sorted(profile.get("custom_tickers") or [])),
    ]
    raw = "|".join(parts)
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{profile.get('risk_tolerance_level','U')}::{h}"


def record_skill(profile: Dict[str, Any], intent: str,
                 critic_feedback: str, critic_score: int,
                 revision_summary: str, snapshot_id: Optional[int] = None) -> int:
    """记录一次成功的修订经验"""
    s = get_session()
    try:
        rec = SkillRecord(
            profile_signature=make_profile_signature(profile),
            risk_level=profile.get("risk_tolerance_level"),
            intent=intent,
            critic_feedback=_truncate(critic_feedback, 2000),
            critic_score=critic_score,
            revision_summary=_truncate(revision_summary, 2000),
            snapshot_id=snapshot_id,
        )
        s.add(rec)
        s.commit()
        return rec.id
    finally:
        s.close()


def recall_similar_skills(profile: Dict[str, Any], limit: int = 3) -> List[Dict[str, Any]]:
    """根据画像签名召回历史修订经验，按 critic_score 倒序"""
    sig = make_profile_signature(profile)
    s = get_session()
    try:
        rows = (
            s.query(SkillRecord)
            .filter_by(profile_signature=sig)
            .order_by(SkillRecord.critic_score.desc(), SkillRecord.id.desc())
            .limit(limit)
            .all()
        )
        # 同步 reuse_count + last_used_at
        for r in rows:
            r.reuse_count = (r.reuse_count or 0) + 1
            r.last_used_at = dt.datetime.utcnow()
        s.commit()

        return [{
            "id": r.id,
            "intent": r.intent,
            "critic_feedback": r.critic_feedback,
            "critic_score": r.critic_score,
            "revision_summary": r.revision_summary,
            "reuse_count": r.reuse_count,
            "created_at": r.created_at.isoformat(),
        } for r in rows]
    finally:
        s.close()


# ============================================================
# 6-3：历史方案 / 归因复盘 / follow 标记
# ============================================================

def get_user_id_by_uuid(user_uuid: str) -> Optional[int]:
    """按 uuid 解析 user_id（找不到返回 None）。"""
    if not user_uuid:
        return None
    s = get_session()
    try:
        user = s.query(User).filter_by(uuid=user_uuid).first()
        return user.id if user else None
    finally:
        s.close()


def _snapshot_to_dict(snap: PortfolioSnapshot) -> Dict[str, Any]:
    return {
        "id": snap.id,
        "user_id": snap.user_id,
        "intent": snap.intent,
        "tickers": snap.tickers,
        "base_weights": snap.base_weights,
        "final_weights": snap.final_weights,
        "total_wealth": snap.total_wealth,
        "timing_reason": snap.timing_reason,
        "advice_date": snap.advice_date.isoformat() if snap.advice_date else None,
        "is_followed": snap.is_followed,
        "created_at": snap.created_at.isoformat() if snap.created_at else None,
    }


def list_user_snapshots(user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    """历史方案时间线：按时间倒序返回完整快照（含 advice_date / is_followed）。"""
    s = get_session()
    try:
        rows = (
            s.query(PortfolioSnapshot)
            .filter_by(user_id=user_id)
            .order_by(PortfolioSnapshot.id.desc())
            .limit(limit)
            .all()
        )
        return [_snapshot_to_dict(r) for r in rows]
    finally:
        s.close()


def get_snapshot(snapshot_id: int) -> Optional[Dict[str, Any]]:
    """取单份快照详情。"""
    s = get_session()
    try:
        snap = s.query(PortfolioSnapshot).filter_by(id=snapshot_id).first()
        return _snapshot_to_dict(snap) if snap else None
    finally:
        s.close()


def mark_snapshot_followed(snapshot_id: int, followed: bool = True) -> bool:
    """用户标记/取消"我已按此方案调仓"。返回是否成功。"""
    s = get_session()
    try:
        snap = s.query(PortfolioSnapshot).filter_by(id=snapshot_id).first()
        if snap is None:
            return False
        snap.is_followed = followed
        s.commit()
        return True
    finally:
        s.close()


def _attribution_to_dict(row: PerformanceAttribution) -> Dict[str, Any]:
    return {
        "snapshot_id": row.snapshot_id,
        "horizon_days": row.horizon_days,
        "realized_return": row.realized_return,
        "realized_volatility": row.realized_volatility,
        "realized_sharpe": row.realized_sharpe,
        "asset_contributions": row.asset_contributions,
        "computed_at": row.computed_at.isoformat() if row.computed_at else None,
        "status": "ok",
    }


def get_cached_attribution(snapshot_id: int, horizon_days: int) -> Optional[Dict[str, Any]]:
    s = get_session()
    try:
        row = (
            s.query(PerformanceAttribution)
            .filter_by(snapshot_id=snapshot_id, horizon_days=horizon_days)
            .order_by(PerformanceAttribution.id.desc())
            .first()
        )
        return _attribution_to_dict(row) if row else None
    finally:
        s.close()


def record_attribution(snapshot_id: int, result: Dict[str, Any]) -> int:
    """把一次 status=ok 的归因结果落库缓存。"""
    s = get_session()
    try:
        row = PerformanceAttribution(
            snapshot_id=snapshot_id,
            horizon_days=result.get("horizon_days", 7),
            realized_return=result.get("realized_return"),
            realized_volatility=result.get("realized_volatility"),
            realized_sharpe=result.get("realized_sharpe"),
        )
        row.asset_contributions = result.get("asset_contributions", {})
        s.add(row)
        s.commit()
        return row.id
    finally:
        s.close()


def get_or_compute_attribution(snapshot_id: int, horizon_days: int = 7,
                               force: bool = False) -> Dict[str, Any]:
    """归因主入口：先查缓存，未命中则回查行情计算；status=ok 才落库缓存。

    pending / unavailable 不缓存（等数据积累后下次可重算）。
    """
    if not force:
        cached = get_cached_attribution(snapshot_id, horizon_days)
        if cached:
            return cached

    snap = get_snapshot(snapshot_id)
    if snap is None:
        return {"status": "unavailable", "reason": "快照不存在", "horizon_days": horizon_days}

    from data_ops.attribution import compute_attribution
    result = compute_attribution(
        asset_names=snap.get("tickers", []),
        weights=snap.get("final_weights") or snap.get("base_weights") or [],
        advice_date=snap.get("advice_date") or snap.get("created_at"),
        horizon_days=horizon_days,
    )
    if result.get("status") == "ok":
        record_attribution(snapshot_id, result)
    return result


def get_snapshot_backtest(snapshot_id: int, window_days: int = 30,
                          rebalance: str = "none") -> Dict[str, Any]:
    """区间净值回测主入口：加载快照 → 把其权重套在「过去 window_days」真实行情上回测。

    不缓存：窗口随当天日期滑动，结果每天都会变。FAILED 方案 final_weights 为空时回退 base_weights。
    """
    snap = get_snapshot(snapshot_id)
    if snap is None:
        return {"status": "unavailable", "reason": "快照不存在", "window_days": window_days}

    # 读取用户风险等级，用于回测时匹配对应的终身组合基准
    risk_level = None
    try:
        s = get_session()
        user = s.query(User).filter_by(id=snap.get("user_id")).first()
        if user:
            risk_level = user.risk_tolerance_level
        s.close()
    except Exception:
        pass

    from data_ops.backtest_review import compute_backtest
    return compute_backtest(
        asset_names=snap.get("tickers", []),
        weights=snap.get("final_weights") or snap.get("base_weights") or [],
        window_days=window_days,
        rebalance=rebalance,
        initial_capital=float(snap.get("total_wealth") or 1_000_000.0),
        risk_level=risk_level,
    )


# ============================================================
# 6-3：Telegram 绑定 + 主动建议（PendingAdvice）
# ============================================================

def set_user_telegram(user_id: int, chat_id: Optional[str]) -> bool:
    """绑定/解绑用户的 Telegram chat_id（传空=解绑）。"""
    s = get_session()
    try:
        user = s.query(User).filter_by(id=user_id).first()
        if user is None:
            return False
        user.telegram_chat_id = (chat_id or "").strip() or None
        s.commit()
        return True
    finally:
        s.close()


def get_user_telegram(user_id: int) -> Optional[str]:
    s = get_session()
    try:
        user = s.query(User).filter_by(id=user_id).first()
        return user.telegram_chat_id if user else None
    finally:
        s.close()


def record_pending_advice(user_id: int, snapshot_id: Optional[int],
                          trigger_reason: str, advice_summary: str) -> int:
    s = get_session()
    try:
        pa = PendingAdvice(
            user_id=user_id,
            snapshot_id=snapshot_id,
            trigger_reason=_truncate(trigger_reason, 256),
            advice_summary=_truncate(advice_summary, 4000),
        )
        s.add(pa)
        s.commit()
        return pa.id
    finally:
        s.close()


def update_pending_notify(advice_id: int, channel: str, notify_result: Dict[str, Any]) -> None:
    """回写一条 PendingAdvice 的推送状态。"""
    s = get_session()
    try:
        pa = s.query(PendingAdvice).filter_by(id=advice_id).first()
        if pa is None:
            return
        pa.notify_channel = channel
        if notify_result.get("ok"):
            pa.notify_status = "sent"
            pa.pushed_at = dt.datetime.utcnow()
        elif notify_result.get("skipped"):
            pa.notify_status = "skipped"
        else:
            pa.notify_status = "failed"
        s.commit()
    finally:
        s.close()


def _pending_to_dict(pa: PendingAdvice) -> Dict[str, Any]:
    return {
        "id": pa.id,
        "user_id": pa.user_id,
        "snapshot_id": pa.snapshot_id,
        "trigger_reason": pa.trigger_reason,
        "advice_summary": pa.advice_summary,
        "created_at": pa.created_at.isoformat() if pa.created_at else None,
        "read_at": pa.read_at.isoformat() if pa.read_at else None,
        "notify_channel": pa.notify_channel,
        "notify_status": pa.notify_status,
        "pushed_at": pa.pushed_at.isoformat() if pa.pushed_at else None,
    }


def list_pending_advice(user_id: int, unread_only: bool = True,
                        limit: int = 30) -> List[Dict[str, Any]]:
    s = get_session()
    try:
        q = s.query(PendingAdvice).filter_by(user_id=user_id)
        if unread_only:
            q = q.filter(PendingAdvice.read_at.is_(None))
        rows = q.order_by(PendingAdvice.id.desc()).limit(limit).all()
        return [_pending_to_dict(r) for r in rows]
    finally:
        s.close()


def mark_pending_read(advice_id: int) -> bool:
    s = get_session()
    try:
        pa = s.query(PendingAdvice).filter_by(id=advice_id).first()
        if pa is None:
            return False
        pa.read_at = dt.datetime.utcnow()
        s.commit()
        return True
    finally:
        s.close()


def snapshots_due_for_review(min_age_days: int = 7, as_of: Optional[dt.date] = None,
                             limit: int = 100) -> List[Dict[str, Any]]:
    """找出 advice_date <= as_of - min_age_days 且尚未生成过 PendingAdvice 的快照。"""
    as_of = as_of or dt.date.today()
    cutoff = as_of - dt.timedelta(days=min_age_days)
    s = get_session()
    try:
        already = s.query(PendingAdvice.snapshot_id).filter(
            PendingAdvice.snapshot_id.isnot(None)
        )
        already_ids = {row[0] for row in already.all()}
        q = (
            s.query(PortfolioSnapshot)
            .filter(PortfolioSnapshot.advice_date.isnot(None))
            .filter(PortfolioSnapshot.advice_date <= cutoff)
            .order_by(PortfolioSnapshot.id.desc())
            .limit(limit)
        )
        return [_snapshot_to_dict(r) for r in q.all() if r.id not in already_ids]
    finally:
        s.close()


def purge_old_turns(session_id: str, keep_last: int = 50) -> int:
    """清理某 session 老对话，只保留最近 keep_last 轮，防止 SQLite 膨胀"""
    s = get_session()
    try:
        cutoff = (
            s.query(ConversationTurn.id)
            .filter_by(session_id=session_id)
            .order_by(ConversationTurn.id.desc())
            .offset(keep_last)
            .first()
        )
        if cutoff is None:
            return 0
        deleted = (
            s.query(ConversationTurn)
            .filter(ConversationTurn.session_id == session_id,
                    ConversationTurn.id <= cutoff[0])
            .delete(synchronize_session=False)
        )
        s.commit()
        return deleted
    finally:
        s.close()
