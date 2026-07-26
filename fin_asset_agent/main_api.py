# main_api.py
import os
import sys

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
from core_brain.workflow import run_fin_agent_pipeline
from config import server as server_config, asset_universe
from memory import state_manager

app = FastAPI()

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")


@app.get("/api/health")
async def health():
    """供本地验收、容器健康检查和部署平台探针使用。"""
    return {"status": "ok", "service": "finagent-os"}


@app.get("/api/categories")
async def list_categories():
    """列出全部可选资产类别 + 每类资产数量，供前端展示选项"""
    universe = asset_universe.get_all()
    counts = {}
    for a in universe:
        counts[a["category"]] = counts.get(a["category"], 0) + 1
    return {
        "categories": [
            {"key": k, "label": v, "count": counts.get(k, 0)}
            for k, v in asset_universe.category_labels.items()
            if counts.get(k, 0) > 0
        ],
        "total_assets": len(universe),
    }


class LLMOverrideMixin(BaseModel):
    """自定义 LLM 接入（均可留空，留空回退 config.py 默认）"""
    base_url: Optional[str] = None        # 自定义 OpenAI 兼容接口地址
    model: Optional[str] = None           # 单模型 ID，覆盖全部三档
    router_model: Optional[str] = None    # 分档覆盖（优先级高于 model）
    chat_model: Optional[str] = None
    primary_model: Optional[str] = None


class AllocationRequest(LLMOverrideMixin):
    api_key: str
    query: str = ""
    user_uuid: Optional[str] = None
    email: Optional[str] = None
    name: str = "用户"
    age: int = 30
    occupation: str = "未填写"
    total_wealth: float = 500000
    risk_tolerance_level: str = "平衡型"
    investment_horizon: str = "中长期"
    financial_goals: str = "资产稳健增值"
    custom_tickers: Optional[List[str]] = None
    preferred_categories: Optional[List[str]] = None  # 偏好的资产类别（多选）


class AuthInitRequest(BaseModel):
    user_uuid: Optional[str] = None
    email: Optional[str] = None
    name: str = "用户"


class ChatRequest(LLMOverrideMixin):
    api_key: str
    message: str
    user_uuid: Optional[str] = None
    email: Optional[str] = None
    session_id: Optional[str] = None
    name: str = "用户"
    age: int = 30
    occupation: str = "未填写"
    total_wealth: float = 500000
    risk_tolerance_level: str = "平衡型"
    investment_horizon: str = "中长期"
    financial_goals: str = "资产稳健增值"
    custom_tickers: Optional[List[str]] = None
    preferred_categories: Optional[List[str]] = None


@app.post("/api/auth/init")
async def init_auth(req: AuthInitRequest):
    try:
        return state_manager.init_user_identity(
            user_uuid=req.user_uuid,
            email=req.email,
            profile={"name": req.name},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _snapshot_weights(result: dict) -> list:
    return result.get("final_weights") or result.get("adjusted_weights") or []


def _assistant_summary(result: dict) -> str:
    if result.get("risk_status") == "CHAT":
        return result.get("timing_reason") or result.get("risk_report") or ""
    assets = result.get("assets") or []
    weights = _snapshot_weights(result)
    pairs = []
    for asset, weight in zip(assets, weights):
        pairs.append(f"{asset}: {weight:.1%}")
    allocation = "；".join(pairs[:6])
    reason = result.get("timing_reason") or result.get("risk_report") or ""
    return f"{reason}\n\n配置摘要：{allocation}" if allocation else reason


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.api_key:
        raise HTTPException(status_code=400, detail="API Key 不能为空")
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")

    try:
        identity = state_manager.init_user_identity(
            user_uuid=req.user_uuid,
            email=req.email,
            profile={"name": req.name},
        )
        user_id = identity["user_id"]
        session_id = req.session_id or identity["session_id"]

        history = state_manager.recall_recent_turns(session_id, limit=10)
        past_snapshots = state_manager.recall_user_snapshots(user_id, limit=3)
        state_manager.record_turn(session_id, "user", req.message, user_id=user_id)

        user_profile = {
            "user_uuid": identity["user_uuid"],
            "email": req.email,
            "name": req.name,
            "age": req.age,
            "occupation": req.occupation,
            "total_wealth": req.total_wealth,
            "risk_tolerance_level": req.risk_tolerance_level,
            "investment_horizon": req.investment_horizon,
            "financial_goals": req.financial_goals,
            "custom_tickers": req.custom_tickers,
            "preferred_categories": req.preferred_categories,
        }
        result = run_fin_agent_pipeline(
            req.message,
            req.api_key,
            user_profile,
            history=history,
            past_snapshots=past_snapshots,
            base_url=req.base_url,
            model=req.model,
            router_model=req.router_model,
            chat_model=req.chat_model,
            primary_model=req.primary_model,
        )

        assistant_text = _assistant_summary(result)
        state_manager.record_turn(
            session_id,
            "assistant",
            assistant_text,
            user_id=user_id,
            intent=result.get("intent"),
        )

        weights = _snapshot_weights(result)
        snapshot_id = None
        if result.get("assets") and weights:
            snapshot_id = state_manager.record_snapshot(
                user_id=user_id,
                intent=result.get("intent", "ASSET_ALLOCATION"),
                tickers=result.get("assets", []),
                base_weights=result.get("base_weights", []),
                final_weights=weights,
                total_wealth=(result.get("asset_snapshot") or {}).get("total_wealth", req.total_wealth),
                timing_reason=result.get("timing_reason"),
            )
            state_manager.record_risk_assessment(
                user_id=user_id,
                snapshot_id=snapshot_id,
                risk_status=result.get("risk_status", ""),
                risk_score=(result.get("user_profile") or {}).get("risk_score"),
                risk_report=result.get("risk_report", ""),
            )

        result.update({
            "assistant_message": assistant_text,
            "session_id": session_id,
            "user_id": user_id,
            "user_uuid": identity["user_uuid"],
            "snapshot_id": snapshot_id,
        })
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/allocate")
async def allocate_portfolio(req: AllocationRequest):
    if not req.api_key:
        raise HTTPException(status_code=400, detail="API Key 不能为空")
    try:
        user_profile = {
            "user_uuid": req.user_uuid,
            "email": req.email,
            "name": req.name,
            "age": req.age,
            "occupation": req.occupation,
            "total_wealth": req.total_wealth,
            "risk_tolerance_level": req.risk_tolerance_level,
            "investment_horizon": req.investment_horizon,
            "financial_goals": req.financial_goals,
            "custom_tickers": req.custom_tickers,
            "preferred_categories": req.preferred_categories,
        }
        result = run_fin_agent_pipeline(
            req.query, req.api_key, user_profile,
            base_url=req.base_url,
            model=req.model,
            router_model=req.router_model,
            chat_model=req.chat_model,
            primary_model=req.primary_model,
        )

        # 落库一份快照，让表单入口也进历史时间线（与 /api/chat 行为对齐）
        try:
            weights = _snapshot_weights(result)
            if result.get("assets") and weights:
                identity = state_manager.init_user_identity(
                    user_uuid=req.user_uuid, email=req.email, profile={"name": req.name},
                )
                user_id = identity["user_id"]
                snapshot_id = state_manager.record_snapshot(
                    user_id=user_id,
                    intent=result.get("intent", "ASSET_ALLOCATION"),
                    tickers=result.get("assets", []),
                    base_weights=result.get("base_weights", []),
                    final_weights=weights,
                    total_wealth=(result.get("asset_snapshot") or {}).get("total_wealth", req.total_wealth),
                    timing_reason=result.get("timing_reason"),
                )
                state_manager.record_risk_assessment(
                    user_id=user_id,
                    snapshot_id=snapshot_id,
                    risk_status=result.get("risk_status", ""),
                    risk_score=(result.get("user_profile") or {}).get("risk_score"),
                    risk_report=result.get("risk_report", ""),
                )
                result["snapshot_id"] = snapshot_id
                result["user_uuid"] = identity["user_uuid"]
        except Exception as persist_err:
            print(f"[allocate] 快照落库失败（已忽略）: {persist_err}")

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class FollowRequest(BaseModel):
    is_followed: bool = True


def _resolve_user_id(user_uuid: str) -> int:
    uid = state_manager.get_user_id_by_uuid(user_uuid)
    if uid is None:
        raise HTTPException(status_code=404, detail="用户不存在或 uuid 无效")
    return uid


@app.get("/api/sessions/{user_uuid}/snapshots")
async def list_snapshots(user_uuid: str, limit: int = 50):
    """#7 历史方案时间线：返回该用户的全部组合快照（倒序）。"""
    try:
        user_id = _resolve_user_id(user_uuid)
        return {"snapshots": state_manager.list_user_snapshots(user_id, limit=limit)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions/{user_uuid}/attribution/{snapshot_id}")
async def snapshot_attribution(user_uuid: str, snapshot_id: int, horizon: int = 7,
                               force: bool = False):
    """#11 某份方案 horizon 天后的真实表现归因（缓存优先，回查行情兜底）。"""
    try:
        _resolve_user_id(user_uuid)
        return state_manager.get_or_compute_attribution(snapshot_id, horizon_days=horizon, force=force)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions/{user_uuid}/backtest/{snapshot_id}")
async def snapshot_backtest(user_uuid: str, snapshot_id: int, window: int = 30,
                            rebalance: str = "none"):
    """某份方案过去 window 天的固定权重净值回测（拉真实行情，估算大致年化收益）。"""
    try:
        _resolve_user_id(user_uuid)
        return state_manager.get_snapshot_backtest(snapshot_id, window_days=window, rebalance=rebalance)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sessions/{user_uuid}/snapshots/{snapshot_id}/follow")
async def follow_snapshot(user_uuid: str, snapshot_id: int, req: FollowRequest):
    """#14 用户标记"我已按此方案调仓"。"""
    try:
        _resolve_user_id(user_uuid)
        ok = state_manager.mark_snapshot_followed(snapshot_id, followed=req.is_followed)
        if not ok:
            raise HTTPException(status_code=404, detail="快照不存在")
        return {"snapshot_id": snapshot_id, "is_followed": req.is_followed}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 6-3：Telegram 主动推送（#12 / #13）
# ============================================================
from config import telegram as telegram_config, frontend_defaults
from data_ops.notifier import verify_bot, send_telegram
from data_ops.review_job import weekly_review_job, push_snapshot_review
from data_ops import net_proxy


@app.get("/api/config")
async def get_frontend_config():
    """前端启动预填用的默认配置（来自 config.yaml），免每次刷新重输 API/TG。"""
    data = frontend_defaults.to_frontend()
    data["telegram"]["enabled"] = telegram_config.enabled
    data["proxy"] = net_proxy.get_state()
    return data


class ProxyRequest(BaseModel):
    enabled: Optional[bool] = None
    url: Optional[str] = None
    redetect: bool = False


@app.get("/api/proxy")
async def get_proxy():
    """当前代理状态：系统检测值 + 运行时开关 + 生效 URL。"""
    return net_proxy.get_state()


@app.post("/api/proxy")
async def set_proxy(req: ProxyRequest):
    """更新代理开关 / URL（前端 TG 下方的「系统代理」开关）。redetect=True 重新探测系统代理。"""
    return net_proxy.set_state(enabled=req.enabled, url=req.url, redetect=req.redetect)


class TelegramBindRequest(BaseModel):
    chat_id: Optional[str] = None
    send_test: bool = True


class NotifyRequest(BaseModel):
    horizon: int = 7
    chat_id: Optional[str] = None   # 不传则用用户绑定的


class CronRequest(BaseModel):
    force: bool = False
    push: bool = True
    min_age_days: int = 7
    horizon: int = 7


@app.get("/api/telegram/status")
async def telegram_status():
    """Bot 是否已配置 + getMe 自检。"""
    if not telegram_config.enabled:
        return {"enabled": False, "reason": "未配置 TELEGRAM_BOT_TOKEN 环境变量或 config.yaml 的 telegram.bot_token"}
    info = verify_bot()
    return {"enabled": True, "bot_ok": info.get("ok"), "username": info.get("username"),
            "error": info.get("error")}


@app.post("/api/sessions/{user_uuid}/telegram")
async def bind_telegram(user_uuid: str, req: TelegramBindRequest):
    """绑定/解绑用户的 Telegram chat_id；可选发一条测试消息确认通道。"""
    try:
        user_id = _resolve_user_id(user_uuid)
        state_manager.set_user_telegram(user_id, req.chat_id)
        test_result = None
        if req.chat_id and req.send_test:
            test_result = send_telegram(
                "🔔 FinAgent OS 已绑定本对话。之后的复盘建议会推送到这里。",
                chat_id=req.chat_id,
            )
        return {"user_uuid": user_uuid, "chat_id": req.chat_id or None, "test": test_result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sessions/{user_uuid}/snapshots/{snapshot_id}/notify")
async def notify_snapshot(user_uuid: str, snapshot_id: int, req: NotifyRequest):
    """对单份方案立即生成复盘建议并推送 Telegram（前端"推送复盘"按钮）。"""
    try:
        _resolve_user_id(user_uuid)
        result = push_snapshot_review(snapshot_id, horizon_days=req.horizon,
                                      chat_id_override=req.chat_id)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions/{user_uuid}/pending")
async def list_pending(user_uuid: str, unread_only: bool = True):
    """#13 未读主动建议列表（前端 📬 徽章）。"""
    try:
        user_id = _resolve_user_id(user_uuid)
        return {"pending": state_manager.list_pending_advice(user_id, unread_only=unread_only)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sessions/{user_uuid}/pending/{advice_id}/ack")
async def ack_pending(user_uuid: str, advice_id: int):
    """标记某条建议为已读。"""
    try:
        _resolve_user_id(user_uuid)
        ok = state_manager.mark_pending_read(advice_id)
        if not ok:
            raise HTTPException(status_code=404, detail="建议不存在")
        return {"advice_id": advice_id, "read": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/cron/weekly-review")
async def trigger_weekly_review(req: CronRequest):
    """手动触发每周复盘任务（也可由系统 cron 调度 scripts/cron_weekly_review.py）。"""
    try:
        return weekly_review_job(min_age_days=req.min_age_days, horizon_days=req.horizon,
                                 push=req.push, force=req.force)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=server_config.host, port=server_config.port)
