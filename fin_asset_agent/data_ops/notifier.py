# data_ops/notifier.py
"""Telegram bot 主动推送通道。

设计要点：
- 统一 send_telegram()，token/chat_id 缺省回退 config.telegram。
- 永不抛异常：网络/配置问题返回 {"ok": False, "error": ...}，不打断主流程。
- 纯 IO，方便单测用 mock 替换 requests。
"""
from typing import Optional, Dict

import requests

from config import telegram as tg_config
from data_ops import net_proxy


def send_telegram(text: str, chat_id: Optional[str] = None,
                  token: Optional[str] = None,
                  parse_mode: Optional[str] = None) -> Dict:
    """发送一条 Telegram 消息。

    Returns: {"ok": bool, "message_id"?: int, "error"?: str, "skipped"?: bool}
    """
    token = token or tg_config.bot_token
    chat_id = chat_id or tg_config.default_chat_id

    if not token:
        return {"ok": False, "skipped": True, "error": "未配置 TELEGRAM_BOT_TOKEN"}
    if not chat_id:
        return {"ok": False, "skipped": True, "error": "无目标 chat_id（用户未绑定）"}

    url = f"{tg_config.api_base}/bot{token}/sendMessage"
    payload = {"chat_id": str(chat_id), "text": text, "disable_web_page_preview": True}
    if parse_mode:
        payload["parse_mode"] = parse_mode

    try:
        # api.telegram.org 国内被墙：代理开关打开时走系统代理，关闭则直连
        resp = net_proxy.build_session().post(url, json=payload, timeout=tg_config.timeout)
        data = resp.json()
        if data.get("ok"):
            return {"ok": True, "message_id": data.get("result", {}).get("message_id")}
        return {"ok": False, "error": data.get("description", "Telegram API 返回 ok=false")}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def verify_bot(token: Optional[str] = None) -> Dict:
    """getMe 自检，返回 {"ok", "username"?, "error"?}。"""
    token = token or tg_config.bot_token
    if not token:
        return {"ok": False, "error": "未配置 TELEGRAM_BOT_TOKEN"}
    try:
        resp = net_proxy.build_session().get(f"{tg_config.api_base}/bot{token}/getMe", timeout=tg_config.timeout)
        data = resp.json()
        if data.get("ok"):
            return {"ok": True, "username": data.get("result", {}).get("username")}
        return {"ok": False, "error": data.get("description")}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
