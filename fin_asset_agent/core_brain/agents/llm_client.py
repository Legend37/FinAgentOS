# core_brain/agents/llm_client.py
import json
import datetime as dt
from typing import Optional, Callable
from openai import OpenAI
from config import llm as llm_config


# 全局会话级 usage 追踪器（每个进程一份；可重置）
_USAGE_LOG = []


def reset_usage_log():
    _USAGE_LOG.clear()


def get_usage_summary() -> dict:
    """返回本进程累计 token 与估算费用"""
    total_prompt = sum(e["prompt_tokens"] for e in _USAGE_LOG)
    total_completion = sum(e["completion_tokens"] for e in _USAGE_LOG)
    total_cost = sum(e["cost_rmb"] for e in _USAGE_LOG)
    by_model = {}
    for e in _USAGE_LOG:
        by_model.setdefault(e["model"], {"calls": 0, "tokens": 0, "cost_rmb": 0.0})
        by_model[e["model"]]["calls"] += 1
        by_model[e["model"]]["tokens"] += e["prompt_tokens"] + e["completion_tokens"]
        by_model[e["model"]]["cost_rmb"] += e["cost_rmb"]
    return {
        "total_calls": len(_USAGE_LOG),
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_cost_rmb": round(total_cost, 6),
        "by_model": by_model,
        "entries": list(_USAGE_LOG),
    }


def _safe_int(v) -> int:
    """安全提取 int，非 int（如 MagicMock）→ 0"""
    try:
        return int(v) if v is not None else 0
    except (TypeError, ValueError):
        return 0


def _record_usage(model: str, role: str, usage_obj) -> dict:
    """从 OpenAI 兼容 response.usage 中提取 token，估算费用并落到全局日志"""
    if usage_obj is None:
        return {}

    prompt_tokens = _safe_int(getattr(usage_obj, "prompt_tokens", 0))
    completion_tokens = _safe_int(getattr(usage_obj, "completion_tokens", 0))

    # DeepSeek 在 usage.prompt_cache_hit_tokens / prompt_tokens_details.cached_tokens 暴露缓存命中
    cached = 0
    details = getattr(usage_obj, "prompt_tokens_details", None)
    if details is not None:
        cached = _safe_int(getattr(details, "cached_tokens", 0))
    if not cached:
        cached = _safe_int(getattr(usage_obj, "prompt_cache_hit_tokens", 0))

    cost = llm_config.estimate_cost(model, prompt_tokens, completion_tokens, cached)
    entry = {
        "ts": dt.datetime.utcnow().isoformat(),
        "model": model,
        "role": role,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached,
        "cost_rmb": cost,
    }
    _USAGE_LOG.append(entry)
    return entry


def call_deepseek(api_key: str, prompt: str, role: str = "primary",
                  model: Optional[str] = None, base_url: Optional[str] = None) -> dict:
    """JSON 强制模式调用（用于 T_t 微调、R_t 审查、Router 路由等结构化场景）

    Args:
        role: 调用方角色标记，用于日志归类（router / primary / risk / ...）
        model: 覆盖 config 默认模型；router 角色自动选 router_model，其余用 primary_model
        base_url: 覆盖 config 默认接口地址；留空回退 config.llm.base_url
    """
    if not api_key:
        raise ValueError("api_key 为空")
    if model is None:
        model = llm_config.router_model if role == "router" else llm_config.primary_model

    client = OpenAI(api_key=api_key, base_url=base_url or llm_config.base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个严谨的金融智能体，你的输出必须是合法的 JSON 格式。"},
            {"role": "user", "content": prompt}
        ],
        response_format={'type': 'json_object'},
        temperature=llm_config.temperature
    )
    _record_usage(model, role, response.usage)
    data = json.loads(response.choices[0].message.content)
    # 容错：个别模型/第三方兼容接口不遵守 json_object，把对象包成数组返回，
    # 解包出第一个 dict，避免下游 res.get(...) 抛 'list' object has no attribute 'get'
    if isinstance(data, list):
        data = next((x for x in data if isinstance(x, dict)), {})
    return data


def call_deepseek_text(api_key: str, system_prompt: str, user_prompt: str,
                       role: str = "chat", model: Optional[str] = None,
                       base_url: Optional[str] = None) -> str:
    """纯文本对话调用（用于 CHIT_CHAT / QA_RAG，便宜的 chat 模型）"""
    if not api_key:
        raise ValueError("api_key 为空")
    if model is None:
        model = llm_config.chat_model

    client = OpenAI(api_key=api_key, base_url=base_url or llm_config.base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=llm_config.chat_temperature,
    )
    _record_usage(model, role, response.usage)
    return response.choices[0].message.content.strip()
