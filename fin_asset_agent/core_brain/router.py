# core_brain/router.py
from pydantic import BaseModel
from typing import Literal
import json
from openai import OpenAI
from config import llm as llm_config
from core_brain.agents.llm_client import _record_usage


class IntentResponse(BaseModel):
    intent: Literal["QA_RAG", "ASSET_ALLOCATION", "PORTFOLIO_REVIEW", "CHIT_CHAT"]
    extracted_entities: dict


class CoreRouter:
    """意图路由器 — 用最便宜的 router_model（deepseek-chat），单次调用 < 200 token"""

    def __init__(self, api_key: str, base_url: str = None, router_model: str = None):
        self.api_key = api_key
        self.model = router_model or llm_config.router_model
        self.client = OpenAI(api_key=api_key, base_url=base_url or llm_config.base_url)

    def route_query(self, user_query: str) -> IntentResponse:
        prompt = (
            f"分析以下金融用户提问的意图：\"{user_query}\"\n\n"
            f"严格按以下 4 个固定枚举之一返回 intent（不要返回其他任何中文或英文词）：\n"
            f"  - ASSET_ALLOCATION  → 用户希望生成新的资产配置方案、推荐买什么、如何分配资金\n"
            f"  - PORTFOLIO_REVIEW  → 用户希望对已有持仓做体检、风险评估、再平衡建议\n"
            f"  - QA_RAG            → 用户在问金融术语、市场常识、政策解读等知识性问题\n"
            f"  - CHIT_CHAT         → 用户在闲聊、问候、感谢，与投资业务无关\n\n"
            f"同时提取关键实体到 extracted_entities（如 time / tickers / sectors / amount）。\n"
            f"返回 JSON 格式必须严格如：{{\"intent\": \"ASSET_ALLOCATION\", \"extracted_entities\": {{...}}}}"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": (
                    "你是一个严格的意图分类器。intent 字段只能是 "
                    "ASSET_ALLOCATION / PORTFOLIO_REVIEW / QA_RAG / CHIT_CHAT 中的一个英文标识，"
                    "不允许翻译成中文、不允许返回其他词。"
                )},
                {"role": "user", "content": prompt},
            ],
            response_format={'type': 'json_object'},
        )
        # 落 token & 费用日志（按 router 角色归类）
        _record_usage(self.model, "router", response.usage)
        data = json.loads(response.choices[0].message.content)
        return IntentResponse(**data)