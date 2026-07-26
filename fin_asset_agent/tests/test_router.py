import pytest
import json
from core_brain.router import CoreRouter, IntentResponse


class TestIntentRouter:
    """意图路由 — mock LLM，验证 4 种意图正确解析"""

    def test_route_to_asset_allocation(self, mocker):
        mock_client = mocker.patch("core_brain.router.OpenAI")
        mock_client.return_value.chat.completions.create.return_value = mocker.MagicMock(
            choices=[
                mocker.MagicMock(
                    message=mocker.MagicMock(
                        content=json.dumps({
                            "intent": "ASSET_ALLOCATION",
                            "extracted_entities": {"tickers": ["茅台"], "horizon": "中长期"},
                        })
                    )
                )
            ]
        )
        router = CoreRouter(api_key="sk-test")
        result = router.route_query("帮我配置一个稳健型组合")

        assert result.intent == "ASSET_ALLOCATION"
        assert result.extracted_entities["tickers"] == ["茅台"]

    def test_route_to_qa_rag(self, mocker):
        mock_client = mocker.patch("core_brain.router.OpenAI")
        mock_client.return_value.chat.completions.create.return_value = mocker.MagicMock(
            choices=[
                mocker.MagicMock(
                    message=mocker.MagicMock(
                        content=json.dumps({
                            "intent": "QA_RAG",
                            "extracted_entities": {"question": "什么是夏普比率"},
                        })
                    )
                )
            ]
        )
        router = CoreRouter(api_key="sk-test")
        result = router.route_query("什么是夏普比率？")

        assert result.intent == "QA_RAG"

    def test_route_to_portfolio_review(self, mocker):
        mock_client = mocker.patch("core_brain.router.OpenAI")
        mock_client.return_value.chat.completions.create.return_value = mocker.MagicMock(
            choices=[
                mocker.MagicMock(
                    message=mocker.MagicMock(
                        content=json.dumps({
                            "intent": "PORTFOLIO_REVIEW",
                            "extracted_entities": {"period": "近三个月"},
                        })
                    )
                )
            ]
        )
        router = CoreRouter(api_key="sk-test")
        result = router.route_query("帮我回顾一下最近三个月的持仓表现")

        assert result.intent == "PORTFOLIO_REVIEW"

    def test_route_to_chit_chat(self, mocker):
        mock_client = mocker.patch("core_brain.router.OpenAI")
        mock_client.return_value.chat.completions.create.return_value = mocker.MagicMock(
            choices=[
                mocker.MagicMock(
                    message=mocker.MagicMock(
                        content=json.dumps({
                            "intent": "CHIT_CHAT",
                            "extracted_entities": {},
                        })
                    )
                )
            ]
        )
        router = CoreRouter(api_key="sk-test")
        result = router.route_query("你好，今天天气怎么样")

        assert result.intent == "CHIT_CHAT"

    def test_returns_pydantic_model(self, mocker):
        """返回的是 Pydantic IntentResponse 实例，字段类型正确"""
        mock_client = mocker.patch("core_brain.router.OpenAI")
        mock_client.return_value.chat.completions.create.return_value = mocker.MagicMock(
            choices=[
                mocker.MagicMock(
                    message=mocker.MagicMock(
                        content=json.dumps({
                            "intent": "CHIT_CHAT",
                            "extracted_entities": {"greeting": "hello"},
                        })
                    )
                )
            ]
        )
        router = CoreRouter(api_key="sk-test")
        result = router.route_query("你好")

        assert isinstance(result, IntentResponse)
        assert result.intent in {"QA_RAG", "ASSET_ALLOCATION", "PORTFOLIO_REVIEW", "CHIT_CHAT"}
        assert isinstance(result.extracted_entities, dict)

    def test_invalid_intent_raises_validation_error(self, mocker):
        """LLM 返回非法意图值 → Pydantic 校验失败"""
        mock_client = mocker.patch("core_brain.router.OpenAI")
        mock_client.return_value.chat.completions.create.return_value = mocker.MagicMock(
            choices=[
                mocker.MagicMock(
                    message=mocker.MagicMock(
                        content=json.dumps({
                            "intent": "UNKNOWN_INTENT",
                            "extracted_entities": {},
                        })
                    )
                )
            ]
        )
        router = CoreRouter(api_key="sk-test")
        with pytest.raises(Exception):
            router.route_query("随便说点什么")
