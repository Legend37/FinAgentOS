"""LLM cost tracking tests — 全部用 mock，零网络调用"""
from unittest.mock import patch, MagicMock
import pytest

from core_brain.agents import llm_client
from config import llm as llm_config


@pytest.fixture(autouse=True)
def _reset_log():
    llm_client.reset_usage_log()
    yield
    llm_client.reset_usage_log()


def _mock_response(prompt_tokens=100, completion_tokens=50, content="{}"):
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.prompt_tokens_details = None
    usage.prompt_cache_hit_tokens = 0
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def test_router_uses_cheap_model(mocker):
    """role='router' → 自动选 router_model（便宜）"""
    cli = MagicMock()
    cli.chat.completions.create.return_value = _mock_response(content='{"ok":1}')
    mocker.patch("core_brain.agents.llm_client.OpenAI", return_value=cli)

    llm_client.call_deepseek("sk-test", "意图分类", role="router")

    args, kwargs = cli.chat.completions.create.call_args
    assert kwargs["model"] == llm_config.router_model


def test_primary_uses_reasoner_model(mocker):
    """role='primary'（默认）→ 自动选 primary_model（reasoner）"""
    cli = MagicMock()
    cli.chat.completions.create.return_value = _mock_response(content='{"ok":1}')
    mocker.patch("core_brain.agents.llm_client.OpenAI", return_value=cli)

    llm_client.call_deepseek("sk-test", "权重微调")

    args, kwargs = cli.chat.completions.create.call_args
    assert kwargs["model"] == llm_config.primary_model
    assert kwargs["model"] == "deepseek-reasoner"


def test_chat_text_uses_chat_model(mocker):
    cli = MagicMock()
    cli.chat.completions.create.return_value = _mock_response(content="你好")
    mocker.patch("core_brain.agents.llm_client.OpenAI", return_value=cli)

    llm_client.call_deepseek_text("sk-test", "system", "user")

    args, kwargs = cli.chat.completions.create.call_args
    assert kwargs["model"] == llm_config.chat_model


def test_usage_logged_per_call(mocker):
    cli = MagicMock()
    cli.chat.completions.create.return_value = _mock_response(
        prompt_tokens=200, completion_tokens=80, content='{"x":1}'
    )
    mocker.patch("core_brain.agents.llm_client.OpenAI", return_value=cli)

    llm_client.call_deepseek("sk-test", "p1", role="router")
    llm_client.call_deepseek("sk-test", "p2", role="primary")

    s = llm_client.get_usage_summary()
    assert s["total_calls"] == 2
    assert s["total_prompt_tokens"] == 400
    assert s["total_completion_tokens"] == 160
    assert set(s["by_model"].keys()) == {llm_config.router_model, llm_config.primary_model}


def test_cost_estimate_chat_model():
    """deepseek-chat: 输入 1M tokens = 0.27 RMB, 输出 1M tokens = 1.10 RMB"""
    cost = llm_config.estimate_cost(
        "deepseek-chat", prompt_tokens=1_000_000, completion_tokens=0
    )
    assert abs(cost - 0.27) < 1e-6

    cost2 = llm_config.estimate_cost(
        "deepseek-chat", prompt_tokens=0, completion_tokens=1_000_000
    )
    assert abs(cost2 - 1.10) < 1e-6


def test_cost_estimate_reasoner_more_expensive():
    """reasoner 应明显贵于 chat"""
    chat_cost = llm_config.estimate_cost("deepseek-chat", 100_000, 50_000)
    reasoner_cost = llm_config.estimate_cost("deepseek-reasoner", 100_000, 50_000)
    assert reasoner_cost > chat_cost * 5  # reasoner 至少贵 5 倍


def test_cost_estimate_uses_cache_discount():
    """缓存命中按 cached 价格计费（便宜很多）"""
    full = llm_config.estimate_cost("deepseek-chat", 1_000_000, 0, cached_tokens=0)
    cached = llm_config.estimate_cost("deepseek-chat", 1_000_000, 0, cached_tokens=1_000_000)
    assert cached < full
    # deepseek-chat 缓存价 0.07 vs 0.27
    assert abs(cached - 0.07) < 1e-6


def test_usage_summary_by_model_aggregation(mocker):
    cli = MagicMock()
    cli.chat.completions.create.return_value = _mock_response(
        prompt_tokens=100, completion_tokens=50, content='{"ok":1}'
    )
    mocker.patch("core_brain.agents.llm_client.OpenAI", return_value=cli)

    llm_client.call_deepseek("sk", "p", role="router")
    llm_client.call_deepseek("sk", "p", role="router")
    llm_client.call_deepseek("sk", "p", role="primary")

    s = llm_client.get_usage_summary()
    router_stats = s["by_model"][llm_config.router_model]
    primary_stats = s["by_model"][llm_config.primary_model]
    assert router_stats["calls"] == 2
    assert primary_stats["calls"] == 1
    # primary 用 reasoner 更贵
    assert primary_stats["cost_rmb"] > router_stats["cost_rmb"] / 2
