"""系统代理自动检测 + 运行时开关测试。"""
import os

import pytest

from data_ops import net_proxy as NP


@pytest.fixture(autouse=True)
def _restore_state():
    # 保存并恢复模块级运行时状态，避免用例间串扰
    saved = dict(NP._state)
    yield
    NP._state.clear()
    NP._state.update(saved)


def test_detect_reads_env(monkeypatch):
    for k in NP._PROXY_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")
    assert NP.detect_system_proxy() == "http://127.0.0.1:7890"


def test_active_proxy_respects_toggle():
    NP.set_state(enabled=True, url="http://127.0.0.1:7890")
    assert NP.active_proxy() == "http://127.0.0.1:7890"
    assert NP.proxies_dict() == {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}
    NP.set_state(enabled=False)
    assert NP.active_proxy() is None
    assert NP.proxies_dict() == {}


def test_enabled_but_no_url_is_direct():
    NP.set_state(enabled=True, url="")
    assert NP.active_proxy() is None


def test_build_session_disabled_ignores_env(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")
    NP.set_state(enabled=False, url="http://127.0.0.1:7890")
    s = NP.build_session()
    assert s.trust_env is False          # 关闭时不读环境变量 → 真直连
    assert s.proxies == {}


def test_build_session_enabled_uses_proxy():
    NP.set_state(enabled=True, url="http://127.0.0.1:7890")
    s = NP.build_session()
    assert s.proxies == {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}


def test_force_direct_strips_and_restores_env(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7890")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")
    with NP.force_direct():
        assert "HTTP_PROXY" not in os.environ
        assert "HTTPS_PROXY" not in os.environ
        # NO_PROXY 设为国内行情域名列表（按后缀匹配），含 eastmoney
        assert "eastmoney.com" in os.environ.get("NO_PROXY", "")
    # 退出后原样恢复
    assert os.environ.get("HTTP_PROXY") == "http://127.0.0.1:7890"
    assert os.environ.get("HTTPS_PROXY") == "http://127.0.0.1:7890"


def test_get_state_shape():
    NP.set_state(enabled=True, url="http://127.0.0.1:7890")
    st = NP.get_state()
    assert set(st.keys()) == {"detected", "url", "enabled", "active"}
    assert st["enabled"] is True
    assert st["active"] == "http://127.0.0.1:7890"
