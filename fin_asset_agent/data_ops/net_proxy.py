# data_ops/net_proxy.py
"""系统代理自动检测 + 运行时开关。

职责：
- detect_system_proxy(): 读操作系统/环境里配置的 HTTP(S) 代理（Windows 下 getproxies() 读系统注册表 + 环境变量）。
- 运行时单例状态：enabled 开关 + 当前代理 URL；启动时自动检测，检测到则默认开启（「自动连接」）。
- proxies_dict() / build_session(): 外网服务（Telegram / yfinance）按开关走代理。
- force_direct(): 上下文管理器，临时清掉进程级代理环境变量，强制国内行情（akshare/eastmoney）直连
  —— A 股数据走境外代理必被重置，必须直连。

不依赖 config，避免循环导入。
"""
import os
import urllib.request
from contextlib import contextmanager
from typing import Optional, Dict

import requests

# 进程级代理环境变量（force_direct 临时摘除的对象）
_PROXY_ENV_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
                   "ALL_PROXY", "all_proxy")

# 国内行情数据源域名：force_direct 时塞进 NO_PROXY，确保即便 Windows 系统/注册表
# 设了代理，requests/akshare 对这些域名也直连（按域名后缀匹配，requests 不认 "*" 通配）。
_DOMESTIC_NO_PROXY = (
    "eastmoney.com,.eastmoney.com,push2his.eastmoney.com,push2.eastmoney.com,"
    "datacenter-web.eastmoney.com,quote.eastmoney.com,"
    "sina.com.cn,.sina.com.cn,sse.com.cn,szse.cn,.szse.cn,"
    "localhost,127.0.0.1"
)


def detect_system_proxy() -> Optional[str]:
    """返回系统配置的 HTTP(S) 代理 URL（优先 https，其次 http），无则 None。

    urllib.request.getproxies() 在 Windows 下读「Internet 选项 / 系统代理」注册表，
    在各平台也读 HTTP_PROXY/HTTPS_PROXY 环境变量。
    """
    try:
        proxies = urllib.request.getproxies()
    except Exception:
        proxies = {}
    return proxies.get("https") or proxies.get("http") or None


# 运行时状态：检测到系统代理则默认 enabled（用户要的「打开自动连接」）
_detected = detect_system_proxy()
_state = {"url": _detected, "enabled": bool(_detected)}


def active_proxy() -> Optional[str]:
    """当前真正生效的代理 URL；开关关闭或无 URL 时返回 None（=直连）。"""
    return _state["url"] if (_state["enabled"] and _state["url"]) else None


def get_state() -> Dict:
    """供 API/前端读取的完整状态。"""
    return {
        "detected": detect_system_proxy(),   # 实时重测，反映系统当前设置
        "url": _state["url"],
        "enabled": _state["enabled"],
        "active": active_proxy(),
    }


def set_state(enabled: Optional[bool] = None, url: Optional[str] = None,
              redetect: bool = False) -> Dict:
    """更新开关 / URL；redetect=True 时重新探测系统代理覆盖 url。"""
    if redetect:
        _state["url"] = detect_system_proxy()
    if url is not None:
        _state["url"] = url.strip() or None
    if enabled is not None:
        _state["enabled"] = bool(enabled)
    return get_state()


def proxies_dict() -> Dict[str, str]:
    """供 requests proxies= 参数用：开关开且有 URL → 显式代理；否则空 dict。"""
    p = active_proxy()
    return {"http": p, "https": p} if p else {}


def build_session() -> requests.Session:
    """构造一个按开关行为的 requests.Session（用于 Telegram 等外网调用）。

    - 开关开 + 有 URL → 显式走该代理；
    - 开关关 → trust_env=False，忽略环境变量，强制直连（否则进程里残留的
      HTTP_PROXY 会让「关」形同虚设）。
    """
    s = requests.Session()
    p = active_proxy()
    if p:
        s.proxies = {"http": p, "https": p}
        s.trust_env = True
    else:
        s.trust_env = False
    return s


@contextmanager
def force_direct():
    """临时清掉进程级代理环境变量，让块内的网络调用（akshare/eastmoney）直连。

    退出时原样恢复。国内行情绝不应走代理，故 akshare 分支统一包在此上下文里。
    """
    keys = _PROXY_ENV_KEYS + ("NO_PROXY", "no_proxy")
    saved = {k: os.environ.get(k) for k in keys}
    for k in _PROXY_ENV_KEYS:
        os.environ.pop(k, None)
    # NO_PROXY 用域名列表（不是 "*"）：requests 按后缀匹配跳过这些域名的代理，
    # 即便系统/注册表里仍有代理也对 eastmoney/sina 直连。
    os.environ["NO_PROXY"] = _DOMESTIC_NO_PROXY
    os.environ["no_proxy"] = _DOMESTIC_NO_PROXY
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextmanager
def apply_proxy():
    """块内把进程级代理 env 设为当前生效代理；开关关闭时等同直连。

    用于 yfinance 这类只认环境变量、不接受显式 proxy 参数（或参数随版本变动）的库。
    退出时原样恢复。
    """
    p = active_proxy()
    if not p:
        with force_direct():
            yield
        return
    keys = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "NO_PROXY", "no_proxy")
    saved = {k: os.environ.get(k) for k in keys}
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ[k] = p
    os.environ.pop("NO_PROXY", None)
    os.environ.pop("no_proxy", None)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
