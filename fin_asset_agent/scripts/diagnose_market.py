# scripts/diagnose_market.py
"""行情可达性诊断：对一个【确定有数据的过去窗口】用三种代理模式各拉一次，
找出在本机网络下哪种方式能连上 eastmoney。把完整输出贴回来即可定方案。

用法（在 fin_asset_agent 目录下）：
    python scripts/diagnose_market.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import akshare as ak  # noqa: E402
from data_ops import net_proxy  # noqa: E402

# 确定有数据的过去窗口（2025 年初），跟系统时钟无关
SYMBOL = "510300"          # 沪深300ETF
START, END = "20250102", "20250201"

PROXY_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy")


def _clear_proxy_env():
    saved = {k: os.environ.get(k) for k in PROXY_KEYS + ("NO_PROXY", "no_proxy")}
    for k in PROXY_KEYS:
        os.environ.pop(k, None)
    return saved


def _restore(saved):
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _try(label):
    try:
        df = ak.fund_etf_hist_em(symbol=SYMBOL, period="daily",
                                 start_date=START, end_date=END, adjust="qfq")
        n = 0 if df is None or df.empty else len(df)
        tail = "" if not n else f"  末行: {df.iloc[-1]['日期']} 收盘={df.iloc[-1]['收盘']}"
        print(f"  [{label}] -> {n} 行{tail}")
        return n > 0
    except Exception as e:
        print(f"  [{label}] -> 失败: {type(e).__name__}: {str(e)[:140]}")
        return False


def main():
    detected = net_proxy.detect_system_proxy()
    print(f"检测到的系统代理 = {detected}")
    print(f"测试标的 {SYMBOL}，窗口 {START}~{END}（fund_etf_hist_em）\n")

    # 模式 1：保持当前 shell 环境（不动）
    print("模式1 · 保持现状（当前 shell 环境变量原样）")
    _try("as-is")

    # 模式 2：强制清空代理 → 直连
    print("\n模式2 · 直连（清空所有代理环境变量）")
    saved = _clear_proxy_env()
    os.environ["NO_PROXY"] = "*"
    try:
        _try("direct")
    finally:
        _restore(saved)

    # 模式 3：显式走系统代理
    print(f"\n模式3 · 走系统代理（HTTP(S)_PROXY = {detected}）")
    if not detected:
        print("  [proxy] -> 跳过：未检测到系统代理")
    else:
        saved = _clear_proxy_env()
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ[k] = detected
        os.environ.pop("NO_PROXY", None); os.environ.pop("no_proxy", None)
        try:
            _try("proxy")
        finally:
            _restore(saved)

    print("\n>>> 把上面三种模式各几行的结果贴回来：哪种 >0 行，就用哪种。")


if __name__ == "__main__":
    main()
