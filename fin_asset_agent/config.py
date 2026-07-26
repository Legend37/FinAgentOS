# config.py — 统一配置中心
import os
from dataclasses import dataclass, field

# ---- config.yaml：本地默认配置（LLM + Telegram），程序启动读取、前端预填 ----
def _load_yaml_config() -> dict:
    try:
        import yaml
    except Exception:
        return {}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[config] config.yaml 解析失败（忽略）: {e}")
        return {}

_YAML = _load_yaml_config()
_YAML_LLM = _YAML.get("llm", {}) or {}
_YAML_TG = _YAML.get("telegram", {}) or {}


@dataclass
class LLMConfig:
    """LLM 模型分层配置 — 简单任务用便宜模型，复杂推理用 reasoner"""
    base_url: str = "https://api.deepseek.com"

    # 三档模型分工
    router_model: str = "deepseek-chat"      # 意图分类，单次 < 100 token，最便宜
    chat_model: str = "deepseek-chat"        # CHIT_CHAT/QA_RAG 自然语言对话
    primary_model: str = "deepseek-reasoner" # T_t 微调 + R_t 风控审查，复杂推理

    temperature: float = 0.1
    chat_temperature: float = 0.7

    # DeepSeek 官方定价（单位：人民币元 / 百万 token）
    # 来源：https://api-docs.deepseek.com/zh-cn/quick_start/pricing
    pricing: dict = field(default_factory=lambda: {
        "deepseek-chat":     {"input": 0.27, "input_cached": 0.07, "output": 1.10},
        "deepseek-reasoner": {"input": 4.00, "input_cached": 1.00, "output": 16.00},
    })

    def estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int,
                      cached_tokens: int = 0) -> float:
        """根据 token 用量估算费用（人民币元）"""
        p = self.pricing.get(model, self.pricing["deepseek-chat"])
        uncached = max(prompt_tokens - cached_tokens, 0)
        cost_input = uncached * p["input"] / 1_000_000
        cost_cached = cached_tokens * p["input_cached"] / 1_000_000
        cost_output = completion_tokens * p["output"] / 1_000_000
        return round(cost_input + cost_cached + cost_output, 6)


@dataclass
class DataConfig:
    lookback_years: int = 2
    cache_dir: str = "data_ops/cache"


@dataclass
class AssetPoolConfig:
    # 按 risk_tolerance_level 分组的资产池
    pools: dict = field(default_factory=lambda: {
        "保守型": {
            "tickers": ["511010.SS", "511880.SS", "600900.SS"],
            "names": {
                "511010.SS": "国债ETF (固收)",
                "511880.SS": "银华日利 (货币)",
                "600900.SS": "长江电力 (高股息)",
            },
        },
        "稳健型": {
            "tickers": ["601398.SS", "600519.SS", "511010.SS"],
            "names": {
                "601398.SS": "工商银行 (大型金融)",
                "600519.SS": "贵州茅台 (消费龙头)",
                "511010.SS": "国债ETF (固收)",
            },
        },
        "平衡型": {
            "tickers": ["600519.SS", "601398.SS", "000858.SZ"],
            "names": {
                "600519.SS": "贵州茅台 (酒类消费)",
                "601398.SS": "工商银行 (大型金融)",
                "000858.SZ": "比亚迪 (新能源车)",
            },
        },
        "成长型": {
            "tickers": ["000858.SZ", "300750.SZ", "159915.SZ"],
            "names": {
                "000858.SZ": "比亚迪 (新能源车)",
                "300750.SZ": "宁德时代 (电池龙头)",
                "159915.SZ": "创业板ETF (成长)",
            },
        },
        "进取型": {
            "tickers": ["159995.SZ", "512880.SS", "300750.SZ"],
            "names": {
                "159995.SZ": "芯片ETF (科技)",
                "512880.SS": "证券ETF (券商)",
                "300750.SZ": "宁德时代 (高成长)",
            },
        },
    })

    def get_pool(self, risk_level: str) -> dict:
        """根据风险等级获取对应资产池（兜底用，LLM 选品不可用时回退）"""
        return self.pools.get(risk_level, self.pools["平衡型"])


@dataclass
class AssetUniverseConfig:
    """全资产宇宙 — 50+ 标的、7 大类，供 LLM 智能选品

    每个标的格式：{ticker, name, category, risk_band: 1-5（越高越激进）, desc}
    """
    universe: list = field(default_factory=lambda: [
        # ──────── A 股 · 大金融 (risk_band 2-3) ────────
        {"ticker": "601398.SS", "name": "工商银行", "category": "stock_a", "risk_band": 2, "desc": "国有四大行，高分红低波动"},
        {"ticker": "600036.SS", "name": "招商银行", "category": "stock_a", "risk_band": 3, "desc": "零售银行龙头"},
        {"ticker": "601318.SS", "name": "中国平安", "category": "stock_a", "risk_band": 3, "desc": "保险+综合金融"},
        {"ticker": "600030.SS", "name": "中信证券", "category": "stock_a", "risk_band": 4, "desc": "券商龙头，弹性大"},

        # ──────── A 股 · 消费白酒/家电 (risk_band 3) ────────
        {"ticker": "600519.SS", "name": "贵州茅台", "category": "stock_a", "risk_band": 3, "desc": "白酒龙头"},
        {"ticker": "000858.SZ", "name": "五粮液", "category": "stock_a", "risk_band": 3, "desc": "白酒次龙头"},
        {"ticker": "000333.SZ", "name": "美的集团", "category": "stock_a", "risk_band": 3, "desc": "白色家电龙头"},
        {"ticker": "600887.SS", "name": "伊利股份", "category": "stock_a", "risk_band": 2, "desc": "乳业龙头，防御性强"},

        # ──────── A 股 · 新能源/科技 (risk_band 4-5) ────────
        {"ticker": "002594.SZ", "name": "比亚迪", "category": "stock_a", "risk_band": 4, "desc": "新能源车龙头"},
        {"ticker": "300750.SZ", "name": "宁德时代", "category": "stock_a", "risk_band": 4, "desc": "动力电池全球龙头"},
        {"ticker": "601012.SS", "name": "隆基绿能", "category": "stock_a", "risk_band": 5, "desc": "光伏龙头"},
        {"ticker": "688981.SS", "name": "中芯国际", "category": "stock_a", "risk_band": 5, "desc": "本土晶圆代工龙头"},

        # ──────── A 股 · 医药 (risk_band 3-4) ────────
        {"ticker": "600276.SS", "name": "恒瑞医药", "category": "stock_a", "risk_band": 3, "desc": "创新药龙头"},
        {"ticker": "300760.SZ", "name": "迈瑞医疗", "category": "stock_a", "risk_band": 4, "desc": "医疗器械龙头"},

        # ──────── A 股 · 高分红/防御 (risk_band 1-2) ────────
        {"ticker": "600900.SS", "name": "长江电力", "category": "stock_a", "risk_band": 1, "desc": "水电龙头，类债券"},
        {"ticker": "601088.SS", "name": "中国神华", "category": "stock_a", "risk_band": 2, "desc": "高股息煤炭"},

        # ──────── A 股 · 房地产板块 (risk_band 4) ────────
        {"ticker": "600048.SS", "name": "保利发展", "category": "reit", "risk_band": 4, "desc": "央企地产龙头"},
        {"ticker": "000002.SZ", "name": "万科A", "category": "reit", "risk_band": 4, "desc": "地产白马"},
        {"ticker": "001979.SZ", "name": "招商蛇口", "category": "reit", "risk_band": 4, "desc": "央企地产"},

        # ──────── ETF · 宽基 (risk_band 3-4) ────────
        {"ticker": "510300.SS", "name": "沪深300ETF", "category": "etf_broad", "risk_band": 3, "desc": "A 股核心宽基"},
        {"ticker": "510500.SS", "name": "中证500ETF", "category": "etf_broad", "risk_band": 4, "desc": "中盘成长"},
        {"ticker": "159915.SZ", "name": "创业板ETF", "category": "etf_broad", "risk_band": 5, "desc": "创业板宽基"},
        {"ticker": "588000.SS", "name": "科创50ETF", "category": "etf_broad", "risk_band": 5, "desc": "科创板宽基"},
        {"ticker": "510050.SS", "name": "上证50ETF", "category": "etf_broad", "risk_band": 2, "desc": "大盘蓝筹"},

        # ──────── ETF · 行业 (risk_band 4-5) ────────
        {"ticker": "159995.SZ", "name": "芯片ETF", "category": "etf_sector", "risk_band": 5, "desc": "半导体一篮子"},
        {"ticker": "515030.SS", "name": "新能源车ETF", "category": "etf_sector", "risk_band": 5, "desc": "新能源车产业链"},
        {"ticker": "512010.SS", "name": "医药ETF", "category": "etf_sector", "risk_band": 3, "desc": "医药行业一篮子"},
        {"ticker": "512800.SS", "name": "银行ETF", "category": "etf_sector", "risk_band": 2, "desc": "银行业一篮子"},
        {"ticker": "512880.SS", "name": "证券ETF", "category": "etf_sector", "risk_band": 5, "desc": "券商弹性"},
        {"ticker": "515100.SS", "name": "红利低波ETF", "category": "etf_sector", "risk_band": 1, "desc": "高股息低波动"},

        # ──────── ETF · 海外 (risk_band 3-4) ────────
        {"ticker": "513100.SS", "name": "纳指ETF", "category": "etf_overseas", "risk_band": 4, "desc": "美股科技纳斯达克"},
        {"ticker": "513500.SS", "name": "标普500ETF", "category": "etf_overseas", "risk_band": 3, "desc": "美股大盘"},
        {"ticker": "159920.SZ", "name": "恒生ETF", "category": "etf_overseas", "risk_band": 4, "desc": "港股蓝筹"},

        # ──────── 债券 / 货币 (risk_band 1) ────────
        {"ticker": "511010.SS", "name": "国债ETF", "category": "bond_money", "risk_band": 1, "desc": "5 年期国债"},
        {"ticker": "511880.SS", "name": "银华日利", "category": "bond_money", "risk_band": 1, "desc": "货币 ETF，类活期"},
        {"ticker": "511090.SS", "name": "30年国债ETF", "category": "bond_money", "risk_band": 2, "desc": "超长债，对冲股票"},
        {"ticker": "511360.SS", "name": "短债ETF", "category": "bond_money", "risk_band": 1, "desc": "短久期债券"},

        # ──────── 商品 · 金银油 (risk_band 3) ────────
        {"ticker": "518880.SS", "name": "黄金ETF", "category": "commodity", "risk_band": 3, "desc": "国内黄金 ETF"},
        {"ticker": "GLD", "name": "SPDR 黄金", "category": "commodity", "risk_band": 3, "desc": "美股黄金 ETF"},
        {"ticker": "SLV", "name": "iShares 白银", "category": "commodity", "risk_band": 4, "desc": "美股白银 ETF"},
        {"ticker": "USO", "name": "美国原油基金", "category": "commodity", "risk_band": 5, "desc": "WTI 原油"},

        # ──────── 美股 (risk_band 3-5) ────────
        {"ticker": "AAPL", "name": "苹果", "category": "stock_us", "risk_band": 3, "desc": "科技龙头"},
        {"ticker": "MSFT", "name": "微软", "category": "stock_us", "risk_band": 3, "desc": "云+AI 双引擎"},
        {"ticker": "GOOGL", "name": "谷歌", "category": "stock_us", "risk_band": 4, "desc": "搜索+AI"},
        {"ticker": "NVDA", "name": "英伟达", "category": "stock_us", "risk_band": 5, "desc": "AI 算力霸主"},
        {"ticker": "TSLA", "name": "特斯拉", "category": "stock_us", "risk_band": 5, "desc": "电动车+自动驾驶"},
        {"ticker": "JPM", "name": "摩根大通", "category": "stock_us", "risk_band": 3, "desc": "美国银行龙头"},
        {"ticker": "KO", "name": "可口可乐", "category": "stock_us", "risk_band": 2, "desc": "防御性消费"},
        {"ticker": "JNJ", "name": "强生", "category": "stock_us", "risk_band": 2, "desc": "医药消费防御"},

        # ──────── 虚拟币代理 (risk_band 5) ────────
        # ⚠️ 国内合规上不能直接买币，用美股代理（持币公司 / 矿股 / 现货 ETF）
        {"ticker": "BTC-USD", "name": "比特币", "category": "crypto", "risk_band": 5, "desc": "BTC 现货（yfinance）"},
        {"ticker": "ETH-USD", "name": "以太坊", "category": "crypto", "risk_band": 5, "desc": "ETH 现货"},
        {"ticker": "IBIT", "name": "贝莱德比特币ETF", "category": "crypto", "risk_band": 5, "desc": "美股合规 BTC ETF"},
        {"ticker": "COIN", "name": "Coinbase", "category": "crypto", "risk_band": 5, "desc": "美股加密交易所龙头"},
        {"ticker": "MSTR", "name": "MicroStrategy", "category": "crypto", "risk_band": 5, "desc": "持币代理（已转型 BTC 国库）"},
    ])

    # 类别中文名映射，用于前端展示和 LLM prompt
    category_labels: dict = field(default_factory=lambda: {
        "stock_a": "A 股个股",
        "stock_us": "美股个股",
        "etf_broad": "宽基 ETF",
        "etf_sector": "行业 ETF",
        "etf_overseas": "海外 ETF",
        "bond_money": "债券 / 货币",
        "commodity": "商品（金银油）",
        "reit": "房地产",
        "crypto": "虚拟币（美股代理）",
    })

    def get_all(self) -> list:
        return list(self.universe)

    def filter_by_categories(self, categories: list) -> list:
        if not categories:
            return self.get_all()
        return [a for a in self.universe if a["category"] in set(categories)]

    def filter_by_risk_band(self, max_band: int) -> list:
        """只保留 risk_band ≤ max_band 的标的（用于保守型用户）"""
        return [a for a in self.universe if a["risk_band"] <= max_band]

    def get_by_ticker(self, ticker: str) -> dict:
        for a in self.universe:
            if a["ticker"] == ticker:
                return a
        return None


@dataclass
class MVOConfig:
    risk_free_rate: float = 0.02


@dataclass
class RiskConfig:
    weight_tolerance_low: float = 0.99
    weight_tolerance_high: float = 1.01
    # 按风险等级的单一资产权重上限（硬门禁）
    max_single_asset_weight: dict = field(default_factory=lambda: {
        "保守型": 0.40,
        "稳健型": 0.50,
        "平衡型": 0.60,
        "成长型": 0.80,
        "进取型": 0.95,
    })

    def get_max_single_weight(self, risk_level: str) -> float:
        return self.max_single_asset_weight.get(risk_level, 0.60)


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass
class TelegramConfig:
    """Telegram bot 主动推送配置。

    bot_token / default_chat_id 优先级：环境变量 > config.yaml。
    enabled 由是否拿到 token 决定，没配则推送静默跳过（不影响主流程）。
    """
    bot_token: str = field(default_factory=lambda: os.environ.get("TELEGRAM_BOT_TOKEN") or _YAML_TG.get("bot_token") or "")
    default_chat_id: str = field(default_factory=lambda: os.environ.get("TELEGRAM_CHAT_ID") or _YAML_TG.get("chat_id") or "")
    api_base: str = "https://api.telegram.org"
    timeout: int = 20

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token)


@dataclass
class FrontendDefaults:
    """供前端启动时预填的默认值（来自 config.yaml），避免每次刷新重输 API/TG。"""
    llm_api_key: str = field(default_factory=lambda: _YAML_LLM.get("api_key", "") or "")
    llm_base_url: str = field(default_factory=lambda: _YAML_LLM.get("base_url", "") or "")
    llm_model: str = field(default_factory=lambda: _YAML_LLM.get("model", "") or "")
    llm_router_model: str = field(default_factory=lambda: _YAML_LLM.get("router_model", "") or "")
    llm_chat_model: str = field(default_factory=lambda: _YAML_LLM.get("chat_model", "") or "")
    llm_primary_model: str = field(default_factory=lambda: _YAML_LLM.get("primary_model", "") or "")
    telegram_chat_id: str = field(default_factory=lambda: _YAML_TG.get("chat_id", "") or "")

    def to_frontend(self) -> dict:
        return {
            "llm": {
                "api_key": self.llm_api_key,
                "base_url": self.llm_base_url,
                "model": self.llm_model,
                "router_model": self.llm_router_model,
                "chat_model": self.llm_chat_model,
                "primary_model": self.llm_primary_model,
            },
            "telegram": {"chat_id": self.telegram_chat_id},
        }


# 全局单例
llm = LLMConfig()
data = DataConfig()
asset_pool = AssetPoolConfig()
asset_universe = AssetUniverseConfig()
mvo = MVOConfig()
risk = RiskConfig()
server = ServerConfig()
telegram = TelegramConfig()
frontend_defaults = FrontendDefaults()
