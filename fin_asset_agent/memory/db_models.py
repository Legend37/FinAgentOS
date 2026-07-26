# memory/db_models.py
"""SQLAlchemy ORM 模型：跨会话持久化用户画像、组合快照、风控评估与对话历史。

使用 SQLite 作为默认存储，零依赖部署；可通过 FINAGENT_DB_URL 环境变量切换到 Postgres。
"""
import os
import json
import datetime as dt
import uuid as uuid_lib
from typing import Optional

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Date, Boolean, Text, ForeignKey,
    create_engine, inspect, text,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session

Base = declarative_base()


class User(Base):
    """用户画像主表（按 name+age 简单去重，生产环境应改为账号体系）"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(String(36), nullable=False, unique=True, index=True, default=lambda: str(uuid_lib.uuid4()))
    email = Column(String(128), nullable=True, unique=True, index=True)
    name = Column(String(64), nullable=False, index=True)
    age = Column(Integer, nullable=True)
    occupation = Column(String(64), nullable=True)
    risk_tolerance_level = Column(String(16), nullable=True)
    investment_horizon = Column(String(32), nullable=True)
    financial_goals = Column(Text, nullable=True)
    last_active_at = Column(DateTime, default=dt.datetime.utcnow, index=True)
    # 🆕 6-3：Telegram 主动推送目标
    telegram_chat_id = Column(String(32), nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow)

    snapshots = relationship("PortfolioSnapshot", back_populates="user", cascade="all, delete-orphan")
    assessments = relationship("RiskAssessment", back_populates="user", cascade="all, delete-orphan")
    conversations = relationship("ConversationTurn", back_populates="user", cascade="all, delete-orphan")


class PortfolioSnapshot(Base):
    """每次 MVO/Review 运行后的组合快照"""
    __tablename__ = "portfolio_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    intent = Column(String(32), nullable=False)
    total_wealth = Column(Float, nullable=True)
    tickers_json = Column(Text, nullable=False, default="[]")
    base_weights_json = Column(Text, nullable=False, default="[]")
    final_weights_json = Column(Text, nullable=False, default="[]")
    timing_reason = Column(Text, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow, index=True)
    # 🆕 6-3：建议日（归因回看锚点）+ 用户是否真的按此方案调仓
    advice_date = Column(Date, nullable=True, index=True, default=lambda: dt.date.today())
    is_followed = Column(Boolean, nullable=True, default=None)

    user = relationship("User", back_populates="snapshots")
    attributions = relationship("PerformanceAttribution", back_populates="snapshot",
                                cascade="all, delete-orphan")

    @property
    def tickers(self) -> list:
        return json.loads(self.tickers_json or "[]")

    @tickers.setter
    def tickers(self, value: list):
        self.tickers_json = json.dumps(value or [], ensure_ascii=False)

    @property
    def base_weights(self) -> list:
        return json.loads(self.base_weights_json or "[]")

    @base_weights.setter
    def base_weights(self, value: list):
        self.base_weights_json = json.dumps(value or [])

    @property
    def final_weights(self) -> list:
        return json.loads(self.final_weights_json or "[]")

    @final_weights.setter
    def final_weights(self, value: list):
        self.final_weights_json = json.dumps(value or [])


class RiskAssessment(Base):
    """每次 R_t 风控审查结果"""
    __tablename__ = "risk_assessments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    snapshot_id = Column(Integer, ForeignKey("portfolio_snapshots.id"), nullable=True)
    risk_status = Column(String(16), nullable=False)
    risk_score = Column(Integer, nullable=True)
    risk_report = Column(Text, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow, index=True)

    user = relationship("User", back_populates="assessments")


class PerformanceAttribution(Base):
    """🆕 6-3：某份 PortfolioSnapshot 在 horizon_days 后的真实表现归因。

    由 GET /api/sessions/{uid}/attribution/{sid} 触发计算并缓存，避免重复回查行情。
    """
    __tablename__ = "performance_attribution"

    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_id = Column(Integer, ForeignKey("portfolio_snapshots.id"), nullable=False, index=True)
    horizon_days = Column(Integer, nullable=False)        # 7 / 30 / 90
    realized_return = Column(Float, nullable=True)
    realized_volatility = Column(Float, nullable=True)
    realized_sharpe = Column(Float, nullable=True)
    asset_contributions_json = Column(Text, nullable=True, default="{}")  # {资产名: 贡献}
    computed_at = Column(DateTime, default=dt.datetime.utcnow, index=True)

    snapshot = relationship("PortfolioSnapshot", back_populates="attributions")

    @property
    def asset_contributions(self) -> dict:
        return json.loads(self.asset_contributions_json or "{}")

    @asset_contributions.setter
    def asset_contributions(self, value: dict):
        self.asset_contributions_json = json.dumps(value or {}, ensure_ascii=False)


class ConversationTurn(Base):
    """多轮对话单轮记录，用于 state_manager 选择性切片"""
    __tablename__ = "conversation_turns"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    role = Column(String(16), nullable=False)  # user / assistant / system
    intent = Column(String(32), nullable=True)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow, index=True)

    user = relationship("User", back_populates="conversations")


class PendingAdvice(Base):
    """🆕 6-3：weekly_review_job 生成的待读主动建议（可经 Telegram 推送）。"""
    __tablename__ = "pending_advice"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    snapshot_id = Column(Integer, ForeignKey("portfolio_snapshots.id"), nullable=True)
    trigger_reason = Column(String(256), nullable=True)   # "7天复盘到期" / "VaR 偏离" / "重大新闻"
    advice_summary = Column(Text, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow, index=True)
    read_at = Column(DateTime, nullable=True)
    # Telegram 推送状态
    notify_channel = Column(String(16), nullable=True)    # "telegram" / None
    notify_status = Column(String(16), nullable=True)     # "sent" / "failed" / "skipped"
    pushed_at = Column(DateTime, nullable=True)


class SkillRecord(Base):
    """Self-evolve 技能记录：(画像模式 + critic 反馈 → 修订策略) 学习成果

    下次遇到相似画像时，可召回历史成功的修订作为 T_t 的种子建议。
    """
    __tablename__ = "skill_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # 画像签名：risk_level + horizon + 年龄段 + custom_tickers_hash
    profile_signature = Column(String(128), nullable=False, index=True)
    risk_level = Column(String(16), nullable=True, index=True)
    intent = Column(String(32), nullable=True)
    # 触发本次学习的 critic 反馈
    critic_feedback = Column(Text, nullable=True)
    critic_score = Column(Integer, nullable=True)
    # 学习成果：从 base → final 的修订要点（自然语言）
    revision_summary = Column(Text, nullable=True)
    # 关联快照
    snapshot_id = Column(Integer, ForeignKey("portfolio_snapshots.id"), nullable=True)
    # 被复用次数
    reuse_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=dt.datetime.utcnow, index=True)
    last_used_at = Column(DateTime, nullable=True)


def get_db_url() -> str:
    """优先读环境变量 FINAGENT_DB_URL，缺省落到 memory/finagent.db"""
    env_url = os.environ.get("FINAGENT_DB_URL")
    if env_url:
        return env_url
    here = os.path.dirname(os.path.abspath(__file__))
    return f"sqlite:///{os.path.join(here, 'finagent.db')}"


_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        url = get_db_url()
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = create_engine(url, connect_args=connect_args, future=True)
        Base.metadata.create_all(_engine)
        _ensure_user_identity_columns(_engine)
        _ensure_snapshot_columns(_engine)
    return _engine


def _ensure_user_identity_columns(engine) -> None:
    """Backfill identity columns for existing SQLite databases."""
    if not str(engine.url).startswith("sqlite"):
        return
    inspector = inspect(engine)
    if "users" not in inspector.get_table_names():
        return
    existing = {c["name"] for c in inspector.get_columns("users")}
    with engine.begin() as conn:
        if "uuid" not in existing:
            conn.execute(text("ALTER TABLE users ADD COLUMN uuid VARCHAR(36)"))
        if "email" not in existing:
            conn.execute(text("ALTER TABLE users ADD COLUMN email VARCHAR(128)"))
        if "last_active_at" not in existing:
            conn.execute(text("ALTER TABLE users ADD COLUMN last_active_at DATETIME"))
        if "telegram_chat_id" not in existing:
            conn.execute(text("ALTER TABLE users ADD COLUMN telegram_chat_id VARCHAR(32)"))

        rows = conn.execute(text("SELECT id FROM users WHERE uuid IS NULL OR uuid = ''")).fetchall()
        for row in rows:
            conn.execute(
                text("UPDATE users SET uuid = :uuid WHERE id = :id"),
                {"uuid": str(uuid_lib.uuid4()), "id": row.id},
            )

        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_users_uuid ON users (uuid)"))
        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_users_email ON users (email)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_users_last_active_at ON users (last_active_at)"))


def _ensure_snapshot_columns(engine) -> None:
    """Backfill 6-3 columns (advice_date / is_followed) for existing SQLite databases."""
    if not str(engine.url).startswith("sqlite"):
        return
    inspector = inspect(engine)
    if "portfolio_snapshots" not in inspector.get_table_names():
        return
    existing = {c["name"] for c in inspector.get_columns("portfolio_snapshots")}
    with engine.begin() as conn:
        if "advice_date" not in existing:
            conn.execute(text("ALTER TABLE portfolio_snapshots ADD COLUMN advice_date DATE"))
            # 老数据回填：建议日 = 快照创建日
            conn.execute(text(
                "UPDATE portfolio_snapshots SET advice_date = date(created_at) "
                "WHERE advice_date IS NULL"
            ))
        if "is_followed" not in existing:
            conn.execute(text("ALTER TABLE portfolio_snapshots ADD COLUMN is_followed BOOLEAN"))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_portfolio_snapshots_advice_date "
            "ON portfolio_snapshots (advice_date)"
        ))


def get_session() -> Session:
    """获取一个新的 SQLAlchemy session，调用方负责 close。"""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, future=True)
    return _SessionLocal()


def reset_engine(db_url: Optional[str] = None):
    """测试用：重置引擎指向新的 URL（如内存 SQLite）"""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
    if db_url is not None:
        os.environ["FINAGENT_DB_URL"] = db_url
