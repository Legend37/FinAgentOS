# skills/base.py
"""Skill 基类与结果协议"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable


@dataclass
class SkillResult:
    """所有 skill 的统一输出协议"""
    skill_name: str
    items: List[Dict[str, Any]]    # 结构化记录，每条 {title, content, date, source, ...}
    summary: str                    # 短文本摘要（喂 LLM prompt）
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_rag_documents(self) -> List[Dict[str, str]]:
        """转成 RAG 引擎可吃的 {text, source} 列表"""
        docs = []
        for item in self.items:
            title = item.get("title", "")
            content = item.get("content", "")
            text = f"{title}\n{content}".strip()
            if text:
                docs.append({
                    "text": text,
                    "source": f"{self.skill_name}::{item.get('date', '')}::{item.get('source', '')}",
                })
        return docs


class BaseSkill:
    """Skill 基类。子类需实现 fetch() 返回 SkillResult。

    属性约定：
      name: 全局唯一标识（kebab-case）
      description: 一句话说明该 skill 提供什么材料
      category: 分类标签 — "market" / "news" / "macro" / "policy" / "filing" / "research"
      key_required: 是否需要外部 API key（用于 cost 决策与降级）
    """
    name: str = "base"
    description: str = ""
    category: str = "misc"
    key_required: bool = False

    def fetch(self, **params) -> SkillResult:
        """抓取数据，必须由子类实现"""
        raise NotImplementedError(f"{self.__class__.__name__}.fetch() 未实现")

    def is_relevant(self, intent: str, profile: Dict[str, Any]) -> bool:
        """该 skill 是否与当前意图/画像相关，用于自动选择。

        默认全部相关；子类可重写做更精细的过滤。
        """
        return True


class SkillRegistry:
    """全局 skill 注册表，按 name 索引、按 category 分组"""

    def __init__(self):
        self._skills: Dict[str, BaseSkill] = {}

    def register(self, skill: BaseSkill):
        if not skill.name or skill.name == "base":
            raise ValueError(f"Skill must have a unique name, got: {skill.name!r}")
        if skill.name in self._skills:
            raise ValueError(f"Skill {skill.name!r} 已存在")
        self._skills[skill.name] = skill

    def get(self, name: str) -> Optional[BaseSkill]:
        return self._skills.get(name)

    def list_all(self) -> List[BaseSkill]:
        return list(self._skills.values())

    def list_by_category(self, category: str) -> List[BaseSkill]:
        return [s for s in self._skills.values() if s.category == category]

    def select_relevant(self, intent: str, profile: Dict[str, Any]) -> List[BaseSkill]:
        """按意图 + 画像挑选相关 skill 集合"""
        return [s for s in self._skills.values() if s.is_relevant(intent, profile)]

    def fetch_many(self, names: List[str], **shared_params) -> Dict[str, SkillResult]:
        """一次性调用多个 skill，返回 {name: SkillResult}"""
        out = {}
        for n in names:
            sk = self.get(n)
            if sk is None:
                out[n] = SkillResult(skill_name=n, items=[], summary="", error=f"skill {n} 未注册")
                continue
            try:
                out[n] = sk.fetch(**shared_params)
            except Exception as e:
                out[n] = SkillResult(skill_name=n, items=[], summary="", error=str(e))
        return out
