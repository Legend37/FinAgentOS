# skills/__init__.py
"""Skill 系统 — 把"不同材料 → 不同 API → 不同 RAG 接口"统一成可注册可调度的能力单元。

每个 skill 是一个继承 BaseSkill 的类，声明：
  - name / description / category
  - 是否需要外部 API key
  - fetch() 抓取数据
  - to_rag_documents() 把抓取结果转成可喂 RAG 引擎的统一文档结构

Workflow / Agent 通过 SkillRegistry.get(name) 或按 category 查询动态选用。
"""
from .base import BaseSkill, SkillRegistry, SkillResult
from .registry import registry

# 触发 skill 自动注册
from . import macro_skill, policy_skill, filing_skill, news_skill  # noqa: F401

__all__ = ["BaseSkill", "SkillRegistry", "SkillResult", "registry"]
