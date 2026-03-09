"""
Skill 系统包初始化

导出所有 Skill 类和工具函数，方便外部统一导入
"""

from .base import (
    BaseSkill,
    SkillDefinition,
    SkillRegistry,
    SkillResult,
    SkillRouter,
)
from .mcp_client import MCPSkillClient, check_mcp_environment
from .meal_planner_skill import MealPlannerSkill
from .nutrition_skill import NutritionSkill
from .recipe_search_skill import RecipeSearchSkill
from .web_recipe_skill import WebRecipeSkill

__all__ = [
    # 框架核心
    "BaseSkill",
    "SkillDefinition",
    "SkillResult",
    "SkillRegistry",
    "SkillRouter",
    # MCP 客户端
    "MCPSkillClient",
    "check_mcp_environment",
    # 具体 Skill
    "WebRecipeSkill",      # MCP: mcp-server-fetch (开源)
    "RecipeSearchSkill",   # 本地 RAG 混合检索
    "MealPlannerSkill",    # LLM 膳食规划
    "NutritionSkill",      # LLM 营养估算
]
