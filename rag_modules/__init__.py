from .data_preparation import DataPreparationModule
from .index_construction import IndexConstructionModule
from .retrieval_optimization import RetrievalOptimizationModule
from .generation_integration import GenerationIntegrationModule
from .conversation_memory import MemoryItem, WorkingMemory, EpisodicMemory, MemoryManager
from .skills import (
    SkillRegistry, SkillRouter, MCPSkillClient, check_mcp_environment,
    WebRecipeSkill, RecipeSearchSkill, MealPlannerSkill, NutritionSkill,
)

__all__ = [
    # ── RAG 核心模块 ──
    'DataPreparationModule',
    'IndexConstructionModule', 
    'RetrievalOptimizationModule',
    'GenerationIntegrationModule',
    # ── 记忆系统 ──
    'MemoryItem',
    'WorkingMemory',
    'EpisodicMemory',
    'MemoryManager',
    # ── Skill 系统 ──
    'SkillRegistry',
    'SkillRouter',
    'MCPSkillClient',
    'check_mcp_environment',
    'WebRecipeSkill',
    'RecipeSearchSkill',
    'MealPlannerSkill',
    'NutritionSkill',
]

__version__ = "4.0.0"  # Milvus + 记忆系统 + MCP Skill 系统
