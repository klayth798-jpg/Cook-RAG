"""
Skill系统核心框架
提供统一的 Skill 定义、注册、路由机制

架构说明:
- SkillDefinition: 描述一个 Skill 的元信息（对齐 MCP Tool Schema）
- SkillResult: Skill 执行结果的统一封装
- BaseSkill: Skill 抽象基类
- SkillRegistry: 全局注册表，管理所有已注册 Skill
- SkillRouter: LLM Function Calling 驱动的智能路由
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


# ────────────────────── 数据结构 ──────────────────────

@dataclass
class SkillDefinition:
    """
    Skill 的元信息定义，对齐 MCP Tool Schema
    可直接转换为 OpenAI function calling 的 tool 描述
    """
    name: str                              # Skill 唯一标识，如 "recipe_search"
    description: str                       # Skill 功能描述（供LLM理解）
    parameters: Dict[str, Any]             # JSON Schema 格式的参数描述
    required_params: List[str] = field(default_factory=list)  # 必填参数
    is_mcp: bool = False                   # 是否来自外部MCP Server

    def to_openai_function(self) -> dict:
        """转换为 OpenAI function calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required_params,
                },
            },
        }


@dataclass
class SkillResult:
    """Skill 执行结果的统一封装"""
    skill_name: str                        # 哪个 Skill 产生的
    success: bool                          # 是否执行成功
    content: str                           # 主要输出内容
    metadata: Dict[str, Any] = field(default_factory=dict)  # 附加元数据
    error: Optional[str] = None            # 错误信息


# ────────────────────── 抽象基类 ──────────────────────

class BaseSkill(ABC):
    """
    Skill 抽象基类
    所有自定义 Skill 和 MCP 适配 Skill 都继承此类
    """

    @abstractmethod
    def get_definition(self) -> SkillDefinition:
        """返回 Skill 元信息"""
        ...

    @abstractmethod
    def execute(self, **kwargs) -> SkillResult:
        """
        执行 Skill

        Args:
            **kwargs: 由 SkillRouter 从 LLM function calling 中解析出的参数

        Returns:
            SkillResult 执行结果
        """
        ...


# ────────────────────── 注册表 ──────────────────────

class SkillRegistry:
    """
    Skill 全局注册表
    管理所有已注册的 Skill 实例，支持按名称检索
    """

    def __init__(self):
        self._skills: Dict[str, BaseSkill] = {}

    def register(self, skill: BaseSkill):
        """注册一个 Skill"""
        defn = skill.get_definition()
        self._skills[defn.name] = skill
        logger.info(f"✅ 已注册 Skill: {defn.name} (MCP={defn.is_mcp})")

    def get(self, name: str) -> Optional[BaseSkill]:
        """按名称获取 Skill"""
        return self._skills.get(name)

    def list_definitions(self) -> List[SkillDefinition]:
        """列出所有已注册 Skill 的定义"""
        return [s.get_definition() for s in self._skills.values()]

    def list_names(self) -> List[str]:
        """列出所有 Skill 名称"""
        return list(self._skills.keys())

    def to_openai_tools(self) -> List[dict]:
        """将所有 Skill 转换为 OpenAI function calling tools 列表"""
        return [d.to_openai_function() for d in self.list_definitions()]


# ────────────────────── 智能路由 ──────────────────────

class SkillRouter:
    """
    LLM Function Calling 驱动的智能路由器

    工作流程:
    1. 将所有注册的 Skill 描述作为 tools 传给 LLM
    2. LLM 分析用户意图，选择最合适的 Skill 并提取参数
    3. Router 调用对应 Skill 的 execute() 方法
    4. 返回 SkillResult

    如果 LLM 选择了闲聊/兜底（无 tool_call），Router 返回 chitchat 类型
    """

    def __init__(self, registry: SkillRegistry, llm: ChatOpenAI):
        self.registry = registry
        self.llm = llm

    def route(self, user_query: str, memory_context: str = "无") -> SkillResult:
        """
        路由用户问题到最佳 Skill

        Args:
            user_query: 用户原始问题
            memory_context: 记忆上下文（可选）

        Returns:
            SkillResult 执行结果
        """
        tools = self.registry.to_openai_tools()

        if not tools:
            logger.warning("⚠️ 无已注册 Skill，直接返回兜底")
            return SkillResult(
                skill_name="chitchat",
                success=True,
                content="抱歉，系统暂未加载任何技能模块。",
            )

        # 构建系统提示
        system_prompt = (
            "你是一个食谱助手的技能路由器。根据用户的问题，判断应该调用哪个工具。\n"
            "- 如果用户问的是本地知识库中的菜谱（做法、食材、步骤），选择 recipe_search\n"
            "- 如果用户想搜索网上的菜谱或提供了URL，选择 web_recipe_fetch\n"
            "- 如果用户想规划一周膳食/搭配饮食，选择 meal_planner\n"
            "- 如果用户问营养成分/热量，选择 nutrition_calc\n"
            "- 如果是闲聊/打招呼/感谢等非食谱问题，不要调用任何工具，直接回复即可\n"
            f"\n当前对话记忆上下文：{memory_context}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]

        try:
            response = self.llm.invoke(
                messages,
                tools=tools,
                tool_choice="auto",
            )
        except Exception as e:
            logger.error(f"LLM路由调用失败: {e}")
            return SkillResult(
                skill_name="chitchat",
                success=True,
                content=f"路由暂时不可用，请稍后重试。({e})",
            )

        # 解析 LLM 响应
        if not response.tool_calls:
            # LLM 没有选择任何工具 → 闲聊/兜底
            logger.info("💬 路由结果: 闲聊（无 tool_call）")
            return SkillResult(
                skill_name="chitchat",
                success=True,
                content=response.content or "你好！有什么关于食谱的问题我可以帮你吗？",
            )

        # 取第一个 tool_call
        tool_call = response.tool_calls[0]
        skill_name = tool_call["name"]
        skill_args = tool_call["args"]

        logger.info(f"🎯 路由结果: {skill_name}, 参数: {json.dumps(skill_args, ensure_ascii=False)}")

        # 查找并执行 Skill
        skill = self.registry.get(skill_name)
        if not skill:
            logger.error(f"❌ 未找到 Skill: {skill_name}")
            return SkillResult(
                skill_name=skill_name,
                success=False,
                content=f"系统错误：未找到技能 '{skill_name}'",
                error=f"Skill '{skill_name}' not registered",
            )

        # 注入原始查询（所有 Skill 都可能需要）
        if "query" not in skill_args:
            skill_args["query"] = user_query

        try:
            result = skill.execute(**skill_args)
            return result
        except Exception as e:
            logger.error(f"Skill '{skill_name}' 执行失败: {e}")
            return SkillResult(
                skill_name=skill_name,
                success=False,
                content=f"技能 '{skill_name}' 执行出错: {str(e)}",
                error=str(e),
            )
