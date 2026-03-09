"""
膳食规划 Skill (自定义 LLM Skill)

当开源 MCP 生态中没有合适的膳食规划 Server 时，自己实现。
利用 LLM + 本地知识库 + 用户记忆，生成个性化的膳食计划。

特色:
- 结合本地食谱知识库（有什么菜可以做）
- 结合用户对话记忆（用户的偏好、禁忌）
- 支持按天数、人数、饮食偏好定制
"""

import logging
from typing import Any, Dict

from langchain_openai import ChatOpenAI

from .base import BaseSkill, SkillDefinition, SkillResult

logger = logging.getLogger(__name__)


class MealPlannerSkill(BaseSkill):
    """
    LLM 驱动的膳食规划 Skill

    工作流:
    1. 从知识库中获取可用菜谱概览
    2. 从记忆中获取用户偏好
    3. LLM 生成个性化膳食计划
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        retrieval_module=None,
        memory_manager=None,
    ):
        self.llm = llm
        self.retrieval = retrieval_module
        self.memory = memory_manager

    def get_definition(self) -> SkillDefinition:
        return SkillDefinition(
            name="meal_planner",
            description=(
                "制定膳食计划。当用户想要规划一天、一周的饮食，"
                "或者需要搭配菜品建议时使用。"
                "例如: '帮我规划一周的晚餐', '明天请客8个人吃什么好', '减脂期一周食谱'"
            ),
            parameters={
                "query": {
                    "type": "string",
                    "description": "用户的膳食规划需求",
                },
                "days": {
                    "type": "integer",
                    "description": "需要规划的天数，默认7天",
                },
                "people": {
                    "type": "integer",
                    "description": "用餐人数，默认2人",
                },
                "preference": {
                    "type": "string",
                    "description": "饮食偏好，如: 清淡/减脂/增肌/素食/无辣",
                },
            },
            required_params=["query"],
            is_mcp=False,
        )

    def execute(self, **kwargs) -> SkillResult:
        query = kwargs.get("query", "")
        days = kwargs.get("days", 7)
        people = kwargs.get("people", 2)
        preference = kwargs.get("preference", "均衡")

        try:
            # 1. 从知识库获取可用菜品列表
            available_dishes = self._get_available_dishes()

            # 2. 从记忆中获取用户偏好
            user_memory = "无特殊偏好记录"
            if self.memory:
                user_memory = self.memory.build_memory_prompt_context(query)

            # 3. LLM 生成膳食计划
            plan = self._generate_meal_plan(
                query=query,
                days=days,
                people=people,
                preference=preference,
                available_dishes=available_dishes,
                user_memory=user_memory,
            )

            return SkillResult(
                skill_name="meal_planner",
                success=True,
                content=plan,
                metadata={
                    "days": days,
                    "people": people,
                    "preference": preference,
                    "available_dishes_count": len(available_dishes.split("\n")),
                },
            )

        except Exception as e:
            logger.error(f"膳食规划失败: {e}")
            return SkillResult(
                skill_name="meal_planner",
                success=False,
                content=f"膳食规划时出错: {str(e)}",
                error=str(e),
            )

    def _get_available_dishes(self) -> str:
        """从知识库获取可用菜品概览"""
        if not self.retrieval:
            return "（知识库未加载，将基于通用知识规划）"

        try:
            # 搜索知识库中的菜品
            categories = ["川菜", "粤菜", "家常菜", "素菜", "汤类", "主食"]
            dishes = []
            for cat in categories:
                results = self.retrieval.hybrid_search(query=cat, top_k=3)
                for doc in results:
                    text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                    # 提取菜名（通常在文档开头）
                    first_line = text.split("\n")[0].strip()
                    if first_line and len(first_line) < 20:
                        dishes.append(f"- {first_line}")

            if dishes:
                return "知识库中的部分菜品:\n" + "\n".join(list(set(dishes))[:20])
            return "（知识库中暂无分类菜品索引）"
        except Exception as e:
            logger.warning(f"获取菜品列表失败: {e}")
            return "（获取菜品列表时出错，将基于通用知识规划）"

    def _generate_meal_plan(
        self,
        query: str,
        days: int,
        people: int,
        preference: str,
        available_dishes: str,
        user_memory: str,
    ) -> str:
        """使用 LLM 生成膳食计划"""
        prompt = f"""你是一位专业的营养师和膳食规划专家。请根据以下信息生成个性化的膳食计划。

## 用户需求
{query}

## 规划参数
- 天数: {days} 天
- 用餐人数: {people} 人
- 饮食偏好: {preference}

## 可参考的本地菜谱
{available_dishes}

## 用户历史偏好（来自对话记忆）
{user_memory}

## 输出要求
请按以下格式输出膳食计划:

### 📅 第X天
- 🌅 早餐: [菜品] (简要说明)
- ☀️ 午餐: [主菜] + [配菜] + [汤/饮品]
- 🌙 晚餐: [主菜] + [配菜] + [汤/饮品]

### 📊 总体建议
- 营养搭配建议
- 采购清单概要
- 注意事项

注意:
1. 尽量使用本地菜谱中已有的菜品，这样用户可以直接查看做法
2. 注意营养均衡：蛋白质、碳水、蔬菜搭配
3. 考虑食材的复用，减少浪费
4. 如果有用户偏好记录，要尊重其口味和禁忌"""

        response = self.llm.invoke(prompt)
        return response.content
