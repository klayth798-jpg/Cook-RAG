"""
营养估算 Skill (自定义 LLM Skill)

估算菜品/食谱的营养成分：热量、蛋白质、碳水化合物、脂肪等。
由于市面上没有合适的开源营养 MCP Server，采用 LLM + 内置知识 实现。

亮点:
- 基于常见食材营养数据库的 LLM 推理
- 支持单菜估算和多菜组合估算
- 可结合本地知识库中的食材清单精确估算
"""

import logging
from typing import Any, Dict

from langchain_openai import ChatOpenAI

from .base import BaseSkill, SkillDefinition, SkillResult

logger = logging.getLogger(__name__)


class NutritionSkill(BaseSkill):
    """
    营养成分估算 Skill

    工作流:
    1. 如果本地知识库有该菜品 → 获取其食材清单
    2. 基于食材清单 + LLM 内置营养知识 → 估算营养
    3. 返回结构化的营养分析报告
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        retrieval_module=None,
    ):
        self.llm = llm
        self.retrieval = retrieval_module

    def get_definition(self) -> SkillDefinition:
        return SkillDefinition(
            name="nutrition_calc",
            description=(
                "估算菜品或食材的营养成分。当用户询问某道菜的热量、"
                "蛋白质含量、是否适合减脂等营养相关问题时使用。"
                "例如: '红烧肉热量多少', '这道菜适合减肥吗', '100克鸡胸肉的蛋白质'"
            ),
            parameters={
                "query": {
                    "type": "string",
                    "description": "用户的营养相关问题",
                },
                "dish_name": {
                    "type": "string",
                    "description": "需要分析营养的菜品名称",
                },
            },
            required_params=["query"],
            is_mcp=False,
        )

    def execute(self, **kwargs) -> SkillResult:
        query = kwargs.get("query", "")
        dish_name = kwargs.get("dish_name", "")

        try:
            # 1. 尝试从知识库获取食材信息
            ingredient_info = self._get_ingredients(dish_name or query)

            # 2. LLM 营养估算
            analysis = self._analyze_nutrition(
                query=query,
                dish_name=dish_name,
                ingredient_info=ingredient_info,
            )

            return SkillResult(
                skill_name="nutrition_calc",
                success=True,
                content=analysis,
                metadata={
                    "dish_name": dish_name,
                    "has_ingredient_data": bool(ingredient_info),
                },
            )

        except Exception as e:
            logger.error(f"营养估算失败: {e}")
            return SkillResult(
                skill_name="nutrition_calc",
                success=False,
                content=f"营养估算时出错: {str(e)}",
                error=str(e),
            )

    def _get_ingredients(self, dish_or_query: str) -> str:
        """从本地知识库获取食材信息"""
        if not self.retrieval:
            return ""

        try:
            results = self.retrieval.hybrid_search(
                query=f"{dish_or_query} 食材 用料",
                top_k=3,
            )
            if results:
                texts = []
                for doc in results:
                    text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                    texts.append(text)
                return "\n---\n".join(texts)
            return ""
        except Exception:
            return ""

    def _analyze_nutrition(
        self, query: str, dish_name: str, ingredient_info: str
    ) -> str:
        """使用 LLM 估算营养成分"""
        context_section = ""
        if ingredient_info:
            context_section = f"""
## 知识库中的食材信息（供参考）
{ingredient_info[:3000]}
"""

        prompt = f"""你是一位注册营养师。请根据以下信息估算菜品的营养成分。

## 用户问题
{query}

## 菜品名称
{dish_name or '（从用户问题中识别）'}
{context_section}

## 输出要求
请提供以下格式的营养分析（以每份/每人为单位估算）:

### 🥗 营养分析: [菜名]

| 营养成分 | 含量（每份约） |
|---------|------------|
| 热量 | XX kcal |
| 蛋白质 | XX g |
| 碳水化合物 | XX g |
| 脂肪 | XX g |
| 膳食纤维 | XX g |
| 钠 | XX mg |

### 💡 健康建议
- 适合人群: ...
- 注意事项: ...
- 搭配建议: ...

### ⚠️ 免责声明
以上数据为基于常见食材的估算值，实际营养成分因食材产地、
烹饪方式、用量等因素而异。如有特殊饮食需求，请咨询专业营养师。

注意:
1. 给出具体数字而非范围（更有参考价值）
2. 如果从知识库获取了食材信息，基于实际用量估算
3. 如果没有食材信息，基于该菜品的常见做法估算"""

        response = self.llm.invoke(prompt)
        return response.content
