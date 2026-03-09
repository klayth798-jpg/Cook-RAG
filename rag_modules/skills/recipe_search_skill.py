"""
本地食谱检索 Skill
将 C10 已有的 RAG 混合检索流水线封装为标准 Skill。
"""

import logging

from .base import BaseSkill, SkillDefinition, SkillResult

logger = logging.getLogger(__name__)


class RecipeSearchSkill(BaseSkill):
    def __init__(self, retrieval_module=None, generation_module=None, index_module=None, memory_manager=None):
        self.retrieval = retrieval_module
        self.generation = generation_module
        self.index = index_module
        self.memory = memory_manager

    def get_definition(self) -> SkillDefinition:
        return SkillDefinition(
            name="recipe_search",
            description=(
                "从本地知识库中搜索食谱信息。当用户询问某道菜怎么做、需要什么食材、"
                "有什么菜推荐等食谱相关问题时使用。"
            ),
            parameters={
                "query": {"type": "string", "description": "用户的食谱搜索问题"},
                "category": {"type": "string", "description": "可选的分类过滤"},
                "difficulty": {"type": "string", "description": "可选的难度过滤"},
            },
            required_params=["query"],
            is_mcp=False,
        )

    def execute(self, **kwargs) -> SkillResult:
        query = kwargs.get("query", "")
        category = kwargs.get("category")
        difficulty = kwargs.get("difficulty")
        if not query:
            return SkillResult(skill_name="recipe_search", success=False, content="请提供查询内容", error="Empty query")

        try:
            memory_context = "无"
            if self.memory:
                memory_context = self.memory.build_memory_prompt_context(query)

            rewritten = query
            if self.generation:
                rewritten = self.generation.query_rewrite(query)
                if memory_context != "无":
                    rewritten = self.generation.context_aware_rewrite(rewritten, memory_context)

            queries = [rewritten]
            if self.generation:
                queries = self.generation.multi_query_expansion(rewritten)

            all_docs = []
            seen_texts = set()
            for item in queries:
                if category or difficulty:
                    filters = {}
                    if category:
                        filters["category"] = category
                    if difficulty:
                        filters["difficulty"] = difficulty
                    docs = self.retrieval.metadata_filtered_search(query=item, filters=filters, top_k=5)
                else:
                    docs = self.retrieval.hybrid_search(query=item, top_k=5)
                for doc in docs:
                    text = getattr(doc, "page_content", str(doc))
                    if text not in seen_texts:
                        seen_texts.add(text)
                        all_docs.append(doc)

            if not all_docs:
                return SkillResult(skill_name="recipe_search", success=True, content=f"抱歉，在本地知识库中没有找到与「{query}」相关的食谱。", metadata={"num_results": 0})

            if self.generation:
                answer = self.generation.generate_memory_aware_answer(
                    query=query,
                    documents=all_docs[:8],
                    memory_context=memory_context,
                )
            else:
                answer = "\n\n---\n\n".join(getattr(doc, "page_content", str(doc)) for doc in all_docs[:5])

            return SkillResult(
                skill_name="recipe_search",
                success=True,
                content=answer,
                metadata={"num_results": len(all_docs), "queries_used": queries},
            )
        except Exception as error:
            logger.error(f"本地食谱检索失败: {error}")
            return SkillResult(skill_name="recipe_search", success=False, content=f"检索过程中出错: {error}", error=str(error))
