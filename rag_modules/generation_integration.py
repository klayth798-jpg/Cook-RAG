"""
生成集成模块
负责 LLM 初始化、查询改写、MQE 和记忆感知回答生成
"""

import logging
import os
from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class GenerationIntegrationModule:
    """生成集成模块 - 负责 LLM 集成和回答生成"""

    def __init__(self, model_name: str = None, temperature: float = 0.1, max_tokens: int = 2048):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None
        self.setup_llm()

    def setup_llm(self):
        if self.model_name is None:
            self.model_name = os.getenv("LLM_MODEL_ID")
        logger.info(f"正在初始化 LLM: {self.model_name}")
        api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL")
        if not api_key:
            raise ValueError("请设置 LLM_API_KEY 环境变量")
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=api_key,
            base_url=base_url,
        )
        logger.info("LLM 初始化完成")

    def _build_context(self, context_docs: List[Document]) -> str:
        if not context_docs:
            return ""
        return "\n\n---\n\n".join(doc.page_content for doc in context_docs)

    def query_router(self, query: str) -> str:
        prompt = PromptTemplate(
            template="""
你是食谱问答路由器。请将用户问题分类为以下三类之一：
- list: 用户想要推荐、列举、筛选多个菜品
- detail: 用户想了解某道菜的详细做法、步骤、食材
- general: 其他一般性说明、技巧、原理问题

用户问题: {query}

只输出一个标签：list / detail / general
""",
            input_variables=["query"],
        )
        chain = ({"query": RunnablePassthrough()} | prompt | self.llm | StrOutputParser())
        result = chain.invoke(query).strip().lower()
        if result not in {"list", "detail", "general"}:
            return "general"
        return result

    def query_rewrite(self, query: str) -> str:
        prompt = PromptTemplate(
            template="""
你是一个智能查询分析助手。请分析用户的查询，判断是否需要重写以提高食谱搜索效果。

原始查询: {query}

如果查询已经足够明确，直接返回原查询；如果查询模糊，请将其改写为更适合食谱检索的表达。
只输出最终查询。
""",
            input_variables=["query"],
        )
        chain = ({"query": RunnablePassthrough()} | prompt | self.llm | StrOutputParser())
        response = chain.invoke(query).strip()
        if response != query:
            logger.info(f"查询已重写: '{query}' → '{response}'")
        return response

    def context_aware_rewrite(self, query: str, memory_context: str) -> str:
        prompt = PromptTemplate(
            template="""
你是一个对话式烹饪助手的查询分析器。请根据历史上下文补全当前提问中的指代和省略。

历史上下文:
{memory_context}

当前提问: {query}

如果当前提问已经完整明确，直接返回原查询。只输出改写后的查询。
""",
            input_variables=["query", "memory_context"],
        )
        chain = (
            {"query": RunnablePassthrough(), "memory_context": lambda _: memory_context}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        result = chain.invoke(query).strip()
        if result != query:
            logger.info(f"上下文感知改写: '{query}' → '{result}'")
        return result

    def multi_query_expansion(self, query: str, n: int = 2) -> List[str]:
        prompt = PromptTemplate(
            template="""
你是食谱检索查询扩展助手。请围绕用户问题生成 {n} 个语义相近但表达不同的检索查询。

原始查询: {query}

每行一个查询，不加序号。只输出扩展结果。
""",
            input_variables=["query", "n"],
        )
        chain = ({"query": RunnablePassthrough(), "n": lambda _: str(n)} | prompt | self.llm | StrOutputParser())
        try:
            result = chain.invoke(query)
            lines = [line.strip("- \t0123456789.") for line in result.splitlines()]
            expansions = [line for line in lines if line and len(line) > 2]
            all_queries = [query]
            for item in expansions[:n]:
                if item not in all_queries:
                    all_queries.append(item)
            return all_queries
        except Exception as error:
            logger.warning(f"MQE扩展失败: {error}")
            return [query]

    def generate_basic_answer(self, query: str, context_docs: List[Document]) -> str:
        context = self._build_context(context_docs)
        prompt = ChatPromptTemplate.from_template(
            """
你是一位专业的烹饪助手。请根据以下食谱信息回答用户的问题。

用户问题: {question}

相关食谱信息:
{context}

请提供详细、实用的回答。如果信息不足，请诚实说明。
"""
        )
        chain = ({"question": RunnablePassthrough(), "context": lambda _: context} | prompt | self.llm | StrOutputParser())
        return chain.invoke(query)

    def generate_list_answer(self, query: str, context_docs: List[Document]) -> str:
        context = self._build_context(context_docs)
        prompt = ChatPromptTemplate.from_template(
            """
你是一位食谱推荐助手。请根据给定食谱内容，为用户列出合适的菜品列表。

用户问题: {question}

候选食谱:
{context}

输出要求：
- 用列表方式推荐
- 每个菜品附一句推荐理由
- 避免冗长
"""
        )
        chain = ({"question": RunnablePassthrough(), "context": lambda _: context} | prompt | self.llm | StrOutputParser())
        return chain.invoke(query)

    def generate_memory_aware_answer(self, query: str, documents: List[Document], memory_context: str = "", route_type: str = "general") -> str:
        context = self._build_context(documents)
        style_instruction = "详细步骤化说明" if route_type == "detail" else "清晰自然地回答"
        prompt = ChatPromptTemplate.from_template(
            """
你是一位专业的烹饪助手，请结合历史对话记忆和检索到的食谱信息来回答用户。

历史记忆:
{memory_context}

用户问题: {question}

检索到的食谱信息:
{context}

回答要求:
- {style_instruction}
- 如果历史记忆里有用户偏好，请优先对齐用户偏好
- 如果检索内容不足，请明确说明
"""
        )
        chain = (
            {
                "question": RunnablePassthrough(),
                "context": lambda _: context,
                "memory_context": lambda _: memory_context or "无",
                "style_instruction": lambda _: style_instruction,
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(query)

    def generate_memory_aware_answer_stream(self, query: str, documents: List[Document], memory_context: str = "", route_type: str = "general"):
        answer = self.generate_memory_aware_answer(query, documents, memory_context, route_type)
        for char in answer:
            yield char
