"""
Web 食谱获取 Skill
优先使用 Tavily 联网搜索，失败时回退到 mcp-server-fetch。
"""

import logging
import os
import urllib.parse

import requests
from langchain_openai import ChatOpenAI

from .base import BaseSkill, SkillDefinition, SkillResult
from .mcp_client import MCPSkillClient

logger = logging.getLogger(__name__)


class WebRecipeSkill(BaseSkill):
    RECIPE_SITES = [
        {"name": "下厨房", "url_template": "https://www.xiachufang.com/search/?keyword={query}"},
        {"name": "美食杰", "url_template": "https://www.meishij.net/chaxun/mingan/{query}/"},
        {"name": "豆果美食", "url_template": "https://www.douguo.com/recipe/search/{query}"},
    ]

    def __init__(self, llm: ChatOpenAI, mcp_client: MCPSkillClient):
        self.llm = llm
        self.mcp_client = mcp_client
        self._tavily_api_key = os.getenv("TAVILY_API_KEY", "").strip()
        self._tavily_enabled = bool(self._tavily_api_key)
        self._mcp_available = mcp_client.is_server_available("fetch")

    def get_definition(self) -> SkillDefinition:
        return SkillDefinition(
            name="web_recipe_fetch",
            description="从互联网获取食谱。当用户想搜索网上的某道菜做法，或提供了链接时使用。",
            parameters={
                "url": {"type": "string", "description": "食谱网页 URL，可选"},
                "query": {"type": "string", "description": "用户原始查询"},
            },
            required_params=["query"],
            is_mcp=True,
        )

    def execute(self, **kwargs) -> SkillResult:
        query = kwargs.get("query", "")
        url = kwargs.get("url", "")

        if not self._mcp_available and not self._tavily_enabled:
            return self._fallback_response(query)

        if url and not self._is_search_engine_url(url):
            return self._fetch_and_extract(query, url)

        if self._tavily_enabled and query:
            tavily_result = self._search_with_tavily(query)
            if tavily_result.success:
                return tavily_result

        if not self._mcp_available:
            return self._fallback_response(query)

        encoded_query = urllib.parse.quote(query)
        for site in self.RECIPE_SITES:
            site_url = site["url_template"].format(query=encoded_query)
            fetch_result = self.mcp_client.call_tool(
                server_name="fetch",
                tool_name="fetch",
                arguments={"url": site_url, "max_length": 8000},
            )
            if fetch_result.success:
                recipe_info = self._extract_recipe_from_content(
                    query=query,
                    web_content=fetch_result.content,
                    source_url=site_url,
                    site_name=site["name"],
                )
                return SkillResult(
                    skill_name="web_recipe_fetch",
                    success=True,
                    content=recipe_info,
                    metadata={
                        "source_url": site_url,
                        "source_site": site["name"],
                        "mcp_server": "mcp-server-fetch",
                    },
                )
        return self._fallback_response(query)

    def _search_with_tavily(self, query: str) -> SkillResult:
        try:
            response = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self._tavily_api_key,
                    "query": f"{query} 食谱 做法",
                    "search_depth": "advanced",
                    "max_results": 5,
                    "include_answer": True,
                    "include_raw_content": True,
                },
                timeout=25,
            )
            if response.status_code != 200:
                return SkillResult(skill_name="web_recipe_fetch", success=False, content=f"Tavily 搜索失败: HTTP {response.status_code}", error=response.text[:300])
            data = response.json()
            results = data.get("results", []) or []
            answer = data.get("answer", "")
            if not results and not answer:
                return SkillResult(skill_name="web_recipe_fetch", success=False, content="Tavily 没有返回可用结果", error="empty results")
            blocks = []
            if answer:
                blocks.append(f"Tavily摘要:\n{answer}")
            for index, item in enumerate(results, 1):
                title = item.get("title", "")
                source_url = item.get("url", "")
                snippet = item.get("raw_content") or item.get("content") or ""
                blocks.append(f"来源{index}: {title}\nURL: {source_url}\n内容:\n{snippet[:1800]}")
            recipe_info = self._extract_recipe_from_content(
                query=query,
                web_content="\n\n---\n\n".join(blocks),
                source_url="https://api.tavily.com/search",
                site_name="Tavily",
            )
            return SkillResult(
                skill_name="web_recipe_fetch",
                success=True,
                content=recipe_info,
                metadata={"source": "tavily", "mcp_server": "tavily-api", "result_count": len(results)},
            )
        except Exception as error:
            logger.error(f"Tavily 搜索异常: {error}")
            return SkillResult(skill_name="web_recipe_fetch", success=False, content=f"Tavily 搜索异常: {error}", error=str(error))

    def _is_search_engine_url(self, url: str) -> bool:
        return any(domain in url for domain in ["google.com", "baidu.com", "bing.com", "sogou.com"])

    def _fetch_and_extract(self, query: str, url: str) -> SkillResult:
        fetch_result = self.mcp_client.call_tool(
            server_name="fetch",
            tool_name="fetch",
            arguments={"url": url, "max_length": 8000},
        )
        if not fetch_result.success:
            return self._fallback_response(query)
        recipe_info = self._extract_recipe_from_content(query, fetch_result.content, url)
        return SkillResult(skill_name="web_recipe_fetch", success=True, content=recipe_info, metadata={"source_url": url, "mcp_server": "mcp-server-fetch"})

    def _extract_recipe_from_content(self, query: str, web_content: str, source_url: str, site_name: str = "") -> str:
        if len(web_content) > 6000:
            web_content = web_content[:6000] + "\n...(内容已截断)"
        prompt = f"""你是一个食谱信息提取专家。请从以下网页内容中提取与用户问题相关的食谱信息。

用户问题: {query}
网页来源: {source_url}

网页内容（Markdown格式）:
---
{web_content}
---

请提取：菜名、简介、食材、步骤、技巧、烹饪时间。
如果网页并非完整食谱，请提取最相关摘要。请用清晰中文输出。"""
        try:
            response = self.llm.invoke(prompt)
            result = response.content
            result += f"\n\n📌 来源: {site_name + ' ' if site_name else ''}{source_url}"
            return result
        except Exception as error:
            return f"获取到了网页内容，但提取食谱信息时出错: {error}\n来源: {source_url}"

    def _fallback_response(self, query: str) -> SkillResult:
        try:
            response = self.llm.invoke(
                f"用户想了解：{query}\n请根据你的知识，简要介绍这道菜的做法。注意说明这是基于模型内置知识。"
            )
            return SkillResult(
                skill_name="web_recipe_fetch",
                success=True,
                content=response.content + "\n\n⚠️ 注: 联网抓取不可用，以上基于模型内置知识",
                metadata={"fallback": True},
            )
        except Exception as error:
            return SkillResult(skill_name="web_recipe_fetch", success=False, content=f"网络食谱获取失败: {error}", error=str(error))
