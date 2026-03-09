"""
RAG系统主程序（Milvus + 记忆 + MCP Skill 版本）
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent))

from config import DEFAULT_CONFIG, RAGConfig
from rag_modules import (
    DataPreparationModule,
    GenerationIntegrationModule,
    IndexConstructionModule,
    MCPSkillClient,
    MealPlannerSkill,
    MemoryManager,
    NutritionSkill,
    RecipeSearchSkill,
    RetrievalOptimizationModule,
    SkillRegistry,
    SkillRouter,
    WebRecipeSkill,
    check_mcp_environment,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class RecipeRAGSystem:
    """食谱 RAG 系统主类"""

    def __init__(self, config: RAGConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None
        self.memory_manager = None
        self.skill_registry: Optional[SkillRegistry] = None
        self.skill_router: Optional[SkillRouter] = None
        self.mcp_client: Optional[MCPSkillClient] = None

        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")
        if not os.getenv("LLM_API_KEY"):
            raise ValueError("请设置 LLM_API_KEY 环境变量")

    def initialize_system(self):
        print("🚀 正在初始化 RAG 系统（Milvus + MCP Skill 版本）...")
        print("📂 初始化数据准备模块...")
        self.data_module = DataPreparationModule(self.config.data_path)

        print("🔗 初始化 Milvus 索引模块...")
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            milvus_host=self.config.milvus_host,
            milvus_port=self.config.milvus_port,
            collection_name=self.config.milvus_collection_name,
            dimension=self.config.milvus_dimension,
        )

        print("🤖 初始化生成集成模块...")
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        if self.config.enable_memory:
            print("🧠 初始化记忆管理模块...")
            self.memory_manager = MemoryManager(config=self.config, index_module=self.index_module)
            self.memory_manager.set_llm(self.generation_module.llm)
            print("✅ 记忆模块已启用")
        else:
            print("⚪ 记忆模块未启用")

        if self.config.enable_skills:
            print("🔌 检查 MCP 环境...")
            self.mcp_client = MCPSkillClient()
            env_status = check_mcp_environment()
            for name, available in env_status.items():
                status = "✅" if available else "⚠️"
                print(f"   {status} {name}: {'可用' if available else '不可用'}")

        print("✅ 系统初始化完成！")

    def build_knowledge_base(self):
        print("\n📚 正在构建知识库...")
        if self.index_module.collection_exists():
            print("✅ Milvus 中已存在向量索引，跳过重建！")
            print("📄 加载食谱文档...")
            self.data_module.load_documents()
            print("✂️ 进行文本分块...")
            chunks = self.data_module.chunk_documents()
        else:
            print("📦 Milvus 中未找到索引，开始构建...")
            print("📄 加载食谱文档...")
            self.data_module.load_documents()
            print("✂️ 进行文本分块...")
            chunks = self.data_module.chunk_documents()
            print("🔨 构建 Milvus 向量索引...")
            self.index_module.build_vector_index(chunks)

        print("🔍 初始化检索优化...")
        self.retrieval_module = RetrievalOptimizationModule(self.index_module, chunks)

        stats = self.data_module.get_statistics()
        milvus_stats = self.index_module.get_collection_stats()
        print("\n📊 知识库统计:")
        print(f"   文档总数: {stats['total_documents']}")
        print(f"   文本块数: {stats['total_chunks']}")
        print(f"   Milvus向量数: {milvus_stats.get('row_count', 'N/A')}")
        print(f"   菜品分类: {list(stats['categories'].keys())}")
        print(f"   难度分布: {stats['difficulties']}")
        print("✅ 知识库构建完成！")

        if self.config.enable_skills:
            self._initialize_skills()

    def _initialize_skills(self):
        print("\n🎯 正在初始化 Skill 系统...")
        self.skill_registry = SkillRegistry()

        self.skill_registry.register(
            RecipeSearchSkill(
                retrieval_module=self.retrieval_module,
                generation_module=self.generation_module,
                index_module=self.index_module,
                memory_manager=self.memory_manager,
            )
        )

        if self.config.enable_mcp_fetch and self.mcp_client:
            self.skill_registry.register(WebRecipeSkill(llm=self.generation_module.llm, mcp_client=self.mcp_client))

        self.skill_registry.register(
            MealPlannerSkill(
                llm=self.generation_module.llm,
                retrieval_module=self.retrieval_module,
                memory_manager=self.memory_manager,
            )
        )
        self.skill_registry.register(
            NutritionSkill(llm=self.generation_module.llm, retrieval_module=self.retrieval_module)
        )
        self.skill_router = SkillRouter(registry=self.skill_registry, llm=self.generation_module.llm)
        print(f"✅ Skill 系统就绪，已注册 {len(self.skill_registry.list_names())} 个技能")

    def ask_question(self, question: str, stream: bool = False):
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")

        print(f"\n❓ 用户问题: {question}")
        if self.config.enable_skills and self.skill_router:
            memory_context = "无"
            if self.memory_manager:
                memory_context = self.memory_manager.build_memory_prompt_context(question)
            result = self.skill_router.route(question, memory_context)
            if result.skill_name == "recipe_search":
                print("🎯 Skill 路由 → 本地知识库检索")
                return self._ask_with_rag_pipeline(question, stream, memory_context)
            if result.skill_name == "chitchat":
                print("💬 Skill 路由 → 闲聊")
                if self.memory_manager:
                    self.memory_manager.add_interaction(question, result.content)
                return result.content
            print(f"🎯 Skill 路由 → {result.skill_name}")
            if not result.success:
                print(f"⚠️ Skill 执行失败: {result.error}")
                print("🔄 降级到本地 RAG 检索...")
                return self._ask_with_rag_pipeline(question, stream, memory_context)
            if self.memory_manager:
                self.memory_manager.add_interaction(question, result.content)
                self.memory_manager.execute_forgetting()
            return result.content

        return self._ask_with_rag_pipeline(question, stream)

    def _ask_with_rag_pipeline(self, question: str, stream: bool = False, memory_context: str = None):
        route_type = self.generation_module.query_router(question)
        print(f"🎯 查询类型: {route_type}")

        if memory_context is None:
            memory_context = "无"
            if self.memory_manager:
                memory_bundle = self.memory_manager.retrieve_context(question)
                memory_context = self.memory_manager.build_memory_prompt_context(memory_bundle)

        if route_type == "list":
            rewritten_query = question
            print(f"📝 列表查询保持原样: {question}")
        else:
            print("🤖 智能分析查询...")
            rewritten_query = self.generation_module.query_rewrite(question)

        if self.memory_manager and memory_context != "无":
            rewritten_query = self.generation_module.context_aware_rewrite(rewritten_query, memory_context)

        expanded_queries = [rewritten_query]
        if self.config.enable_mqe and route_type != "list":
            expanded_queries = self.generation_module.multi_query_expansion(rewritten_query, n=self.config.mqe_expansions)
            print(f"🔎 启用 MQE 扩展检索，共 {len(expanded_queries)} 个查询")

        print("🔍 检索相关文档...")
        filters = self._extract_filters_from_query(question)
        aggregated_chunks = []
        seen = set()
        for query in expanded_queries:
            if filters:
                print(f"🏷️ 应用过滤条件: {filters}")
                docs = self.retrieval_module.metadata_filtered_search(query, filters, top_k=self.config.top_k)
            else:
                docs = self.retrieval_module.hybrid_search(query, top_k=self.config.top_k)
            for doc in docs:
                key = (
                    doc.metadata.get("parent_id", ""),
                    doc.metadata.get("dish_name", ""),
                    doc.metadata.get("chunk_index", -1),
                    doc.page_content[:80],
                )
                if key not in seen:
                    seen.add(key)
                    aggregated_chunks.append(doc)

        relevant_chunks = aggregated_chunks[: max(self.config.top_k, len(aggregated_chunks))]
        if not relevant_chunks:
            return "抱歉，没有找到相关的食谱信息。请尝试其他菜品名称或关键词。"

        if route_type == "list":
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)
            answer = self.generation_module.generate_list_answer(question, relevant_docs)
            if self.memory_manager:
                self.memory_manager.add_interaction(question, answer)
                self.memory_manager.execute_forgetting()
            return answer

        relevant_docs = self.data_module.get_parent_documents(relevant_chunks)
        if stream:
            stream_gen = self.generation_module.generate_memory_aware_answer_stream(
                question,
                relevant_docs,
                memory_context=memory_context,
                route_type="detail" if route_type == "detail" else "general",
            )

            def _stream_and_store():
                chunks = []
                for chunk in stream_gen:
                    chunks.append(chunk)
                    yield chunk
                final_answer = "".join(chunks).strip()
                if self.memory_manager and final_answer:
                    self.memory_manager.add_interaction(question, final_answer)
                    self.memory_manager.execute_forgetting()

            return _stream_and_store()

        answer = self.generation_module.generate_memory_aware_answer(
            question,
            relevant_docs,
            memory_context=memory_context,
            route_type="detail" if route_type == "detail" else "general",
        )
        if self.memory_manager:
            self.memory_manager.add_interaction(question, answer)
            self.memory_manager.execute_forgetting()
        return answer

    def clear_working_memory(self):
        if self.memory_manager:
            self.memory_manager.working_memory.clear()
            return "✅ 已清空当前会话记忆"
        return "⚪ 记忆模块未启用"

    def _extract_filters_from_query(self, query: str) -> dict:
        filters = {}
        for category in DataPreparationModule.get_supported_categories():
            if category in query:
                filters["category"] = category
                break
        for difficulty in sorted(DataPreparationModule.get_supported_difficulties(), key=len, reverse=True):
            if difficulty in query:
                filters["difficulty"] = difficulty
                break
        return filters

    def search_by_category(self, category: str, query: str = "") -> List[str]:
        if not self.retrieval_module:
            raise ValueError("请先构建知识库")
        search_query = query if query else category
        docs = self.retrieval_module.metadata_filtered_search(search_query, {"category": category}, top_k=10)
        names = []
        for doc in docs:
            dish_name = doc.metadata.get("dish_name", "未知菜品")
            if dish_name not in names:
                names.append(dish_name)
        return names

    def get_ingredients_list(self, dish_name: str) -> str:
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")
        docs = self.retrieval_module.hybrid_search(dish_name, top_k=3)
        return self.generation_module.generate_basic_answer(f"{dish_name}需要什么食材？", docs)

    def run_interactive(self):
        print("=" * 60)
        print("🍽️  尝尝咸淡RAG系统（Milvus + MCP Skill 版）  🍽️")
        print("=" * 60)
        print("💡 解决您的选择困难症，告别'今天吃什么'的世纪难题！")
        print("🔗 向量数据库: Milvus | 检索策略: BM25 + 向量 + RRF")

        self.initialize_system()
        self.build_knowledge_base()

        print("\n📌 可用命令: /skills /clear /mcp 退出")
        while True:
            try:
                user_input = input("\n您的问题: ").strip()
                if user_input.lower() in ["退出", "quit", "exit", ""]:
                    break
                if user_input.lower() in ["/clear", "clear", "清空记忆"]:
                    print(self.clear_working_memory())
                    continue
                if user_input.lower() in ["/skills", "skills"]:
                    self._show_skills()
                    continue
                if user_input.lower() in ["/mcp", "mcp"]:
                    self._show_mcp_status()
                    continue
                stream_choice = input("是否使用流式输出? (y/n, 默认y): ").strip().lower()
                use_stream = stream_choice != "n"
                print("\n回答:")
                if use_stream:
                    result = self.ask_question(user_input, stream=True)
                    if hasattr(result, "__iter__") and not isinstance(result, str):
                        for chunk in result:
                            print(chunk, end="", flush=True)
                        print("\n")
                    else:
                        print(f"{result}\n")
                else:
                    print(f"{self.ask_question(user_input, stream=False)}\n")
            except KeyboardInterrupt:
                break
            except Exception as error:
                print(f"处理问题时出错: {error}")
        print("\n感谢使用尝尝咸淡RAG系统（Milvus + MCP Skill 版）！")

    def _show_skills(self):
        if not self.skill_registry:
            print("⚪ Skill 系统未启用")
            return
        print(f"\n🎯 已注册 {len(self.skill_registry.list_definitions())} 个技能:")
        for definition in self.skill_registry.list_definitions():
            mcp_tag = " [MCP开源]" if definition.is_mcp else " [自定义]"
            print(f"  📌 {definition.name}{mcp_tag}")

    def _show_mcp_status(self):
        print("\n🔌 MCP 环境状态:")
        for name, available in check_mcp_environment().items():
            print(f"  {name}: {'✅ 可用' if available else '❌ 不可用'}")


def main():
    try:
        RecipeRAGSystem().run_interactive()
    except Exception as error:
        logger.error(f"系统运行出错: {error}")
        print(f"系统错误: {error}")


if __name__ == "__main__":
    main()
