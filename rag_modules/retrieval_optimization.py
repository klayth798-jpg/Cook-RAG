"""
检索优化模块（Milvus版本）
改动点：
  1. 构造函数接收 index_module（IndexConstructionModule）替代 FAISS vectorstore
  2. 向量检索直接调用 index_module.similarity_search
  3. metadata_filtered_search 升级为 Milvus 原生过滤 + BM25 混合
"""

import logging
from typing import List, Dict, Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class RetrievalOptimizationModule:
    """检索优化模块 - 基于Milvus + BM25的混合检索"""

    def __init__(self, index_module, chunks: List[Document]):
        """
        初始化检索优化模块

        Args:
            index_module: IndexConstructionModule实例（Milvus版本）
            chunks: 文档块列表（用于BM25检索）
        """
        self.index_module = index_module
        self.chunks = chunks
        self.bm25_retriever = None
        self.setup_retrievers()

    def setup_retrievers(self):
        """设置BM25稀疏检索器（向量检索由index_module直接提供）"""
        logger.info("正在设置检索器...")

        # BM25检索器（与C8完全一致）
        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks,
            k=5,
        )

        logger.info("检索器设置完成（Milvus向量检索 + BM25稀疏检索）")

    def hybrid_search(self, query: str, top_k: int = 3) -> List[Document]:
        """
        混合检索 - 结合Milvus向量检索和BM25检索，使用RRF重排

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            检索到的文档列表
        """
        # Milvus向量检索（替代C8的vector_retriever.invoke）
        vector_docs = self.index_module.similarity_search(query, k=5)

        # BM25稀疏检索（与C8一致）
        bm25_docs = self.bm25_retriever.invoke(query)

        # 使用RRF重排（与C8完全一致）
        reranked_docs = self._rrf_rerank(vector_docs, bm25_docs)
        return reranked_docs[:top_k]

    def metadata_filtered_search(
        self, query: str, filters: Dict[str, Any], top_k: int = 5
    ) -> List[Document]:
        """
        带元数据过滤的检索
        升级点：利用Milvus原生filter表达式在数据库层面过滤，
              同时BM25结果仍走Python层面过滤，最终RRF融合

        Args:
            query: 查询文本
            filters: 元数据过滤条件，如 {"category": "荤菜", "difficulty": "简单"}
            top_k: 返回结果数量

        Returns:
            过滤后的文档列表
        """
        # ── 1. 构建Milvus过滤表达式 ──
        filter_expr = self._build_filter_expr(filters)

        # ── 2. Milvus原生过滤 + 向量检索 ──
        if filter_expr:
            vector_docs = self.index_module.similarity_search_with_filter(
                query, filter_expr, k=top_k * 3
            )
            logger.info(f"Milvus过滤检索: filter='{filter_expr}', 返回 {len(vector_docs)} 个结果")
        else:
            vector_docs = self.index_module.similarity_search(query, k=top_k * 3)

        # ── 3. BM25检索（不支持原生过滤，走Python层面过滤） ──
        bm25_docs = self.bm25_retriever.invoke(query)
        bm25_filtered = self._python_filter(bm25_docs, filters)

        # ── 4. RRF融合重排 ──
        reranked_docs = self._rrf_rerank(vector_docs, bm25_filtered)
        return reranked_docs[:top_k]

    # ───────────────────── 内部方法 ─────────────────────

    @staticmethod
    def _build_filter_expr(filters: Dict[str, Any]) -> str:
        """
        将Python字典转换为Milvus过滤表达式

        Args:
            filters: 过滤条件字典

        Returns:
            Milvus filter表达式字符串

        示例:
            {"category": "荤菜"} → 'category == "荤菜"'
            {"category": "素菜", "difficulty": "简单"} → 'category == "素菜" and difficulty == "简单"'
        """
        if not filters:
            return ""

        conditions = []
        for key, value in filters.items():
            if isinstance(value, list):
                # 列表条件：使用 in 操作符
                values_str = ", ".join([f'"{v}"' for v in value])
                conditions.append(f'{key} in [{values_str}]')
            else:
                conditions.append(f'{key} == "{value}"')

        return " and ".join(conditions)

    @staticmethod
    def _python_filter(
        docs: List[Document], filters: Dict[str, Any]
    ) -> List[Document]:
        """
        Python层面的元数据过滤（用于BM25结果后处理）

        Args:
            docs: 待过滤文档列表
            filters: 过滤条件

        Returns:
            过滤后的文档列表
        """
        if not filters:
            return docs

        filtered = []
        for doc in docs:
            match = True
            for key, value in filters.items():
                if key in doc.metadata:
                    if isinstance(value, list):
                        if doc.metadata[key] not in value:
                            match = False
                            break
                    else:
                        if doc.metadata[key] != value:
                            match = False
                            break
                else:
                    match = False
                    break
            if match:
                filtered.append(doc)
        return filtered

    def _rrf_rerank(
        self,
        vector_docs: List[Document],
        bm25_docs: List[Document],
        k: int = 60,
    ) -> List[Document]:
        """
        使用RRF (Reciprocal Rank Fusion) 算法重排文档（与C8完全一致）

        Args:
            vector_docs: 向量检索结果
            bm25_docs: BM25检索结果
            k: RRF参数，用于平滑排名

        Returns:
            重排后的文档列表
        """
        doc_scores = {}
        doc_objects = {}

        # 计算向量检索结果的RRF分数
        for rank, doc in enumerate(vector_docs):
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
            logger.debug(f"向量检索 - 文档{rank+1}: RRF分数 = {rrf_score:.4f}")

        # 计算BM25检索结果的RRF分数
        for rank, doc in enumerate(bm25_docs):
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
            logger.debug(f"BM25检索 - 文档{rank+1}: RRF分数 = {rrf_score:.4f}")

        # 按最终RRF分数排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # 构建最终结果
        reranked_docs = []
        for doc_id, final_score in sorted_docs:
            if doc_id in doc_objects:
                doc = doc_objects[doc_id]
                doc.metadata["rrf_score"] = final_score
                reranked_docs.append(doc)
                logger.debug(
                    f"最终排序 - 文档: {doc.page_content[:50]}... "
                    f"最终RRF分数: {final_score:.4f}"
                )

        logger.info(
            f"RRF重排完成: 向量检索{len(vector_docs)}个文档, "
            f"BM25检索{len(bm25_docs)}个文档, 合并后{len(reranked_docs)}个文档"
        )

        return reranked_docs
