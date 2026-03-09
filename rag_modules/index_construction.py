"""
索引构建模块（Milvus版本）
核心改写：将 FAISS 替换为 Milvus 向量数据库
"""

import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

logger = logging.getLogger(__name__)


class IndexConstructionModule:
    """索引构建模块 - 基于 Milvus 的向量索引管理"""

    MAX_VARCHAR_LENGTH = 65535
    METADATA_MAX_LENGTH = 256

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        milvus_host: str = "localhost",
        milvus_port: int = 19530,
        collection_name: str = "recipe_knowledge",
        dimension: int = 512,
    ):
        self.model_name = model_name
        self.milvus_uri = f"http://{milvus_host}:{milvus_port}"
        self.collection_name = collection_name
        self.dimension = dimension
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.client: Optional[MilvusClient] = None
        self.setup_embeddings()
        self._connect()

    def setup_embeddings(self):
        logger.info(f"正在初始化嵌入模型: {self.model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("嵌入模型初始化完成")

    def _connect(self):
        logger.info(f"正在连接 Milvus: {self.milvus_uri}")
        self.client = MilvusClient(uri=self.milvus_uri)
        logger.info("Milvus 连接成功")

    def _create_collection(self):
        if self.client.has_collection(self.collection_name):
            logger.info(f"Collection '{self.collection_name}' 已存在，将删除后重建")
            self.client.drop_collection(self.collection_name)

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=self.MAX_VARCHAR_LENGTH),
            FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="dish_name", dtype=DataType.VARCHAR, max_length=self.METADATA_MAX_LENGTH),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=self.METADATA_MAX_LENGTH),
            FieldSchema(name="difficulty", dtype=DataType.VARCHAR, max_length=self.METADATA_MAX_LENGTH),
            FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
        ]
        schema = CollectionSchema(fields=fields, description="食谱知识库 - Milvus版本")

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info(f"Collection '{self.collection_name}' 创建成功")

    @staticmethod
    def _safe_truncate(text: str, max_length: int) -> str:
        if text is None:
            return ""
        return text[:max_length] if len(text) > max_length else text

    def build_vector_index(self, chunks: List[Document]):
        if not chunks:
            raise ValueError("文档块列表不能为空")

        logger.info(f"正在构建 Milvus 向量索引，共 {len(chunks)} 个文档块...")
        self._create_collection()

        texts = [chunk.page_content for chunk in chunks]
        vectors = self.embeddings.embed_documents(texts)
        logger.info(f"向量生成完成，维度: {len(vectors[0])}")

        data = []
        for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
            meta = chunk.metadata
            data.append(
                {
                    "id": meta.get("chunk_id", f"chunk_{index}"),
                    "vector": vector,
                    "text": self._safe_truncate(chunk.page_content, self.MAX_VARCHAR_LENGTH),
                    "parent_id": self._safe_truncate(str(meta.get("parent_id", "")), 64),
                    "dish_name": self._safe_truncate(str(meta.get("dish_name", "")), self.METADATA_MAX_LENGTH),
                    "category": self._safe_truncate(str(meta.get("category", "")), self.METADATA_MAX_LENGTH),
                    "difficulty": self._safe_truncate(str(meta.get("difficulty", "")), self.METADATA_MAX_LENGTH),
                    "doc_type": self._safe_truncate(str(meta.get("doc_type", "child")), 32),
                    "chunk_index": int(meta.get("chunk_index", index)),
                }
            )

        batch_size = 1000
        for start in range(0, len(data), batch_size):
            batch = data[start:start + batch_size]
            self.client.insert(collection_name=self.collection_name, data=batch)
            logger.info(f"已插入 {min(start + batch_size, len(data))}/{len(data)} 条数据")

        logger.info(f"Milvus 向量索引构建完成，共插入 {len(data)} 条数据")

    def collection_exists(self) -> bool:
        if not self.client.has_collection(self.collection_name):
            return False
        stats = self.client.get_collection_stats(self.collection_name)
        row_count = int(stats.get("row_count", 0))
        if row_count > 0:
            logger.info(f"Collection '{self.collection_name}' 已存在，包含 {row_count} 条数据")
            return True
        return False

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        if not self.client.has_collection(self.collection_name):
            raise ValueError("Collection不存在，请先构建索引")

        query_vector = self.embeddings.embed_query(query)
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=k,
            output_fields=["text", "parent_id", "dish_name", "category", "difficulty", "doc_type", "chunk_index"],
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
        )
        return self._convert_search_results(results)

    def similarity_search_with_filter(self, query: str, filter_expr: str, k: int = 5) -> List[Document]:
        if not self.client.has_collection(self.collection_name):
            raise ValueError("Collection不存在，请先构建索引")

        query_vector = self.embeddings.embed_query(query)
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=k,
            filter=filter_expr,
            output_fields=["text", "parent_id", "dish_name", "category", "difficulty", "doc_type", "chunk_index"],
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
        )
        return self._convert_search_results(results)

    def _convert_search_results(self, results) -> List[Document]:
        docs: List[Document] = []
        if not results or not results[0]:
            return docs
        for hit in results[0]:
            entity = hit.get("entity", {})
            doc = Document(
                page_content=entity.get("text", ""),
                metadata={
                    "parent_id": entity.get("parent_id", ""),
                    "dish_name": entity.get("dish_name", ""),
                    "category": entity.get("category", ""),
                    "difficulty": entity.get("difficulty", ""),
                    "doc_type": entity.get("doc_type", "child"),
                    "chunk_index": entity.get("chunk_index", 0),
                    "score": hit.get("distance", 0.0),
                },
            )
            docs.append(doc)
        return docs

    def get_collection_stats(self):
        if not self.client.has_collection(self.collection_name):
            return {"row_count": 0}
        return self.client.get_collection_stats(self.collection_name)
