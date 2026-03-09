"""
对话记忆模块
包含: WorkingMemory, EpisodicMemory, MemoryManager
复用 Milvus 作为长期情景记忆存储。
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

logger = logging.getLogger(__name__)


class MemoryItem:
    def __init__(self, query: str, answer: str, importance: float = 0.5, timestamp: float = None):
        self.id = str(uuid.uuid4())
        self.query = query
        self.answer = answer
        self.importance = importance
        self.timestamp = timestamp or time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "answer": self.answer,
            "importance": self.importance,
            "timestamp": self.timestamp,
        }


class WorkingMemory:
    def __init__(self, max_turns: int = 10, ttl_minutes: int = 30):
        self.max_turns = max_turns
        self.ttl_minutes = ttl_minutes
        self.memories: List[MemoryItem] = []
        self.last_active_time = time.time()

    def add(self, memory_item: MemoryItem):
        self._expire_old_memories()
        if len(self.memories) >= self.max_turns:
            self.memories.pop(0)
        self.memories.append(memory_item)
        self.last_active_time = time.time()

    def retrieve_all(self) -> List[MemoryItem]:
        self._expire_old_memories()
        return self.memories

    def get_langchain_messages(self) -> List[BaseMessage]:
        self._expire_old_memories()
        messages: List[BaseMessage] = []
        for memory in self.memories:
            messages.append(HumanMessage(content=memory.query))
            messages.append(AIMessage(content=memory.answer))
        return messages

    def clear(self):
        self.memories.clear()

    def _expire_old_memories(self):
        current_time = time.time()
        if current_time - self.last_active_time > self.ttl_minutes * 60:
            self.memories.clear()
        self.last_active_time = current_time


class EpisodicMemory:
    MAX_VARCHAR_LENGTH = 65535

    def __init__(
        self,
        milvus_client: MilvusClient,
        embeddings: HuggingFaceEmbeddings,
        collection_name: str = "recipe_memory",
        dimension: int = 512,
        session_id: str = "default_session",
    ):
        self.client = milvus_client
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.dimension = dimension
        self.session_id = session_id
        self._init_collection()

    def _safe_truncate(self, text: str, max_length: int) -> str:
        if text is None:
            return ""
        return text[:max_length] if len(text) > max_length else text

    def _init_collection(self):
        if self.client.has_collection(self.collection_name):
            return
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="query", dtype=DataType.VARCHAR, max_length=self.MAX_VARCHAR_LENGTH),
            FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=self.MAX_VARCHAR_LENGTH),
            FieldSchema(name="importance", dtype=DataType.FLOAT),
            FieldSchema(name="timestamp", dtype=DataType.FLOAT),
            FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=64),
        ]
        schema = CollectionSchema(fields=fields, description="食谱RAG-情景对话记忆")
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 8, "efConstruction": 100},
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info(f"情景记忆 Collection '{self.collection_name}' 创建成功")

    def add(self, memory_item: MemoryItem):
        vector = self.embeddings.embed_query(memory_item.query)
        row = {
            "id": memory_item.id,
            "vector": vector,
            "query": self._safe_truncate(memory_item.query, self.MAX_VARCHAR_LENGTH),
            "answer": self._safe_truncate(memory_item.answer, self.MAX_VARCHAR_LENGTH),
            "importance": float(memory_item.importance),
            "timestamp": float(memory_item.timestamp),
            "session_id": self._safe_truncate(self.session_id, 64),
        }
        self.client.insert(collection_name=self.collection_name, data=[row])

    def retrieve(self, query: str, limit: int = 3, min_importance: float = 0.0) -> List[MemoryItem]:
        if not self.client.has_collection(self.collection_name):
            return []
        try:
            stats = self.client.get_collection_stats(self.collection_name)
            if int(stats.get("row_count", 0)) == 0:
                return []

            query_vector = self.embeddings.embed_query(query)
            filter_expr = f"importance >= {min_importance}" if min_importance > 0 else ""
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                limit=limit * 2,
                filter=filter_expr,
                output_fields=["query", "answer", "importance", "timestamp", "session_id"],
                search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            )
            if not results or not results[0]:
                return []

            scored_memories = []
            current_time = time.time()
            for hit in results[0]:
                entity = hit.get("entity", {})
                vec_score = hit.get("distance", 0.0)
                mem_time = float(entity.get("timestamp", current_time))
                age_days = (current_time - mem_time) / (3600 * 24)
                recency_score = max(0.1, 0.5 ** (age_days / 10))
                importance = float(entity.get("importance", 0.5))
                importance_weight = 0.8 + importance * 0.4
                final_score = (vec_score * 0.8 + recency_score * 0.2) * importance_weight

                item = MemoryItem(query=entity.get("query", ""), answer=entity.get("answer", ""))
                item.id = hit.get("id")
                item.importance = importance
                item.timestamp = mem_time
                scored_memories.append((final_score, item))

            scored_memories.sort(key=lambda pair: pair[0], reverse=True)
            return [item for _, item in scored_memories[:limit]]
        except Exception as error:
            logger.error(f"情景记忆检索失败: {error}")
            return []

    def forget(self, threshold: float = 0.3):
        if not self.client.has_collection(self.collection_name):
            return
        try:
            self.client.delete(collection_name=self.collection_name, filter=f"importance < {threshold}")
            logger.info(f"执行遗忘机制: 删除了重要性 < {threshold} 的记忆")
        except Exception as error:
            logger.warning(f"遗忘机制执行失败: {error}")


class MemoryManager:
    def __init__(self, config: Any, index_module: Any, session_id: str = None):
        self.config = config
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.working_memory = WorkingMemory(
            max_turns=config.memory_max_turns,
            ttl_minutes=config.memory_ttl_minutes,
        )
        self.episodic_memory: Optional[EpisodicMemory] = None
        if config.enable_memory and hasattr(index_module, "client") and hasattr(index_module, "embeddings"):
            self.episodic_memory = EpisodicMemory(
                milvus_client=index_module.client,
                embeddings=index_module.embeddings,
                collection_name=config.memory_collection_name,
                dimension=config.milvus_dimension,
                session_id=self.session_id,
            )
        self.llm = None

    def set_llm(self, llm):
        self.llm = llm

    def _evaluate_importance(self, query: str, answer: str) -> float:
        importance = 0.5
        if len(query) < 5 or query in ["你好", "谢谢", "好的", "没问题", "退出"]:
            return 0.2
        preferences = ["喜欢", "不吃", "不要", "多放", "少放", "我是", "能吃", "过敏"]
        if any(token in query for token in preferences):
            return 0.9
        if len(answer) > 200:
            importance = max(importance, 0.7)
        return importance

    def add_interaction(self, query: str, answer: str):
        if not self.config.enable_memory:
            return
        importance = self._evaluate_importance(query, answer)
        item = MemoryItem(query=query, answer=answer, importance=importance)
        self.working_memory.add(item)
        if self.episodic_memory and importance >= self.config.memory_importance_threshold:
            self.episodic_memory.add(item)

    def retrieve_context(self, query: str) -> Dict[str, Any]:
        working = self.working_memory.retrieve_all()
        episodic = []
        if self.episodic_memory:
            episodic = self.episodic_memory.retrieve(
                query,
                limit=3,
                min_importance=self.config.memory_forget_threshold,
            )
        return {"working": working, "episodic": episodic}

    def build_memory_prompt_context(self, query_or_bundle) -> str:
        if isinstance(query_or_bundle, str):
            bundle = self.retrieve_context(query_or_bundle)
        else:
            bundle = query_or_bundle or {}
        working = bundle.get("working", [])
        episodic = bundle.get("episodic", [])
        parts = []
        if working:
            parts.append("## 最近对话")
            for item in working[-4:]:
                parts.append(f"用户: {item.query}\n助手: {item.answer}")
        if episodic:
            parts.append("## 历史偏好/重要记忆")
            for item in episodic:
                parts.append(f"用户: {item.query}\n助手: {item.answer}")
        return "\n\n".join(parts) if parts else "无"

    def execute_forgetting(self):
        if self.episodic_memory:
            self.episodic_memory.forget(threshold=self.config.memory_forget_threshold)
