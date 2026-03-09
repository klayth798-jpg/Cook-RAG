"""
数据准备模块
（与C8逻辑一致，负责文档加载、元数据增强与结构化分块）
"""

import hashlib
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

logger = logging.getLogger(__name__)


class DataPreparationModule:
    """数据准备模块 - 负责数据加载、清洗和预处理"""

    CATEGORY_MAPPING = {
        "meat_dish": "荤菜",
        "vegetable_dish": "素菜",
        "soup": "汤品",
        "dessert": "甜品",
        "breakfast": "早餐",
        "staple": "主食",
        "aquatic": "水产",
        "condiment": "调料",
        "drink": "饮品",
    }
    CATEGORY_LABELS = list(set(CATEGORY_MAPPING.values()))
    DIFFICULTY_LABELS = ["非常简单", "简单", "中等", "困难", "非常困难"]

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.documents: List[Document] = []
        self.chunks: List[Document] = []
        self.parent_child_map: Dict[str, str] = {}
        self.parent_documents_by_id: Dict[str, Document] = {}

    def load_documents(self) -> List[Document]:
        logger.info(f"正在从 {self.data_path} 加载文档...")
        documents: List[Document] = []
        data_root = Path(self.data_path)

        for md_file in data_root.rglob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
                try:
                    relative_path = md_file.resolve().relative_to(data_root.resolve()).as_posix()
                except Exception:
                    relative_path = md_file.as_posix()
                parent_id = hashlib.md5(relative_path.encode("utf-8")).hexdigest()
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(md_file),
                        "parent_id": parent_id,
                        "doc_type": "parent",
                    },
                )
                self._enhance_metadata(doc)
                documents.append(doc)
                self.parent_documents_by_id[parent_id] = doc
            except Exception as error:
                logger.warning(f"读取文件 {md_file} 失败: {error}")

        self.documents = documents
        logger.info(f"成功加载 {len(documents)} 个文档")
        return documents

    def _enhance_metadata(self, doc: Document):
        file_path = Path(doc.metadata.get("source", ""))
        path_parts = file_path.parts

        doc.metadata["category"] = "其他"
        for key, value in self.CATEGORY_MAPPING.items():
            if key in path_parts:
                doc.metadata["category"] = value
                break

        doc.metadata["dish_name"] = file_path.stem

        content = doc.page_content
        if "★★★★★" in content:
            doc.metadata["difficulty"] = "非常困难"
        elif "★★★★" in content:
            doc.metadata["difficulty"] = "困难"
        elif "★★★" in content:
            doc.metadata["difficulty"] = "中等"
        elif "★★" in content:
            doc.metadata["difficulty"] = "简单"
        elif "★" in content:
            doc.metadata["difficulty"] = "非常简单"
        else:
            doc.metadata["difficulty"] = "未知"

    @classmethod
    def get_supported_categories(cls) -> List[str]:
        return cls.CATEGORY_LABELS

    @classmethod
    def get_supported_difficulties(cls) -> List[str]:
        return cls.DIFFICULTY_LABELS

    def chunk_documents(self) -> List[Document]:
        logger.info("正在进行 Markdown 结构感知分块...")
        if not self.documents:
            raise ValueError("请先加载文档")

        chunks = self._markdown_header_split()
        for index, chunk in enumerate(chunks):
            chunk.metadata.setdefault("chunk_id", str(uuid.uuid4()))
            chunk.metadata["batch_index"] = index
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        self.chunks = chunks
        logger.info(f"Markdown 分块完成，共生成 {len(chunks)} 个 chunk")
        return chunks

    def _markdown_header_split(self) -> List[Document]:
        headers_to_split_on = [
            ("#", "主标题"),
            ("##", "二级标题"),
            ("###", "三级标题"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )

        all_chunks: List[Document] = []
        for doc in self.documents:
            try:
                md_chunks = markdown_splitter.split_text(doc.page_content)
                if not md_chunks:
                    md_chunks = [Document(page_content=doc.page_content, metadata={})]

                parent_id = doc.metadata["parent_id"]
                for index, chunk in enumerate(md_chunks):
                    child_id = str(uuid.uuid4())
                    chunk.metadata.update(doc.metadata)
                    chunk.metadata.update(
                        {
                            "chunk_id": child_id,
                            "parent_id": parent_id,
                            "doc_type": "child",
                            "chunk_index": index,
                        }
                    )
                    self.parent_child_map[child_id] = parent_id
                all_chunks.extend(md_chunks)
            except Exception as error:
                logger.warning(f"文档 {doc.metadata.get('source', '未知')} Markdown分割失败: {error}")
                fallback_chunk = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
                fallback_chunk.metadata.update(
                    {
                        "chunk_id": str(uuid.uuid4()),
                        "doc_type": "child",
                        "chunk_index": 0,
                    }
                )
                self.parent_child_map[fallback_chunk.metadata["chunk_id"]] = doc.metadata["parent_id"]
                all_chunks.append(fallback_chunk)

        logger.info(f"Markdown 结构分割完成，生成 {len(all_chunks)} 个结构化块")
        return all_chunks

    def get_parent_documents(self, child_chunks: List[Document]) -> List[Document]:
        parent_ids = []
        for chunk in child_chunks:
            parent_id = chunk.metadata.get("parent_id")
            if parent_id and parent_id not in parent_ids:
                parent_ids.append(parent_id)

        parent_docs = []
        for parent_id in parent_ids:
            doc = self.parent_documents_by_id.get(parent_id)
            if doc is not None:
                parent_docs.append(doc)
        return parent_docs

    def filter_documents_by_category(self, category: str) -> List[Document]:
        return [doc for doc in self.documents if doc.metadata.get("category") == category]

    def filter_documents_by_difficulty(self, difficulty: str) -> List[Document]:
        return [doc for doc in self.documents if doc.metadata.get("difficulty") == difficulty]

    def get_statistics(self) -> Dict[str, Any]:
        categories: Dict[str, int] = {}
        difficulties: Dict[str, int] = {}
        for doc in self.documents:
            categories[doc.metadata.get("category", "其他")] = categories.get(doc.metadata.get("category", "其他"), 0) + 1
            difficulties[doc.metadata.get("difficulty", "未知")] = difficulties.get(doc.metadata.get("difficulty", "未知"), 0) + 1
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "categories": categories,
            "difficulties": difficulties,
        }
