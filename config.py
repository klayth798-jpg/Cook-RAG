"""
RAG系统配置文件（Milvus版本）
基于C8项目迁移，将FAISS替换为Milvus向量数据库
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class RAGConfig:
    """RAG系统配置类"""

    # 路径配置
    data_path: str = "../../data/C8/cook"

    # Milvus配置（替代原FAISS本地文件）
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection_name: str = "recipe_knowledge"
    milvus_dimension: int = 512  # BGE-small-zh-v1.5 的输出维度

    # 模型配置
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_model: str = None  # 从环境变量读取

    # 检索配置
    top_k: int = 3

    # 生成配置
    temperature: float = 0.1
    max_tokens: int = 2048

    # ── 记忆系统配置（hello-agents启发） ──
    enable_memory: bool = True                          # 是否启用记忆系统
    memory_collection_name: str = "recipe_memory"        # 情景记忆在Milvus中的Collection名
    memory_max_turns: int = 10                           # 工作记忆最大轮次
    memory_ttl_minutes: int = 30                         # 工作记忆TTL（分钟）
    memory_importance_threshold: float = 0.7             # 整合阈值：重要性≥此值才升级为情景记忆
    memory_max_episodic: int = 200                       # 情景记忆最大条数（超出触发遗忘）
    memory_forget_threshold: float = 0.3                 # 遗忘阈值：重要性<此值优先被遗忘
    enable_mqe: bool = True                              # 是否启用MQE多查询扩展
    mqe_expansions: int = 2                              # MQE扩展查询数量

    # ── Skill 系统配置 ──
    enable_skills: bool = True                            # 是否启用 Skill 路由
    enable_mcp_fetch: bool = True                         # 是否启用 mcp-server-fetch
    enable_tavily_search: bool = True                     # 是否允许使用 Tavily API 联网搜索

    def __post_init__(self):
        """初始化后的处理"""
        pass
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RAGConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'data_path': self.data_path,
            'milvus_host': self.milvus_host,
            'milvus_port': self.milvus_port,
            'milvus_collection_name': self.milvus_collection_name,
            'milvus_dimension': self.milvus_dimension,
            'embedding_model': self.embedding_model,
            'llm_model': self.llm_model,
            'top_k': self.top_k,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'enable_memory': self.enable_memory,
            'memory_collection_name': self.memory_collection_name,
            'memory_max_turns': self.memory_max_turns,
            'memory_ttl_minutes': self.memory_ttl_minutes,
            'memory_importance_threshold': self.memory_importance_threshold,
            'memory_max_episodic': self.memory_max_episodic,
            'memory_forget_threshold': self.memory_forget_threshold,
            'enable_mqe': self.enable_mqe,
            'mqe_expansions': self.mqe_expansions,
            'enable_skills': self.enable_skills,
            'enable_mcp_fetch': self.enable_mcp_fetch,
            'enable_tavily_search': self.enable_tavily_search,
        }

# 默认配置实例
DEFAULT_CONFIG = RAGConfig()
