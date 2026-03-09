# Cook-RAG

基于 `Milvus + 混合检索 + 对话记忆 + Skill 路由` 的食谱领域 RAG 智能体系统。

项目面向“吃什么、怎么做、怎么搭配、营养如何”这类厨房场景问题，结合本地食谱知识库、向量检索、工作记忆 / 情景记忆，以及 MCP 联网抓取能力，构建一个可交互的食谱助手。

## 项目亮点

- `Milvus` 替代本地向量索引，支持更稳定的向量存储与检索。
- `BM25 + 向量检索 + RRF` 混合召回，提高菜名、食材、做法等查询命中率。
- 引入 `MemoryManager`，支持工作记忆、情景记忆和遗忘机制。
- 提供 `SkillRouter`，可自动把问题路由到本地检索、网页食谱抓取、膳食规划、营养分析等能力。
- 支持 `MCP` 与 `mcp-server-fetch`，可抓取网页食谱并交给 LLM 提取结构化内容。
- 支持 `MQE`（Multi-Query Expansion）和上下文感知查询改写，提升复杂对话场景下的检索质量。

## 系统架构

整体流程如下：

1. 加载本地 Markdown 食谱数据。
2. 进行结构化分块与元数据增强。
3. 将文本向量写入 `Milvus`，同时保留 BM25 稀疏检索能力。
4. 用户提问后，系统先进行问题路由：
   - 本地食谱检索
   - 网页食谱抓取
   - 膳食规划
   - 营养分析
   - 闲聊兜底
5. 若走本地 RAG 流程，则执行：查询分类 → 查询改写 → 多查询扩展 → 混合检索 → 重排 → 生成回答。
6. 回答完成后，将当前对话写入记忆模块，用于后续多轮问答。

## 核心能力

### 1. 本地食谱检索

- 支持按菜名、食材、分类、难度进行检索。
- 支持推荐类问题，如“有什么简单的素菜”“推荐几道川菜”。
- 支持步骤类问题，如“红烧肉怎么做”“鱼香肉丝要什么食材”。

### 2. 记忆增强对话

- 记录当前会话中的偏好、禁忌、延续问题。
- 支持指代消解，如“它需要什么食材”“再推荐一个”。
- 支持根据对话历史调整回答风格和内容。

### 3. Skill 路由

当前默认会注册以下能力：

- `recipe_search`：本地知识库检索
- `web_recipe_fetch`：联网抓取网页食谱（依赖 MCP）
- `meal_planner`：膳食规划
- `nutrition_calc`：营养估算

### 4. 联网食谱抓取

- 优先通过 `mcp-server-fetch` 抓取网页内容。
- 若配置了 `TAVILY_API_KEY`，可先走 Tavily 搜索，再解析结果。
- 抓取结果会交由 LLM 进行食谱信息提取与整理。

## 项目结构

```text
C10/
├── config.py                     # 全局配置
├── docker-compose.yml            # Milvus 本地部署
├── main.py                       # 程序入口 / 交互式 CLI
├── requirements.txt              # Python 依赖
└── rag_modules/
	├── data_preparation.py       # 数据加载、元数据增强、分块
	├── index_construction.py     # Milvus 向量索引构建
	├── retrieval_optimization.py # 混合检索与 RRF 重排
	├── generation_integration.py # LLM 调用、改写、回答生成
	├── conversation_memory.py    # 工作记忆 / 情景记忆 / 遗忘
	└── skills/
		├── base.py               # Skill 框架与路由基类
		├── mcp_client.py         # MCP 客户端
		├── recipe_search_skill.py
		├── web_recipe_skill.py
		├── meal_planner_skill.py
		└── nutrition_skill.py
```

## 环境要求

- Python `3.10+`（推荐 `3.11` 或 `3.12`）
- Docker / Docker Compose（用于启动 Milvus）
- 可访问大模型 API 的网络环境

## 安装步骤

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd C10
```

### 2. 创建 Python 环境并安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 3. 启动 Milvus

```bash
docker compose up -d
```

Milvus 默认暴露以下端口：

- `19530`：Milvus 服务端口
- `9091`：健康检查端口
- `9000 / 9001`：MinIO 相关端口

### 4. 配置环境变量

在项目根目录创建 `.env` 文件，例如：

```bash
LLM_API_KEY=your_api_key
LLM_MODEL_ID=gpt-4o-mini
LLM_BASE_URL=https://your-llm-endpoint/v1

# 可选：启用 Tavily 搜索
TAVILY_API_KEY=your_tavily_api_key
```

说明：

- `LLM_API_KEY`：必填，大模型接口密钥。
- `LLM_MODEL_ID`：必填或建议显式配置，对应实际使用的模型名。
- `LLM_BASE_URL`：可选，用于兼容 OpenAI 兼容接口。
- `TAVILY_API_KEY`：可选，用于联网搜索增强。

## 数据准备

默认数据目录在 [config.py](config.py) 中配置为：

```python
data_path = "../../data/C8/cook"
```

也就是说，当前项目默认依赖一个外部食谱 Markdown 数据目录。如果你的数据不在这个位置，可以直接修改 [config.py](config.py) 中的 `RAGConfig.data_path`。

建议数据格式：

- 使用 Markdown 存储食谱
- 每道菜一个文件
- 文件中尽量包含标题、食材、步骤、分类、难度等信息

## 运行方式

### 交互式启动

```bash
python main.py
```

启动后系统会自动执行：

1. 初始化数据模块、Milvus 模块、LLM 模块
2. 检查并初始化记忆系统
3. 检查并初始化 Skill / MCP 环境
4. 若 Milvus 中不存在索引，则自动构建知识库
5. 进入命令行对话模式

### 交互命令

程序运行后支持以下命令：

- `/skills`：查看当前已注册的 Skill
- `/mcp`：查看 MCP 环境状态
- `/clear`：清空当前会话记忆
- `退出` / `quit` / `exit`：退出程序

## 典型问题示例

你可以尝试以下问题：

```text
红烧肉怎么做？
推荐几个简单的素菜。
鱼香肉丝需要什么食材？
帮我规划一周晚餐。
这道菜适合减脂吗？
帮我看看这个网页里的食谱：https://example.com/recipe
```

## 配置说明

主要配置位于 [config.py](config.py)，包括：

- `milvus_host` / `milvus_port`：Milvus 连接地址
- `milvus_collection_name`：食谱知识库 Collection 名称
- `embedding_model`：嵌入模型
- `top_k`：检索返回数量
- `temperature` / `max_tokens`：生成参数
- `enable_memory`：是否启用记忆系统
- `enable_mqe`：是否启用多查询扩展
- `enable_skills`：是否启用 Skill 路由
- `enable_mcp_fetch`：是否启用网页抓取 Skill
- `enable_tavily_search`：是否启用 Tavily 搜索

## 依赖说明

项目主要依赖包括：

- `langchain` / `langchain-openai`
- `pymilvus`
- `sentence-transformers`
- `rank_bm25`
- `mcp` / `mcp-server-fetch`
- `python-dotenv`
- `requests`

## 常见问题

### 1. 报错“请设置 LLM_API_KEY 环境变量”

说明没有正确配置 `.env` 或当前 shell 环境变量。

### 2. 报错“数据路径不存在”

说明 [config.py](config.py) 中的 `data_path` 指向的目录不存在，请修改为你的本地食谱数据目录。

### 3. 无法连接 Milvus

请确认：

- 已执行 `docker compose up -d`
- `19530` 端口未被占用
- 容器状态正常，可通过 `docker ps` 查看

### 4. 网页抓取不可用

请确认：

- 已安装 `mcp-server-fetch`
- `mcp` 相关依赖安装成功
- 当前环境可启动 MCP 子进程

## 后续可扩展方向

- 增加 Web UI 或 API 服务层
- 引入更细粒度的食材结构化抽取
- 支持用户画像持久化
- 支持多数据源食谱融合检索
- 增加自动评测与检索质量分析

## License

本项目使用仓库中的 [LICENSE](LICENSE) 许可文件。
