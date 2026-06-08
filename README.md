# MCP RAG Server

MCP RAG Server 是一个符合 Model Context Protocol (MCP) 标准的 RAG（Retrieval-Augmented Generation）功能 Python 服务器。它以多种格式的文档（如 Markdown、文本、PowerPoint、PDF 等）作为数据源，使用 multilingual-e5-large 模型进行索引，并提供通过向量检索获取相关信息的功能。

## 概述

本项目在 MCP 服务器的基本实现之上，还提供了 RAG 功能。它可以对多种格式的文档进行索引，并根据自然语言查询检索相关信息。

## 功能

- **MCP 服务器基本实现**
  - 基于官方 MCP Python SDK 构建
  - 支持 JSON-RPC over **stdio**（默认）和 **HTTP SSE** 两种传输方式
  - 提供工具注册和执行机制
  - 错误处理和日志记录

- **RAG 功能**
  - 支持多种格式文档（Markdown、文本、PowerPoint、PDF）的读取和解析
  - 支持具有层级结构的源目录
  - 使用 markitdown 库将 PowerPoint 和 PDF 转换为 Markdown
  - 使用可选的嵌入模型（multilingual-e5-large、ruri 等）生成嵌入向量
  - 使用 PostgreSQL 的 pgvector 实现向量数据库
  - 通过向量检索获取相关信息
  - 获取前后分块功能（确保上下文连续性）
  - 获取文档全文功能（提供完整上下文）
  - 增量索引功能（仅处理新增和变更的文件）

- **工具**
  - 向量检索工具（MCP）
  - 文档数量获取工具（MCP）
  - 索引管理工具（CLI）

## 前提条件

- Python 3.10 以上
- PostgreSQL 14 以上（带 pgvector 扩展）

## 安装

### 安装依赖

```bash
# 如果尚未安装 uv，请先安装
# pip install uv

# 基本安装（包含本地模型）
uv sync

# 如果也要使用 OpenAI 兼容 API
uv sync --extra openai
```

### 设置 PostgreSQL 和 pgvector

#### 使用 Docker 的情况

```bash
# 启动包含 pgvector 的 PostgreSQL 容器
docker run --name postgres-pgvector -e POSTGRES_PASSWORD=password -p 5432:5432 -d pgvector/pgvector:pg17
```

#### 创建数据库

启动 PostgreSQL 容器后，使用以下命令创建数据库：

```bash
# 创建 ragdb 数据库
docker exec -it postgres-pgvector psql -U postgres -c "CREATE DATABASE ragdb;"
```

#### 在现有 PostgreSQL 上安装 pgvector 的情况

```sql
-- 安装 pgvector 扩展
CREATE EXTENSION vector;
```

### 设置环境变量

创建 `.env` 文件，并设置以下环境变量：

```
# PostgreSQL 连接信息
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password
POSTGRES_DB=ragdb

# 文档目录
SOURCE_DIR=./data/source
PROCESSED_DIR=./data/processed

# 嵌入模型设置
EMBEDDING_MODEL=intfloat/multilingual-e5-large
EMBEDDING_DIM=1024
EMBEDDING_PREFIX_QUERY="query: "
EMBEDDING_PREFIX_EMBEDDING="passage: "
```

## 嵌入模型设置

本服务器可以通过环境变量选择嵌入模型和提供者。

### 环境变量一览

| 变量名 | 默认值 | 说明 |
|--------|-----------|------|
| `EMBEDDING_PROVIDER` | `local` | 提供者选择：`local`（本地模型）或 `openai`（OpenAI 兼容 API） |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | 模型名称。env 的值优先于参数 |
| `EMBEDDING_DIM` | `1024` | 向量维度数。根据模型输出维度设置 |
| `EMBEDDING_PREFIX_QUERY` | `""` | 添加到检索查询的前缀 |
| `EMBEDDING_PREFIX_EMBEDDING` | `""` | 添加到索引对象文档的前缀 |
| `EMBEDDING_API_KEY` | *(无)* | OpenAI 模式用 API 密钥。使用官方端点时必须设置 |
| `EMBEDDING_BASE_URL` | *(无)* | OpenAI 模式用基础 URL。未设置时使用官方 OpenAI 端点 |
| `EMBEDDING_API_BATCH_SIZE` | `64` | OpenAI 模式用批处理大小（一次 API 调用包含的最大文本数）|

### 提供者：local（默认）

使用 `sentence-transformers` 进行本地推理。首次启动时会下载模型。

#### multilingual-e5-large（默认）

```env
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=intfloat/multilingual-e5-large
EMBEDDING_DIM=1024
EMBEDDING_PREFIX_QUERY="query: "
EMBEDDING_PREFIX_EMBEDDING="passage: "
```

#### cl-nagoya/ruri-v3-30m（日语专用、轻量）

```env
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=cl-nagoya/ruri-v3-30m
EMBEDDING_DIM=256
EMBEDDING_PREFIX_QUERY="検索クエリ: "
EMBEDDING_PREFIX_EMBEDDING="検索文書: "
```

### 提供者：openai（OpenAI 兼容 API）

使用 OpenAI 或兼容 API（Xinference / vLLM / LocalAI / Ollama 等）。无需本地 GPU/CPU 推理。

#### 安装

```bash
uv sync --extra openai
```

#### 使用官方 OpenAI 的情况

`EMBEDDING_API_KEY` 是必需的。无需设置 `EMBEDDING_BASE_URL`。

```env
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536
EMBEDDING_API_KEY=sk-...
```

#### 使用自托管兼容服务的情况

设置 `EMBEDDING_BASE_URL` 后会被判定为自托管模式，`EMBEDDING_API_KEY` 可以省略。

```env
EMBEDDING_PROVIDER=openai
EMBEDDING_BASE_URL=http://localhost:9997/v1
EMBEDDING_MODEL=bge-m3
EMBEDDING_DIM=1024
# EMBEDDING_API_KEY 可省略（对于不需要认证的服务）
```

如果 `EMBEDDING_BASE_URL` 明确设置为 `https://api.openai.com/v1`，则判定为官方端点，需要 `EMBEDDING_API_KEY`。

#### 调整批处理大小

在为大量文档建立索引时，如果遇到 API 速率限制，请减小批处理大小。

```env
EMBEDDING_API_BATCH_SIZE=16
```

### 关于前缀

对于 E5 系列、Ruri 系列等模型，根据文本类型添加相应的前缀可以提高检索精度。

- `EMBEDDING_PREFIX_QUERY` — 自动添加到检索查询（例：`"query: "`）
- `EMBEDDING_PREFIX_EMBEDDING` — 自动添加到索引对象文档（例：`"passage: "`）

前缀由 `EmbeddingGenerator` 内部处理，MCP 客户端无需特别处理。在 OpenAI 模式下使用 E5 系列以外的模型时，请将前缀设置为空（默认为空）。

### 更改提供者或模型时的注意事项

更改提供者或模型可能会导致向量维度发生变化。由于维度不匹配会导致现有索引无法使用，因此**必须**清除后重新创建。

```bash
# 1. 清除索引
python -m src.cli clear

# 2. 更新 .env 为新的模型设置后重新索引
python -m src.cli index
```

如果维度不匹配，在首次调用 `generate_embedding` 时会出现以下错误：

```
ValueError: Embedding dimension mismatch (generate_embedding): got 1536, expected EMBEDDING_DIM=1024.
Run `python -m src.cli clear` then re-index after updating EMBEDDING_DIM.
```


## 使用方法

### 启动 MCP 服务器

#### stdio 模式（默认，适用于 Claude Desktop / Cline 等 MCP 主机）

```bash
# 使用 uv（推荐）
uv run python -m src.main

# 指定服务器名称等选项
uv run python -m src.main --name "my-rag-server" --version "1.0.0" --description "My RAG Server"

# 使用普通 Python
python -m src.main
```

#### HTTP SSE 模式（适用于容器化部署、多客户端共享等场景）

```bash
# 在本地 8000 端口启动
uv run python -m src.main --transport sse --port 8000

# 指定监听地址
uv run python -m src.main --transport sse --host 0.0.0.0 --port 8000
```

启动后，客户端可通过以下端点连接：
- `GET  http://localhost:8000/sse` — 建立 SSE 连接，接收 JSON-RPC 响应
- `POST http://localhost:8000/messages/?session_id=<id>` — 发送 JSON-RPC 请求

#### 可用选项一览

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--transport` | `stdio` | 传输方式：`stdio` 或 `sse` |
| `--host` | `0.0.0.0` | SSE 模式的监听地址 |
| `--port` | `8000` | SSE 模式的监听端口 |
| `--name` | `mcp-rag-server` | 服务器名称 |
| `--version` | `0.1.0` | 版本号 |
| `--description` | *(默认说明)* | 服务器描述 |
| `--module` | *(无)* | 额外工具模块（如 `myapp.tools`） |

### 命令行工具（CLI）的使用方法

提供了用于清除索引和建立索引的命令行工具。

#### 显示帮助

```bash
python -m src.cli --help
```

#### 清除索引

```bash
python -m src.cli clear
```

#### 为文档建立索引

```bash
# 使用默认设置建立索引（./data/source 目录）
python -m src.cli index

# 为特定目录建立索引
python -m src.cli index --directory ./path/to/documents

# 指定分块大小和重叠量建立索引
python -m src.cli index --directory ./data/source --chunk-size 300 --chunk-overlap 50
# 或使用简短形式
python -m src.cli index -d ./data/source -s 300 -o 50

# 增量索引（仅处理新增和变更的文件）
python -m src.cli index --incremental
# 或使用简短形式
python -m src.cli index -i
```

#### 获取索引中的文档数量

```bash
python -m src.cli count
```

### 在 MCP 主机中的设置

要在 MCP 主机（Claude Desktop、Cline、Cursor 等）中使用本服务器，请进行如下设置。关于设置用的 JSON 文件，请参考各 MCP 主机的文档。

#### stdio 模式（推荐）

```json
{
  "mcpServers": {
    "mcp-rag-server": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/mcp-rag-server",
        "python",
        "-m",
        "src.main"
      ]
    }
  }
}
```

#### SSE 模式（服务器单独部署的情况）

先单独启动服务器：

```bash
uv run python -m src.main --transport sse --port 8000
```

然后在 MCP 主机中指定 SSE 端点（各主机的具体写法可能有所不同）：

```json
{
  "mcpServers": {
    "mcp-rag-server": {
      "url": "http://localhost:8000/sse"
    }
  }
}
```

#### 不使用 uv 的情况

在未安装 uv 的环境中，可以使用普通 Python（stdio 模式）：

```json
{
  "command": "python",
  "args": ["-m", "src.main"],
  "cwd": "/path/to/mcp-rag-server"
}
```

## RAG 工具的使用方法

### search

进行向量检索。

```json
{
  "jsonrpc": "2.0",
  "method": "search",
  "params": {
    "query": "Python 的生成器是什么？",
    "limit": 5,
    "with_context": true,
    "context_size": 1,
    "full_document": false
  },
  "id": 1
}
```

#### 参数说明

- `query`：检索查询（必需）
- `limit`：返回结果的数量（默认：5）
- `with_context`：是否获取前后分块（默认：true）
- `context_size`：获取前后分块的数量（默认：1）
- `full_document`：是否获取文档全文（默认：false）

#### 检索结果的改进

本工具通过以下功能提供更好的检索结果：

1. **获取前后分块功能**：
   - 获取检索命中分块的前后分块并包含在结果中
   - 可通过 `with_context` 参数启用/禁用
   - 可通过 `context_size` 参数调整获取前后分块的数量

2. **获取文档全文功能**：
   - 获取检索命中文档的全文并包含在结果中
   - 可通过 `full_document` 参数启用/禁用
   - 特别适用于处理短文档或需要完整上下文的文档

3. **结果格式改进**：
   - 按文件分组检索结果
   - 视觉上区分"检索命中"、"前后上下文"和"文档全文"
   - 按分块索引排序以保持文档的连贯性

### get_document_count

获取索引中的文档数量。

```json
{
  "jsonrpc": "2.0",
  "method": "get_document_count",
  "params": {},
  "id": 2
}
```

## 使用示例

1. 将文档文件放入 `data/source` 目录。支持的文件格式如下：
   - Markdown（.md, .markdown）
   - 文本（.txt）
   - PowerPoint（.ppt, .pptx）
   - Word（.doc, .docx）
   - PDF（.pdf）

2. 使用 CLI 命令为文档建立索引：
   ```bash
   # 首次建立全量索引
   python -m src.cli index

   # 之后使用增量索引高效更新
   python -m src.cli index -i
   ```

3. 启动 MCP 服务器：
   ```bash
   uv run python -m src.main
   ```

4. 使用 `search` 工具进行检索。

## 备份与恢复

要在其他 PC 上使用已建立索引的数据库，请按以下步骤进行备份和恢复。

### 最小备份（仅 PostgreSQL 数据库）

如果只是想在其他 PC 上使用 RAG 检索功能，只需备份 PostgreSQL 数据库即可。因为所有向量化数据都存储在数据库中。

#### PostgreSQL 数据库备份

要备份 PostgreSQL 数据库，请在 Docker 容器内使用 `pg_dump` 命令：

```bash
# 在 Docker 容器内备份数据库
docker exec -it postgres-pgvector pg_dump -U postgres -d ragdb -F c -f /tmp/ragdb_backup.dump

# 将备份文件从容器复制到主机
docker cp postgres-pgvector:/tmp/ragdb_backup.dump ./ragdb_backup.dump
```

这样会在当前目录创建 PostgreSQL 数据库的备份文件（例如：239MB）。

#### 最小恢复步骤

1. 在新 PC 上设置 PostgreSQL 和 pgvector：

```bash
# 使用 Docker 的情况
docker run --name postgres-pgvector -e POSTGRES_PASSWORD=password -p 5432:5432 -d pgvector/pgvector:pg17

# 创建数据库
docker exec -it postgres-pgvector psql -U postgres -c "CREATE DATABASE ragdb;"
```

2. 从备份恢复数据库：

```bash
# 将备份文件复制到容器
docker cp ./ragdb_backup.dump postgres-pgvector:/tmp/ragdb_backup.dump

# 在容器内恢复数据库
docker exec -it postgres-pgvector pg_restore -U postgres -d ragdb -c /tmp/ragdb_backup.dump
```

3. 确认环境设置：

在新 PC 上，请确认 `.env` 文件中的 PostgreSQL 连接信息设置正确。

4. 验证运行：

```bash
python -m src.cli count
```

这将显示索引中的文档数量。如果显示的数量与原 PC 相同，则表示恢复成功。

### 完全备份（可选）

如果将来计划添加新文档，或需要使用增量索引功能，建议进行以下额外备份：

#### 备份已处理的文档

备份已处理文档目录：

```bash
# 将已处理文档目录备份为 ZIP 文件
zip -r processed_data_backup.zip data/processed/
```

#### 备份环境设置文件

备份 `.env` 文件：

```bash
# 复制 .env 文件
cp .env env_backup.txt
```

#### 完全恢复步骤

1. 前提条件

新 PC 上需要安装以下软件：

- Python 3.10 以上
- PostgreSQL 14 以上（带 pgvector 扩展）
- mcp-rag-server 代码库

2. 按上述"最小恢复步骤"恢复 PostgreSQL 数据库。

3. 恢复已处理的文档：

```bash
# 解压 ZIP 文件
unzip processed_data_backup.zip -d /path/to/mcp-rag-server/
```

4. 恢复环境设置文件：

```bash
# 恢复 .env 文件
cp env_backup.txt /path/to/mcp-rag-server/.env
```

根据需要，编辑 `.env` 文件的设置（特别是 PostgreSQL 连接信息）以适应新 PC 环境。

5. 验证运行：

```bash
python -m src.cli count
```

### 注意事项

- PostgreSQL 版本和 pgvector 版本需要在原 PC 和新 PC 之间保持兼容。
- 如果数据量较大，备份和恢复可能需要较长时间。
- 在新 PC 上，需要预先安装所需的 Python 包（`sentence-transformers`、`psycopg2-binary` 等）。

## 目录结构

```
mcp-rag-server/
├── data/
│   ├── source/        # 原始文档（支持层级结构）
│   │   ├── markdown/  # Markdown 文件
│   │   ├── docs/      # 文档文件
│   │   └── slides/    # 演示文稿文件
│   └── processed/     # 已处理文件（文本已提取）
│       └── file_registry.json  # 已处理文件信息（用于增量索引）
├── docs/
│   └── design.md      # 需求与设计文档
├── logs/              # 日志文件
├── src/
│   ├── __init__.py
│   ├── document_processor.py  # 文档处理模块
│   ├── embedding_generator.py # 嵌入向量生成模块（支持本地/OpenAI 兼容 API）
│   ├── example_tool.py        # 示例工具模块（旧版插件示例）
│   ├── main.py                # 主入口点（支持 --transport stdio/sse）
│   ├── mcp_server.py          # 旧版 JSON-RPC over stdio 服务器（保留兼容）
│   ├── rag_service.py         # RAG 服务模块
│   ├── rag_tools.py           # RAG 工具模块（含 SDK 路径）
│   ├── server.py              # SDK 服务器工厂、ToolRegistry、传输运行函数
│   └── vector_database.py     # 向量数据库模块
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_document_processor.py
│   ├── test_embedding_generator.py
│   ├── test_example_tool.py
│   ├── test_mcp_server.py       # 旧版服务器单元测试
│   ├── test_rag_service.py
│   ├── test_rag_tools.py
│   ├── test_server.py           # SDK 服务器单元测试（anyio）
│   ├── test_sse_transport.py    # SSE 传输集成测试（真实 HTTP 服务器）
│   └── test_vector_database.py
├── .env           # 环境变量设置文件
├── .gitignore
├── LICENSE
├── pyproject.toml
└── README.md
```

## 许可证

本项目基于 MIT 许可证发布。详情请参阅 [LICENSE](LICENSE) 文件。
