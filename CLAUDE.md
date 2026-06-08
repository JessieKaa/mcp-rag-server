# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指导。

## 项目概述

MCP RAG Server 是一个实现了 Model Context Protocol (MCP) 和 RAG (Retrieval-Augmented Generation) 功能的 Python 服务器。它提供了一个支持多种文档格式的向量检索系统。

## 主要命令

### 开发环境设置
```bash
# 安装依赖 (使用 uv)
uv sync

# 需要设置 PostgreSQL 和 pgvector
# 使用 Docker 启动 PostgreSQL:
docker run -d \
  --name pgvector-db \
  -e POSTGRES_USER=your_user \
  -e POSTGRES_PASSWORD=your_password \
  -e POSTGRES_DB=your_database \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# 需要设置 .env 文件
```

### 执行命令
```bash
# 启动 MCP 服务器（stdio 模式，默认）
uv run python -m src.main

# 启动 MCP 服务器（HTTP SSE 模式）
uv run python -m src.main --transport sse --host 0.0.0.0 --port 8000

# 指定服务器名称等选项
uv run python -m src.main --name "my-rag-server" --version "1.0.0" --description "My RAG Server"

# 使用 CLI 为文档建立索引
uv run python -m src.cli index
uv run python -m src.cli index --incremental  # 增量索引

# 清除索引
uv run python -m src.cli clear

# 确认文档数量
uv run python -m src.cli count
```

### 运行测试
```bash
# 使用 pytest 运行测试
uv run pytest
```

### Lint 和格式化
```bash
# 使用 ruff 进行 lint 检查
uv run ruff check --line-length=127

# 使用 ruff 自动格式化
uv run ruff format --line-length=127

# 检查格式化（显示差异）
uv run ruff format --check --diff --line-length=127
```

### Pull Request (PR)

#### 创建 PR 时
- 当被要求创建 PR 时，请先用 git 命令确认差异，然后使用 `gh pr` 命令创建 PR
- PR 的 description 请参考 .github/pull_request_template.md 的格式

#### PR 审查时
请按以下步骤为每个文件添加评论：

1. 检查要点请参考 .github/pull_request_template.md
2. 确认 PR 差异：
   ```bash
   gh pr diff <PR号>
   ```

3. 在确认每个文件的完整内容和 PR 差异后，添加审查评论：
   ```bash
   gh api repos/<owner>/<repo>/pulls/<PR号>/comments \
     -F body="审查评论" \
     -F commit_id="$(gh pr view <PR号> --json headRefOid --jq .headRefOid)" \
     -F path="目标文件路径" \
     -F position=<diff行号>
   ```

   参数说明：
   - position: diff 的行号（新文件从 1 开始）
   - commit_id: 自动获取 PR 的最新 commit ID

## 架构概要

### 核心构成
- **SDK 服务器层**: `src/server.py` 基于官方 MCP Python SDK，统一管理工具注册和传输方式
- **传统服务器层（保留兼容）**: `src/mcp_server.py` 原有的 JSON-RPC over stdio 实现，现有测试继续使用
- **RAG 服务层**: `src/rag_service.py` 统管文档处理和检索
- **数据层**: 使用 PostgreSQL + pgvector 实现向量数据库

### 主要模块
- `src/main.py`: 入口点，支持 `--transport [stdio|sse]` 切换传输方式
- `src/server.py`: SDK 服务器工厂、`ToolRegistry`（工具注册中心）、stdio/SSE 传输运行函数
- `src/rag_tools.py`: RAG 工具定义，包含旧版 `register_rag_tools()` 和新版 `register_rag_tools_sdk()`
- `src/mcp_server.py`: 旧版手写 JSON-RPC 服务器（保留供测试和旧插件使用）
- `src/document_processor.py`: 文档解析和分块
- `src/embedding_generator.py`: 嵌入向量生成（支持本地模型和 OpenAI 兼容 API）
- `src/vector_database.py`: PostgreSQL/pgvector 接口

### 数据流
1. 读取 `data/source/` 下的文档
2. 使用 markitdown 进行文本转换、分块
3. 使用 sentence-transformers 或 OpenAI 兼容 API 生成嵌入向量
4. 将向量与文档一起保存到 PostgreSQL
5. 通过 MCP 工具（stdio 或 HTTP SSE）提供语义检索

### 重要设计模式
- **ToolRegistry**：防止重复注册，所有工具统一通过 `registry.register(tool, handler)` 注册后一次性挂载到 SDK 服务器
- **统一执行锁**：`server.tool_execution_lock` 在内置 RAG 工具和旧版插件工具间共享，保证 psycopg2 单连接的并发安全
- **旧版插件桥接**：实现 `register_tools(shim)` 的旧版插件通过 `_adapt_legacy_tools()` 自动适配到新 SDK 路径
- 增量索引：通过文件哈希检测变更
- 重叠分块：为保持上下文连贯性而设置重叠
- 相邻分块获取：可获取检索结果的前后上下文

## 环境变量设置

在 `.env` 文件中设置以下内容：
```
# PostgreSQL 连接信息
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
POSTGRES_DB=your_database

# 路径设置
SOURCE_DIR=data/source
PROCESSED_DIR=data/processed
```

## 支持的文档格式
- Markdown (.md)
- 文本 (.txt)
- PowerPoint (.pptx)
- PDF (.pdf)
- Word (.docx)

## 开发注意事项
- 如需添加新的文档格式，请扩展 `document_processor.py`
- 如需修改向量数据库架构，请更新 `vector_database.py` 的 `create_tables()`
- 如需添加 MCP 工具（新方式），在 `rag_tools.py` 的 `register_rag_tools_sdk()` 中调用 `registry.register(tool, async_handler)`
- 如需开发插件模块，优先实现 `register_tools_sdk(registry: ToolRegistry)` 接口；旧版 `register_tools(server: MCPServer)` 也受支持（自动桥接）
- `src/mcp_server.py` 保持不变，其测试 `tests/test_mcp_server.py` 必须始终通过