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
# 启动 MCP 服务器
uv run python -m src.main

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
- **MCP 服务器层**: `src/mcp_server.py` 处理 JSON-RPC 通信
- **RAG 服务层**: `src/rag_service.py` 统管文档处理和检索
- **数据层**: 使用 PostgreSQL + pgvector 实现向量数据库

### 主要模块
- `src/main.py`: 入口点
- `src/rag_tools.py`: MCP 用的检索工具定义
- `src/document_processor.py`: 文档解析和分块
- `src/embedding_generator.py`: 使用 multilingual-e5-large 模型生成嵌入向量
- `src/vector_database.py`: PostgreSQL/pgvector 接口

### 数据流
1. 读取 `data/source/` 下的文档
2. 使用 markitdown 进行文本转换、分块
3. 使用 sentence-transformers 生成嵌入向量
4. 将向量与文档一起保存到 PostgreSQL
5. 通过 MCP 工具提供语义检索

### 重要设计模式
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
- 如需添加 MCP 工具，请在 `rag_tools.py` 中添加工具定义