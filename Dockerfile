# MCP RAG Server — 本地模型版本（包含 sentence-transformers / torch）
FROM python:3.11-slim

# uv 公式インストーラー
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Docker layer cache: 先に依存だけインストール
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# アプリケーションソースをコピー
COPY src/ src/

# データディレクトリを作成
RUN mkdir -p data/source data/processed logs

EXPOSE 8000

CMD ["uv", "run", "python", "-m", "src.main", "--transport", "sse", "--host", "0.0.0.0", "--port", "8000"]
