"""
RAG 工具模块

提供用于注册到 MCP 服务器的 RAG 相关工具。
"""

import logging
import os

from typing import TYPE_CHECKING, Dict, Any

import anyio
from mcp import types as mcp_types

from .document_processor import DocumentProcessor
from .embedding_generator import EmbeddingGenerator
from .vector_database import VectorDatabase
from .rag_service import RAGService
from .reranker import create_reranker_from_env

from dotenv import load_dotenv

if TYPE_CHECKING:
    from .server import ToolRegistry

load_dotenv()


# 使用 server 模块中的统一执行锁，确保内置工具和旧版插件工具都经过同一串行化路径
# 延迟导入避免循环依赖
def _get_tool_lock():
    from .server import tool_execution_lock

    return tool_execution_lock


def register_rag_tools(server, rag_service: RAGService):
    """
    将 RAG 相关工具注册到 MCP 服务器。

    Args:
        server: MCP 服务器实例
        rag_service: RAG 服务实例
    """
    # 注册搜索工具
    server.register_tool(
        name="search",
        description="进行向量检索",
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索查询",
                },
                "limit": {
                    "type": "integer",
                    "description": "返回结果的数量（默认：5）",
                    "default": 5,
                },
                "with_context": {
                    "type": "boolean",
                    "description": "是否获取前后分块（默认：true）",
                    "default": True,
                },
                "context_size": {
                    "type": "integer",
                    "description": "获取前后分块的数量（默认：1）",
                    "default": 1,
                },
                "full_document": {
                    "type": "boolean",
                    "description": "是否获取文档全文（默认：false）",
                    "default": False,
                },
            },
            "required": ["query"],
        },
        handler=lambda params: search_handler(params, rag_service),
    )

    # 注册文档数量获取工具
    server.register_tool(
        name="get_document_count",
        description="获取索引中的文档数量",
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
        },
        handler=lambda params: get_document_count_handler(params, rag_service),
    )


def search_handler(params: Dict[str, Any], rag_service: RAGService) -> Dict[str, Any]:
    """
    进行向量检索的处理函数

    Args:
        params: 参数
            - query: 检索查询
            - limit: 返回结果的数量（默认：5）
            - with_context: 是否获取前后分块（默认：true）
            - context_size: 获取前后分块的数量（默认：1）
            - full_document: 是否获取文档全文（默认：false）
        rag_service: RAG 服务实例

    Returns:
        检索结果
    """
    query = params.get("query")
    limit = params.get("limit", 5)
    with_context = params.get("with_context", True)
    context_size = params.get("context_size", 1)
    full_document = params.get("full_document", False)

    if not query:
        return {
            "content": [
                {
                    "type": "text",
                    "text": "错误：未指定检索查询",
                }
            ],
            "isError": True,
        }

    try:
        # 确认文档数量
        doc_count = rag_service.get_document_count()
        if doc_count == 0:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "索引中不存在文档。请使用 CLI 命令 `python -m src.cli index` 为文档建立索引。",
                    }
                ],
                "isError": True,
            }

        # 执行检索（获取前后分块、获取文档全文）
        results = rag_service.search(query, limit, with_context, context_size, full_document)

        if not results:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"未找到与查询 '{query}' 匹配的结果",
                    }
                ]
            }

        # 按文件分组结果
        file_groups = {}
        for result in results:
            file_path = result["file_path"]
            if file_path not in file_groups:
                file_groups[file_path] = []
            file_groups[file_path].append(result)

        # 在各组内按分块索引排序
        for file_path in file_groups:
            file_groups[file_path].sort(key=lambda x: x["chunk_index"])

        # 有重排序时按每组最优 rerank_score 降序排列文件分组
        has_rerank = any("rerank_score" in r for r in results)
        if has_rerank:
            sorted_groups = sorted(
                file_groups.items(),
                key=lambda item: max(r.get("rerank_score", float("-inf")) for r in item[1]),
                reverse=True,
            )
        else:
            sorted_groups = list(file_groups.items())

        # 格式化结果
        content_items = [
            {
                "type": "text",
                "text": f"查询 '{query}' 的检索结果（{len(results)} 条）:",
            }
        ]

        # 按文件显示结果
        for i, (file_path, group) in enumerate(sorted_groups):
            file_name = os.path.basename(file_path)

            # 文件标题
            content_items.append(
                {
                    "type": "text",
                    "text": f"\n[{i + 1}] 文件：{file_name}",
                }
            )

            # 显示各个分块
            for j, result in enumerate(group):
                is_context = result.get("is_context", False)
                is_full_document = result.get("is_full_document", False)

                if "rerank_score" in result:
                    score_str = f"重排分数：{result['rerank_score']:.4f}"
                else:
                    score_str = f"相似度：{result.get('similarity', 0) * 100:.2f}%"

                # 根据全文文档、上下文分块、检索命中分块改变显示
                if is_full_document:
                    content_items.append(
                        {
                            "type": "text",
                            "text": f"\n+++ 文档全文（分块 {result['chunk_index']}) +++\n{result['content']}",
                        }
                    )
                elif is_context:
                    content_items.append(
                        {
                            "type": "text",
                            "text": f"\n--- 前后上下文（分块 {result['chunk_index']}) ---\n{result['content']}",
                        }
                    )
                else:
                    content_items.append(
                        {
                            "type": "text",
                            "text": f"\n=== 检索命中（分块 {result['chunk_index']}, {score_str}) ===\n{result['content']}",
                        }
                    )

        return {"content": content_items}

    except Exception as e:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"检索过程中发生错误：{str(e)}",
                }
            ],
            "isError": True,
        }


def get_document_count_handler(params: Dict[str, Any], rag_service: RAGService) -> Dict[str, Any]:
    """
    获取索引中文档数量的处理函数

    Args:
        params: 参数（未使用）
        rag_service: RAG 服务实例

    Returns:
        文档数量
    """
    try:
        # 获取文档数量
        count = rag_service.get_document_count()

        return {
            "content": [
                {
                    "type": "text",
                    "text": f"索引中的文档数量：{count}",
                }
            ]
        }

    except Exception as e:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"获取文档数量时发生错误：{str(e)}",
                }
            ],
            "isError": True,
        }


def _sync_result_to_content(result: Dict[str, Any]) -> list:
    """将同步处理函数返回的字典转换为 SDK TextContent 列表，isError=True 时抛出 RuntimeError。"""
    if isinstance(result, dict) and "content" in result:
        if result.get("isError"):
            text_parts = [item.get("text", str(item)) for item in result["content"] if isinstance(item, dict)]
            raise RuntimeError(" ".join(text_parts))
        return [
            mcp_types.TextContent(type="text", text=item.get("text", str(item)))
            if isinstance(item, dict) and item.get("type") == "text"
            else mcp_types.TextContent(type="text", text=str(item))
            for item in result["content"]
        ]
    return [mcp_types.TextContent(type="text", text=str(result))]


def register_rag_tools_sdk(registry: "ToolRegistry", rag_service: RAGService) -> None:
    """
    将 RAG 工具注册到 ToolRegistry（SDK 路径）。

    每个处理函数通过 anyio.to_thread.run_sync + _tool_lock 包装同步调用。
    """

    search_tool = mcp_types.Tool(
        name="search",
        description="进行向量检索",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索查询"},
                "limit": {"type": "integer", "description": "返回结果的数量（默认：5）", "default": 5},
                "with_context": {"type": "boolean", "description": "是否获取前后分块（默认：true）", "default": True},
                "context_size": {"type": "integer", "description": "获取前后分块的数量（默认：1）", "default": 1},
                "full_document": {"type": "boolean", "description": "是否获取文档全文（默认：false）", "default": False},
            },
            "required": ["query"],
        },
    )

    count_tool = mcp_types.Tool(
        name="get_document_count",
        description="获取索引中的文档数量",
        inputSchema={"type": "object", "properties": {}, "required": []},
    )

    async def _search_async(arguments: dict) -> list:
        def _run():
            with _get_tool_lock():
                return search_handler(arguments, rag_service)

        result = await anyio.to_thread.run_sync(_run)
        return _sync_result_to_content(result)

    async def _count_async(arguments: dict) -> list:
        def _run():
            with _get_tool_lock():
                return get_document_count_handler(arguments, rag_service)

        result = await anyio.to_thread.run_sync(_run)
        return _sync_result_to_content(result)

    registry.register(search_tool, _search_async)
    registry.register(count_tool, _count_async)


def create_rag_service_from_env() -> RAGService:
    """
    从环境变量创建 RAG 服务。

    Returns:
        RAG 服务实例
    """
    # 从环境变量获取连接信息
    postgres_host = os.environ.get("POSTGRES_HOST", "localhost")
    postgres_port = os.environ.get("POSTGRES_PORT", "5432")
    postgres_user = os.environ.get("POSTGRES_USER", "postgres")
    postgres_password = os.environ.get("POSTGRES_PASSWORD", "password")
    postgres_db = os.environ.get("POSTGRES_DB", "ragdb")

    embedding_model = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")

    # 创建组件
    document_processor = DocumentProcessor()
    embedding_generator = EmbeddingGenerator(model_name=embedding_model)
    vector_database = VectorDatabase(
        {
            "host": postgres_host,
            "port": postgres_port,
            "user": postgres_user,
            "password": postgres_password,
            "database": postgres_db,
        }
    )

    # 创建 RAG 服务
    rerank_factor = 3
    rerank_factor_str = os.environ.get("RERANK_FACTOR", "3")
    try:
        rerank_factor = int(rerank_factor_str)
    except ValueError:
        logging.getLogger("rag_tools").warning(f"RERANK_FACTOR 值 '{rerank_factor_str}' 无效，使用默认值 3")
        rerank_factor = 3

    rag_service = RAGService(
        document_processor,
        embedding_generator,
        vector_database,
        rerank_factor=rerank_factor,
    )

    # 延迟初始化重排序器，仅在首次搜索时加载模型
    rag_service.set_reranker_factory(create_reranker_from_env)

    return rag_service
