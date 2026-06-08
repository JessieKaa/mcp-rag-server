"""
基于 MCP Python SDK 的服务器模块，支持 stdio 和 HTTP SSE 两种传输方式。
"""

import logging
import threading
from typing import Callable

import anyio
from mcp.server.lowlevel import Server as LowLevelServer
from mcp import types as mcp_types

from .rag_tools import register_rag_tools_sdk
from .rag_service import RAGService

logger = logging.getLogger(__name__)

# 所有 SDK 暴露的同步处理函数共用此锁（psycopg2 单连接非线程安全）
# rag_tools.py 也导入此锁，确保内置工具和旧版插件工具不会并发执行
tool_execution_lock = threading.Lock()


class ToolRegistry:
    """
    聚合工具定义和处理函数，统一注册到 SDK 服务器。

    防止多次注册 list_tools/call_tool 导致覆盖问题。
    """

    def __init__(self):
        self._tools: dict[str, mcp_types.Tool] = {}
        self._handlers: dict[str, Callable] = {}

    def register(self, tool: mcp_types.Tool, handler: Callable) -> None:
        """注册工具，若名称已存在则抛出 ValueError。"""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
        self._handlers[tool.name] = handler

    def wire(self, server: LowLevelServer) -> None:
        """在 SDK 服务器上安装唯一的 list_tools 和 call_tool 处理函数对。"""
        tools = list(self._tools.values())
        handlers = dict(self._handlers)

        @server.list_tools()
        async def list_tools() -> list[mcp_types.Tool]:
            return tools

        @server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[mcp_types.TextContent]:
            if name not in handlers:
                raise ValueError(f"Unknown tool: {name}")
            return await handlers[name](arguments)


def _legacy_item_to_sdk(item) -> mcp_types.TextContent | mcp_types.ImageContent | mcp_types.EmbeddedResource:
    """将单个旧版内容项转换为对应的 SDK 类型。"""
    if not isinstance(item, dict):
        return mcp_types.TextContent(type="text", text=str(item))
    item_type = item.get("type")
    if item_type == "text":
        return mcp_types.TextContent(type="text", text=item.get("text", ""))
    if item_type == "image":
        return mcp_types.ImageContent(type="image", data=item.get("data", ""), mimeType=item.get("mimeType", "image/png"))
    if item_type == "resource":
        resource = item.get("resource", {})
        if "blob" in resource:
            res_contents = mcp_types.BlobResourceContents(uri=resource.get("uri", ""), blob=resource["blob"], mimeType=resource.get("mimeType"))
        else:
            res_contents = mcp_types.TextResourceContents(uri=resource.get("uri", ""), text=resource.get("text", ""), mimeType=resource.get("mimeType"))
        return mcp_types.EmbeddedResource(type="resource", resource=res_contents)
    # unknown type — preserve as text
    return mcp_types.TextContent(type="text", text=str(item))


def _legacy_result_to_content(result) -> list:
    """
    将旧版 MCPServer 处理函数的返回值转换为 SDK 内容列表。

    多项错误有效载荷合并为单一 RuntimeError 字符串。
    """
    if isinstance(result, dict) and "content" in result:
        if result.get("isError"):
            text_parts = [
                item.get("text", str(item))
                for item in result["content"]
                if isinstance(item, dict)
            ]
            raise RuntimeError(" ".join(text_parts))
        return [_legacy_item_to_sdk(item) for item in result["content"]]
    return [mcp_types.TextContent(type="text", text=str(result))]


def _adapt_legacy_tools(shim, registry: ToolRegistry) -> None:
    """将旧版 register_tools(shim: MCPServer) 插件桥接到 ToolRegistry。"""
    for name, tool_def in shim.tools.items():
        handler_fn = shim.tool_handlers[name]
        tool = mcp_types.Tool(
            name=name,
            description=tool_def.get("description", ""),
            inputSchema=tool_def.get("inputSchema", {"type": "object", "properties": {}}),
        )

        def _make_async_handler(fn):
            async def _handler(arguments: dict) -> list[mcp_types.TextContent]:
                def _run():
                    with tool_execution_lock:
                        return fn(arguments)
                result = await anyio.to_thread.run_sync(_run)
                return _legacy_result_to_content(result)
            return _handler

        registry.register(tool, _make_async_handler(handler_fn))


def create_sdk_server(
    name: str,
    version: str,
    description: str,
    rag_service: RAGService,
    extra_module=None,
) -> LowLevelServer:
    """
    创建并配置 SDK 服务器。

    Args:
        name: 服务器名称
        version: 版本号
        description: 服务器描述（映射为 SDK instructions）
        rag_service: RAG 服务实例
        extra_module: 可选的额外工具模块
    """
    registry = ToolRegistry()
    register_rag_tools_sdk(registry, rag_service)

    if extra_module is not None:
        if hasattr(extra_module, "register_tools_sdk"):
            extra_module.register_tools_sdk(registry)
        elif hasattr(extra_module, "register_tools"):
            from .mcp_server import MCPServer
            shim = MCPServer()
            extra_module.register_tools(shim)
            _adapt_legacy_tools(shim, registry)
        else:
            logger.warning("插件模块没有 register_tools_sdk 或 register_tools 函数，跳过")

    server = LowLevelServer(name, version=version, instructions=description)
    registry.wire(server)
    return server


async def run_stdio(server: LowLevelServer) -> None:
    """以 stdio 传输方式运行 SDK 服务器。"""
    from mcp.server.stdio import stdio_server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def create_sse_app(server: LowLevelServer):
    """
    创建 Starlette ASGI 应用，提供 /sse 和 /messages/ 端点。

    注意：SseServerTransport 不会在 SSE 断开时清理会话，
    过期会话将持续保留直到进程重启。
    """
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.requests import Request

    sse_transport = SseServerTransport("/messages/")

    async def handle_sse(request: Request):
        async with sse_transport.connect_sse(
            request.scope, request.receive, request._send
        ) as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    async def handle_messages(request: Request):
        await sse_transport.handle_post_message(request.scope, request.receive, request._send)

    return Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Route("/messages/", endpoint=handle_messages, methods=["POST"]),
        ]
    )


def run_sse(server: LowLevelServer, host: str, port: int) -> None:
    """以 SSE 传输方式运行 SDK 服务器。"""
    import uvicorn
    app = create_sse_app(server)
    uvicorn.run(app, host=host, port=port)
