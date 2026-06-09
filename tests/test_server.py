"""
SDK 服务器单元测试
"""

import pytest
from unittest.mock import MagicMock
from mcp import types

from src.server import create_sdk_server, ToolRegistry
from src.rag_service import RAGService


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def mock_rag_service():
    svc = MagicMock(spec=RAGService)
    svc.get_document_count.return_value = 1
    svc.search.return_value = [
        {
            "file_path": "data/source/test.md",
            "chunk_index": 0,
            "content": "Test content",
            "similarity": 0.9,
            "is_context": False,
            "is_full_document": False,
        }
    ]
    return svc


@pytest.fixture
def sdk_server(mock_rag_service):
    return create_sdk_server(
        name="test-server",
        version="0.1.0",
        description="Test",
        rag_service=mock_rag_service,
    )


@pytest.mark.anyio
async def test_list_tools_returns_two_tools(sdk_server):
    handler = sdk_server.request_handlers[types.ListToolsRequest]
    req = types.ListToolsRequest(method="tools/list", params=None)
    result = await handler(req)
    tool_names = [t.name for t in result.root.tools]
    assert "search" in tool_names
    assert "get_document_count" in tool_names


@pytest.mark.anyio
async def test_call_tool_search_success(sdk_server):
    handler = sdk_server.request_handlers[types.CallToolRequest]
    req = types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name="search", arguments={"query": "test"}),
    )
    result = await handler(req)
    assert result.root.isError is False
    assert len(result.root.content) > 0
    assert any("test" in c.text.lower() for c in result.root.content)


@pytest.mark.anyio
async def test_call_tool_get_document_count(sdk_server, mock_rag_service):
    mock_rag_service.get_document_count.return_value = 42
    handler = sdk_server.request_handlers[types.CallToolRequest]
    req = types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name="get_document_count", arguments={}),
    )
    result = await handler(req)
    assert result.root.isError is False
    assert any("42" in c.text for c in result.root.content)


@pytest.mark.anyio
async def test_call_tool_missing_query_returns_error(sdk_server):
    handler = sdk_server.request_handlers[types.CallToolRequest]
    req = types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name="search", arguments={}),
    )
    result = await handler(req)
    assert result.root.isError is True


@pytest.mark.anyio
async def test_call_tool_unknown_tool_returns_error(sdk_server):
    handler = sdk_server.request_handlers[types.CallToolRequest]
    req = types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(name="nonexistent_tool", arguments={}),
    )
    result = await handler(req)
    assert result.root.isError is True


def test_tool_registry_duplicate_raises():
    registry = ToolRegistry()
    tool = types.Tool(name="dup", description="d", inputSchema={"type": "object", "properties": {}})

    async def handler(args):
        return []

    registry.register(tool, handler)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(tool, handler)
