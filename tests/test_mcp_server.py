"""
MCP 服务器测试
"""

import json
from unittest.mock import patch

from src.mcp_server import MCPServer


def test_mcp_server_initialization():
    """测试 MCP 服务器初始化"""
    server = MCPServer()
    assert server.tools == {}
    assert server.tool_handlers == {}


def test_register_tool():
    """测试工具注册"""
    server = MCPServer()

    # 测试用处理函数
    def test_handler(params):
        return {"result": "test"}

    # 注册工具
    server.register_tool(
        name="test_tool",
        description="Test tool",
        input_schema={
            "type": "object",
            "properties": {
                "param1": {"type": "string"},
            },
            "required": ["param1"],
        },
        handler=test_handler,
    )

    # 确认工具已注册
    assert "test_tool" in server.tools
    assert "test_tool" in server.tool_handlers
    assert server.tools["test_tool"]["name"] == "test_tool"
    assert server.tools["test_tool"]["description"] == "Test tool"
    assert server.tool_handlers["test_tool"] == test_handler


@patch("sys.stdout")
def test_send_response(mock_stdout):
    """测试响应发送"""
    server = MCPServer()

    # 发送响应
    response = {"jsonrpc": "2.0", "result": "test", "id": 1}
    server._send_response(response)

    # 确认标准输出中输出了正确的 JSON
    # 注：根据实现不同，write 可能被调用 2 次（分别写入 JSON 和换行符）
    mock_stdout.write.assert_any_call(json.dumps(response))
    mock_stdout.flush.assert_called_once()


@patch("src.mcp_server.MCPServer._send_result")
def test_handle_tools_call(mock_send_result):
    """测试 tools/call 方法处理"""
    server = MCPServer()

    # 测试用处理函数
    def test_handler(params):
        return {"content": [{"type": "text", "text": f"Result: {params.get('param1')}"}]}

    # 注册工具
    server.register_tool(
        name="test_tool",
        description="Test tool",
        input_schema={
            "type": "object",
            "properties": {
                "param1": {"type": "string"},
            },
            "required": ["param1"],
        },
        handler=test_handler,
    )

    # 调用 tools/call 方法
    params = {
        "name": "test_tool",
        "arguments": {
            "param1": "test_value",
        },
    }
    server._handle_tools_call(params, 1)

    # 确认 _send_result 被正确调用
    mock_send_result.assert_called_once_with({"content": [{"type": "text", "text": "Result: test_value"}]}, 1)


@patch("src.mcp_server.MCPServer._send_result")
def test_handle_tools_call_error(mock_send_result):
    """测试 tools/call 方法的错误处理"""
    server = MCPServer()

    # 测试用处理函数（抛出异常）
    def test_handler(params):
        raise ValueError("Test error")

    # 注册工具
    server.register_tool(
        name="test_tool",
        description="Test tool",
        input_schema={
            "type": "object",
            "properties": {
                "param1": {"type": "string"},
            },
            "required": ["param1"],
        },
        handler=test_handler,
    )

    # 调用 tools/call 方法
    params = {
        "name": "test_tool",
        "arguments": {
            "param1": "test_value",
        },
    }
    server._handle_tools_call(params, 1)

    # 确认 _send_result 被正确调用
    mock_send_result.assert_called_once()
    args, _ = mock_send_result.call_args
    assert args[0]["isError"] is True
    assert "Test error" in args[0]["content"][0]["text"]


@patch("src.mcp_server.MCPServer._send_result")
def test_handle_notifications_initialized(mock_send_result):
    """测试 notifications/initialized 方法处理"""
    server = MCPServer()

    # 调用 notifications/initialized 方法
    params = {}
    request_id = 1
    server._handle_notifications_initialized(params, request_id)

    # 确认指定 request_id 时会调用 _send_result
    mock_send_result.assert_called_once_with({}, request_id)

    # 确认 request_id 为 None 时不会调用 _send_result
    mock_send_result.reset_mock()
    server._handle_notifications_initialized(params, None)
    mock_send_result.assert_not_called()


@patch("src.mcp_server.MCPServer._send_result")
@patch("src.mcp_server.MCPServer._get_resources")
def test_handle_resources_list(mock_get_resources, mock_send_result):
    """测试 resources/list 方法处理"""
    server = MCPServer()

    # 设置 mock 返回值
    mock_resources = [{"name": "test_resource", "uri": "test://resource"}]
    mock_get_resources.return_value = mock_resources

    # 调用 resources/list 方法
    request_id = 1
    server._handle_resources_list(request_id)

    # 确认 _get_resources 被调用
    mock_get_resources.assert_called_once()

    # 确认 _send_result 被正确调用
    mock_send_result.assert_called_once_with({"resources": mock_resources}, request_id)


@patch("src.mcp_server.MCPServer._send_result")
@patch("src.mcp_server.MCPServer._get_resource_templates")
def test_handle_resources_templates_list(mock_get_resource_templates, mock_send_result):
    """测试 resources/templates/list 方法处理"""
    server = MCPServer()

    # 设置 mock 返回值
    mock_templates = [{"name": "test_template", "schema": {}}]
    mock_get_resource_templates.return_value = mock_templates

    # 调用 resources/templates/list 方法
    request_id = 1
    server._handle_resources_templates_list(request_id)

    # 确认 _get_resource_templates 被调用
    mock_get_resource_templates.assert_called_once()

    # 确认 _send_result 被正确调用
    mock_send_result.assert_called_once_with({"templates": mock_templates}, request_id)


def test_get_resource_templates():
    """测试 _get_resource_templates 方法"""
    server = MCPServer()

    # 调用 _get_resource_templates 方法
    templates = server._get_resource_templates()

    # 确认返回空列表
    assert templates == []
