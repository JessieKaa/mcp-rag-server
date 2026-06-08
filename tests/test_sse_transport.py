"""
SSE 传输层集成测试

使用 uvicorn 在后台线程中启动真实 HTTP 服务器，避免 starlette TestClient
缓冲 SSE 流导致的阻塞问题。
"""

import queue
import socket
import threading
import time

import pytest
import requests
import uvicorn
from unittest.mock import MagicMock

from src.server import create_sdk_server, create_sse_app
from src.rag_service import RAGService


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class _LiveServer:
    """在后台线程中运行 uvicorn 服务器。"""

    def __init__(self, app, port: int):
        self.port = port
        self._server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
        )
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def start(self, timeout: float = 5.0) -> None:
        self._thread.start()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(0.05)
            try:
                requests.get(f"http://127.0.0.1:{self.port}/sse", timeout=0.5, stream=True).close()
                return
            except requests.exceptions.ConnectionError:
                pass
        raise RuntimeError(f"Server did not start on port {self.port} within {timeout}s")

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=3)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@pytest.fixture(scope="module")
def live_server():
    mock_svc = MagicMock(spec=RAGService)
    mock_svc.get_document_count.return_value = 1
    mock_svc.search.return_value = [
        {
            "file_path": "data/source/test.md",
            "chunk_index": 0,
            "content": "Test content",
            "similarity": 0.9,
            "is_context": False,
            "is_full_document": False,
        }
    ]

    sdk_server = create_sdk_server(
        name="test-sse-server",
        version="0.1.0",
        description="Test SSE",
        rag_service=mock_svc,
    )
    app = create_sse_app(sdk_server)

    port = _find_free_port()
    server = _LiveServer(app, port)
    server.start()
    yield server
    server.stop()


def _collect_sse_events(
    base_url: str,
    url_queue: queue.Queue,
    message_queue: queue.Queue,
    done_event: threading.Event,
) -> None:
    """
    读取 SSE 流。

    - 将 endpoint 事件的 data 放入 url_queue
    - 将 message 事件的 data 放入 message_queue（JSON-RPC 响应）
    - 在 done_event 被设置之前保持连接
    """
    try:
        with requests.get(f"{base_url}/sse", stream=True, timeout=10) as resp:
            current_event = None
            for raw in resp.iter_lines(decode_unicode=True):
                line = raw if isinstance(raw, str) else raw.decode()
                if line.startswith("event:"):
                    current_event = line[6:].strip()
                elif line.startswith("data:"):
                    data = line[5:].strip()
                    if current_event == "endpoint" or current_event is None:
                        url_queue.put(data)
                        current_event = None
                    elif current_event == "message":
                        message_queue.put(data)
                        current_event = None
                elif line == "":
                    current_event = None
                if done_event.is_set():
                    break
            done_event.wait(timeout=10)
    except Exception as exc:
        url_queue.put(exc)


def test_sse_endpoint_returns_endpoint_event(live_server):
    """确认 GET /sse 返回包含 endpoint 事件的 SSE 流。"""
    done = threading.Event()
    url_q: queue.Queue = queue.Queue()
    msg_q: queue.Queue = queue.Queue()

    t = threading.Thread(
        target=_collect_sse_events,
        args=(live_server.base_url, url_q, msg_q, done),
        daemon=True,
    )
    t.start()

    result = url_q.get(timeout=10)
    done.set()

    assert isinstance(result, str), f"Expected endpoint URL, got exception: {result}"
    assert "session_id" in result


def _post_rpc(base_url: str, endpoint_path: str, payload: dict, timeout: int = 5) -> requests.Response:
    return requests.post(f"{base_url}{endpoint_path}", json=payload, timeout=timeout)


def test_sse_round_trip_tools_list(live_server):
    """
    确认完成 MCP 初始化握手后，tools/list 响应
    作为 SSE message 事件到达。
    """
    import json as _json

    done = threading.Event()
    url_q: queue.Queue = queue.Queue()
    msg_q: queue.Queue = queue.Queue()

    t = threading.Thread(
        target=_collect_sse_events,
        args=(live_server.base_url, url_q, msg_q, done),
        daemon=True,
    )
    t.start()

    endpoint_path = url_q.get(timeout=10)
    assert isinstance(endpoint_path, str), f"Expected endpoint URL: {endpoint_path}"

    # 1. initialize handshake
    resp = _post_rpc(live_server.base_url, endpoint_path, {
        "jsonrpc": "2.0",
        "method": "initialize",
        "id": 0,
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "0.0.1"},
        },
    })
    assert resp.status_code == 202

    init_msg = _json.loads(msg_q.get(timeout=10))
    assert init_msg.get("id") == 0
    assert "result" in init_msg

    # 2. notifications/initialized (no response expected)
    _post_rpc(live_server.base_url, endpoint_path, {
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
    })

    # 3. tools/list
    resp = _post_rpc(live_server.base_url, endpoint_path, {
        "jsonrpc": "2.0",
        "method": "tools/list",
        "id": 1,
    })
    assert resp.status_code == 202

    # Assert the JSON-RPC response arrives as a message event on the SSE stream
    raw_message = msg_q.get(timeout=10)
    done.set()

    payload = _json.loads(raw_message)
    assert payload.get("id") == 1
    tool_names = [tool["name"] for tool in payload["result"]["tools"]]
    assert "search" in tool_names
    assert "get_document_count" in tool_names


def test_post_without_session_id_returns_400(live_server):
    """确认没有 session_id 的 POST 返回 400。"""
    resp = requests.post(
        f"{live_server.base_url}/messages/",
        json={"jsonrpc": "2.0", "method": "tools/list", "id": 1},
        timeout=5,
    )
    assert resp.status_code == 400


def test_post_with_unknown_session_returns_404(live_server):
    """确认向不存在的 session_id 发送 POST 返回 404。"""
    resp = requests.post(
        f"{live_server.base_url}/messages/?session_id=00000000000000000000000000000000",
        json={"jsonrpc": "2.0", "method": "tools/list", "id": 1},
        timeout=5,
    )
    assert resp.status_code == 404
