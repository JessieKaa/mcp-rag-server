#!/usr/bin/env python
"""
MCP RAG Server

符合 Model Context Protocol (MCP) 标准的 RAG 功能 Python 服务器
"""

import sys
import os
import argparse
import importlib
import logging
import anyio
from dotenv import load_dotenv

from .rag_tools import create_rag_service_from_env
from .server import create_sdk_server, run_stdio, run_sse


def main():
    """
    主函数

    解析命令行参数并启动 MCP 服务器。
    """
    parser = argparse.ArgumentParser(
        description="MCP RAG Server - 符合 Model Context Protocol (MCP) 标准的 RAG 功能 Python 服务器"
    )
    parser.add_argument("--name", default="mcp-rag-server", help="服务器名称")
    parser.add_argument("--version", default="0.1.0", help="服务器版本")
    parser.add_argument("--description", default="MCP RAG Server - 支持多格式文档的 RAG 检索", help="服务器描述")
    parser.add_argument("--module", help="额外的工具模块（例：myapp.tools）")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio", help="传输方式（默认：stdio）")
    parser.add_argument("--host", default="0.0.0.0", help="SSE 服务器监听地址（默认：0.0.0.0）")
    parser.add_argument("--port", type=int, default=8000, help="SSE 服务器监听端口（默认：8000）")
    args = parser.parse_args()

    load_dotenv()

    os.makedirs("logs", exist_ok=True)
    os.makedirs(os.environ.get("SOURCE_DIR", "data/source"), exist_ok=True)
    os.makedirs(os.environ.get("PROCESSED_DIR", "data/processed"), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(os.path.join("logs", "mcp_rag_server.log"), encoding="utf-8"),
        ],
    )
    logger = logging.getLogger("main")

    try:
        logger.info("正在初始化 RAG 服务...")
        rag_service = create_rag_service_from_env()
        logger.info("RAG 服务已初始化")

        loaded_module = None
        if args.module:
            try:
                loaded_module = importlib.import_module(args.module)
                logger.info(f"已加载模块 '{args.module}'")
            except ImportError as e:
                logger.warning(f"加载模块 '{args.module}' 失败：{str(e)}")

        sdk_server = create_sdk_server(
            name=args.name,
            version=args.version,
            description=args.description,
            rag_service=rag_service,
            extra_module=loaded_module,
        )
        logger.info("SDK 服务器已创建")

        if args.transport == "stdio":
            anyio.run(run_stdio, sdk_server)
        else:
            logger.info(f"SSE 服务器启动中，监听 {args.host}:{args.port}")
            run_sse(sdk_server, args.host, args.port)

    except KeyboardInterrupt:
        print("服务器正在退出。", file=sys.stderr)
        sys.exit(0)

    except Exception as e:
        print(f"发生错误：{str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
