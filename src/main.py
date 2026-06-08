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
from dotenv import load_dotenv

from .mcp_server import MCPServer
from .rag_tools import register_rag_tools, create_rag_service_from_env


def main():
    """
    主函数

    解析命令行参数并启动 MCP 服务器。
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="MCP RAG Server - 符合 Model Context Protocol (MCP) 标准的 RAG 功能 Python 服务器"
    )
    parser.add_argument("--name", default="mcp-rag-server", help="服务器名称")
    parser.add_argument("--version", default="0.1.0", help="服务器版本")
    parser.add_argument("--description", default="MCP RAG Server - 支持多格式文档的 RAG 检索", help="服务器描述")
    parser.add_argument("--module", help="额外的工具模块（例：myapp.tools）")
    args = parser.parse_args()

    # 加载环境变量
    load_dotenv()

    # 创建目录
    os.makedirs("logs", exist_ok=True)
    os.makedirs(os.environ.get("SOURCE_DIR", "data/source"), exist_ok=True)
    os.makedirs(os.environ.get("PROCESSED_DIR", "data/processed"), exist_ok=True)

    # 设置日志
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
        # 创建 MCP 服务器
        server = MCPServer()

        # 创建并注册 RAG 服务
        logger.info("正在初始化 RAG 服务...")
        rag_service = create_rag_service_from_env()
        register_rag_tools(server, rag_service)
        logger.info("RAG 工具已注册")

        # 如果有额外的工具模块则加载
        if args.module:
            try:
                module = importlib.import_module(args.module)
                if hasattr(module, "register_tools"):
                    module.register_tools(server)
                    print(f"已从模块 '{args.module}' 注册工具", file=sys.stderr)
                else:
                    print(f"警告：在模块 '{args.module}' 中未找到 register_tools 函数", file=sys.stderr)
            except ImportError as e:
                print(f"警告：加载模块 '{args.module}' 失败：{str(e)}", file=sys.stderr)

        # 启动 MCP 服务器
        server.start(args.name, args.version, args.description)

    except KeyboardInterrupt:
        print("服务器正在退出。", file=sys.stderr)
        sys.exit(0)

    except Exception as e:
        print(f"发生错误：{str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
