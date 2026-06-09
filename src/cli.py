#!/usr/bin/env python
"""
MCP RAG Server CLI

用于清除索引和建立索引的命令行界面
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

from .rag_tools import create_rag_service_from_env


def setup_logging():
    """
    设置日志
    """
    # 创建日志目录
    os.makedirs("logs", exist_ok=True)

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join("logs", "mcp_rag_cli.log"), encoding="utf-8"),
        ],
    )
    return logging.getLogger("cli")


def clear_index():
    """
    清除索引
    """
    logger = setup_logging()
    logger.info("正在清除索引...")

    # 加载环境变量
    load_dotenv()

    # 创建 RAG 服务
    rag_service = create_rag_service_from_env()

    # 已处理目录的路径
    processed_dir = os.environ.get("PROCESSED_DIR", "data/processed")

    # 删除文件注册表
    registry_path = Path(processed_dir) / "file_registry.json"
    if registry_path.exists():
        try:
            registry_path.unlink()
            logger.info(f"已删除文件注册表：{registry_path}")
            print(f"已删除文件注册表：{registry_path}")
        except Exception as e:
            logger.error(f"删除文件注册表失败：{str(e)}")
            print(f"删除文件注册表失败：{str(e)}")

    # 清除索引
    result = rag_service.clear_index()

    if result["success"]:
        logger.info(f"索引已清除（删除了 {result['deleted_count']} 个文档）")
        print(f"索引已清除（删除了 {result['deleted_count']} 个文档）")
    else:
        logger.error(f"清除索引失败：{result.get('error', '未知错误')}")
        print(f"清除索引失败：{result.get('error', '未知错误')}")
        sys.exit(1)


def index_documents(directory_path, chunk_size=500, chunk_overlap=100, incremental=False):
    """
    为文档建立索引

    Args:
        directory_path: 包含要索引文档的目录路径
        chunk_size: 分块大小（字符数）
        chunk_overlap: 分块间的重叠量（字符数）
        incremental: 是否仅进行增量索引
    """
    logger = setup_logging()
    if incremental:
        logger.info(f"正在为目录 '{directory_path}' 内的增量文件建立索引...")
    else:
        logger.info(f"正在为目录 '{directory_path}' 内的文档建立索引...")

    # 加载环境变量
    load_dotenv()

    # 确认目录存在
    if not os.path.exists(directory_path):
        logger.error(f"未找到目录 '{directory_path}'")
        print(f"错误：未找到目录 '{directory_path}'")
        sys.exit(1)

    if not os.path.isdir(directory_path):
        logger.error(f"'{directory_path}' 不是目录")
        print(f"错误：'{directory_path}' 不是目录")
        sys.exit(1)

    # 创建 RAG 服务
    rag_service = create_rag_service_from_env()

    # 已处理目录的路径
    processed_dir = os.environ.get("PROCESSED_DIR", "data/processed")

    # 执行索引化
    if incremental:
        print(f"正在为目录 '{directory_path}' 内的增量文件建立索引...")
    else:
        print(f"正在为目录 '{directory_path}' 内的文档建立索引...")

    # 用于显示进度的计数器
    processed_files = 0

    # 处理前获取文件数量
    total_files = 0
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file_path)[1].lower()
            if ext in [".md", ".markdown", ".txt", ".pdf", ".ppt", ".pptx", ".doc", ".docx"]:
                total_files += 1

    print(f"共找到 {total_files} 个文件...")

    # 在调用原始 RAGService 的 index_documents 方法之前，
    # 覆盖 DocumentProcessor 的 process_directory 方法以显示进度
    original_process_directory = rag_service.document_processor.process_directory

    def process_directory_with_progress(source_dir, processed_dir, chunk_size=500, overlap=100, incremental=False):
        nonlocal processed_files
        results = []
        source_directory = Path(source_dir)

        if not source_directory.exists() or not source_directory.is_dir():
            logger.error(f"未找到目录 '{source_dir}' 或不是目录")
            raise FileNotFoundError(f"未找到目录 '{source_dir}' 或不是目录")

        # 获取所有支持的文件扩展名
        all_extensions = []
        for ext_list in rag_service.document_processor.SUPPORTED_EXTENSIONS.values():
            all_extensions.extend(ext_list)

        # 搜索文件
        files = []
        for ext in all_extensions:
            files.extend(list(source_directory.glob(f"**/*{ext}")))

        logger.info(f"在目录 '{source_dir}' 内找到 {len(files)} 个文件")

        # 增量处理时，加载文件注册表
        if incremental:
            file_registry = rag_service.document_processor.load_file_registry(processed_dir)
            logger.info(f"从文件注册表读取了 {len(file_registry)} 条文件信息")

            # 确定要处理的文件
            files_to_process = []
            for file_path in files:
                str_path = str(file_path)
                # 获取文件元数据
                current_metadata = rag_service.document_processor.get_file_metadata(str_path)

                # 仅当注册表中不存在或哈希值已更改时才处理
                if (
                    str_path not in file_registry
                    or file_registry[str_path]["hash"] != current_metadata["hash"]
                    or file_registry[str_path]["mtime"] != current_metadata["mtime"]
                    or file_registry[str_path]["size"] != current_metadata["size"]
                ):
                    files_to_process.append(file_path)
                    # 更新注册表
                    file_registry[str_path] = current_metadata

            print(f"待处理文件数：{len(files_to_process)} / {len(files)}")

            # 处理各个文件
            for i, file_path in enumerate(files_to_process):
                try:
                    file_results = rag_service.document_processor.process_file(
                        str(file_path), processed_dir, chunk_size, overlap
                    )
                    results.extend(file_results)
                    processed_files += 1
                    print(
                        f"处理中... {processed_files}/{len(files_to_process)} 个文件 ({(processed_files / len(files_to_process) * 100):.1f}%)：{file_path}"
                    )
                except Exception as e:
                    logger.error(f"处理文件 '{file_path}' 时发生错误：{str(e)}")
                    # 发生错误仍继续处理
                    continue

            # 保存文件注册表
            rag_service.document_processor.save_file_registry(processed_dir, file_registry)
        else:
            # 非增量处理时处理所有文件
            for i, file_path in enumerate(files):
                try:
                    file_results = rag_service.document_processor.process_file(
                        str(file_path), processed_dir, chunk_size, overlap
                    )
                    results.extend(file_results)
                    processed_files += 1
                    print(
                        f"处理中... {processed_files}/{total_files} 个文件 ({(processed_files / total_files * 100):.1f}%)：{file_path}"
                    )
                except Exception as e:
                    logger.error(f"处理文件 '{file_path}' 时发生错误：{str(e)}")
                    # 发生错误仍继续处理
                    continue

            # 全文件处理时也创建并保存新注册表
            file_registry = {}
            for file_path in files:
                str_path = str(file_path)
                file_registry[str_path] = rag_service.document_processor.get_file_metadata(str_path)
            rag_service.document_processor.save_file_registry(processed_dir, file_registry)

        logger.info(f"已处理目录 '{source_dir}' 内的文件（共 {len(results)} 个分块）")
        return results

    # 替换为带进度显示的处理方法
    rag_service.document_processor.process_directory = process_directory_with_progress

    # 执行索引化
    result = rag_service.index_documents(directory_path, processed_dir, chunk_size, chunk_overlap, incremental)

    # 恢复原始方法
    rag_service.document_processor.process_directory = original_process_directory

    if result["success"]:
        incremental_text = "增量" if incremental else "全部"
        logger.info(
            f"索引化完成（处理了{incremental_text}文件，{result['document_count']} 个文档，{result['processing_time']:.2f} 秒）"
        )
        print(
            f"索引化完成（处理了{incremental_text}文件）\n"
            f"- 文档数：{result['document_count']}\n"
            f"- 处理时间：{result['processing_time']:.2f} 秒\n"
            f"- 消息：{result.get('message', '')}"
        )
    else:
        logger.error(f"索引化失败：{result.get('error', '未知错误')}")
        print(f"索引化失败\n- 错误：{result.get('error', '未知错误')}\n- 处理时间：{result['processing_time']:.2f} 秒")
        sys.exit(1)


def get_document_count():
    """
    获取索引中的文档数量
    """
    logger = setup_logging()
    logger.info("正在获取索引中的文档数量...")

    # 加载环境变量
    load_dotenv()

    # 创建 RAG 服务
    rag_service = create_rag_service_from_env()

    # 获取文档数量
    try:
        count = rag_service.get_document_count()
        logger.info(f"索引中的文档数量：{count}")
        print(f"索引中的文档数量：{count}")
    except Exception as e:
        logger.error(f"获取文档数量时发生错误：{str(e)}")
        print(f"获取文档数量时发生错误：{str(e)}")
        sys.exit(1)


def main():
    """
    主函数

    解析命令行参数并执行相应的处理。
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="MCP RAG Server CLI - 用于清除索引和建立索引的命令行界面")
    subparsers = parser.add_subparsers(dest="command", help="要执行的命令")

    # clear 命令
    subparsers.add_parser("clear", help="清除索引")

    # index 命令
    index_parser = subparsers.add_parser("index", help="为文档建立索引")
    index_parser.add_argument(
        "--directory",
        "-d",
        default=os.environ.get("SOURCE_DIR", "./data/source"),
        help="包含要索引文档的目录路径",
    )
    index_parser.add_argument("--chunk-size", "-s", type=int, default=500, help="分块大小（字符数）")
    index_parser.add_argument("--chunk-overlap", "-o", type=int, default=100, help="分块间的重叠量（字符数）")
    index_parser.add_argument("--incremental", "-i", action="store_true", help="仅进行增量索引")

    # count 命令
    subparsers.add_parser("count", help="获取索引中的文档数量")

    args = parser.parse_args()

    # 根据命令执行相应的处理
    if args.command == "clear":
        clear_index()
    elif args.command == "index":
        index_documents(args.directory, args.chunk_size, args.chunk_overlap, args.incremental)
    elif args.command == "count":
        get_document_count()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
