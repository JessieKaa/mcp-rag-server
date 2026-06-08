"""
文档处理模块

负责读取和解析 Markdown、文本、PowerPoint、PDF 等文件，并进行分块处理。
"""

import logging
import os
import json
from pathlib import Path
from typing import List, Dict, Any
import hashlib
import time

import markitdown


class DocumentProcessor:
    """
    文档处理类

    负责读取和解析 Markdown、文本、PowerPoint、PDF 等文件，并进行分块处理。

    Attributes:
        logger: 日志记录器
    """

    # 支持的文件扩展名
    SUPPORTED_EXTENSIONS = {
        "text": [".txt", ".md", ".markdown"],
        "office": [".ppt", ".pptx", ".doc", ".docx"],
        "pdf": [".pdf"],
    }

    def __init__(self):
        """
        DocumentProcessor 的构造函数
        """
        # 设置日志记录器
        self.logger = logging.getLogger("document_processor")
        self.logger.setLevel(logging.INFO)

    def read_file(self, file_path: str) -> str:
        """
        读取文件。

        Args:
            file_path: 文件路径

        Returns:
            文件内容

        Raises:
            FileNotFoundError: 文件不存在时
            IOError: 文件读取失败时
        """
        try:
            # 获取文件扩展名
            ext = Path(file_path).suffix.lower()

            # 文本文件（包含 Markdown）的情况
            if ext in self.SUPPORTED_EXTENSIONS["text"]:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    # 删除 NUL 字符
                    content = content.replace("\x00", "")
                self.logger.info(f"已读取文本文件 '{file_path}'")
                return content

            # PowerPoint、Word、PDF 的情况使用 markitdown 进行转换
            elif ext in self.SUPPORTED_EXTENSIONS["office"] or ext in self.SUPPORTED_EXTENSIONS["pdf"]:
                return self.convert_to_markdown(file_path)

            # 不支持的扩展名的情况
            else:
                self.logger.warning(f"不支持的文件格式：{file_path}")
                return ""

        except FileNotFoundError:
            self.logger.error(f"未找到文件 '{file_path}'")
            raise
        except IOError as e:
            self.logger.error(f"读取文件 '{file_path}' 失败：{str(e)}")
            raise

    def convert_to_markdown(self, file_path: str) -> str:
        """
        将 PowerPoint、Word、PDF 等文件转换为 Markdown。

        Args:
            file_path: 文件路径

        Returns:
            转换后的 Markdown 内容

        Raises:
            Exception: 转换失败时
        """
        try:
            # 创建文件 URI
            file_uri = f"file://{os.path.abspath(file_path)}"

            # 使用 markitdown 进行转换
            markdown_content = markitdown.MarkItDown().convert_uri(file_uri).markdown
            # 删除 NUL 字符
            markdown_content = markdown_content.replace("\x00", "")

            self.logger.info(f"已将文件 '{file_path}' 转换为 Markdown")
            return markdown_content
        except Exception as e:
            self.logger.error(f"将文件 '{file_path}' 转换为 Markdown 失败：{str(e)}")
            raise

    def split_into_chunks(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """
        将文本分割为分块。

        Args:
            text: 要分割的文本
            chunk_size: 分块大小（字符数）
            overlap: 分块间的重叠量（字符数）

        Returns:
            分块列表
        """
        if not text:
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + chunk_size, text_length)

            # 调整以避免在句子中间切断
            if end < text_length:
                # 查找下一个换行符或句号
                next_newline = text.find("\n", end)
                next_period = text.find("。", end)

                if next_newline != -1 and (next_period == -1 or next_newline < next_period):
                    end = next_newline + 1  # 包含换行符
                elif next_period != -1:
                    end = next_period + 1  # 包含句号

            chunks.append(text[start:end])
            start = end - overlap if end - overlap > start else end

            # 终止条件
            if start >= text_length:
                break

        self.logger.info(f"已将文本分割为 {len(chunks)} 个分块")
        return chunks

    def calculate_file_hash(self, file_path: str) -> str:
        """
        计算文件的哈希值。

        Args:
            file_path: 文件路径

        Returns:
            文件的 SHA-256 哈希值
        """
        try:
            with open(file_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            return file_hash
        except Exception as e:
            self.logger.error(f"计算文件 '{file_path}' 的哈希值失败：{str(e)}")
            # 发生错误时使用时间戳作为哈希值
            return f"timestamp-{int(time.time())}"

    def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        获取文件的元数据。

        Args:
            file_path: 文件路径

        Returns:
            文件的元数据（哈希值、最后修改时间等）
        """
        file_stat = os.stat(file_path)
        return {
            "hash": self.calculate_file_hash(file_path),
            "mtime": file_stat.st_mtime,
            "size": file_stat.st_size,
            "path": file_path,
        }

    def load_file_registry(self, processed_dir: str) -> Dict[str, Dict[str, Any]]:
        """
        加载已处理文件的注册表。

        Args:
            processed_dir: 保存已处理文件的目录路径

        Returns:
            已处理文件的注册表（以文件路径为键的元数据字典）
        """
        registry_path = Path(processed_dir) / "file_registry.json"
        if not registry_path.exists():
            return {}

        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"加载文件注册表失败：{str(e)}")
            return {}

    def save_file_registry(self, processed_dir: str, registry: Dict[str, Dict[str, Any]]) -> None:
        """
        保存已处理文件的注册表。

        Args:
            processed_dir: 保存已处理文件的目录路径
            registry: 已处理文件的注册表
        """
        registry_path = Path(processed_dir) / "file_registry.json"
        try:
            # 如果已处理目录不存在则创建
            os.makedirs(Path(processed_dir), exist_ok=True)

            with open(registry_path, "w", encoding="utf-8") as f:
                json.dump(registry, f, ensure_ascii=False, indent=2)
            self.logger.info(f"已保存文件注册表：{registry_path}")
        except Exception as e:
            self.logger.error(f"保存文件注册表失败：{str(e)}")

    def process_file(
        self, file_path: str, processed_dir: str, chunk_size: int = 500, overlap: int = 100
    ) -> List[Dict[str, Any]]:
        """
        处理文件。

        Args:
            file_path: 文件路径
            processed_dir: 保存已处理文件的目录路径
            chunk_size: 分块大小（字符数）
            overlap: 分块间的重叠量（字符数）

        Returns:
            处理结果列表（每个元素是包含分块信息的字典）
        """
        try:
            # 读取文件
            content = self.read_file(file_path)
            if not content:
                return []

            # 从文件路径获取目录结构
            file_path_obj = Path(file_path)
            relative_path = file_path_obj.relative_to(Path(file_path_obj.parts[0]) / Path(file_path_obj.parts[1]))
            parent_dirs = relative_path.parent.parts

            # 使用目录名作为后缀
            dir_suffix = "_".join(parent_dirs) if parent_dirs else ""

            # 生成已处理文件名
            processed_file_name = f"{file_path_obj.stem}{('_' + dir_suffix) if dir_suffix else ''}.md"
            processed_file_path = Path(processed_dir) / processed_file_name

            # 如果已处理目录不存在则创建
            os.makedirs(Path(processed_dir), exist_ok=True)

            # 写入已处理文件
            with open(processed_file_path, "w", encoding="utf-8") as f:
                f.write(content)

            self.logger.info(f"已保存已处理文件：{processed_file_path}")

            # 分割为分块
            chunks = self.split_into_chunks(content, chunk_size, overlap)

            # 创建结果
            results = []
            for i, chunk in enumerate(chunks):
                document_id = f"{processed_file_name}_{i}"
                results.append(
                    {
                        "document_id": document_id,
                        "content": chunk,
                        "file_path": str(processed_file_path),
                        "original_file_path": file_path,
                        "chunk_index": i,
                        "metadata": {
                            "file_name": file_path_obj.name,
                            "directory": str(file_path_obj.parent),
                            "directory_suffix": dir_suffix,
                        },
                    }
                )

            self.logger.info(f"已处理文件 '{file_path}'（{len(results)} 个分块）")
            return results

        except Exception as e:
            self.logger.error(f"处理文件 '{file_path}' 时发生错误：{str(e)}")
            raise

    def process_directory(
        self, source_dir: str, processed_dir: str, chunk_size: int = 500, overlap: int = 100, incremental: bool = False
    ) -> List[Dict[str, Any]]:
        """
        处理目录内的文件。

        Args:
            source_dir: 包含原始文件的目录路径
            processed_dir: 保存已处理文件的目录路径
            chunk_size: 分块大小（字符数）
            overlap: 分块间的重叠量（字符数）
            incremental: 是否仅处理增量

        Returns:
            处理结果列表（每个元素是包含分块信息的字典）
        """
        results = []
        source_directory = Path(source_dir)

        if not source_directory.exists() or not source_directory.is_dir():
            self.logger.error(f"未找到目录 '{source_dir}' 或不是目录")
            raise FileNotFoundError(f"未找到目录 '{source_dir}' 或不是目录")

        # 获取所有支持的文件扩展名
        all_extensions = []
        for ext_list in self.SUPPORTED_EXTENSIONS.values():
            all_extensions.extend(ext_list)

        # 搜索文件
        files = []
        for ext in all_extensions:
            files.extend(list(source_directory.glob(f"**/*{ext}")))

        self.logger.info(f"在目录 '{source_dir}' 内找到 {len(files)} 个文件")

        # 增量处理时，加载文件注册表
        if incremental:
            file_registry = self.load_file_registry(processed_dir)
            self.logger.info(f"从文件注册表读取了 {len(file_registry)} 条文件信息")
        else:
            file_registry = {}

        # 确定要处理的文件
        files_to_process = []
        for file_path in files:
            str_path = str(file_path)
            if incremental:
                # 获取文件元数据
                current_metadata = self.get_file_metadata(str_path)

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
            else:
                # 非增量处理时处理所有文件
                files_to_process.append(file_path)
                # 更新注册表
                file_registry[str_path] = self.get_file_metadata(str_path)

        self.logger.info(f"待处理文件数：{len(files_to_process)} / {len(files)}")

        # 处理各个文件
        for file_path in files_to_process:
            try:
                file_results = self.process_file(str(file_path), processed_dir, chunk_size, overlap)
                results.extend(file_results)
            except Exception as e:
                self.logger.error(f"处理文件 '{file_path}' 时发生错误：{str(e)}")
                # 发生错误仍继续处理
                continue

        # 保存文件注册表
        self.save_file_registry(processed_dir, file_registry)

        self.logger.info(f"已处理目录 '{source_dir}' 内的文件（共 {len(results)} 个分块）")
        return results
