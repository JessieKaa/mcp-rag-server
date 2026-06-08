"""
RAG 服务模块

集成文档处理、嵌入向量生成和向量数据库，提供索引化和检索功能。
"""

import os
import time
import logging
from typing import List, Dict, Any

from .document_processor import DocumentProcessor
from .embedding_generator import EmbeddingGenerator
from .vector_database import VectorDatabase


class RAGService:
    """
    RAG 服务类

    集成文档处理、嵌入向量生成和向量数据库，提供索引化和检索功能。

    Attributes:
        document_processor: 文档处理类实例
        embedding_generator: 嵌入向量生成类实例
        vector_database: 向量数据库类实例
        logger: 日志记录器
    """

    def __init__(
        self, document_processor: DocumentProcessor, embedding_generator: EmbeddingGenerator, vector_database: VectorDatabase
    ):
        """
        RAGService 的构造函数

        Args:
            document_processor: 文档处理类实例
            embedding_generator: 嵌入向量生成类实例
            vector_database: 向量数据库类实例
        """
        # 设置日志记录器
        self.logger = logging.getLogger("rag_service")
        self.logger.setLevel(logging.INFO)

        # 设置组件
        self.document_processor = document_processor
        self.embedding_generator = embedding_generator
        self.vector_database = vector_database

        # 初始化数据库
        try:
            self.vector_database.initialize_database()
        except Exception as e:
            self.logger.error(f"数据库初始化失败：{str(e)}")
            raise

    def index_documents(
        self,
        source_dir: str,
        processed_dir: str = None,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        incremental: bool = False,
    ) -> Dict[str, Any]:
        """
        为目录内的文件建立索引。

        Args:
            source_dir: 包含要索引文件的目录路径
            processed_dir: 保存已处理文件的目录路径（未指定时为 data/processed）
            chunk_size: 分块大小（字符数）
            chunk_overlap: 分块间的重叠量（字符数）
            incremental: 是否仅进行增量索引

        Returns:
            索引化结果
                - document_count: 已索引的文档数
                - processing_time: 处理时间（秒）
                - success: 是否成功
                - error: 错误消息（发生错误时）
        """
        start_time = time.time()
        document_count = 0

        # 已处理目录的默认值
        if processed_dir is None:
            processed_dir = "data/processed"

        try:
            # 处理目录内的文件
            if incremental:
                self.logger.info(f"正在为目录 '{source_dir}' 内的增量文件建立索引...")
            else:
                self.logger.info(f"正在为目录 '{source_dir}' 内的文件建立索引...")

            chunks = self.document_processor.process_directory(
                source_dir, processed_dir, chunk_size, chunk_overlap, incremental
            )

            if not chunks:
                self.logger.warning(f"在目录 '{source_dir}' 内未找到可处理的文件")
                return {
                    "document_count": 0,
                    "processing_time": time.time() - start_time,
                    "success": True,
                    "message": f"在目录 '{source_dir}' 内未找到可处理的文件",
                }

            # 从分块内容生成嵌入向量
            self.logger.info(f"正在为 {len(chunks)} 个分块生成嵌入向量...")
            texts = [chunk["content"] for chunk in chunks]
            embeddings = self.embedding_generator.generate_embeddings(texts)

            # 将文档插入数据库
            self.logger.info(f"正在将 {len(chunks)} 个分块插入数据库...")
            documents = []
            for i, chunk in enumerate(chunks):
                documents.append(
                    {
                        "document_id": chunk["document_id"],
                        "content": chunk["content"],
                        "file_path": chunk["file_path"],
                        "chunk_index": chunk["chunk_index"],
                        "embedding": embeddings[i],
                        "metadata": {
                            "file_name": os.path.basename(chunk["file_path"]),
                            "directory": os.path.dirname(chunk["file_path"]),
                            "original_file_path": chunk.get("original_file_path", ""),
                            "directory_suffix": chunk.get("metadata", {}).get("directory_suffix", ""),
                        },
                    }
                )

            self.vector_database.batch_insert_documents(documents)
            document_count = len(documents)

            processing_time = time.time() - start_time
            self.logger.info(f"索引化完成（{document_count} 个文档，{processing_time:.2f} 秒）")

            return {
                "document_count": document_count,
                "processing_time": processing_time,
                "success": True,
                "message": f"已为 {document_count} 个文档建立索引",
            }

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"索引化过程中发生错误：{str(e)}")

            return {"document_count": document_count, "processing_time": processing_time, "success": False, "error": str(e)}

    def search(
        self, query: str, limit: int = 5, with_context: bool = False, context_size: int = 1, full_document: bool = False
    ) -> List[Dict[str, Any]]:
        """
        进行向量检索。

        Args:
            query: 检索查询
            limit: 返回结果的数量（默认：5）
            with_context: 是否获取前后分块（默认：False）
            context_size: 获取前后分块的数量（默认：1）
            full_document: 是否获取文档全文（默认：False）

        Returns:
            检索结果列表（按相关度排序）
                - document_id: 文档 ID
                - content: 内容
                - file_path: 文件路径
                - similarity: 相似度
                - metadata: 元数据
                - is_context: 是否为上下文分块（前后分块时为 True）
                - is_full_document: 是否为全文文档（文档全文时为 True）
        """
        try:
            # 从查询生成嵌入向量
            self.logger.info(f"正在为查询 '{query}' 生成嵌入向量...")
            query_embedding = self.embedding_generator.generate_search_embedding(query)

            # 向量检索
            self.logger.info(f"正在执行查询 '{query}' 的向量检索...")
            results = self.vector_database.search(query_embedding, limit)

            # 获取前后分块的情况
            if with_context and context_size > 0:
                context_results = []
                processed_files = set()  # 记录已处理的文件和分块组合

                for result in results:
                    file_path = result["file_path"]
                    chunk_index = result["chunk_index"]
                    file_chunk_key = f"{file_path}_{chunk_index}"

                    # 跳过已处理的文件和分块组合
                    if file_chunk_key in processed_files:
                        continue

                    processed_files.add(file_chunk_key)

                    # 获取前后分块
                    adjacent_chunks = self.vector_database.get_adjacent_chunks(file_path, chunk_index, context_size)
                    context_results.extend(adjacent_chunks)

                # 合并结果
                all_results = results.copy()

                # 为避免重复，记录已包含的文档 ID
                existing_doc_ids = {result["document_id"] for result in all_results}

                # 仅添加不重复的上下文分块
                for context in context_results:
                    if context["document_id"] not in existing_doc_ids:
                        all_results.append(context)
                        existing_doc_ids.add(context["document_id"])

                # 按文件路径和分块索引排序
                all_results.sort(key=lambda x: (x["file_path"], x["chunk_index"]))

                self.logger.info(f"检索结果（含上下文）：{len(all_results)} 条")

                # 获取文档全文的情况
                if full_document:
                    full_doc_results = []
                    processed_files = set()  # 记录已处理的文件

                    # 获取检索结果中包含的文件全文
                    for result in all_results:
                        file_path = result["file_path"]

                        # 跳过已处理的文件
                        if file_path in processed_files:
                            continue

                        processed_files.add(file_path)

                        # 获取文件全文
                        full_doc_chunks = self.vector_database.get_document_by_file_path(file_path)
                        full_doc_results.extend(full_doc_chunks)

                    # 合并结果
                    merged_results = all_results.copy()

                    # 为避免重复，记录已包含的文档 ID
                    existing_doc_ids = {result["document_id"] for result in merged_results}

                    # 仅添加不重复的全文分块
                    for doc_chunk in full_doc_results:
                        if doc_chunk["document_id"] not in existing_doc_ids:
                            merged_results.append(doc_chunk)
                            existing_doc_ids.add(doc_chunk["document_id"])

                    # 按文件路径和分块索引排序
                    merged_results.sort(key=lambda x: (x["file_path"], x["chunk_index"]))

                    self.logger.info(f"检索结果（含全文）：{len(merged_results)} 条")
                    return merged_results
                else:
                    return all_results
            else:
                # 获取文档全文的情况
                if full_document:
                    full_doc_results = []
                    processed_files = set()  # 记录已处理的文件

                    # 获取检索结果中包含的文件全文
                    for result in results:
                        file_path = result["file_path"]

                        # 跳过已处理的文件
                        if file_path in processed_files:
                            continue

                        processed_files.add(file_path)

                        # 获取文件全文
                        full_doc_chunks = self.vector_database.get_document_by_file_path(file_path)
                        full_doc_results.extend(full_doc_chunks)

                    # 合并结果
                    merged_results = results.copy()

                    # 为避免重复，记录已包含的文档 ID
                    existing_doc_ids = {result["document_id"] for result in merged_results}

                    # 仅添加不重复的全文分块
                    for doc_chunk in full_doc_results:
                        if doc_chunk["document_id"] not in existing_doc_ids:
                            merged_results.append(doc_chunk)
                            existing_doc_ids.add(doc_chunk["document_id"])

                    # 按文件路径和分块索引排序
                    merged_results.sort(key=lambda x: (x["file_path"], x["chunk_index"]))

                    self.logger.info(f"检索结果（含全文）：{len(merged_results)} 条")
                    return merged_results
                else:
                    self.logger.info(f"检索结果：{len(results)} 条")
                    return results

        except Exception as e:
            self.logger.error(f"检索过程中发生错误：{str(e)}")
            raise

    def clear_index(self) -> Dict[str, Any]:
        """
        清除索引。

        Returns:
            清除结果
                - deleted_count: 已删除的文档数
                - success: 是否成功
                - error: 错误消息（发生错误时）
        """
        try:
            # 清除数据库
            self.logger.info("正在清除索引...")
            deleted_count = self.vector_database.clear_database()

            self.logger.info(f"索引已清除（删除了 {deleted_count} 个文档）")
            return {"deleted_count": deleted_count, "success": True, "message": f"已删除 {deleted_count} 个文档"}

        except Exception as e:
            self.logger.error(f"清除索引时发生错误：{str(e)}")

            return {"deleted_count": 0, "success": False, "error": str(e)}

    def get_document_count(self) -> int:
        """
        获取索引中的文档数量。

        Returns:
            文档数量
        """
        try:
            # 获取文档数量
            count = self.vector_database.get_document_count()
            self.logger.info(f"索引中的文档数量：{count}")
            return count

        except Exception as e:
            self.logger.error(f"获取文档数量时发生错误：{str(e)}")
            raise
