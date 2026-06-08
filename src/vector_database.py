"""
向量数据库模块

使用 PostgreSQL 和 pgvector 进行向量的保存和检索。
"""

import logging
import psycopg2
import json
import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# 加载 .env
load_dotenv()
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))


class VectorDatabase:
    """
    向量数据库类

    使用 PostgreSQL 和 pgvector 进行向量的保存和检索。

    Attributes:
        connection_params: 连接参数
        connection: 数据库连接
        logger: 日志记录器
    """

    def __init__(self, connection_params: Dict[str, Any]):
        """
        VectorDatabase 的构造函数

        Args:
            connection_params: 连接参数
                - host: 主机名
                - port: 端口号
                - user: 用户名
                - password: 密码
                - database: 数据库名
        """
        # 设置日志记录器
        self.logger = logging.getLogger("vector_database")
        self.logger.setLevel(logging.INFO)

        # 保存连接参数
        self.connection_params = connection_params
        self.connection = None

    def connect(self) -> None:
        """
        连接数据库。

        Raises:
            Exception: 连接失败时
        """
        try:
            self.connection = psycopg2.connect(**self.connection_params)
            self.logger.info("已连接数据库")
        except Exception as e:
            self.logger.error(f"连接数据库失败：{str(e)}")
            raise

    def disconnect(self) -> None:
        """
        断开数据库连接。
        """
        if self.connection:
            self.connection.close()
            self.connection = None
            self.logger.info("已断开数据库连接")

    def initialize_database(self) -> None:
        """
        初始化数据库。

        创建表和索引。

        Raises:
            Exception: 初始化失败时
        """
        try:
            # 如果没有连接则连接
            if not self.connection:
                self.connect()

            # 创建游标
            cursor = self.connection.cursor()

            # 启用 pgvector 扩展
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # 创建文档表
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    document_id TEXT UNIQUE NOT NULL,
                    content TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    metadata JSONB,
                    embedding vector({EMBEDDING_DIM}),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 创建索引
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_document_id ON documents (document_id);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_file_path ON documents (file_path);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents USING ivfflat (embedding vector_cosine_ops);
            """)

            # 提交
            self.connection.commit()
            self.logger.info("数据库已初始化")

        except Exception as e:
            # 回滚
            if self.connection:
                self.connection.rollback()
            self.logger.error(f"数据库初始化失败：{str(e)}")
            raise

        finally:
            # 关闭游标
            if "cursor" in locals() and cursor:
                cursor.close()

    def insert_document(
        self,
        document_id: str,
        content: str,
        file_path: str,
        chunk_index: int,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        插入文档。

        Args:
            document_id: 文档 ID
            content: 文档内容
            file_path: 文件路径
            chunk_index: 分块索引
            embedding: 嵌入向量
            metadata: 元数据（可选）

        Raises:
            Exception: 插入失败时
        """
        try:
            # 如果没有连接则连接
            if not self.connection:
                self.connect()

            # 创建游标
            cursor = self.connection.cursor()

            # 将元数据转换为 JSON 格式
            metadata_json = json.dumps(metadata) if metadata else None

            # 插入文档
            cursor.execute(
                """
                INSERT INTO documents (document_id, content, file_path, chunk_index, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (document_id)
                DO UPDATE SET
                    content = EXCLUDED.content,
                    file_path = EXCLUDED.file_path,
                    chunk_index = EXCLUDED.chunk_index,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    created_at = CURRENT_TIMESTAMP;
            """,
                (document_id, content, file_path, chunk_index, embedding, metadata_json),
            )

            # 提交
            self.connection.commit()
            self.logger.debug(f"已插入文档 '{document_id}'")

        except Exception as e:
            # 回滚
            if self.connection:
                self.connection.rollback()
            self.logger.error(f"插入文档失败：{str(e)}")
            raise

        finally:
            # 关闭游标
            if "cursor" in locals() and cursor:
                cursor.close()

    def batch_insert_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        批量插入多个文档。

        Args:
            documents: 文档列表
                每个文档是包含以下键的字典：
                - document_id: 文档 ID
                - content: 文档内容
                - file_path: 文件路径
                - chunk_index: 分块索引
                - embedding: 嵌入向量
                - metadata: 元数据（可选）

        Raises:
            Exception: 插入失败时
        """
        if not documents:
            self.logger.warning("没有要插入的文档")
            return

        try:
            # 如果没有连接则连接
            if not self.connection:
                self.connect()

            # 创建游标
            cursor = self.connection.cursor()

            # 创建批量插入数据
            values = []
            for doc in documents:
                metadata_json = json.dumps(doc.get("metadata")) if doc.get("metadata") else None
                values.append(
                    (doc["document_id"], doc["content"], doc["file_path"], doc["chunk_index"], doc["embedding"], metadata_json)
                )

            # 批量插入
            cursor.executemany(
                """
                INSERT INTO documents (document_id, content, file_path, chunk_index, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (document_id)
                DO UPDATE SET
                    content = EXCLUDED.content,
                    file_path = EXCLUDED.file_path,
                    chunk_index = EXCLUDED.chunk_index,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    created_at = CURRENT_TIMESTAMP;
            """,
                values,
            )

            # 提交
            self.connection.commit()
            self.logger.info(f"已插入 {len(documents)} 个文档")

        except Exception as e:
            # 回滚
            if self.connection:
                self.connection.rollback()
            self.logger.error(f"批量插入文档失败：{str(e)}")
            raise

        finally:
            # 关闭游标
            if "cursor" in locals() and cursor:
                cursor.close()

    def search(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """
        进行向量检索。

        Args:
            query_embedding: 查询的嵌入向量
            limit: 返回结果的数量（默认：5）

        Returns:
            检索结果列表（按相关度排序）

        Raises:
            Exception: 检索失败时
        """
        try:
            # 如果没有连接则连接
            if not self.connection:
                self.connect()

            # 创建游标
            cursor = self.connection.cursor()

            # 将查询嵌入向量转换为 PostgreSQL 数组语法
            embedding_str = str(query_embedding)
            embedding_array = f"ARRAY{embedding_str}::vector"

            # 向量检索
            cursor.execute(
                f"""
                SELECT
                    document_id,
                    content,
                    file_path,
                    chunk_index,
                    metadata,
                    1 - (embedding <=> {embedding_array}) AS similarity
                FROM
                    documents
                WHERE
                    embedding IS NOT NULL
                ORDER BY
                    embedding <=> {embedding_array}
                LIMIT %s;
                """,
                (limit,),
            )

            # 获取结果
            results = []
            for row in cursor.fetchall():
                document_id, content, file_path, chunk_index, metadata_json, similarity = row

                # 从 JSON 解码元数据
                if metadata_json:
                    if isinstance(metadata_json, str):
                        try:
                            metadata = json.loads(metadata_json)
                        except json.JSONDecodeError:
                            metadata = {}
                    else:
                        # 如果已经是字典类型则直接使用
                        metadata = metadata_json
                else:
                    metadata = {}

                results.append(
                    {
                        "document_id": document_id,
                        "content": content,
                        "file_path": file_path,
                        "chunk_index": chunk_index,
                        "metadata": metadata,
                        "similarity": similarity,
                    }
                )

            self.logger.info(f"查询找到 {len(results)} 条结果")
            return results

        except Exception as e:
            self.logger.error(f"向量检索过程中发生错误：{str(e)}")
            raise

        finally:
            # 关闭游标
            if "cursor" in locals() and cursor:
                cursor.close()

    def delete_document(self, document_id: str) -> bool:
        """
        删除文档。

        Args:
            document_id: 要删除的文档 ID

        Returns:
            删除成功时返回 True，未找到文档时返回 False

        Raises:
            Exception: 删除失败时
        """
        try:
            # 如果没有连接则连接
            if not self.connection:
                self.connect()

            # 创建游标
            cursor = self.connection.cursor()

            # 删除文档
            cursor.execute("DELETE FROM documents WHERE document_id = %s;", (document_id,))

            # 获取删除的行数
            deleted_rows = cursor.rowcount

            # 提交
            self.connection.commit()

            if deleted_rows > 0:
                self.logger.info(f"已删除文档 '{document_id}'")
                return True
            else:
                self.logger.warning(f"未找到文档 '{document_id}'")
                return False

        except Exception as e:
            # 回滚
            if self.connection:
                self.connection.rollback()
            self.logger.error(f"删除文档时发生错误：{str(e)}")
            raise

        finally:
            # 关闭游标
            if "cursor" in locals() and cursor:
                cursor.close()

    def delete_by_file_path(self, file_path: str) -> int:
        """
        根据文件路径删除文档。

        Args:
            file_path: 要删除文档的文件路径

        Returns:
            删除的文档数量

        Raises:
            Exception: 删除失败时
        """
        try:
            # 如果没有连接则连接
            if not self.connection:
                self.connect()

            # 创建游标
            cursor = self.connection.cursor()

            # 删除文档
            cursor.execute("DELETE FROM documents WHERE file_path = %s;", (file_path,))

            # 获取删除的行数
            deleted_rows = cursor.rowcount

            # 提交
            self.connection.commit()

            self.logger.info(f"已删除与文件路径 '{file_path}' 相关的 {deleted_rows} 个文档")
            return deleted_rows

        except Exception as e:
            # 回滚
            if self.connection:
                self.connection.rollback()
            self.logger.error(f"删除文档时发生错误：{str(e)}")
            raise

        finally:
            # 关闭游标
            if "cursor" in locals() and cursor:
                cursor.close()

    def clear_database(self) -> int:
        """
        清除数据库（删除所有文档）。

        Raises:
            Exception: 清除失败时

        Returns:
            删除的文档数量。由于会 DROP 表，因此返回删除前的数量。
        """
        try:
            # 如果没有连接则连接
            if not self.connection:
                self.connect()

            # 获取删除前的文档数量
            count_before_delete = self.get_document_count()

            # 创建游标
            cursor = self.connection.cursor()

            # 删除表以清除架构
            cursor.execute("DROP TABLE IF EXISTS documents;")

            # 提交
            self.connection.commit()

            if count_before_delete > 0:
                self.logger.info(
                    f"数据库已清除（删除了 documents 表，目标文档数为 {count_before_delete} 个）"
                )
            else:
                self.logger.info("数据库已清除（删除了 documents 表）")
            return count_before_delete

        except Exception as e:
            # 回滚
            if self.connection:
                self.connection.rollback()
            self.logger.error(f"清除数据库时发生错误：{str(e)}")
            raise

        finally:
            # 关闭游标
            if "cursor" in locals() and cursor:
                cursor.close()

    def get_document_count(self) -> int:
        """
        获取数据库中的文档数量。

        Returns:
            文档数量

        Raises:
            Exception: 获取失败时
        """
        try:
            # 如果没有连接则连接
            if not self.connection:
                self.connect()

            # 创建游标
            cursor = self.connection.cursor()

            # 获取文档数量
            cursor.execute("SELECT COUNT(*) FROM documents;")
            count = cursor.fetchone()[0]

            self.logger.info(f"数据库中的文档数量：{count}")
            return count

        except psycopg2.errors.UndefinedTable:
            # 表不存在时返回 0
            self.connection.rollback()  # 重置错误状态
            self.logger.info("documents 表不存在，文档数量为 0")
            return 0
        except Exception as e:
            self.logger.error(f"获取文档数量时发生错误：{str(e)}")
            raise

    def get_adjacent_chunks(self, file_path: str, chunk_index: int, context_size: int = 1) -> List[Dict[str, Any]]:
        """
        获取指定分块的前后分块。

        Args:
            file_path: 文件路径
            chunk_index: 分块索引
            context_size: 获取前后分块的数量（默认：1）

        Returns:
            前后分块的列表

        Raises:
            Exception: 获取失败时
        """
        try:
            # 如果没有连接则连接
            if not self.connection:
                self.connect()

            # 创建游标
            cursor = self.connection.cursor()

            # 获取前后分块
            min_index = max(0, chunk_index - context_size)
            max_index = chunk_index + context_size

            cursor.execute(
                """
                SELECT
                    document_id,
                    content,
                    file_path,
                    chunk_index,
                    metadata,
                    1 AS similarity
                FROM
                    documents
                WHERE
                    file_path = %s
                    AND chunk_index >= %s
                    AND chunk_index <= %s
                    AND chunk_index != %s
                ORDER BY
                    chunk_index
                """,
                (file_path, min_index, max_index, chunk_index),
            )

            # 获取结果
            results = []
            for row in cursor.fetchall():
                document_id, content, file_path, chunk_index, metadata_json, similarity = row

                # 从 JSON 解码元数据
                if metadata_json:
                    if isinstance(metadata_json, str):
                        try:
                            metadata = json.loads(metadata_json)
                        except json.JSONDecodeError:
                            metadata = {}
                    else:
                        # 如果已经是字典类型则直接使用
                        metadata = metadata_json
                else:
                    metadata = {}

                results.append(
                    {
                        "document_id": document_id,
                        "content": content,
                        "file_path": file_path,
                        "chunk_index": chunk_index,
                        "metadata": metadata,
                        "similarity": similarity,
                        "is_context": True,  # 标识为上下文分块的标志
                    }
                )

            self.logger.info(
                f"获取了文件 '{file_path}' 的分块 {chunk_index} 前后 {len(results)} 个分块"
            )
            return results

        except Exception as e:
            self.logger.error(f"获取前后分块时发生错误：{str(e)}")
            raise

        finally:
            # 关闭游标
            if "cursor" in locals() and cursor:
                cursor.close()

    def get_document_by_file_path(self, file_path: str) -> List[Dict[str, Any]]:
        """
        根据指定的文件路径获取整个文档。

        Args:
            file_path: 文件路径

        Returns:
            整个文档的分块列表

        Raises:
            Exception: 获取失败时
        """
        try:
            # 如果没有连接则连接
            if not self.connection:
                self.connect()

            # 创建游标
            cursor = self.connection.cursor()

            # 根据文件路径获取文档
            cursor.execute(
                """
                SELECT
                    document_id,
                    content,
                    file_path,
                    chunk_index,
                    metadata,
                    1 AS similarity
                FROM
                    documents
                WHERE
                    file_path = %s
                ORDER BY
                    chunk_index
                """,
                (file_path,),
            )

            # 获取结果
            results = []
            for row in cursor.fetchall():
                document_id, content, file_path, chunk_index, metadata_json, similarity = row

                # 从 JSON 解码元数据
                if metadata_json:
                    if isinstance(metadata_json, str):
                        try:
                            metadata = json.loads(metadata_json)
                        except json.JSONDecodeError:
                            metadata = {}
                    else:
                        # 如果已经是字典类型则直接使用
                        metadata = metadata_json
                else:
                    metadata = {}

                results.append(
                    {
                        "document_id": document_id,
                        "content": content,
                        "file_path": file_path,
                        "chunk_index": chunk_index,
                        "metadata": metadata,
                        "similarity": similarity,
                        "is_full_document": True,  # 标识为全文文档的标志
                    }
                )

            self.logger.info(f"获取了文件 '{file_path}' 的全文 {len(results)} 个分块")
            return results

        except Exception as e:
            self.logger.error(f"获取文档全文时发生错误：{str(e)}")
            raise

        finally:
            # 关闭游标
            if "cursor" in locals() and cursor:
                cursor.close()
