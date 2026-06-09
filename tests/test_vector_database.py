import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# 将 `src` 目录添加到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# 全局 mock psycopg2
sys.modules["psycopg2"] = MagicMock()


class TestVectorDatabase(unittest.TestCase):
    def setUp(self):
        # 在每个测试前卸载 vector_database 模块
        if "vector_database" in sys.modules:
            del sys.modules["vector_database"]

    @patch.dict(os.environ, {"EMBEDDING_DIM": "512"})
    def test_create_table_with_custom_embedding_dim(self):
        """测试从环境变量读取 EMBEDDING_DIM，以及 CREATE TABLE 查询是否正确格式化"""
        # 在 patch 环境下导入模块
        from vector_database import VectorDatabase, EMBEDDING_DIM

        mock_connect = MagicMock()
        with patch("vector_database.psycopg2.connect", return_value=mock_connect):
            mock_cursor = MagicMock()
            mock_connect.cursor.return_value = mock_cursor

            db = VectorDatabase(connection_params={"dbname": "test_db"})
            db.initialize_database()

            # 获取 CREATE TABLE 的 SQL 语句
            create_table_sql = mock_cursor.execute.call_args_list[1][0][0]

            # 确认 SQL 中包含正确的向量维度
            self.assertEqual(EMBEDDING_DIM, 512)
            self.assertIn("embedding vector(512)", create_table_sql)

            # 确认 execute 被调用了 5 次
            self.assertEqual(mock_cursor.execute.call_count, 5)

    @patch("dotenv.load_dotenv")  # 禁用 .env 加载
    def test_create_table_with_default_embedding_dim(self, mock_load_dotenv):
        """测试环境变量不存在时使用默认的 EMBEDDING_DIM 创建表"""
        # 清除环境变量
        with patch.dict(os.environ, {}, clear=True):
            # 在 patch 环境下导入模块
            from vector_database import VectorDatabase, EMBEDDING_DIM

            mock_connect = MagicMock()
            with patch("vector_database.psycopg2.connect", return_value=mock_connect):
                mock_cursor = MagicMock()
                mock_connect.cursor.return_value = mock_cursor

                db = VectorDatabase(connection_params={"dbname": "test_db"})
                db.initialize_database()

                # 获取 CREATE TABLE 的 SQL 语句
                create_table_sql = mock_cursor.execute.call_args_list[1][0][0]

                # 确认使用默认维度（1024）
                self.assertEqual(EMBEDDING_DIM, 1024)
                self.assertIn("embedding vector(1024)", create_table_sql)


if __name__ == "__main__":
    unittest.main()
