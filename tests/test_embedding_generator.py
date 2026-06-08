import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from embedding_generator import EmbeddingGenerator


class TestLocalEmbeddingGenerator(unittest.TestCase):
    """本地 (sentence-transformers) 后端测试"""

    def setUp(self):
        self.fake_st = MagicMock()
        self.fake_model = MagicMock()
        self.fake_st.SentenceTransformer.return_value = self.fake_model
        # encode(str) → 1D array; encode(list) → 2D array; side_effect handles both
        def _encode_side_effect(inp):
            if isinstance(inp, str):
                return np.array([0.1, 0.2, 0.3])
            return np.array([[0.1, 0.2, 0.3]] * len(inp))
        self.fake_model.encode.side_effect = _encode_side_effect
        self._modules = patch.dict(sys.modules, {"sentence_transformers": self.fake_st})
        self._modules.start()

    def tearDown(self):
        self._modules.stop()

    def _make_env(self, extra=None, clear=True):
        env = {"EMBEDDING_DIM": "3"}
        if extra:
            env.update(extra)
        return env

    def test_initialization_with_env_variables(self):
        test_env = self._make_env({
            "EMBEDDING_MODEL": "test-model",
            "EMBEDDING_PREFIX_QUERY": "query: ",
            "EMBEDDING_PREFIX_EMBEDDING": "passage: ",
        })
        with patch.dict(os.environ, test_env, clear=True):
            generator = EmbeddingGenerator()
            self.assertEqual(generator.model_name, "test-model")
            self.assertEqual(generator.prefix_query, "query: ")
            self.assertEqual(generator.prefix_embedding, "passage: ")
            self.fake_st.SentenceTransformer.assert_called_once_with("test-model")

    def test_initialization_with_defaults(self):
        with patch.dict(os.environ, self._make_env(), clear=True):
            generator = EmbeddingGenerator()
            self.assertEqual(generator.model_name, "intfloat/multilingual-e5-large")
            self.assertEqual(generator.prefix_query, "")
            self.assertEqual(generator.prefix_embedding, "")
            self.fake_st.SentenceTransformer.assert_called_once_with("intfloat/multilingual-e5-large")

    def test_add_prefix(self):
        with patch.dict(os.environ, self._make_env(), clear=True):
            generator = EmbeddingGenerator()
            self.assertEqual(generator._add_prefix("text", "prefix: "), "prefix: text")
            self.assertEqual(generator._add_prefix("prefix: text", "prefix: "), "prefix: text")
            self.assertEqual(generator._add_prefix("text", ""), "text")
            self.assertEqual(generator._add_prefix("TEXT", "prefix: "), "prefix: TEXT")

    def test_generate_embedding_with_prefix(self):
        env = self._make_env({"EMBEDDING_PREFIX_EMBEDDING": "passage: "})
        with patch.dict(os.environ, env, clear=True):
            generator = EmbeddingGenerator()
            generator.generate_embedding("my text")
            self.fake_model.encode.assert_called_with("passage: my text")

    def test_generate_embeddings_with_prefix(self):
        env = self._make_env({"EMBEDDING_PREFIX_EMBEDDING": "passage: "})
        with patch.dict(os.environ, env, clear=True):
            generator = EmbeddingGenerator()
            generator.generate_embeddings(["text1", "text2"])
            self.fake_model.encode.assert_called_with(["passage: text1", "passage: text2"])

    def test_generate_query_embedding_with_prefix(self):
        env = self._make_env({"EMBEDDING_PREFIX_QUERY": "query: "})
        with patch.dict(os.environ, env, clear=True):
            generator = EmbeddingGenerator()
            generator.generate_search_embedding("my query")
            self.fake_model.encode.assert_called_with("query: my query")


class TestOpenAIEmbeddingGenerator(unittest.TestCase):
    """OpenAI 兼容 API 后端测试"""

    def _make_fake_openai(self):
        fake_openai = MagicMock()
        mock_client = MagicMock()
        fake_openai.OpenAI.return_value = mock_client
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response
        return fake_openai

    def _start_openai_mock(self, fake_openai):
        self._modules = patch.dict(sys.modules, {"openai": fake_openai})
        self._modules.start()
        return self._modules

    def tearDown(self):
        if hasattr(self, "_modules"):
            self._modules.stop()

    def test_openai_init_with_api_key_only(self):
        """仅设置 API_KEY → 官方端点 → 成功"""
        fake = self._make_fake_openai()
        self._start_openai_mock(fake)
        env = {
            "EMBEDDING_PROVIDER": "openai",
            "EMBEDDING_API_KEY": "sk-test",
            "EMBEDDING_MODEL": "text-embedding-3-small",
            "EMBEDDING_DIM": "3",
        }
        with patch.dict(os.environ, env, clear=True):
            EmbeddingGenerator()
            fake.OpenAI.assert_called_once()
            call_kwargs = fake.OpenAI.call_args
            self.assertEqual(call_kwargs.kwargs["api_key"], "sk-test")

    def test_openai_init_with_base_url_no_key(self):
        """非官方 BASE_URL + 无 KEY → 自托管 → 成功（使用占位符 KEY）"""
        fake = self._make_fake_openai()
        self._start_openai_mock(fake)
        env = {
            "EMBEDDING_PROVIDER": "openai",
            "EMBEDDING_BASE_URL": "http://localhost:9997/v1",
            "EMBEDDING_MODEL": "bge-m3",
            "EMBEDDING_DIM": "3",
        }
        with patch.dict(os.environ, env, clear=True):
            EmbeddingGenerator()
            call_kwargs = fake.OpenAI.call_args
            self.assertEqual(call_kwargs.kwargs["api_key"], "no-api-key")

    def test_openai_init_no_key_no_url_raises_value_error(self):
        """无 KEY + 无 URL → 官方端点 → ValueError"""
        fake = self._make_fake_openai()
        self._start_openai_mock(fake)
        env = {
            "EMBEDDING_PROVIDER": "openai",
            "EMBEDDING_MODEL": "text-embedding-3-small",
            "EMBEDDING_DIM": "3",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError) as ctx:
                EmbeddingGenerator()
            self.assertIn("EMBEDDING_API_KEY", str(ctx.exception))

    def test_openai_init_official_base_url_no_key_raises(self):
        """明确指定官方 BASE_URL + 无 KEY → 仍按官方端点处理 → ValueError"""
        fake = self._make_fake_openai()
        self._start_openai_mock(fake)
        env = {
            "EMBEDDING_PROVIDER": "openai",
            "EMBEDDING_BASE_URL": "https://api.openai.com/v1",
            "EMBEDDING_MODEL": "text-embedding-3-small",
            "EMBEDDING_DIM": "3",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError) as ctx:
                EmbeddingGenerator()
            self.assertIn("EMBEDDING_API_KEY", str(ctx.exception))

    def test_openai_generate_embedding_calls_create(self):
        """generate_embedding 调用 embeddings.create"""
        fake = self._make_fake_openai()
        self._start_openai_mock(fake)
        env = {
            "EMBEDDING_PROVIDER": "openai",
            "EMBEDDING_API_KEY": "sk-test",
            "EMBEDDING_MODEL": "text-embedding-3-small",
            "EMBEDDING_DIM": "3",
        }
        with patch.dict(os.environ, env, clear=True):
            generator = EmbeddingGenerator()
            result = generator.generate_embedding("hello world")
            self.assertEqual(result, [0.1, 0.2, 0.3])
            mock_client = fake.OpenAI.return_value
            mock_client.embeddings.create.assert_called_once_with(
                input=["hello world"], model="text-embedding-3-small"
            )

    def test_openai_generate_embeddings_batches(self):
        """100 个文本 / batch_size=32 → create 被调用 4 次"""
        fake_openai = MagicMock()
        mock_client = MagicMock()
        fake_openai.OpenAI.return_value = mock_client

        def make_response(n):
            return MagicMock(data=[MagicMock(embedding=[0.1, 0.2, 0.3])] * n)

        mock_client.embeddings.create.side_effect = [
            make_response(32), make_response(32), make_response(32), make_response(4)
        ]

        self._start_openai_mock(fake_openai)
        env = {
            "EMBEDDING_PROVIDER": "openai",
            "EMBEDDING_API_KEY": "sk-test",
            "EMBEDDING_MODEL": "text-embedding-3-small",
            "EMBEDDING_DIM": "3",
            "EMBEDDING_API_BATCH_SIZE": "32",
        }
        with patch.dict(os.environ, env, clear=True):
            generator = EmbeddingGenerator()
            texts = [f"text{i}" for i in range(100)]
            result = generator.generate_embeddings(texts)
            self.assertEqual(len(result), 100)
            self.assertEqual(mock_client.embeddings.create.call_count, 4)

    def test_openai_no_prefix_by_default(self):
        """默认无前缀"""
        fake = self._make_fake_openai()
        self._start_openai_mock(fake)
        env = {
            "EMBEDDING_PROVIDER": "openai",
            "EMBEDDING_API_KEY": "sk-test",
            "EMBEDDING_MODEL": "text-embedding-3-small",
            "EMBEDDING_DIM": "3",
        }
        with patch.dict(os.environ, env, clear=True):
            generator = EmbeddingGenerator()
            self.assertEqual(generator.prefix_query, "")
            self.assertEqual(generator.prefix_embedding, "")
            generator.generate_embedding("hello")
            mock_client = fake.OpenAI.return_value
            mock_client.embeddings.create.assert_called_once_with(
                input=["hello"], model="text-embedding-3-small"
            )

    def test_openai_missing_package_raises_helpful_import_error(self):
        """未安装 openai 包 → 带安装指南的 ImportError"""
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("Simulated: no module named 'openai'")
            return real_import(name, *args, **kwargs)

        env = {
            "EMBEDDING_PROVIDER": "openai",
            "EMBEDDING_API_KEY": "sk-test",
            "EMBEDDING_MODEL": "text-embedding-3-small",
            "EMBEDDING_DIM": "3",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch("builtins.__import__", side_effect=fake_import):
                with self.assertRaises(ImportError) as ctx:
                    EmbeddingGenerator()
                self.assertIn("pip install mcp-rag-server[openai]", str(ctx.exception))

    def test_openai_response_count_mismatch_raises(self):
        """API 响应数量 ≠ 输入数量 → RuntimeError"""
        fake_openai = MagicMock()
        mock_client = MagicMock()
        fake_openai.OpenAI.return_value = mock_client
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]  # 仅 1 条
        mock_client.embeddings.create.return_value = mock_response

        self._start_openai_mock(fake_openai)
        env = {
            "EMBEDDING_PROVIDER": "openai",
            "EMBEDDING_API_KEY": "sk-test",
            "EMBEDDING_MODEL": "text-embedding-3-small",
            "EMBEDDING_DIM": "3",
        }
        with patch.dict(os.environ, env, clear=True):
            generator = EmbeddingGenerator()
            with self.assertRaises(RuntimeError) as ctx:
                generator.generate_embeddings(["text1", "text2"])
            self.assertIn("returned 1 embeddings for 2 inputs", str(ctx.exception))

    def test_openai_invalid_batch_size_raises_value_error(self):
        """无效批处理大小 → ValueError"""
        fake = self._make_fake_openai()
        self._start_openai_mock(fake)
        for bad_val in ["0", "-1", "abc"]:
            with self.subTest(batch_size=bad_val):
                env = {
                    "EMBEDDING_PROVIDER": "openai",
                    "EMBEDDING_API_KEY": "sk-test",
                    "EMBEDDING_MODEL": "text-embedding-3-small",
                    "EMBEDDING_DIM": "3",
                    "EMBEDDING_API_BATCH_SIZE": bad_val,
                }
                with patch.dict(os.environ, env, clear=True):
                    with self.assertRaises(ValueError):
                        EmbeddingGenerator()

    def test_openai_malformed_base_url_raises_value_error(self):
        """格式错误的 EMBEDDING_BASE_URL → ValueError"""
        fake = self._make_fake_openai()
        self._start_openai_mock(fake)
        for bad_url in ["not-a-url", "localhost:9997", "ftp://bad-scheme.com"]:
            with self.subTest(base_url=bad_url):
                env = {
                    "EMBEDDING_PROVIDER": "openai",
                    "EMBEDDING_API_KEY": "sk-test",
                    "EMBEDDING_MODEL": "text-embedding-3-small",
                    "EMBEDDING_DIM": "3",
                    "EMBEDDING_BASE_URL": bad_url,
                }
                with patch.dict(os.environ, env, clear=True):
                    with self.assertRaises(ValueError) as ctx:
                        EmbeddingGenerator()
                    self.assertIn("malformed", str(ctx.exception))


class TestProviderAndDimensionValidation(unittest.TestCase):
    """通用验证测试（适用于两种后端）"""

    def test_invalid_provider_raises_value_error(self):
        """未知的 EMBEDDING_PROVIDER → ValueError"""
        fake_st = MagicMock()
        with patch.dict(sys.modules, {"sentence_transformers": fake_st}):
            env = {
                "EMBEDDING_PROVIDER": "anthropic",
                "EMBEDDING_DIM": "3",
            }
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(ValueError) as ctx:
                    EmbeddingGenerator()
                self.assertIn("'local' or 'openai'", str(ctx.exception))

    def test_dimension_mismatch_raises_clear_error(self):
        """维度不一致 → 包含 clear and re-index 的 ValueError"""
        fake_st = MagicMock()
        fake_model = MagicMock()
        fake_st.SentenceTransformer.return_value = fake_model
        # 返回 3 维但期望 EMBEDDING_DIM=1024
        fake_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        with patch.dict(sys.modules, {"sentence_transformers": fake_st}):
            env = {
                "EMBEDDING_DIM": "1024",
                "EMBEDDING_PROVIDER": "local",
            }
            with patch.dict(os.environ, env, clear=True):
                generator = EmbeddingGenerator()
                with self.assertRaises(ValueError) as ctx:
                    generator.generate_embedding("test")
                self.assertIn("dimension mismatch", str(ctx.exception))
                self.assertIn("clear", str(ctx.exception))
                self.assertIn("re-index", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
