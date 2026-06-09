"""
嵌入向量生成模块

从文本生成嵌入向量。
"""

import logging
import os
from typing import List
from dotenv import load_dotenv
from urllib.parse import urlparse

# 加载 .env
load_dotenv()


class _LocalBackend:
    """使用 sentence-transformers 的本地推理后端"""

    def __init__(self, model_name: str, logger: logging.Logger):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "EMBEDDING_PROVIDER=local requires 'sentence-transformers' package. "
                "Install with: pip install mcp-rag-server[local]"
            )
        self.model = SentenceTransformer(model_name)
        self._logger = logger

    def encode_one(self, text: str) -> List[float]:
        result = self.model.encode(text)
        return result.tolist()

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        result = self.model.encode(texts)
        return result.tolist()


class _OpenAIBackend:
    """使用 OpenAI 兼容 API 的后端"""

    def __init__(self, model_name: str, logger: logging.Logger):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "EMBEDDING_PROVIDER=openai requires 'openai' package. Install with: pip install mcp-rag-server[openai]"
            )

        base_url = os.getenv("EMBEDDING_BASE_URL", "").strip() or None
        api_key = os.getenv("EMBEDDING_API_KEY")

        if base_url is not None:
            parsed = urlparse(base_url)
            if parsed.scheme not in ("http", "https") or not parsed.hostname:
                raise ValueError(
                    f"EMBEDDING_BASE_URL is malformed: {base_url!r}. Expected a full URL, e.g. 'http://localhost:9997/v1'."
                )

        is_official = base_url is None or urlparse(base_url).hostname == "api.openai.com"

        if is_official and not api_key:
            raise ValueError(
                "EMBEDDING_PROVIDER=openai targeting the official OpenAI endpoint requires "
                "EMBEDDING_API_KEY. For self-hosted compatible services, set EMBEDDING_BASE_URL "
                "to a non-OpenAI host."
            )
        api_key = api_key or "no-api-key"

        try:
            batch_size = int(os.getenv("EMBEDDING_API_BATCH_SIZE", "64"))
        except ValueError as e:
            raise ValueError(f"EMBEDDING_API_BATCH_SIZE must be an integer, got: {e}")
        if batch_size <= 0:
            raise ValueError(f"EMBEDDING_API_BATCH_SIZE must be a positive integer, got: {batch_size}")
        self.batch_size = batch_size

        self.client = OpenAI(api_key=api_key, base_url=base_url, max_retries=3)
        self.model_name = model_name
        self._logger = logger

    def encode_one(self, text: str) -> List[float]:
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        results = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            resp = self.client.embeddings.create(input=chunk, model=self.model_name)
            if len(resp.data) != len(chunk):
                raise RuntimeError(
                    f"OpenAI API returned {len(resp.data)} embeddings for {len(chunk)} inputs "
                    f"(model={self.model_name}). Possible backend bug or rate-limit truncation."
                )
            results.extend([d.embedding for d in resp.data])
        return results


class EmbeddingGenerator:
    """
    嵌入向量生成类

    从文本生成嵌入向量。

    Attributes:
        model_name: 模型名称
        prefix_query: 查询用前缀
        prefix_embedding: 嵌入向量用前缀
        logger: 日志记录器
    """

    def __init__(self, model_name: str = None):
        """
        EmbeddingGenerator 的构造函数

        Args:
            model_name: 使用的模型名称（.env 优先）
        """
        self.model_name = os.getenv("EMBEDDING_MODEL") or model_name or "intfloat/multilingual-e5-large"
        self.prefix_query = os.getenv("EMBEDDING_PREFIX_QUERY", "")
        self.prefix_embedding = os.getenv("EMBEDDING_PREFIX_EMBEDDING", "")
        self.expected_dim = int(os.getenv("EMBEDDING_DIM", "1024"))

        self.logger = logging.getLogger("embedding_generator")
        self.logger.setLevel(logging.INFO)

        provider = os.getenv("EMBEDDING_PROVIDER", "local").lower()
        if provider not in ("local", "openai"):
            raise ValueError(f"EMBEDDING_PROVIDER must be 'local' or 'openai', got: {provider!r}")

        self.logger.info(f"正在使用提供者 '{provider}' 初始化模型 '{self.model_name}'...")
        try:
            if provider == "local":
                self._backend = _LocalBackend(self.model_name, self.logger)
            else:
                self._backend = _OpenAIBackend(self.model_name, self.logger)
            self.logger.info(f"已加载模型 '{self.model_name}'")
        except Exception as e:
            self.logger.error(f"加载模型 '{self.model_name}' 失败：{str(e)}")
            raise

    def _validate_dim(self, embedding: List[float], context: str):
        actual = len(embedding)
        if actual != self.expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch ({context}): "
                f"got {actual}, expected EMBEDDING_DIM={self.expected_dim}. "
                f"Run `python -m src.cli clear` then re-index after updating EMBEDDING_DIM."
            )

    def _add_prefix(self, text: str, prefix: str) -> str:
        """
        为文本添加适当的前缀

        Args:
            text: 原始文本
            prefix: 要添加的前缀

        Returns:
            带前缀的文本
        """
        if not prefix:
            return text

        if text.startswith(prefix):
            return text

        return f"{prefix}{text}"

    def generate_embedding(self, text: str) -> List[float]:
        """
        从文本生成嵌入向量。

        Args:
            text: 要生成嵌入向量的文本

        Returns:
            嵌入向量（浮点数列表）
        """
        if not text:
            self.logger.warning("尝试从空文本生成嵌入向量")
            return []

        try:
            processed_text = self._add_prefix(text, self.prefix_embedding)
            embedding = self._backend.encode_one(processed_text)
            self._validate_dim(embedding, "generate_embedding")
            self.logger.debug(f"已生成文本 '{text[:50]}...' 的嵌入向量")
            return embedding
        except Exception as e:
            self.logger.error(f"生成嵌入向量时发生错误：{str(e)}")
            raise

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        从多个文本生成嵌入向量。

        Args:
            texts: 要生成嵌入向量的文本列表

        Returns:
            嵌入向量列表
        """
        if not texts:
            self.logger.warning("尝试从空文本列表生成嵌入向量")
            return []

        try:
            processed_texts = [self._add_prefix(text, self.prefix_embedding) for text in texts]
            embeddings = self._backend.encode_batch(processed_texts)
            if embeddings:
                self._validate_dim(embeddings[0], "generate_embeddings")
            self.logger.info(f"已生成 {len(texts)} 个文本的嵌入向量")
            return embeddings
        except Exception as e:
            self.logger.error(f"生成嵌入向量时发生错误：{str(e)}")
            raise

    def generate_search_embedding(self, query: str) -> List[float]:
        """
        从搜索查询生成嵌入向量。

        Args:
            query: 搜索查询

        Returns:
            嵌入向量（浮点数列表）
        """
        if not query:
            self.logger.warning("尝试从空查询生成嵌入向量")
            return []

        try:
            processed_query = self._add_prefix(query, self.prefix_query)
            embedding = self._backend.encode_one(processed_query)
            self._validate_dim(embedding, "generate_search_embedding")
            self.logger.debug(f"已生成查询 '{query}' 的嵌入向量")
            return embedding
        except Exception as e:
            self.logger.error(f"生成查询嵌入向量时发生错误：{str(e)}")
            raise
