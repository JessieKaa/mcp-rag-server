"""
エンベディング生成モジュール

テキストからエンベディングを生成します。
"""

import logging
import os
from typing import List
from dotenv import load_dotenv
from urllib.parse import urlparse

# .envの読み込み
load_dotenv()


class _LocalBackend:
    """sentence-transformers を使用したローカル推論バックエンド"""

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
    """OpenAI 互換 API を使用したバックエンド"""

    def __init__(self, model_name: str, logger: logging.Logger):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "EMBEDDING_PROVIDER=openai requires 'openai' package. "
                "Install with: pip install mcp-rag-server[openai]"
            )

        base_url = os.getenv("EMBEDDING_BASE_URL", "").strip() or None
        api_key = os.getenv("EMBEDDING_API_KEY")

        if base_url is not None:
            parsed = urlparse(base_url)
            if parsed.scheme not in ("http", "https") or not parsed.hostname:
                raise ValueError(
                    f"EMBEDDING_BASE_URL is malformed: {base_url!r}. "
                    "Expected a full URL, e.g. 'http://localhost:9997/v1'."
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
    エンベディング生成クラス

    テキストからエンベディングを生成します。

    Attributes:
        model_name: モデル名
        prefix_query: クエリ用プレフィックス
        prefix_embedding: エンベディング用プレフィックス
        logger: ロガー
    """

    def __init__(self, model_name: str = None):
        """
        EmbeddingGeneratorのコンストラクタ

        Args:
            model_name: 使用するモデル名（.env優先）
        """
        self.model_name = os.getenv("EMBEDDING_MODEL") or model_name or "intfloat/multilingual-e5-large"
        self.prefix_query = os.getenv("EMBEDDING_PREFIX_QUERY", "")
        self.prefix_embedding = os.getenv("EMBEDDING_PREFIX_EMBEDDING", "")
        self.expected_dim = int(os.getenv("EMBEDDING_DIM", "1024"))

        self.logger = logging.getLogger("embedding_generator")
        self.logger.setLevel(logging.INFO)

        provider = os.getenv("EMBEDDING_PROVIDER", "local").lower()
        if provider not in ("local", "openai"):
            raise ValueError(
                f"EMBEDDING_PROVIDER must be 'local' or 'openai', got: {provider!r}"
            )

        self.logger.info(f"プロバイダ '{provider}' でモデル '{self.model_name}' を初期化しています...")
        try:
            if provider == "local":
                self._backend = _LocalBackend(self.model_name, self.logger)
            else:
                self._backend = _OpenAIBackend(self.model_name, self.logger)
            self.logger.info(f"モデル '{self.model_name}' を読み込みました")
        except Exception as e:
            self.logger.error(f"モデル '{self.model_name}' の読み込みに失敗しました: {str(e)}")
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
        テキストに適切なプレフィックスを追加する

        Args:
            text: 元のテキスト
            prefix: 追加するプレフィックス

        Returns:
            プレフィックス付きのテキスト
        """
        if not prefix:
            return text

        if text.startswith(prefix):
            return text

        return f"{prefix}{text}"

    def generate_embedding(self, text: str) -> List[float]:
        """
        テキストからエンベディングを生成します。

        Args:
            text: エンベディングを生成するテキスト

        Returns:
            エンベディング（浮動小数点数のリスト）
        """
        if not text:
            self.logger.warning("空のテキストからエンベディングを生成しようとしています")
            return []

        try:
            processed_text = self._add_prefix(text, self.prefix_embedding)
            embedding = self._backend.encode_one(processed_text)
            self._validate_dim(embedding, "generate_embedding")
            self.logger.debug(f"テキスト '{text[:50]}...' のエンベディングを生成しました")
            return embedding
        except Exception as e:
            self.logger.error(f"エンベディングの生成中にエラーが発生しました: {str(e)}")
            raise

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        複数のテキストからエンベディングを生成します。

        Args:
            texts: エンベディングを生成するテキストのリスト

        Returns:
            エンベディングのリスト
        """
        if not texts:
            self.logger.warning("空のテキストリストからエンベディングを生成しようとしています")
            return []

        try:
            processed_texts = [self._add_prefix(text, self.prefix_embedding) for text in texts]
            embeddings = self._backend.encode_batch(processed_texts)
            if embeddings:
                self._validate_dim(embeddings[0], "generate_embeddings")
            self.logger.info(f"{len(texts)} 個のテキストのエンベディングを生成しました")
            return embeddings
        except Exception as e:
            self.logger.error(f"エンベディングの生成中にエラーが発生しました: {str(e)}")
            raise

    def generate_search_embedding(self, query: str) -> List[float]:
        """
        検索クエリからエンベディングを生成します。

        Args:
            query: 検索クエリ

        Returns:
            エンベディング（浮動小数点数のリスト）
        """
        if not query:
            self.logger.warning("空のクエリからエンベディングを生成しようとしています")
            return []

        try:
            processed_query = self._add_prefix(query, self.prefix_query)
            embedding = self._backend.encode_one(processed_query)
            self._validate_dim(embedding, "generate_search_embedding")
            self.logger.debug(f"クエリ '{query}' のエンベディングを生成しました")
            return embedding
        except Exception as e:
            self.logger.error(f"クエリエンベディングの生成中にエラーが発生しました: {str(e)}")
            raise
