"""
重排序模块的单元测试
"""

import os
import sys
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.getcwd())

from src.reranker import LocalReranker, create_reranker_from_env


def _make_candidates(count: int) -> list:
    """生成指定数量的候选结果。"""
    return [
        {"content": f"passage {i}", "document_id": f"id_{i}", "file_path": f"file_{i}.txt", "chunk_index": i}
        for i in range(count)
    ]


class TestLocalReranker:
    """LocalReranker 测试组。"""

    @patch("src.reranker._LocalCrossEncoderBackend")
    def test_reranks_order(self, mock_backend_cls):
        """按 rerank 分数降序排列。"""
        mock_backend = MagicMock()
        mock_backend.score.return_value = np.array([0.1, 0.9, 0.5])
        mock_backend_cls.return_value = mock_backend

        reranker = LocalReranker(model_name="test-model")
        candidates = _make_candidates(3)
        result = reranker.rerank("query", candidates, k=3)

        assert len(result) == 3
        assert result[0]["content"] == "passage 1"
        assert result[1]["content"] == "passage 2"
        assert result[2]["content"] == "passage 0"

    @patch("src.reranker._LocalCrossEncoderBackend")
    def test_truncates_to_k(self, mock_backend_cls):
        """输入 10 条，k=3，输出 3 条。"""
        mock_backend = MagicMock()
        mock_backend.score.return_value = np.random.rand(10)
        mock_backend_cls.return_value = mock_backend

        reranker = LocalReranker(model_name="test-model")
        candidates = _make_candidates(10)
        result = reranker.rerank("query", candidates, k=3)

        assert len(result) == 3

    @patch("src.reranker._LocalCrossEncoderBackend")
    def test_injects_rerank_score(self, mock_backend_cls):
        """结果含 rerank_score 字段。"""
        mock_backend = MagicMock()
        mock_backend.score.return_value = np.array([0.42, 0.87])
        mock_backend_cls.return_value = mock_backend

        reranker = LocalReranker(model_name="test-model")
        candidates = _make_candidates(2)
        result = reranker.rerank("query", candidates, k=2)

        for r in result:
            assert "rerank_score" in r
        assert result[0]["rerank_score"] == pytest.approx(0.87)
        assert result[1]["rerank_score"] == pytest.approx(0.42)

    @patch("src.reranker._LocalCrossEncoderBackend")
    def test_empty_candidates(self, mock_backend_cls):
        """空候选列表返回空列表。"""
        mock_backend = MagicMock()
        mock_backend_cls.return_value = mock_backend

        reranker = LocalReranker(model_name="test-model")
        result = reranker.rerank("query", [], k=5)

        assert result == []
        mock_backend.score.assert_not_called()


class TestCreateRerankerFromEnv:
    """create_reranker_from_env 工厂函数测试组。"""

    def test_provider_none_returns_none(self):
        """RERANKER_PROVIDER=none 返回 None。"""
        with patch.dict(os.environ, {"RERANKER_PROVIDER": "none"}):
            assert create_reranker_from_env() is None

    def test_provider_default_returns_none(self):
        """未设置 RERANKER_PROVIDER 时返回 None。"""
        with patch.dict(os.environ, {}, clear=True):
            # 确保没有 RERANKER_PROVIDER
            os.environ.pop("RERANKER_PROVIDER", None)
            assert create_reranker_from_env() is None

    @patch("src.reranker._LocalCrossEncoderBackend")
    def test_provider_local_returns_local_reranker(self, mock_backend_cls):
        """RERANKER_PROVIDER=local 返回 LocalReranker 实例。"""
        mock_backend_cls.return_value = MagicMock()
        with patch.dict(os.environ, {"RERANKER_PROVIDER": "local"}):
            reranker = create_reranker_from_env()
            assert isinstance(reranker, LocalReranker)

    def test_provider_unknown_raises(self):
        """未知的 provider 抛出 ValueError。"""
        with patch.dict(os.environ, {"RERANKER_PROVIDER": "unknown"}):
            with pytest.raises(ValueError, match="Unknown RERANKER_PROVIDER"):
                create_reranker_from_env()
