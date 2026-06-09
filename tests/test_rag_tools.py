"""
RAG 工具模块的单元测试（含重排序分数显示）
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.getcwd())

from src.rag_tools import search_handler


def _mock_rag_service(results=None, doc_count=10):
    """创建模拟的 RAG 服务。"""
    svc = MagicMock()
    svc.get_document_count.return_value = doc_count
    svc.search.return_value = results or []
    return svc


class TestSearchHandlerScoreDisplay:
    """search_handler 分数显示测试组。"""

    def test_shows_rerank_score(self):
        """结果含 rerank_score 时显示"重排分数"。"""
        results = [
            {
                "document_id": "id_0",
                "content": "hello world",
                "file_path": "test.txt",
                "chunk_index": 0,
                "rerank_score": 0.8765,
                "is_context": False,
                "is_full_document": False,
            }
        ]
        svc = _mock_rag_service(results=results)
        result = search_handler({"query": "test"}, svc)

        texts = [item["text"] for item in result["content"]]
        assert any("重排分数" in t for t in texts)
        assert not any("相似度" in t for t in texts)

    def test_shows_similarity_without_rerank(self):
        """结果不含 rerank_score 时显示"相似度: XX.XX%"。"""
        results = [
            {
                "document_id": "id_0",
                "content": "hello world",
                "file_path": "test.txt",
                "chunk_index": 0,
                "similarity": 0.9234,
                "is_context": False,
                "is_full_document": False,
            }
        ]
        svc = _mock_rag_service(results=results)
        result = search_handler({"query": "test"}, svc)

        texts = [item["text"] for item in result["content"]]
        assert any("相似度" in t for t in texts)
        assert not any("重排分数" in t for t in texts)

    def test_shows_similarity_as_percentage(self):
        """相似度以百分比形式显示。"""
        results = [
            {
                "document_id": "id_0",
                "content": "hello world",
                "file_path": "test.txt",
                "chunk_index": 0,
                "similarity": 0.9234,
                "is_context": False,
                "is_full_document": False,
            }
        ]
        svc = _mock_rag_service(results=results)
        result = search_handler({"query": "test"}, svc)

        texts = [item["text"] for item in result["content"]]
        assert any("92.34%" in t for t in texts)
