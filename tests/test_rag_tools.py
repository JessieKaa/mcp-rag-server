"""
RAG 工具模块的单元测试（含重排序分数显示）
"""

import os
import sys
from unittest.mock import MagicMock


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


class TestSearchHandlerSorting:
    """search_handler 排序行为测试组。"""

    def test_groups_sorted_by_max_rerank_score_desc(self):
        """多文件组按每组最大 rerank_score 降序排列。"""
        results = [
            {
                "document_id": "a1",
                "content": "a1",
                "file_path": "a.txt",
                "chunk_index": 0,
                "rerank_score": 0.5,
                "is_context": False,
                "is_full_document": False,
            },
            {
                "document_id": "b1",
                "content": "b1",
                "file_path": "b.txt",
                "chunk_index": 0,
                "rerank_score": 0.9,
                "is_context": False,
                "is_full_document": False,
            },
            {
                "document_id": "a2",
                "content": "a2",
                "file_path": "a.txt",
                "chunk_index": 1,
                "rerank_score": 0.3,
                "is_context": False,
                "is_full_document": False,
            },
        ]
        svc = _mock_rag_service(results=results)
        result = search_handler({"query": "test"}, svc)

        texts = [item["text"] for item in result["content"]]
        # b.txt (max score 0.9) 应在 a.txt (max score 0.5) 之前
        b_idx = next(i for i, t in enumerate(texts) if "b.txt" in t)
        a_idx = next(i for i, t in enumerate(texts) if "a.txt" in t)
        assert b_idx < a_idx

    def test_chunks_sorted_by_chunk_index_within_group(self):
        """同一文件组内按 chunk_index 排序。"""
        results = [
            {
                "document_id": "a2",
                "content": "chunk2",
                "file_path": "a.txt",
                "chunk_index": 2,
                "rerank_score": 0.9,
                "is_context": False,
                "is_full_document": False,
            },
            {
                "document_id": "a0",
                "content": "chunk0",
                "file_path": "a.txt",
                "chunk_index": 0,
                "rerank_score": 0.5,
                "is_context": False,
                "is_full_document": False,
            },
            {
                "document_id": "a1",
                "content": "chunk1",
                "file_path": "a.txt",
                "chunk_index": 1,
                "rerank_score": 0.7,
                "is_context": False,
                "is_full_document": False,
            },
        ]
        svc = _mock_rag_service(results=results)
        result = search_handler({"query": "test"}, svc)

        texts = [item["text"] for item in result["content"]]
        # 找到包含 chunk_index 的文本，验证顺序
        chunk_texts = [t for t in texts if "检索命中" in t]
        assert "分块 0," in chunk_texts[0]
        assert "分块 1," in chunk_texts[1]
        assert "分块 2," in chunk_texts[2]
