"""
RAG 服务模块的单元测试（含重排序逻辑）
"""

import os
import sys
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, os.getcwd())

from src.rag_service import RAGService, _MAX_FETCH_LIMIT


def _mock_rag_components():
    """创建 RAGService 所需的基础 mock 组件。"""
    dp = MagicMock()
    eg = MagicMock()
    vd = MagicMock()
    vd.initialize_database.return_value = None
    return dp, eg, vd


def _make_results(count: int, file_prefix: str = "file") -> list:
    """生成指定数量的检索结果。"""
    return [
        {
            "document_id": f"id_{i}",
            "content": f"content {i}",
            "file_path": f"{file_prefix}_{i % 2}.txt",
            "chunk_index": i,
            "similarity": 0.9 - i * 0.05,
            "metadata": {},
        }
        for i in range(count)
    ]


class TestSearchWithoutReranker:
    """无重排序时的检索测试。"""

    def test_search_uses_limit_directly(self):
        """reranker=None 时调用 vector_database.search(limit)。"""
        dp, eg, vd = _mock_rag_components()
        eg.generate_search_embedding.return_value = [0.1] * 128
        vd.search.return_value = _make_results(5)

        svc = RAGService(dp, eg, vd)
        result = svc.search("test query", limit=5)

        vd.search.assert_called_once_with([0.1] * 128, 5)
        assert len(result) == 5

    def test_search_sorts_by_file_and_chunk_with_context(self):
        """with_context=True 无重排序时结果按 (file_path, chunk_index) 排序。"""
        dp, eg, vd = _mock_rag_components()
        eg.generate_search_embedding.return_value = [0.1] * 128
        vd.search.return_value = _make_results(4)
        vd.get_adjacent_chunks.return_value = []

        svc = RAGService(dp, eg, vd)
        result = svc.search("test query", limit=4, with_context=True)

        for i in range(len(result) - 1):
            assert (result[i]["file_path"], result[i]["chunk_index"]) <= (
                result[i + 1]["file_path"],
                result[i + 1]["chunk_index"],
            )


class TestSearchWithReranker:
    """有重排序时的检索测试。"""

    def test_calls_larger_fetch_then_rerank(self):
        """先以 limit * rerank_factor 调用 DB search，再调用 reranker.rerank。"""
        dp, eg, vd = _mock_rag_components()
        eg.generate_search_embedding.return_value = [0.1] * 128
        initial = _make_results(15)
        vd.search.return_value = initial

        mock_reranker = MagicMock()
        reranked = _make_results(5)
        for i, r in enumerate(reranked):
            r["rerank_score"] = 0.9 - i * 0.1
        mock_reranker.rerank.return_value = reranked

        svc = RAGService(dp, eg, vd, reranker=mock_reranker, rerank_factor=3)
        result = svc.search("test query", limit=5)

        vd.search.assert_called_once_with([0.1] * 128, 15)
        mock_reranker.rerank.assert_called_once_with("test query", initial, k=5)
        assert len(result) == 5

    def test_reranker_order_preserved_with_context(self):
        """with_context=True 时合并相邻块后仍按 _rerank_order 排列。"""
        dp, eg, vd = _mock_rag_components()
        eg.generate_search_embedding.return_value = [0.1] * 128

        # 初始检索结果
        initial = _make_results(2)
        initial[0]["file_path"] = "a.txt"
        initial[0]["chunk_index"] = 2
        initial[1]["file_path"] = "b.txt"
        initial[1]["chunk_index"] = 1
        vd.search.return_value = initial

        # reranker 返回重排后的结果
        reranked = [initial[1], initial[0]]  # b.txt 在前
        for i, r in enumerate(reranked):
            r["rerank_score"] = 0.9 - i * 0.1
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = reranked

        # 相邻分块
        vd.get_adjacent_chunks.return_value = []

        svc = RAGService(dp, eg, vd, reranker=mock_reranker, rerank_factor=3)
        result = svc.search("test query", limit=2, with_context=True, context_size=1)

        # _rerank_order 应为 0, 1（与 reranker 返回顺序一致）
        assert result[0]["_rerank_order"] == 0
        assert result[1]["_rerank_order"] == 1
        assert result[0]["file_path"] == "b.txt"
        assert result[1]["file_path"] == "a.txt"

    def test_fetch_limit_capped_at_max(self):
        """limit=100, factor=10 → fetch_limit 被限制为 _MAX_FETCH_LIMIT (200)。"""
        dp, eg, vd = _mock_rag_components()
        eg.generate_search_embedding.return_value = [0.1] * 128
        vd.search.return_value = _make_results(20)

        mock_reranker = MagicMock()
        reranked = _make_results(5)
        for i, r in enumerate(reranked):
            r["rerank_score"] = 0.9 - i * 0.1
        mock_reranker.rerank.return_value = reranked

        svc = RAGService(dp, eg, vd, reranker=mock_reranker, rerank_factor=10)
        result = svc.search("test query", limit=100)

        vd.search.assert_called_once_with([0.1] * 128, _MAX_FETCH_LIMIT)

    def test_rerank_factor_clamped(self):
        """rerank_factor 超出范围时被 clamp 到 1-10。"""
        dp, eg, vd = _mock_rag_components()
        eg.generate_search_embedding.return_value = [0.1] * 128
        vd.search.return_value = _make_results(5)

        mock_reranker = MagicMock()
        reranked = _make_results(5)
        for i, r in enumerate(reranked):
            r["rerank_score"] = 0.9 - i * 0.1
        mock_reranker.rerank.return_value = reranked

        # factor=20 → 被clamp到10
        svc = RAGService(dp, eg, vd, reranker=mock_reranker, rerank_factor=20)
        svc.search("test query", limit=5)

        assert svc.rerank_factor == 10

    def test_same_file_multiple_hits_context_ordering(self):
        """同一文件多个命中点时，上下文分块按各自父命中的 _rerank_order 排列。"""
        dp, eg, vd = _mock_rag_components()
        eg.generate_search_embedding.return_value = [0.1] * 128

        # reranker 返回：file_a chunk5 (order=0), file_b chunk3 (order=1), file_a chunk10 (order=2)
        hit_a = {"document_id": "id_a", "content": "a", "file_path": "a.txt", "chunk_index": 5, "similarity": 0.9}
        hit_b = {"document_id": "id_b", "content": "b", "file_path": "b.txt", "chunk_index": 3, "similarity": 0.8}
        hit_c = {"document_id": "id_c", "content": "c", "file_path": "a.txt", "chunk_index": 10, "similarity": 0.7}

        reranked = [
            {**hit_a, "rerank_score": 0.95},
            {**hit_b, "rerank_score": 0.85},
            {**hit_c, "rerank_score": 0.75},
        ]
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = reranked
        vd.search.return_value = [hit_a, hit_b, hit_c]

        # adjacent chunks: hit_a (a.txt chunk5) → chunk4, chunk6; hit_c (a.txt chunk10) → chunk9, chunk11
        def _adjacent(fp, ci, size):
            chunks = []
            for offset in range(-size, size + 1):
                idx = ci + offset
                if idx == ci:
                    continue
                chunks.append({
                    "document_id": f"adj_{fp}_{idx}",
                    "content": f"adj {fp} {idx}",
                    "file_path": fp,
                    "chunk_index": idx,
                    "similarity": 0.5,
                    "metadata": {},
                })
            return chunks

        vd.get_adjacent_chunks.side_effect = _adjacent

        svc = RAGService(dp, eg, vd, reranker=mock_reranker, rerank_factor=3)
        result = svc.search("test query", limit=3, with_context=True, context_size=1)

        # 找出所有 _rerank_order
        orders = [(r.get("_rerank_order", 999), r["file_path"], r["chunk_index"]) for r in result]

        # 同一 file_a 的上下文分块应按各自父命中的 order 分组
        # hit_a (order=0) 的上下文 chunk4, chunk6 应在 hit_c (order=2) 的 chunk9, chunk11 之前
        order_a_5_ctx = [o for o in orders if o[1] == "a.txt" and o[2] in (4, 5, 6)]
        order_a_10_ctx = [o for o in orders if o[1] == "a.txt" and o[2] in (9, 10, 11)]

        # 所有 hit_a 的上下文分块应继承 order=0
        assert all(o[0] == 0 for o in order_a_5_ctx), f"Expected order=0, got {order_a_5_ctx}"
        # 所有 hit_c 的上下文分块应继承 order=2
        assert all(o[0] == 2 for o in order_a_10_ctx), f"Expected order=2, got {order_a_10_ctx}"

    def test_overlapping_context_keeps_best_parent_order(self):
        """重叠上下文窗口中，共享分块应继承最优（最小）的 _rerank_order。"""
        dp, eg, vd = _mock_rag_components()
        eg.generate_search_embedding.return_value = [0.1] * 128

        # 两个命中点：chunk3 (order=0, 高优先级) 和 chunk5 (order=1, 低优先级)
        # context_size=2 时，chunk5 的上下文窗口 [3,4,6,7] 与 chunk3 的 [1,2,4,5] 重叠于 chunk4
        hit_3 = {"document_id": "id_3", "content": "hit3", "file_path": "a.txt", "chunk_index": 3, "similarity": 0.9}
        hit_5 = {"document_id": "id_5", "content": "hit5", "file_path": "a.txt", "chunk_index": 5, "similarity": 0.7}

        reranked = [
            {**hit_3, "rerank_score": 0.95},
            {**hit_5, "rerank_score": 0.75},
        ]
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = reranked
        vd.search.return_value = [hit_3, hit_5]

        def _adjacent(fp, ci, size):
            chunks = []
            for offset in range(-size, size + 1):
                idx = ci + offset
                if idx == ci:
                    continue
                chunks.append({
                    "document_id": f"adj_{idx}",
                    "content": f"adj {idx}",
                    "file_path": fp,
                    "chunk_index": idx,
                    "similarity": 0.5,
                    "metadata": {},
                })
            return chunks

        vd.get_adjacent_chunks.side_effect = _adjacent

        svc = RAGService(dp, eg, vd, reranker=mock_reranker, rerank_factor=3)
        result = svc.search("test query", limit=2, with_context=True, context_size=2)

        by_doc_id = {r["document_id"]: r for r in result}

        # 命中分块保持自己的 _rerank_order
        assert by_doc_id["id_3"]["_rerank_order"] == 0
        assert by_doc_id["id_5"]["_rerank_order"] == 1

        # chunk4 在两个窗口的重叠区（adj_4 被 hit_3 和 hit_5 的上下文都覆盖）
        # adj_4 先被 hit_3 (order=0) 注册，再被 hit_5 (order=1) 尝试覆盖
        # 因 1 > 0，不覆盖，应保留 order=0（最优）
        assert by_doc_id["adj_4"]["_rerank_order"] == 0, f"Expected 0, got {by_doc_id['adj_4']['_rerank_order']}"
