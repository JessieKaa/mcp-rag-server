"""
重排序模块

提供基于 Cross-encoder 的两阶段检索精排功能。
默认关闭（RERANKER_PROVIDER=none），对存量部署零影响。
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np


class Reranker:
    """重排序抽象基类，定义 rerank 接口。"""

    def rerank(self, query: str, candidates: List[Dict], k: int) -> List[Dict]:
        """
        对候选结果进行重排序。

        Args:
            query: 检索查询
            candidates: 候选结果列表（含 content 字段）
            k: 返回前 k 条结果

        Returns:
            按 rerank 分数降序排列的前 k 条结果，每条含 rerank_score 字段
        """
        raise NotImplementedError


class _LocalCrossEncoderBackend:
    """使用 sentence-transformers CrossEncoder 的本地推理后端。"""

    def __init__(self, model_name: str, logger: logging.Logger):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "RERANKER_PROVIDER=local requires 'sentence-transformers' package. "
                "Install with: pip install mcp-rag-server[local]"
            )
        self.model = CrossEncoder(model_name)
        self._logger = logger

    def score(self, query: str, passages: List[str]) -> np.ndarray:
        pairs = [(query, passage) for passage in passages]
        return self.model.predict(pairs)


class LocalReranker(Reranker):
    """使用本地 Cross-encoder 模型的重排序器。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self.logger = logging.getLogger("reranker")
        self.logger.setLevel(logging.INFO)
        self.logger.info(f"正在加载重排序模型 '{model_name}'...")
        self._backend = _LocalCrossEncoderBackend(model_name, self.logger)
        self.logger.info(f"已加载重排序模型 '{model_name}'")

    def rerank(self, query: str, candidates: List[Dict], k: int) -> List[Dict]:
        if not candidates:
            return []

        passages = [c["content"] for c in candidates]
        scores = self._backend.score(query, passages)

        scored = []
        for i, candidate in enumerate(candidates):
            entry = dict(candidate)
            entry["rerank_score"] = float(scores[i])
            scored.append(entry)

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:k]


def create_reranker_from_env() -> Optional[Reranker]:
    """
    根据 RERANKER_PROVIDER 环境变量返回 Reranker 实例或 None。

    Returns:
        Reranker 实例（启用时）或 None（禁用时）
    """
    provider = os.getenv("RERANKER_PROVIDER", "none").lower()
    if provider == "none":
        return None
    if provider == "local":
        model = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
        return LocalReranker(model_name=model)
    raise ValueError(f"Unknown RERANKER_PROVIDER: {provider}")
