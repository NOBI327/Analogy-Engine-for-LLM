"""sentence-transformers埋め込みクライアント — DI対応"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.config import EMBEDDING_MODEL


class EmbeddingClient:
    """sentence-transformersの薄いラッパー。Semantic SME (Step 3) で使う"""

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name or EMBEDDING_MODEL)

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        """テキストリストをベクトル化して返す"""
        return self._model.encode(texts, convert_to_numpy=True)

    def similarity(self, text_a: str, text_b: str) -> float:
        """2テキスト間のコサイン類似度を返す"""
        vecs = self.encode([text_a, text_b])
        cos = np.dot(vecs[0], vecs[1]) / (
            np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1])
        )
        return float(cos)
