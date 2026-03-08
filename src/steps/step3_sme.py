"""Step 3: Semantic SME — sentence-transformersベースの構造マッチング・ランキング"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.clients.embedding_client import EmbeddingClient
from src.models import Structure, ScoredAnalogy
from src.config import SME_SCORE_THRESHOLD


def _relations_to_texts(structure: Structure) -> list[str]:
    """構造内の関係を、埋め込み可能なテキストに変換"""
    texts = []
    for r in structure["first_order_relations"]:
        texts.append(f"{r['source']}が{r['target']}を{r['predicate']}")
    for hr in structure["higher_order_relations"]:
        texts.append(f"{hr['source']}が{hr['target']}を{hr['type']}")
    return texts


def _score_pair(
    base_texts: list[str],
    target_texts: list[str],
    embedder: EmbeddingClient,
) -> tuple[float, list[dict]]:
    """ベースとターゲットの関係テキスト間の排他的マッチングスコアを計算

    ハンガリアン法（scipy.optimize.linear_sum_assignment）で
    全体最適の1対1マッチングを解く。Gentnerの構造写像理論に準拠。
    """
    if not base_texts or not target_texts:
        return 0.0, []

    base_vecs = embedder.encode(base_texts)
    target_vecs = embedder.encode(target_texts)

    import numpy as np
    from scipy.optimize import linear_sum_assignment

    # コサイン類似度行列
    norms_b = np.linalg.norm(base_vecs, axis=1, keepdims=True)
    norms_t = np.linalg.norm(target_vecs, axis=1, keepdims=True)
    sim_matrix = (base_vecs @ target_vecs.T) / (norms_b @ norms_t.T + 1e-8)

    # ハンガリアン法: コスト行列 = 1 - 類似度（最小化問題に変換）
    cost_matrix = 1.0 - sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched = []
    total_score = 0.0
    for i, j in zip(row_ind, col_ind):
        score = float(sim_matrix[i, j])
        if score >= SME_SCORE_THRESHOLD:
            matched.append({
                "base": base_texts[i],
                "target": target_texts[j],
                "score": score,
            })
            total_score += score

    avg_score = total_score / len(base_texts) if base_texts else 0.0
    return avg_score, matched


def rank_analogies(
    base: Structure,
    candidates: list[Structure],
    embedder: EmbeddingClient,
) -> list[ScoredAnalogy]:
    """全候補をスコアリングしてランキング順に返す"""
    base_texts = _relations_to_texts(base)
    results: list[ScoredAnalogy] = []

    for candidate in candidates:
        target_texts = _relations_to_texts(candidate)
        score, matched = _score_pair(base_texts, target_texts, embedder)
        if matched:  # マッチが1つもなければ除外
            results.append({
                "source": candidate,
                "score": score,
                "matched_relations": matched,
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results
