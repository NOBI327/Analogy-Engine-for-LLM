"""Step 3: Semantic SME — テキスト埋め込み + グラフ構造埋め込みの複合ランキング

テキスト: 関係テキストの埋め込み × ハンガリアン法（関係レベルの類似度）
グラフ:   GCN でグラフ全体の構造的特徴を埋め込み（接続パターンの類似度）
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.clients.embedding_client import EmbeddingClient
from src.models import Structure, ScoredAnalogy
from src.config import SME_SCORE_THRESHOLD, GRAPH_WEIGHT


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
    graph_weight: float | None = None,
) -> list[ScoredAnalogy]:
    """全候補をスコアリングしてランキング順に返す

    複合スコア = (1 - graph_weight) × テキストスコア + graph_weight × グラフスコア
    graph_weight=0 でテキストのみ（従来動作）。
    """
    if graph_weight is None:
        graph_weight = GRAPH_WEIGHT

    base_texts = _relations_to_texts(base)
    results: list[ScoredAnalogy] = []

    # グラフ類似度を一括計算（weight > 0 の場合のみ）
    graph_scores: list[float] = []
    if graph_weight > 0:
        from src.graph_embedding import graph_similarity
        graph_scores = [
            graph_similarity(base, cand, embedder) for cand in candidates
        ]
    else:
        graph_scores = [0.0] * len(candidates)

    for i, candidate in enumerate(candidates):
        target_texts = _relations_to_texts(candidate)
        text_score, matched = _score_pair(base_texts, target_texts, embedder)
        if matched:  # マッチが1つもなければ除外
            combined = (1 - graph_weight) * text_score + graph_weight * graph_scores[i]
            results.append({
                "source": candidate,
                "score": combined,
                "matched_relations": matched,
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results
