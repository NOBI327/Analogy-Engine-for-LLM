"""Step 3: Semantic SME — sentence-transformersベースの構造マッチング・ランキング"""

from __future__ import annotations

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
    """ベースとターゲットの関係テキスト間のマッチングスコアを計算

    現実装: Greedy Matching（1対多）。Phase 2でハンガリアン法に置換予定。
    """
    if not base_texts or not target_texts:
        return 0.0, []

    base_vecs = embedder.encode(base_texts)
    target_vecs = embedder.encode(target_texts)

    import numpy as np

    # コサイン類似度行列
    norms_b = np.linalg.norm(base_vecs, axis=1, keepdims=True)
    norms_t = np.linalg.norm(target_vecs, axis=1, keepdims=True)
    sim_matrix = (base_vecs @ target_vecs.T) / (norms_b @ norms_t.T + 1e-8)

    matched = []
    total_score = 0.0
    for i, base_text in enumerate(base_texts):
        best_j = int(np.argmax(sim_matrix[i]))
        best_score = float(sim_matrix[i, best_j])
        if best_score >= SME_SCORE_THRESHOLD:
            matched.append({
                "base": base_text,
                "target": target_texts[best_j],
                "score": best_score,
            })
            total_score += best_score

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
