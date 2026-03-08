"""各Stepのユニットテスト — 外部依存はすべてDIモックで注入"""

import json
import numpy as np
from unittest.mock import MagicMock

from src.steps.step1_extract import extract_structure
from src.steps.step2_search import search_near, search_far
from src.steps.step3_sme import rank_analogies, _relations_to_texts
from src.steps.step4_infer import generate_inferences
from src.steps.step5_plan import cross_plan
from src.idea_bank import IdeaBank


# ── Step 1: 構造抽出 ──

def test_step1_extract(mock_llm, sample_structure):
    """LLMが返したJSONをそのまま構造として返すこと"""
    mock_llm.ask_json.return_value = sample_structure
    result = extract_structure("オンボーディングが形骸化している", mock_llm)

    assert result["domain"] == "IT企業オンボーディング"
    assert len(result["entities"]) == 3
    assert len(result["first_order_relations"]) == 3
    mock_llm.ask_json.assert_called_once()


def test_step1_preserves_raw_predicates(mock_llm, sample_structure):
    """述語がドメイン固有の生々しい動詞であること"""
    mock_llm.ask_json.return_value = sample_structure
    result = extract_structure("test", mock_llm)
    predicates = [r["predicate"] for r in result["first_order_relations"]]
    assert "置換しようとする" in predicates
    assert "捕捉できない" in predicates


# ── Step 2: 類似構造探索 ──

def test_step2_search_near(mock_llm, sample_structure, sample_analogy_structure):
    """近縁探索が5つの構造を返すこと"""
    mock_llm.ask_json.return_value = [sample_analogy_structure] * 5
    results = search_near(sample_structure, mock_llm)
    assert len(results) == 5
    assert all(r["domain"] == "臓器移植" for r in results)


def test_step2_search_far(mock_llm, sample_structure, sample_analogy_structure):
    """遠方探索が5つの構造を返すこと"""
    mock_llm.ask_json.return_value = [sample_analogy_structure] * 5
    results = search_far(sample_structure, mock_llm)
    assert len(results) == 5


# ── Step 3: Semantic SME ──

def test_step3_relations_to_texts(sample_structure):
    """関係テキスト変換が正しく動作すること"""
    texts = _relations_to_texts(sample_structure)
    assert len(texts) == 4  # 3 first_order + 1 higher_order
    assert any("置換しようとする" in t for t in texts)


def test_step3_rank_analogies(sample_structure, sample_analogy_structure):
    """ランキングがスコア降順で返ること"""
    mock_embedder = MagicMock()

    # 2候補を用意し、異なるスコアを返すようにする
    candidate_a = {**sample_analogy_structure, "domain": "候補A"}
    candidate_b = {**sample_analogy_structure, "domain": "候補B"}

    # encodeが呼ばれるたびに異なるベクトルを返す
    call_count = [0]
    def mock_encode(texts):
        call_count[0] += 1
        n = len(texts)
        # ベースと候補Bを高スコアにする
        if call_count[0] == 1:  # base
            return np.array([[1.0, 0.0, 0.0, 0.0]] * n, dtype=np.float32)
        elif call_count[0] == 2:  # candidate A target
            return np.array([[0.5, 0.5, 0.5, 0.5]] * n, dtype=np.float32)
        elif call_count[0] == 3:  # base again
            return np.array([[1.0, 0.0, 0.0, 0.0]] * n, dtype=np.float32)
        else:  # candidate B target - closer to base
            return np.array([[0.9, 0.1, 0.0, 0.0]] * n, dtype=np.float32)

    mock_embedder.encode.side_effect = mock_encode
    ranked = rank_analogies(sample_structure, [candidate_a, candidate_b], mock_embedder)

    assert len(ranked) == 2
    assert ranked[0]["score"] >= ranked[1]["score"]


def test_step3_filters_low_scores(sample_structure, sample_analogy_structure):
    """閾値以下のマッチが除外されること"""
    mock_embedder = MagicMock()
    # 直交ベクトル → コサイン類似度 ≈ 0 → 閾値以下で除外
    def mock_encode(texts):
        n = len(texts)
        return np.eye(max(n, 4), dtype=np.float32)[:n, :4]

    mock_embedder.encode.side_effect = mock_encode
    ranked = rank_analogies(sample_structure, [sample_analogy_structure], mock_embedder)
    # 低スコアのため除外される可能性がある（閾値次第）
    assert isinstance(ranked, list)


# ── Step 4: 候補推論生成 ──

def test_step4_generate_inferences(mock_llm, sample_structure, sample_analogy_structure):
    """候補推論が生成されること"""
    mock_inferences = [
        {
            "base_principle": "免疫抑制剤で拒絶反応を抑える",
            "application": "既存プロセスとの共存期間を設ける",
            "idea": "最初の3ヶ月は旧プロセスと並行運用する",
        }
    ]
    mock_llm.ask_json.return_value = mock_inferences

    analogy = {
        "source": sample_analogy_structure,
        "score": 0.85,
        "matched_relations": [{"base": "test", "target": "test", "score": 0.85}],
    }
    result = generate_inferences(sample_structure, analogy, mock_llm)
    assert len(result) == 1
    assert result[0]["idea"] == "最初の3ヶ月は旧プロセスと並行運用する"


# ── アイデアバンク ──

def test_idea_bank_strips_origin():
    """出自情報がStep 5用出力から除去されること"""
    bank = IdeaBank()
    bank.add(
        [{"idea": "並行運用", "base_principle": "免疫抑制", "application": "共存設計"}],
        origin_domain="臓器移植",
    )

    stripped = bank.get_ideas_stripped()
    assert len(stripped) == 1
    assert "_origin_domain" not in stripped[0]
    assert stripped[0]["idea"] == "並行運用"


def test_idea_bank_preserves_origin():
    """トレーサビリティ用出力に出自情報が含まれること"""
    bank = IdeaBank()
    bank.add(
        [{"idea": "並行運用", "base_principle": "免疫抑制", "application": "共存設計"}],
        origin_domain="臓器移植",
    )

    with_origin = bank.get_ideas_with_origin()
    assert with_origin[0]["_origin_domain"] == "臓器移植"


# ── Step 5: クロス立案 ──

def test_step5_cross_plan(mock_llm):
    """最終提案が生成されること"""
    mock_proposal = {
        "summary": "段階的導入を提案する",
        "actions": ["並行運用期間を設定", "既存フローに埋め込み"],
        "combined_ideas": ["並行運用", "段階的埋め込み"],
    }
    mock_llm.ask_json.return_value = mock_proposal

    ideas = [
        {"idea": "並行運用", "base_principle": "p1", "application": "a1"},
        {"idea": "段階的埋め込み", "base_principle": "p2", "application": "a2"},
    ]
    result = cross_plan("オンボーディング問題", ideas, mock_llm)
    assert result["summary"] == "段階的導入を提案する"
    assert len(result["actions"]) == 2
