"""パイプライン統合テスト — 全Stepを通してEnd-to-Endで動作確認"""

import json
from unittest.mock import MagicMock, patch
import numpy as np


def test_pipeline_end_to_end(mock_llm, sample_structure, sample_analogy_structure):
    """パイプライン全体がモックで通ること"""
    from src.pipeline import run_pipeline

    # mock_llm.ask_json の呼び出し順序に応じて異なる値を返す
    call_results = [
        # Step 1: 構造抽出
        sample_structure,
        # Step 2: 近縁探索
        [sample_analogy_structure] * 5,
        # Step 2: 遠方探索
        [{**sample_analogy_structure, "domain": f"遠方{i}"} for i in range(5)],
        # Step 4: 候補推論（ランキング結果の数だけ呼ばれるが、ここでは最大10回分用意）
        *[[{"idea": f"アイデア{i}", "base_principle": f"原理{i}", "application": f"適用{i}"}]
          for i in range(10)],
        # Step 5: クロス立案
        {"summary": "テスト提案", "actions": ["アクション1"], "combined_ideas": ["アイデア1"]},
    ]
    mock_llm.ask_json.side_effect = call_results

    # Embedderモック: 全ペアで中程度のスコアを返す
    mock_embedder = MagicMock()
    def mock_encode(texts):
        n = len(texts)
        vecs = np.random.rand(n, 384).astype(np.float32)
        # 正規化して一定のコサイン類似度を保証
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    mock_embedder.encode.side_effect = mock_encode

    result = run_pipeline("テスト課題", mock_llm, mock_embedder, verbose=False)

    assert "proposal" in result
    assert "idea_bank" in result
    assert "steps" in result
    assert result["proposal"]["summary"] == "テスト提案"
    assert len(result["idea_bank"]) > 0


def test_pipeline_verbose_runs(mock_llm, sample_structure, sample_analogy_structure, capsys):
    """verbose=Trueで中間結果が出力されること"""
    from src.pipeline import run_pipeline

    call_results = [
        sample_structure,
        [sample_analogy_structure] * 5,
        [{**sample_analogy_structure, "domain": f"遠方{i}"} for i in range(5)],
        *[[{"idea": f"アイデア{i}", "base_principle": f"原理{i}", "application": f"適用{i}"}]
          for i in range(10)],
        {"summary": "テスト", "actions": [], "combined_ideas": []},
    ]
    mock_llm.ask_json.side_effect = call_results

    mock_embedder = MagicMock()
    mock_embedder.encode.side_effect = lambda texts: np.random.rand(len(texts), 384).astype(np.float32)

    run_pipeline("テスト課題", mock_llm, mock_embedder, verbose=True)

    captured = capsys.readouterr()
    assert "Step 1" in captured.out
    assert "Step 5" in captured.out
