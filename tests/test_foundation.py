"""基盤テスト — config, models, クライアントDIの動作確認"""

import json
from unittest.mock import MagicMock, patch


def test_config_defaults():
    from src.config import LLM_MODEL, EMBEDDING_MODEL, SME_SCORE_THRESHOLD
    assert LLM_MODEL  # 空でない
    assert EMBEDDING_MODEL == "paraphrase-multilingual-MiniLM-L12-v2"
    assert 0 < SME_SCORE_THRESHOLD < 1


def test_structure_type(sample_structure):
    """sample_structure fixtureがGentner形式を満たしているか"""
    assert "domain" in sample_structure
    assert len(sample_structure["entities"]) > 0
    assert len(sample_structure["first_order_relations"]) > 0
    assert len(sample_structure["higher_order_relations"]) > 0
    # 述語がドメイン固有の生々しい動詞であること（抽象動詞でないこと）
    predicates = [r["predicate"] for r in sample_structure["first_order_relations"]]
    abstract_verbs = {"cause", "affect", "relate"}
    for p in predicates:
        assert p not in abstract_verbs, f"述語 '{p}' が抽象的すぎる"


def test_mock_llm_ask(mock_llm):
    """LLMモックのDI動作確認"""
    mock_llm.ask.return_value = "テスト応答"
    result = mock_llm.ask("テストプロンプト")
    assert result == "テスト応答"
    mock_llm.ask.assert_called_once_with("テストプロンプト")


def test_mock_llm_ask_json(mock_llm):
    """LLMモックのJSON応答テスト"""
    expected = {"domain": "test", "entities": []}
    mock_llm.ask_json.return_value = expected
    result = mock_llm.ask_json("JSON返して")
    assert result == expected


def test_mock_embedder_similarity(mock_embedder):
    """EmbedderモックのDI動作確認"""
    score = mock_embedder.similarity("テストA", "テストB")
    assert isinstance(score, float)
    assert 0 <= score <= 1


def test_llm_client_ask_json_strips_codeblock():
    """LLMClient.ask_jsonがコードブロックを除去してパースできるか"""
    from src.clients.llm_client import LLMClient

    client = LLMClient.__new__(LLMClient)
    client.model = "test"
    client._client = MagicMock()

    # Claude APIがコードブロック付きで返す場合をシミュレート
    raw_response = '```json\n{"key": "value"}\n```'
    mock_content = MagicMock()
    mock_content.text = raw_response
    client._client.messages.create.return_value = MagicMock(content=[mock_content])

    result = client.ask_json("test")
    assert result == {"key": "value"}
