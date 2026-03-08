"""共有テストfixtures — DI対応のモッククライアントを提供"""

import json
import pytest
import numpy as np
from unittest.mock import MagicMock


@pytest.fixture
def mock_llm():
    """LLMClientのモック。ask / ask_json の戻り値をテスト側で設定可能"""
    client = MagicMock()
    client.ask.return_value = ""
    client.ask_json.return_value = {}
    return client


@pytest.fixture
def mock_embedder():
    """EmbeddingClientのモック。固定ベクトルを返す"""
    client = MagicMock()
    # デフォルト: 384次元のランダムベクトル（MiniLM-L12と同次元）
    client.encode.return_value = np.random.rand(2, 384).astype(np.float32)
    client.similarity.return_value = 0.75
    return client


@pytest.fixture
def sample_structure():
    """テスト用の構造データ（オンボーディング問題の例）"""
    return {
        "domain": "IT企業オンボーディング",
        "entities": [
            {"name": "自動化システム", "attributes": ["新規導入", "全社展開"]},
            {"name": "現場社員", "attributes": ["既存業務に習熟"]},
            {"name": "暗黙知", "attributes": ["非言語化", "属人的"]},
        ],
        "first_order_relations": [
            {"source": "自動化システム", "target": "現場社員", "predicate": "置換しようとする"},
            {"source": "現場社員", "target": "暗黙知", "predicate": "保有する"},
            {"source": "自動化システム", "target": "暗黙知", "predicate": "捕捉できない"},
        ],
        "higher_order_relations": [
            {
                "type": "cause",
                "source": "捕捉できない",
                "target": "置換しようとする",
            },
        ],
    }


@pytest.fixture
def sample_analogy_structure():
    """テスト用のアナロジー候補構造（臓器移植の拒絶反応）"""
    return {
        "domain": "臓器移植",
        "entities": [
            {"name": "移植臓器", "attributes": ["外来", "機能的"]},
            {"name": "免疫系", "attributes": ["防御機構", "自己/非自己判別"]},
            {"name": "生体組織", "attributes": ["既存", "適応済み"]},
        ],
        "first_order_relations": [
            {"source": "移植臓器", "target": "生体組織", "predicate": "置換しようとする"},
            {"source": "免疫系", "target": "生体組織", "predicate": "保護する"},
            {"source": "免疫系", "target": "移植臓器", "predicate": "拒絶する"},
        ],
        "higher_order_relations": [
            {
                "type": "cause",
                "source": "拒絶する",
                "target": "置換しようとする",
            },
        ],
    }
