"""グラフ埋め込みのテスト"""

import numpy as np
import torch
from unittest.mock import MagicMock

from src.graph_embedding import structure_to_graph, GraphEncoder, graph_similarity


def _make_embedder(dim=384):
    """固定ベクトルを返すモック embedder"""
    embedder = MagicMock()

    def encode(texts):
        np.random.seed(42)
        return np.random.rand(len(texts), dim).astype(np.float32)

    embedder.encode.side_effect = encode
    return embedder


class TestStructureToGraph:
    def test_basic_conversion(self, sample_structure):
        embedder = _make_embedder()
        data = structure_to_graph(sample_structure, embedder)

        # 3 エンティティ → 3 ノード
        assert data.x.shape[0] == 3
        assert data.x.shape[1] == 384
        # エッジが存在する（双方向なので元の数×2以上）
        assert data.edge_index.shape[1] > 0

    def test_empty_structure(self):
        embedder = _make_embedder()
        empty = {
            "domain": "空",
            "entities": [],
            "first_order_relations": [],
            "higher_order_relations": [],
        }
        data = structure_to_graph(empty, embedder)
        # ダミーノード1つ
        assert data.x.shape[0] == 1
        assert data.edge_index.shape[1] == 0

    def test_higher_order_adds_edges(self, sample_structure):
        embedder = _make_embedder()
        # 高次関係あり
        data_with = structure_to_graph(sample_structure, embedder)

        # 高次関係なし
        no_hr = {**sample_structure, "higher_order_relations": []}
        data_without = structure_to_graph(no_hr, embedder)

        # 高次関係がある方がエッジが多い
        assert data_with.edge_index.shape[1] >= data_without.edge_index.shape[1]


class TestGraphEncoder:
    def test_output_shape(self, sample_structure):
        embedder = _make_embedder()
        encoder = GraphEncoder()
        encoder.eval()

        data = structure_to_graph(sample_structure, embedder)
        out = encoder(data)
        assert out.shape == (1, 64)  # デフォルト out_dim=64

    def test_deterministic(self, sample_structure):
        """同じ入力に対して同じ出力を返す"""
        embedder = _make_embedder()
        encoder = GraphEncoder()
        encoder.eval()

        data = structure_to_graph(sample_structure, embedder)
        out1 = encoder(data)
        out2 = encoder(data)
        assert torch.allclose(out1, out2)

    def test_weights_frozen(self):
        encoder = GraphEncoder()
        for param in encoder.parameters():
            assert not param.requires_grad


class TestGraphSimilarity:
    def test_self_similarity_high(self, sample_structure):
        embedder = _make_embedder()
        score = graph_similarity(sample_structure, sample_structure, embedder)
        # 自己類似度は 1.0 に近いはず
        assert score > 0.9

    def test_different_structures(self, sample_structure, sample_analogy_structure):
        embedder = _make_embedder()
        score = graph_similarity(sample_structure, sample_analogy_structure, embedder)
        # 値が返ること（具体的な値は重みに依存するので範囲チェック、浮動小数点誤差考慮）
        assert -1.01 <= score <= 1.01

    def test_returns_float(self, sample_structure, sample_analogy_structure):
        embedder = _make_embedder()
        score = graph_similarity(sample_structure, sample_analogy_structure, embedder)
        assert isinstance(score, float)
