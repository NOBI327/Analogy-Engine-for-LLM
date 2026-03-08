"""グラフ埋め込み — Structure を PyG グラフに変換し、構造レベルの類似度を算出

テキスト埋め込みでは捉えきれないグラフ構造（接続パターン、高次関係の連鎖）を
GCN で伝播させて構造的特徴量を得る。学習不要 — 固定重みの GCN でも
ノード特徴量 (sentence-transformers) × 構造伝播で意味のある埋め込みが出る。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

if TYPE_CHECKING:
    from src.clients.embedding_client import EmbeddingClient
from src.models import Structure


def structure_to_graph(structure: Structure, embedder: EmbeddingClient) -> Data:
    """Structure を PyG の Data オブジェクトに変換する

    ノード: エンティティ（特徴量 = sentence-transformers 埋め込み）
    エッジ: first_order_relations + higher_order_relations から推定
    """
    entities = structure.get("entities", [])
    if not entities:
        # 空グラフ: ダミーノード1つ
        return Data(
            x=torch.zeros(1, 384),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
        )

    # エンティティ名 → インデックス
    name_to_idx = {e["name"]: i for i, e in enumerate(entities)}

    # ノード特徴量: "名前: 属性1, 属性2" のテキスト埋め込み
    node_texts = []
    for e in entities:
        attrs = ", ".join(e.get("attributes", []))
        node_texts.append(f"{e['name']}: {attrs}" if attrs else e["name"])
    node_vecs = embedder.encode(node_texts)
    x = torch.tensor(node_vecs, dtype=torch.float32)

    # エッジ構築
    src_indices = []
    dst_indices = []

    for r in structure.get("first_order_relations", []):
        s = name_to_idx.get(r["source"])
        t = name_to_idx.get(r["target"])
        if s is not None and t is not None:
            # 双方向（無向グラフとして扱い、GCN の伝播を対称にする）
            src_indices.extend([s, t])
            dst_indices.extend([t, s])

    # 高次関係: predicate 名から関連エンティティを逆引きしてエッジ追加
    predicate_to_entities = {}
    for r in structure.get("first_order_relations", []):
        s = name_to_idx.get(r["source"])
        t = name_to_idx.get(r["target"])
        if s is not None and t is not None:
            predicate_to_entities[r["predicate"]] = (s, t)

    for hr in structure.get("higher_order_relations", []):
        src_pair = predicate_to_entities.get(hr["source"])
        tgt_pair = predicate_to_entities.get(hr["target"])
        if src_pair and tgt_pair:
            # 高次関係で結ばれた関係のエンティティ間にエッジを張る
            for s in src_pair:
                for t in tgt_pair:
                    if s != t:
                        src_indices.extend([s, t])
                        dst_indices.extend([t, s])

    if src_indices:
        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


class GraphEncoder(nn.Module):
    """固定重み 2 層 GCN — 構造伝播で位相的特徴を混ぜ込む

    学習しない。Xavier 初期化の固定重みで、同じ構造パターンには
    類似した埋め込みを出力する（random features の原理）。
    """

    def __init__(self, in_dim: int = 384, hidden_dim: int = 128, out_dim: int = 64):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.relu = nn.ReLU()

        # 重みを固定（学習しない）
        for param in self.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def forward(self, data: Data) -> torch.Tensor:
        """グラフレベル埋め込みを返す (1, out_dim)"""
        x = data.x
        edge_index = data.edge_index

        x = self.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        # global mean pooling — バッチなしなので全ノード平均
        batch = torch.zeros(x.size(0), dtype=torch.long)
        return global_mean_pool(x, batch)  # (1, out_dim)


# シングルトン — 毎回再生成しない
_encoder: GraphEncoder | None = None


def _get_encoder() -> GraphEncoder:
    global _encoder
    if _encoder is None:
        _encoder = GraphEncoder()
        _encoder.eval()
    return _encoder


def graph_similarity(
    base: Structure,
    candidate: Structure,
    embedder: EmbeddingClient,
) -> float:
    """2 つの Structure 間のグラフ構造類似度を返す (0.0〜1.0)"""
    encoder = _get_encoder()

    g_base = structure_to_graph(base, embedder)
    g_cand = structure_to_graph(candidate, embedder)

    emb_base = encoder(g_base).squeeze(0)   # (out_dim,)
    emb_cand = encoder(g_cand).squeeze(0)

    cos = nn.functional.cosine_similarity(emb_base.unsqueeze(0), emb_cand.unsqueeze(0))
    return float(cos.item())
