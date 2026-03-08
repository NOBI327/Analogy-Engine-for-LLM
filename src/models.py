"""データ構造 — 帰納的に作る。まずdictで動かし、パターンが見えたら型を足す"""

from __future__ import annotations
from typing import TypedDict


class Entity(TypedDict):
    name: str
    attributes: list[str]


class Relation(TypedDict):
    source: str
    target: str
    predicate: str  # ドメイン固有の生々しい動詞を保持


class HigherOrderRelation(TypedDict):
    type: str       # cause, depend, constrain, etc.
    source: str     # 一次関係のpredicateまたは別の高次関係
    target: str


class Structure(TypedDict):
    """Step 1の出力: Gentner形式の構造表現"""
    domain: str
    entities: list[Entity]
    first_order_relations: list[Relation]
    higher_order_relations: list[HigherOrderRelation]


class ScoredAnalogy(TypedDict):
    """Step 3の出力: スコア付きアナロジー"""
    source: Structure
    score: float
    matched_relations: list[dict]


class CandidateInference(TypedDict):
    """Step 4の出力: 候補推論"""
    idea: str
    base_principle: str
    application: str


class IdeaBankEntry(TypedDict):
    """アイデアバンクの1エントリ（出自情報は内部保持、外部には非公開）"""
    idea: str
    base_principle: str
    application: str
    _origin_domain: str  # 内部トレーサビリティ用


class Proposal(TypedDict):
    """Step 5の出力: 最終提案"""
    summary: str
    actions: list[str]
    combined_ideas: list[str]
