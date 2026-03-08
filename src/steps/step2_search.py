"""Step 2: 類似構造探索 — 近縁 + 遠方の構造的類似物を探索"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.clients.llm_client import LLMClient
from src.models import Structure
from src.config import NEAR_COUNT, FAR_COUNT


def _make_system_prompt(mode: str, count: int) -> str:
    """探索モードと件数に応じたシステムプロンプトを生成"""
    if mode == "near":
        domain_instruction = "**同分野・隣接分野**から"
        extra_rules = ""
    else:
        domain_instruction = "**表面的に最も遠い分野**から"
        extra_rules = """- 表面的に無関係な分野を優先せよ（生物学、物理学、音楽、料理、軍事、宗教、スポーツ、etc.）
- 「新大陸の発見」を目指す：人間が通常思いつかない組み合わせを積極的に探す
"""

    return f"""\
あなたはGentnerの構造写像理論の専門家です。
与えられた構造表現に対して、{domain_instruction}構造的に類似した事例を{count}つ生成してください。

## ルール
- 表面的な類似（同じ単語が出てくる）ではなく、**関係の構造が類似した**事例を選ぶ
{extra_rules}- 各事例はGentner形式のJSON構造として出力する
- predicateはドメイン固有の生々しい動詞を保持すること

## 出力形式
```json
[
  {{"domain": "...", "entities": [...], "first_order_relations": [...], "higher_order_relations": [...]}},
  ...
]
```
{count}つの構造をJSON配列で返してください。説明文は不要です。"""


def search_near(structure: Structure, llm: LLMClient, count: int | None = None) -> list[Structure]:
    """近縁領域から構造的類似物を探索"""
    import json
    n = count if count is not None else NEAR_COUNT
    prompt = f"以下の構造と類似した事例を近縁分野から探してください:\n\n{json.dumps(structure, ensure_ascii=False, indent=2)}"
    return llm.ask_json(prompt, system=_make_system_prompt("near", n))


def search_far(structure: Structure, llm: LLMClient, count: int | None = None) -> list[Structure]:
    """遠方領域から構造的類似物を探索"""
    import json
    n = count if count is not None else FAR_COUNT
    prompt = f"以下の構造と類似した事例を、表面的に最も遠い分野から探してください:\n\n{json.dumps(structure, ensure_ascii=False, indent=2)}"
    return llm.ask_json(prompt, system=_make_system_prompt("far", n))
