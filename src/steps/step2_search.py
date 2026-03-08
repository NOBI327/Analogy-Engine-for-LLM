"""Step 2: 類似構造探索 — 近縁5 + 遠方5の構造的類似物を探索"""

from __future__ import annotations

from src.clients.llm_client import LLMClient
from src.models import Structure

NEAR_SYSTEM_PROMPT = """\
あなたはGentnerの構造写像理論の専門家です。
与えられた構造表現に対して、**同分野・隣接分野**から構造的に類似した事例を5つ生成してください。

## ルール
- 表面的な類似（同じ単語が出てくる）ではなく、**関係の構造が類似した**事例を選ぶ
- 各事例はGentner形式のJSON構造として出力する
- predicateはドメイン固有の生々しい動詞を保持すること

## 出力形式
```json
[
  {"domain": "...", "entities": [...], "first_order_relations": [...], "higher_order_relations": [...]},
  ...
]
```
5つの構造をJSON配列で返してください。説明文は不要です。"""

FAR_SYSTEM_PROMPT = """\
あなたはGentnerの構造写像理論の専門家です。
与えられた構造表現に対して、**表面的に最も遠い分野**から構造的に類似した事例を5つ生成してください。

## ルール
- 表面的に無関係な分野を優先せよ（生物学、物理学、音楽、料理、軍事、宗教、スポーツ、etc.）
- **オブジェクトや属性は全く異なるが、関係の構造（因果、依存、制約のパターン）が類似している**事例を選ぶ
- predicateはドメイン固有の生々しい動詞を保持すること
- 「新大陸の発見」を目指す：人間が通常思いつかない組み合わせを積極的に探す

## 出力形式
```json
[
  {"domain": "...", "entities": [...], "first_order_relations": [...], "higher_order_relations": [...]},
  ...
]
```
5つの構造をJSON配列で返してください。説明文は不要です。"""


def search_near(structure: Structure, llm: LLMClient) -> list[Structure]:
    """近縁領域から構造的類似物を5つ探索"""
    import json
    prompt = f"以下の構造と類似した事例を近縁分野から探してください:\n\n{json.dumps(structure, ensure_ascii=False, indent=2)}"
    return llm.ask_json(prompt, system=NEAR_SYSTEM_PROMPT)


def search_far(structure: Structure, llm: LLMClient) -> list[Structure]:
    """遠方領域から構造的類似物を5つ探索"""
    import json
    prompt = f"以下の構造と類似した事例を、表面的に最も遠い分野から探してください:\n\n{json.dumps(structure, ensure_ascii=False, indent=2)}"
    return llm.ask_json(prompt, system=FAR_SYSTEM_PROMPT)
