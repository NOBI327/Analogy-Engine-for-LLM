"""Step 4: 候補推論生成 — 構造写像に基づくアイデア生成"""

from __future__ import annotations

import json

from src.clients.llm_client import LLMClient
from src.models import Structure, ScoredAnalogy, CandidateInference

SYSTEM_PROMPT = """\
あなたはGentnerの構造写像理論に基づいて、アナロジーから実用的なアイデアを生成する専門家です。

ベース領域（課題）とターゲット領域（アナロジー元）の構造的対応が示されます。
ターゲット領域の法則・原理をベース領域に転写し、具体的なアイデアを生成してください。

## 出力ルール
- 各アイデアは3段構成:
  1. base_principle: ターゲット領域で機能している法則・原理
  2. application: その原理をベース領域に適用した場合の具体的な解釈
  3. idea: 実行可能な具体的アイデア（1-2文）
- 1つのアナロジーから1〜3個のアイデアを生成
- アイデアは実務的・具体的であること（抽象論は不要）

## 出力形式
```json
[
  {"base_principle": "...", "application": "...", "idea": "..."},
  ...
]
```
JSONのみを返してください。"""


def generate_inferences(
    base: Structure,
    analogy: ScoredAnalogy,
    llm: LLMClient,
) -> list[CandidateInference]:
    """1つのアナロジーから候補推論を生成"""
    prompt = f"""## ベース領域（課題）
{json.dumps(base, ensure_ascii=False, indent=2)}

## ターゲット領域（アナロジー元: {analogy['source']['domain']}）
{json.dumps(analogy['source'], ensure_ascii=False, indent=2)}

## マッチした関係
{json.dumps(analogy['matched_relations'], ensure_ascii=False, indent=2)}

上記の構造的対応に基づいて、ベース領域の課題を解決するアイデアを生成してください。"""

    return llm.ask_json(prompt, system=SYSTEM_PROMPT)


def generate_all_inferences(
    base: Structure,
    ranked_analogies: list[ScoredAnalogy],
    llm: LLMClient,
    verbose: bool = False,
) -> list[tuple[str, list[CandidateInference]]]:
    """全アナロジーから候補推論を生成。(domain, inferences)のリストを返す"""
    results = []
    total = len(ranked_analogies)
    for i, analogy in enumerate(ranked_analogies, 1):
        domain = analogy["source"]["domain"]
        if verbose:
            print(f"  [{i}/{total}] {domain} から推論生成中...", flush=True)
        inferences = generate_inferences(base, analogy, llm)
        results.append((domain, inferences))
    return results
