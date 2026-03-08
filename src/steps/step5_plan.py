"""Step 5: クロス立案 — 出自を忘れて実効性のみで最終提案を構成"""

from __future__ import annotations

import json

from src.clients.llm_client import LLMClient
from src.models import Proposal

SYSTEM_PROMPT = """\
あなたは実務的なソリューション設計の専門家です。

以下のアイデアリストが与えられます。これらのアイデアの出自や発想元は一切気にしないでください。
純粋に「この課題に対して最も実効性の高い提案」を構成してください。

## ルール
- 複数のアイデアを組み合わせてもよいし、一部だけ採用してもよい
- 実行可能性を最優先する（理論的に美しいが実行不能な提案は不要）
- 具体的なアクション項目を含めること
- 提案は1つに絞る（複数案の併記ではなく、最善の1案を構成する）

## 出力形式
```json
{
  "summary": "提案の要約（2-3文）",
  "actions": ["具体的アクション1", "具体的アクション2", ...],
  "combined_ideas": ["採用したアイデアの要約1", "採用したアイデアの要約2", ...]
}
```
JSONのみを返してください。"""


def cross_plan(
    challenge: str,
    ideas: list[dict],
    llm: LLMClient,
) -> Proposal:
    """出自剥離済みアイデアから最終提案を構成"""
    prompt = f"""## 課題
{challenge}

## アイデアリスト（{len(ideas)}件）
{json.dumps(ideas, ensure_ascii=False, indent=2)}

上記のアイデアを参考に、この課題に対する最も実効性の高いソリューション提案を構成してください。"""

    return llm.ask_json(prompt, system=SYSTEM_PROMPT)
