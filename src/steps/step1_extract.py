"""Step 1: 構造抽出 — 課題テキストからGentner形式の構造表現を抽出"""

from __future__ import annotations

from src.clients.llm_client import LLMClient
from src.models import Structure

SYSTEM_PROMPT = """\
あなたはGentnerの構造写像理論（Structure-Mapping Theory）の専門家です。
与えられたテキストから、以下の構造を抽出してJSON形式で返してください。

## 抽出ルール
- **entities**: テキストに登場する主要なオブジェクト。nameとattributes（性質のリスト）を持つ
- **first_order_relations**: オブジェクト間の関係。source, target, predicateを持つ
  - **predicateはドメイン固有の生々しい動詞を保持すること**（「影響する」「関係する」等の抽象動詞に変換しない）
  - 例: 「貪食する」「証券化する」「拒絶する」「形骸化する」
- **higher_order_relations**: 関係間の関係（因果、依存、制約など）。type, source, targetを持つ
  - sourceとtargetはfirst_order_relationsのpredicateまたは別のhigher_order_relation

## 体系性原理
孤立した関係ではなく、高次の関係で結ばれた関係のシステムを優先的に抽出してください。

## 出力形式（厳密に従うこと）
```json
{
  "domain": "領域名",
  "entities": [{"name": "...", "attributes": ["...", "..."]}],
  "first_order_relations": [{"source": "...", "target": "...", "predicate": "..."}],
  "higher_order_relations": [{"type": "cause|depend|constrain|enable", "source": "...", "target": "..."}]
}
```

JSONのみを返してください。説明文は不要です。"""


def extract_structure(challenge: str, llm: LLMClient) -> Structure:
    """課題テキストからGentner形式の構造を抽出する

    Args:
        challenge: 課題の自然言語記述
        llm: LLMClient（DI）
    Returns:
        Structure型のdict
    """
    result = llm.ask_json(challenge, system=SYSTEM_PROMPT)
    return result
