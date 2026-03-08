"""アイデアバンク — 出自情報を剥離してアイデアをフラットに格納

設計原則: アナロジーは発見のツールであり、正当化のツールにしない。
出自に引きずられると「魅了（フェーズ2）」に入るリスクがある。
内部にはトレーサビリティ用の出自情報を保持するが、Step 5への入力時には除去する。
"""

from __future__ import annotations

from src.models import CandidateInference, IdeaBankEntry


class IdeaBank:
    def __init__(self):
        self._entries: list[IdeaBankEntry] = []

    def add(self, inferences: list[CandidateInference], origin_domain: str) -> None:
        """候補推論をバンクに追加。出自情報は内部保持のみ"""
        for inf in inferences:
            self._entries.append({
                "idea": inf["idea"],
                "base_principle": inf["base_principle"],
                "application": inf["application"],
                "_origin_domain": origin_domain,
            })

    def get_ideas_stripped(self) -> list[dict]:
        """Step 5用: 出自情報を剥離したアイデアリストを返す"""
        return [
            {
                "idea": e["idea"],
                "base_principle": e["base_principle"],
                "application": e["application"],
            }
            for e in self._entries
        ]

    def get_ideas_with_origin(self) -> list[IdeaBankEntry]:
        """トレーサビリティ用: 出自情報付きの全エントリを返す"""
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)
