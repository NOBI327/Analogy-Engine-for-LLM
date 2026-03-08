"""Claude APIクライアント — DI対応。テスト時はモックを注入可能"""

from __future__ import annotations

import json
import re

import anthropic

from src.config import ANTHROPIC_API_KEY, LLM_MODEL


class LLMClient:
    """Claude APIの薄いラッパー。全Step共通で使う"""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.model = model or LLM_MODEL
        self._client = anthropic.Anthropic(api_key=api_key or ANTHROPIC_API_KEY)

    def ask(self, prompt: str, system: str = "") -> str:
        """テキスト応答を返す"""
        messages = [{"role": "user", "content": prompt}]
        kwargs = {"model": self.model, "max_tokens": 4096, "messages": messages}
        if system:
            kwargs["system"] = system
        response = self._client.messages.create(**kwargs)
        return response.content[0].text

    def ask_json(self, prompt: str, system: str = "", max_retries: int = 2) -> dict | list:
        """JSON応答をパースして返す（リトライ付き）"""
        last_error = None
        for attempt in range(max_retries + 1):
            if attempt > 0:
                print(f"  [LLM] JSONパース失敗、リトライ {attempt}/{max_retries}...", flush=True)
            raw = self.ask(prompt, system)
            text = self._extract_json(raw)
            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                last_error = e
        raise RuntimeError(
            f"JSONパースに{max_retries + 1}回失敗しました: {last_error}\n応答テキスト: {raw[:500]}"
        )

    @staticmethod
    def _extract_json(raw: str) -> str:
        """LLM応答からJSON部分を抽出する"""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass
        match = re.search(r'(\[[\s\S]*\]|\{[\s\S]*\})', text)
        if match:
            return match.group(1)
        return text
