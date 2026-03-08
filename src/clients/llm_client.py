"""Claude APIクライアント — DI対応。テスト時はモックを注入可能"""

from __future__ import annotations

import json
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

    def ask_json(self, prompt: str, system: str = "") -> dict | list:
        """JSON応答をパースして返す。パース失敗時は例外"""
        raw = self.ask(prompt, system)
        # コードブロックで囲まれている場合を処理
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]  # ```json を除去
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        return json.loads(text)
