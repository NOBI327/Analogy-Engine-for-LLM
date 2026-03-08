"""Claude Code CLIクライアント — Maxプランの定額トークンを使用

claude -p "prompt" --output-format text をサブプロセスで呼び出す。
LLMClientと同じインターフェース（ask / ask_json）を持つため、DIで差し替え可能。

注意: Claude Codeセッション内からはネスト制限により動作しない。
ターミナルから直接 run.py を実行する場合に使う。
"""

from __future__ import annotations

import json
import os
import subprocess


class ClaudeCodeClient:
    """claude CLIの薄いラッパー。Maxプラン定額で使う"""

    def __init__(self, model: str | None = None, timeout: int = 300):
        self.model = model  # None = CLIデフォルト（Maxプランのモデル）
        self.timeout = timeout

    def ask(self, prompt: str, system: str = "") -> str:
        """テキスト応答を返す"""
        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        cmd = ["claude", "-p", full_prompt, "--output-format", "text", "--max-turns", "1"]
        if self.model:
            cmd.extend(["--model", self.model])

        env = os.environ.copy()
        env.pop("CLAUDECODE", None)  # ネスト制限を回避

        print(f"  [claude CLI] リクエスト送信中...", flush=True)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self.timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"claude CLI がタイムアウトしました（{self.timeout}秒）。"
                "ネットワーク状況を確認するか、再実行してください。"
            )

        if result.returncode != 0:
            raise RuntimeError(
                f"claude CLI failed (exit {result.returncode}): {result.stderr}"
            )

        print(f"  [claude CLI] 応答受信完了", flush=True)
        return result.stdout.strip()

    def ask_json(self, prompt: str, system: str = "") -> dict | list:
        """JSON応答をパースして返す"""
        raw = self.ask(prompt, system)
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        return json.loads(text)
