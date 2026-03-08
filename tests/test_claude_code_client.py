"""ClaudeCodeClientのユニットテスト — subprocess.runをモック"""

import json
from unittest.mock import patch, MagicMock

from src.clients.claude_code_client import ClaudeCodeClient


def test_ask_calls_cli():
    """claude CLIが正しい引数で呼ばれること"""
    client = ClaudeCodeClient()

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "テスト応答"

    with patch("src.clients.claude_code_client.subprocess.run", return_value=mock_result) as mock_run:
        result = client.ask("テストプロンプト")

        assert result == "テスト応答"
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert cmd[0] == "claude"
        assert "-p" in cmd
        assert "--output-format" in cmd
        # CLAUDECODE環境変数が除去されていること
        env = call_args[1]["env"]
        assert "CLAUDECODE" not in env


def test_ask_with_system_prompt():
    """systemプロンプトがpromptに結合されること"""
    client = ClaudeCodeClient()

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "応答"

    with patch("src.clients.claude_code_client.subprocess.run", return_value=mock_result) as mock_run:
        client.ask("ユーザー入力", system="システム指示")
        cmd = mock_run.call_args[0][0]
        prompt_arg = cmd[cmd.index("-p") + 1]
        assert "システム指示" in prompt_arg
        assert "ユーザー入力" in prompt_arg


def test_ask_with_model():
    """モデル指定が--modelフラグとして渡ること"""
    client = ClaudeCodeClient(model="claude-sonnet-4-6")

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "応答"

    with patch("src.clients.claude_code_client.subprocess.run", return_value=mock_result) as mock_run:
        client.ask("テスト")
        cmd = mock_run.call_args[0][0]
        assert "--model" in cmd
        assert "claude-sonnet-4-6" in cmd


def test_ask_raises_on_failure():
    """CLIが非ゼロ終了したらRuntimeError"""
    client = ClaudeCodeClient()

    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "error message"

    with patch("src.clients.claude_code_client.subprocess.run", return_value=mock_result):
        try:
            client.ask("テスト")
            assert False, "例外が発生すべき"
        except RuntimeError as e:
            assert "error message" in str(e)


def test_ask_json_parses_response():
    """JSON応答がパースされること"""
    client = ClaudeCodeClient()

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = '{"key": "value"}'

    with patch("src.clients.claude_code_client.subprocess.run", return_value=mock_result):
        result = client.ask_json("JSON返して")
        assert result == {"key": "value"}


def test_ask_json_strips_codeblock():
    """コードブロック付きJSON応答がパースされること"""
    client = ClaudeCodeClient()

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = '```json\n{"key": "value"}\n```'

    with patch("src.clients.claude_code_client.subprocess.run", return_value=mock_result):
        result = client.ask_json("JSON返して")
        assert result == {"key": "value"}


def test_di_compatibility(mock_llm):
    """LLMClientと同じインターフェースを持つこと（DIで差し替え可能）"""
    client = ClaudeCodeClient()
    # ask, ask_json メソッドが存在すること
    assert hasattr(client, "ask")
    assert hasattr(client, "ask_json")
    assert callable(client.ask)
    assert callable(client.ask_json)
