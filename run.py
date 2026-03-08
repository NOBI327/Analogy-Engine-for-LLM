"""エントリポイント — パイプラインをCLIから実行

使い方:
  # APIキー方式（従量課金）
  python run.py "課題テキスト"

  # Claude Code方式（Maxプラン定額）
  python run.py --claude-code "課題テキスト"
"""

import sys
import json

from src.config import ANTHROPIC_API_KEY
from src.clients.embedding_client import EmbeddingClient
from src.pipeline import run_pipeline


def main():
    args = sys.argv[1:]
    use_claude_code = "--claude-code" in args
    if use_claude_code:
        args.remove("--claude-code")

    if use_claude_code:
        from src.clients.claude_code_client import ClaudeCodeClient
        llm = ClaudeCodeClient()
        print("[Claude Code CLIモード（Maxプラン定額）]")
    else:
        if not ANTHROPIC_API_KEY:
            print("エラー: ANTHROPIC_API_KEY が設定されていません。")
            print("  方法1: .env にAPIキーを設定（従量課金）")
            print("  方法2: --claude-code フラグを使用（Maxプラン定額）")
            sys.exit(1)
        from src.clients.llm_client import LLMClient
        llm = LLMClient()
        print("[APIキーモード（従量課金）]")

    if args:
        challenge = " ".join(args)
    else:
        print("課題を入力してください（Ctrl+Dで送信）:")
        challenge = sys.stdin.read().strip()

    if not challenge:
        print("エラー: 課題が入力されていません。")
        sys.exit(1)

    print(f"\n課題: {challenge}")
    print("パイプラインを実行中...\n")

    embedder = EmbeddingClient()
    result = run_pipeline(challenge, llm, embedder, verbose=True)

    print("\n" + "=" * 60)
    print("最終提案")
    print("=" * 60)
    print(json.dumps(result["proposal"], ensure_ascii=False, indent=2))

    # トレーサビリティ
    print("\n" + "-" * 60)
    print("トレーサビリティ（各アイデアの出自）")
    print("-" * 60)
    for entry in result["idea_bank"].get_ideas_with_origin():
        print(f"  [{entry['_origin_domain']}] {entry['idea']}")


if __name__ == "__main__":
    main()
