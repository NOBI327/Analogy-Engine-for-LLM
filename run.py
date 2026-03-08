"""エントリポイント — パイプラインをCLIから実行

使い方:
  # APIキー方式（従量課金）
  python run.py "課題テキスト"

  # Claude Code方式（Maxプラン定額）
  python run.py --claude-code "課題テキスト"
"""

import sys
import json

from src.db import init_db, load_runs, load_run, save_feedback, load_feedback_summary


def show_history():
    """過去の実行結果一覧を表示"""
    init_db()
    runs = load_runs()
    if not runs:
        print("実行履歴がありません。")
        return
    print(f"{'ID':>4}  {'日時':<25}  課題")
    print("-" * 70)
    for r in runs:
        print(f"{r['id']:>4}  {r['created_at'][:19]:<25}  {r['challenge'][:40]}")


def show_run_detail(run_id: int):
    """特定の実行結果を表示"""
    init_db()
    result = load_run(run_id)
    if not result:
        print(f"Run ID {run_id} が見つかりません。")
        return
    print(f"課題: {result['challenge']}")
    print(f"日時: {result['created_at']}")
    print()
    if result["proposal"]:
        print(json.dumps(result["proposal"], ensure_ascii=False, indent=2))
    print()
    print("アイデア出自:")
    for idea in result["ideas"]:
        print(f"  [{idea['_origin_domain']}] {idea['idea']}")


def rate_run(run_id: int, score: int, comment: str = ""):
    """実行結果を評価する"""
    init_db()
    try:
        fb_id = save_feedback(run_id, score, comment)
        print(f"フィードバックを記録しました (ID: {fb_id}, Run: {run_id}, スコア: {score}/5)")
        if comment:
            print(f"  コメント: {comment}")
    except ValueError as e:
        print(f"エラー: {e}")
        sys.exit(1)


def show_insights():
    """フィードバック統計を表示"""
    init_db()
    summary = load_feedback_summary()

    if not summary["total_feedback"]:
        print("フィードバックがまだありません。")
        print("  使い方: python run.py --rate <run_id> <1-5> [コメント]")
        return

    print(f"フィードバック件数: {summary['total_feedback']}")
    print(f"平均スコア: {summary['avg_score']}/5")

    if summary["score_distribution"]:
        print("\nスコア分布:")
        for score in range(1, 6):
            count = summary["score_distribution"].get(score, 0)
            bar = "█" * count
            print(f"  {score}: {bar} ({count})")

    if summary["domain_stats"]:
        print(f"\n{'ドメイン':<20} {'回数':>4} {'平均スコア':>8}")
        print("-" * 36)
        for d in summary["domain_stats"]:
            print(f"  {d['domain']:<18} {d['run_count']:>4} {d['avg_score']:>8.1f}")


def main():
    args = sys.argv[1:]

    # --rate <run_id> <score> [comment]: 評価を記録
    if "--rate" in args:
        idx = args.index("--rate")
        rate_args = args[idx + 1:]
        if len(rate_args) < 2:
            print("使い方: python run.py --rate <run_id> <1-5> [コメント]")
            sys.exit(1)
        run_id = int(rate_args[0])
        score = int(rate_args[1])
        comment = " ".join(rate_args[2:]) if len(rate_args) > 2 else ""
        rate_run(run_id, score, comment)
        return

    # --insights: フィードバック統計
    if "--insights" in args:
        show_insights()
        return

    # --history: 過去の実行一覧
    if "--history" in args:
        args.remove("--history")
        if args:
            show_run_detail(int(args[0]))
        else:
            show_history()
        return

    # ここから先はパイプライン実行なので重い依存をインポート
    from src.config import ANTHROPIC_API_KEY
    from src.clients.embedding_client import EmbeddingClient
    from src.pipeline import run_pipeline

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

    from src.clients.memory_client import create_memory_client
    embedder = EmbeddingClient()
    memory = create_memory_client()
    result = run_pipeline(challenge, llm, embedder, verbose=True, memory=memory)

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
