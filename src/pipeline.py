"""メインパイプライン — End-to-End実行"""

from __future__ import annotations

import json

from src.db import init_db, save_run
from src.idea_bank import IdeaBank
from src.steps.step1_extract import extract_structure
from src.steps.step2_search import search_near, search_far
from src.steps.step3_sme import rank_analogies
from src.steps.step4_infer import generate_all_inferences
from src.steps.step5_plan import cross_plan


def run_pipeline(
    challenge: str,
    llm: LLMClient,
    embedder: EmbeddingClient,
    verbose: bool = False,
) -> dict:
    """パイプライン全体を実行

    Args:
        challenge: 課題の自然言語記述
        llm: LLMClient（DI）
        embedder: EmbeddingClient（DI）
        verbose: 中間結果を表示するか

    Returns:
        {
            "proposal": Proposal,
            "idea_bank": IdeaBank（トレーサビリティ用）,
            "steps": {各ステップの中間結果}
        }
    """
    def log(step: str, msg: str):
        if verbose:
            print(f"\n{'='*60}\n[{step}] {msg}\n{'='*60}")

    # Step 1: 構造抽出
    log("Step 1", "構造抽出")
    if verbose:
        print("  構造抽出中...")
    structure = extract_structure(challenge, llm)
    if verbose:
        print(json.dumps(structure, ensure_ascii=False, indent=2))

    # Step 2: 類似構造探索
    from src.config import NEAR_COUNT, FAR_COUNT
    log("Step 2", f"類似構造探索（近縁{NEAR_COUNT} + 遠方{FAR_COUNT}）")
    if verbose:
        print("  近縁探索中...")
    near = search_near(structure, llm)
    if verbose:
        print("  遠方探索中...")
    far = search_far(structure, llm)
    candidates = near + far
    if verbose:
        near_domains = [c.get("domain", "?") for c in near]
        far_domains = [c.get("domain", "?") for c in far]
        print(f"  近縁: {near_domains}")
        print(f"  遠方: {far_domains}")

    # Step 3: Semantic SME
    log("Step 3", f"Semantic SME（{len(candidates)}候補をスコアリング）")
    ranked = rank_analogies(structure, candidates, embedder)
    if verbose:
        for i, r in enumerate(ranked):
            print(f"  Rank {i+1}: {r['source']['domain']} (score: {r['score']:.3f})")

    # Step 4: 候補推論生成
    log("Step 4", f"候補推論生成（{len(ranked)}領域）")
    all_inferences = generate_all_inferences(structure, ranked, llm, verbose=verbose)

    # アイデアバンクに格納（出自剥離）
    bank = IdeaBank()
    for domain, inferences in all_inferences:
        bank.add(inferences, origin_domain=domain)
        if verbose:
            print(f"  → {domain}: {len(inferences)}件のアイデア")

    log("アイデアバンク", f"合計 {len(bank)} 件（出自剥離済み）")

    # Step 5: クロス立案
    log("Step 5", "クロス立案（出自を忘れて実効性のみで評価）")
    if verbose:
        print("  最終提案を構成中...")
    ideas_stripped = bank.get_ideas_stripped()
    proposal = cross_plan(challenge, ideas_stripped, llm)
    if verbose:
        print(json.dumps(proposal, ensure_ascii=False, indent=2))

    result = {
        "proposal": proposal,
        "idea_bank": bank,
        "steps": {
            "structure": structure,
            "candidates": candidates,
            "ranked": ranked,
            "inferences": all_inferences,
        },
    }

    # SQLite永続化
    try:
        init_db()
        run_id = save_run(result, challenge)
        if verbose:
            print(f"\n  [DB] 実行結果を保存しました (run_id={run_id})")
    except Exception as e:
        if verbose:
            print(f"\n  [DB] 保存失敗（パイプライン結果には影響なし）: {e}")

    return result
