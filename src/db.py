"""SQLite永続化 — 薄いDAO層

パイプライン実行結果を保存し、振り返り・フィードバックループの基盤を提供する。
TypedDictはJSON直列化可能なので、複雑な構造はJSON列に格納して正規化しすぎない。
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# デフォルトDBパス: プロジェクトルート/data/runs.db
DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "runs.db"


def _get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """スキーマ初期化（冪等）"""
    conn = _get_connection(db_path)
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                challenge   TEXT NOT NULL,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS structures (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id   INTEGER NOT NULL REFERENCES runs(id),
                domain   TEXT NOT NULL,
                is_base  INTEGER NOT NULL DEFAULT 0,
                data_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS scored_analogies (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id                INTEGER NOT NULL REFERENCES runs(id),
                candidate_domain      TEXT NOT NULL,
                score                 REAL NOT NULL,
                matched_relations_json TEXT NOT NULL,
                candidate_json        TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ideas (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id          INTEGER NOT NULL REFERENCES runs(id),
                idea            TEXT NOT NULL,
                base_principle  TEXT NOT NULL,
                application     TEXT NOT NULL,
                origin_domain   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS proposals (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id              INTEGER NOT NULL REFERENCES runs(id),
                summary             TEXT NOT NULL,
                actions_json        TEXT NOT NULL,
                combined_ideas_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id      INTEGER NOT NULL REFERENCES runs(id),
                score       INTEGER NOT NULL CHECK(score BETWEEN 1 AND 5),
                comment     TEXT NOT NULL DEFAULT '',
                created_at  TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_structures_run ON structures(run_id);
            CREATE INDEX IF NOT EXISTS idx_scored_run ON scored_analogies(run_id);
            CREATE INDEX IF NOT EXISTS idx_ideas_run ON ideas(run_id);
            CREATE INDEX IF NOT EXISTS idx_proposals_run ON proposals(run_id);
            CREATE INDEX IF NOT EXISTS idx_feedback_run ON feedback(run_id);
        """)
        conn.commit()
    finally:
        conn.close()


def save_run(result: dict, challenge: str, db_path: Optional[Path] = None) -> int:
    """パイプライン結果をDBに保存し、run_idを返す"""
    conn = _get_connection(db_path)
    try:
        now = datetime.now(timezone.utc).isoformat()

        # runs
        cur = conn.execute(
            "INSERT INTO runs (challenge, created_at) VALUES (?, ?)",
            (challenge, now),
        )
        run_id = cur.lastrowid

        # base structure
        structure = result["steps"]["structure"]
        conn.execute(
            "INSERT INTO structures (run_id, domain, is_base, data_json) VALUES (?, ?, 1, ?)",
            (run_id, structure["domain"], json.dumps(structure, ensure_ascii=False)),
        )

        # candidate structures
        for cand in result["steps"]["candidates"]:
            conn.execute(
                "INSERT INTO structures (run_id, domain, is_base, data_json) VALUES (?, ?, 0, ?)",
                (run_id, cand.get("domain", ""), json.dumps(cand, ensure_ascii=False)),
            )

        # scored analogies
        for sa in result["steps"]["ranked"]:
            conn.execute(
                "INSERT INTO scored_analogies (run_id, candidate_domain, score, matched_relations_json, candidate_json) VALUES (?, ?, ?, ?, ?)",
                (
                    run_id,
                    sa["source"]["domain"],
                    sa["score"],
                    json.dumps(sa["matched_relations"], ensure_ascii=False),
                    json.dumps(sa["source"], ensure_ascii=False),
                ),
            )

        # ideas (from idea_bank with origin)
        for entry in result["idea_bank"].get_ideas_with_origin():
            conn.execute(
                "INSERT INTO ideas (run_id, idea, base_principle, application, origin_domain) VALUES (?, ?, ?, ?, ?)",
                (run_id, entry["idea"], entry["base_principle"], entry["application"], entry["_origin_domain"]),
            )

        # proposal
        proposal = result["proposal"]
        conn.execute(
            "INSERT INTO proposals (run_id, summary, actions_json, combined_ideas_json) VALUES (?, ?, ?, ?)",
            (
                run_id,
                proposal["summary"],
                json.dumps(proposal["actions"], ensure_ascii=False),
                json.dumps(proposal["combined_ideas"], ensure_ascii=False),
            ),
        )

        conn.commit()
        return run_id
    finally:
        conn.close()


def save_feedback(
    run_id: int, score: int, comment: str = "", db_path: Optional[Path] = None
) -> int:
    """実行結果に対するフィードバックを保存し、feedback_idを返す"""
    if not 1 <= score <= 5:
        raise ValueError(f"score must be between 1 and 5, got {score}")
    conn = _get_connection(db_path)
    try:
        # run_idの存在チェック
        row = conn.execute("SELECT id FROM runs WHERE id = ?", (run_id,)).fetchone()
        if not row:
            raise ValueError(f"run_id {run_id} does not exist")
        now = datetime.now(timezone.utc).isoformat()
        cur = conn.execute(
            "INSERT INTO feedback (run_id, score, comment, created_at) VALUES (?, ?, ?, ?)",
            (run_id, score, comment, now),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def load_feedback_summary(db_path: Optional[Path] = None) -> dict:
    """フィードバック統計を返す: 全体統計 + ドメイン別平均スコア"""
    conn = _get_connection(db_path)
    try:
        # 全体統計
        overall = conn.execute("""
            SELECT COUNT(*) as cnt, AVG(score) as avg_score
            FROM feedback
        """).fetchone()

        # ドメイン別: アイデアの出自ドメインごとの平均スコア
        domain_rows = conn.execute("""
            SELECT i.origin_domain,
                   COUNT(DISTINCT f.run_id) as run_count,
                   AVG(f.score) as avg_score
            FROM feedback f
            JOIN ideas i ON i.run_id = f.run_id
            GROUP BY i.origin_domain
            ORDER BY avg_score DESC
        """).fetchall()

        # スコア分布
        dist_rows = conn.execute("""
            SELECT score, COUNT(*) as cnt
            FROM feedback
            GROUP BY score
            ORDER BY score
        """).fetchall()

        return {
            "total_feedback": overall["cnt"],
            "avg_score": round(overall["avg_score"], 2) if overall["avg_score"] else None,
            "score_distribution": {row["score"]: row["cnt"] for row in dist_rows},
            "domain_stats": [
                {
                    "domain": row["origin_domain"],
                    "run_count": row["run_count"],
                    "avg_score": round(row["avg_score"], 2),
                }
                for row in domain_rows
            ],
        }
    finally:
        conn.close()


def load_top_domains(min_score: float = 4.0, db_path: Optional[Path] = None) -> list[str]:
    """高評価ドメインのリストを返す（Step 2のヒント用）"""
    conn = _get_connection(db_path)
    try:
        rows = conn.execute("""
            SELECT i.origin_domain, AVG(f.score) as avg_score
            FROM feedback f
            JOIN ideas i ON i.run_id = f.run_id
            GROUP BY i.origin_domain
            HAVING avg_score >= ?
            ORDER BY avg_score DESC
            LIMIT 10
        """, (min_score,)).fetchall()
        return [row["origin_domain"] for row in rows]
    finally:
        conn.close()


def load_runs(limit: int = 20, db_path: Optional[Path] = None) -> list[dict]:
    """過去の実行一覧を返す（新しい順）"""
    conn = _get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT id, challenge, created_at FROM runs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def load_run(run_id: int, db_path: Optional[Path] = None) -> Optional[dict]:
    """特定の実行結果を復元する"""
    conn = _get_connection(db_path)
    try:
        run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if not run:
            return None

        structures = conn.execute(
            "SELECT domain, is_base, data_json FROM structures WHERE run_id = ?",
            (run_id,),
        ).fetchall()

        scored = conn.execute(
            "SELECT candidate_domain, score, matched_relations_json, candidate_json FROM scored_analogies WHERE run_id = ? ORDER BY score DESC",
            (run_id,),
        ).fetchall()

        ideas = conn.execute(
            "SELECT idea, base_principle, application, origin_domain FROM ideas WHERE run_id = ?",
            (run_id,),
        ).fetchall()

        proposal = conn.execute(
            "SELECT summary, actions_json, combined_ideas_json FROM proposals WHERE run_id = ?",
            (run_id,),
        ).fetchone()

        base_structure = None
        candidate_structures = []
        for s in structures:
            data = json.loads(s["data_json"])
            if s["is_base"]:
                base_structure = data
            else:
                candidate_structures.append(data)

        ranked = []
        for s in scored:
            ranked.append({
                "source": json.loads(s["candidate_json"]),
                "score": s["score"],
                "matched_relations": json.loads(s["matched_relations_json"]),
            })

        idea_list = []
        for i in ideas:
            idea_list.append({
                "idea": i["idea"],
                "base_principle": i["base_principle"],
                "application": i["application"],
                "_origin_domain": i["origin_domain"],
            })

        result = {
            "id": run["id"],
            "challenge": run["challenge"],
            "created_at": run["created_at"],
            "proposal": {
                "summary": proposal["summary"],
                "actions": json.loads(proposal["actions_json"]),
                "combined_ideas": json.loads(proposal["combined_ideas_json"]),
            } if proposal else None,
            "steps": {
                "structure": base_structure,
                "candidates": candidate_structures,
                "ranked": ranked,
            },
            "ideas": idea_list,
        }
        return result
    finally:
        conn.close()
