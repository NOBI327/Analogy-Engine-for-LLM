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

            CREATE INDEX IF NOT EXISTS idx_structures_run ON structures(run_id);
            CREATE INDEX IF NOT EXISTS idx_scored_run ON scored_analogies(run_id);
            CREATE INDEX IF NOT EXISTS idx_ideas_run ON ideas(run_id);
            CREATE INDEX IF NOT EXISTS idx_proposals_run ON proposals(run_id);
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
