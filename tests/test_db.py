"""SQLite永続化のテスト"""

from pathlib import Path

import pytest

from src.db import init_db, save_run, load_runs, load_run
from src.idea_bank import IdeaBank


@pytest.fixture
def db_path(tmp_path):
    """テスト用の一時DBパス"""
    path = tmp_path / "test.db"
    init_db(path)
    return path


@pytest.fixture
def sample_result():
    """パイプライン結果のサンプル"""
    bank = IdeaBank()
    bank.add(
        [
            {"idea": "免疫抑制剤的アプローチ", "base_principle": "拒絶反応の制御", "application": "段階的導入"},
            {"idea": "自己化プロセス", "base_principle": "生体適合性", "application": "カスタマイズ期間"},
        ],
        origin_domain="臓器移植",
    )
    bank.add(
        [{"idea": "菌根ネットワーク", "base_principle": "共生関係", "application": "メンター制度"}],
        origin_domain="森林生態系",
    )

    return {
        "proposal": {
            "summary": "段階的導入とメンター制度の組み合わせ",
            "actions": ["フェーズ1: パイロット導入", "フェーズ2: 全社展開"],
            "combined_ideas": ["免疫抑制剤的アプローチ + 菌根ネットワーク"],
        },
        "idea_bank": bank,
        "steps": {
            "structure": {
                "domain": "IT企業オンボーディング",
                "entities": [{"name": "新人", "attributes": ["未経験"]}],
                "first_order_relations": [
                    {"source": "新人", "target": "チーム", "predicate": "参加する"}
                ],
                "higher_order_relations": [],
            },
            "candidates": [
                {
                    "domain": "臓器移植",
                    "entities": [{"name": "移植臓器", "attributes": ["外来"]}],
                    "first_order_relations": [
                        {"source": "移植臓器", "target": "生体", "predicate": "置換する"}
                    ],
                    "higher_order_relations": [],
                },
                {
                    "domain": "森林生態系",
                    "entities": [{"name": "菌根菌", "attributes": ["共生"]}],
                    "first_order_relations": [
                        {"source": "菌根菌", "target": "樹木", "predicate": "養分を供給する"}
                    ],
                    "higher_order_relations": [],
                },
            ],
            "ranked": [
                {
                    "source": {
                        "domain": "臓器移植",
                        "entities": [{"name": "移植臓器", "attributes": ["外来"]}],
                        "first_order_relations": [
                            {"source": "移植臓器", "target": "生体", "predicate": "置換する"}
                        ],
                        "higher_order_relations": [],
                    },
                    "score": 0.85,
                    "matched_relations": [
                        {"base": "参加する", "target": "置換する", "score": 0.72}
                    ],
                },
            ],
            "inferences": [
                ("臓器移植", [{"idea": "免疫抑制剤的アプローチ", "base_principle": "拒絶反応の制御", "application": "段階的導入"}]),
            ],
        },
    }


class TestInitDb:
    def test_creates_tables(self, db_path):
        """テーブルが作成される"""
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [t[0] for t in tables]
        conn.close()
        assert "runs" in table_names
        assert "structures" in table_names
        assert "scored_analogies" in table_names
        assert "ideas" in table_names
        assert "proposals" in table_names

    def test_idempotent(self, db_path):
        """複数回呼んでもエラーにならない"""
        init_db(db_path)
        init_db(db_path)


class TestSaveAndLoad:
    def test_save_returns_run_id(self, db_path, sample_result):
        run_id = save_run(sample_result, "オンボーディング課題", db_path)
        assert isinstance(run_id, int)
        assert run_id >= 1

    def test_load_runs_lists_saved(self, db_path, sample_result):
        save_run(sample_result, "課題A", db_path)
        save_run(sample_result, "課題B", db_path)
        runs = load_runs(db_path=db_path)
        assert len(runs) == 2
        assert runs[0]["challenge"] == "課題B"  # 新しい順
        assert runs[1]["challenge"] == "課題A"

    def test_load_run_restores_data(self, db_path, sample_result):
        run_id = save_run(sample_result, "オンボーディング課題", db_path)
        loaded = load_run(run_id, db_path)

        assert loaded is not None
        assert loaded["challenge"] == "オンボーディング課題"

        # proposal
        assert loaded["proposal"]["summary"] == "段階的導入とメンター制度の組み合わせ"
        assert len(loaded["proposal"]["actions"]) == 2

        # structure
        assert loaded["steps"]["structure"]["domain"] == "IT企業オンボーディング"

        # candidates
        assert len(loaded["steps"]["candidates"]) == 2

        # ranked
        assert len(loaded["steps"]["ranked"]) == 1
        assert loaded["steps"]["ranked"][0]["score"] == 0.85

        # ideas with origin
        assert len(loaded["ideas"]) == 3
        origins = [i["_origin_domain"] for i in loaded["ideas"]]
        assert "臓器移植" in origins
        assert "森林生態系" in origins

    def test_load_nonexistent_returns_none(self, db_path):
        assert load_run(999, db_path) is None

    def test_load_runs_respects_limit(self, db_path, sample_result):
        for i in range(5):
            save_run(sample_result, f"課題{i}", db_path)
        runs = load_runs(limit=3, db_path=db_path)
        assert len(runs) == 3
