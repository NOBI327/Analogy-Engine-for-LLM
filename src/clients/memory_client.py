"""感情メモリクライアント — amygdalaとの統合アダプタ

amygdalaの SearchEngine / DatabaseManager を直接利用し、
パイプラインの実行履歴を感情ベクトル付きで記憶・想起する。

DI設計: amygdala未インストール時はNullMemoryClientにフォールバック。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class MemoryClient:
    """amygdalaの記憶システムへのアダプタ"""

    def __init__(self, db_path: str | None = None):
        from amygdala.config import Config as AmyConfig
        from amygdala.db import DatabaseManager
        from amygdala.search_engine import SearchEngine

        config = AmyConfig()
        path = db_path or config.DB_PATH
        self._db = DatabaseManager(path)
        self._db.init()
        self._search = SearchEngine(config, self._db)
        self._config = config

    def recall(self, query: str, emotions: dict | None = None, top_n: int = 5) -> list[dict]:
        """感情ベクトルベースで過去の記憶を検索する

        Returns:
            [{"id": int, "content": str, "score": float}, ...]
        """
        emotion_vec = emotions or {ax: 0.0 for ax in
                                    list(self._config.EMOTION_AXES) + list(self._config.META_AXES)}
        results = self._search.search_memories(emotion_vec, scenes=["work"], top_k=top_n)
        return [
            {"id": r["id"], "content": r["content"], "score": r["score"]}
            for r in results
        ]

    def store(
        self,
        text: str,
        emotions: dict | None = None,
        scenes: list[str] | None = None,
        context: str = "",
    ) -> int:
        """テキストを感情ベクトル付きで記憶に保存する

        Returns:
            memory_id
        """
        emotion_vec = emotions or {}
        scene_list = scenes or ["work"]

        conn = self._db.get_connection()
        cur = conn.execute(
            """INSERT INTO memories
               (content, raw_input, scenes,
                joy, sadness, anger, fear, surprise, disgust, trust, anticipation,
                importance, urgency)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                text,
                context,
                json.dumps(scene_list, ensure_ascii=False),
                emotion_vec.get("joy", 0.0),
                emotion_vec.get("sadness", 0.0),
                emotion_vec.get("anger", 0.0),
                emotion_vec.get("fear", 0.0),
                emotion_vec.get("surprise", 0.0),
                emotion_vec.get("disgust", 0.0),
                emotion_vec.get("trust", 0.0),
                emotion_vec.get("anticipation", 0.0),
                emotion_vec.get("importance", 0.0),
                emotion_vec.get("urgency", 0.0),
            ),
        )
        conn.commit()
        return cur.lastrowid

    def close(self):
        self._db.close()


class NullMemoryClient:
    """amygdala未インストール時のフォールバック — 何もしない"""

    def recall(self, query: str, emotions: dict | None = None, top_n: int = 5) -> list[dict]:
        return []

    def store(self, text: str, emotions: dict | None = None,
              scenes: list[str] | None = None, context: str = "") -> int:
        return -1

    def close(self):
        pass


def create_memory_client(db_path: str | None = None) -> MemoryClient | NullMemoryClient:
    """amygdalaの有無に応じてクライアントを生成するファクトリ"""
    try:
        return MemoryClient(db_path)
    except ImportError:
        logger.info("amygdala未インストール — メモリ機能は無効")
        return NullMemoryClient()
    except Exception as e:
        logger.warning(f"メモリクライアント初期化失敗（パイプラインには影響なし）: {e}")
        return NullMemoryClient()
