"""感情メモリクライアントのテスト"""

import pytest
from unittest.mock import MagicMock, patch

from src.clients.memory_client import NullMemoryClient, create_memory_client


class TestNullMemoryClient:
    """amygdala未インストール時のフォールバック動作"""

    def test_recall_returns_empty(self):
        client = NullMemoryClient()
        assert client.recall("test") == []

    def test_store_returns_negative(self):
        client = NullMemoryClient()
        assert client.store("test") == -1

    def test_close_does_nothing(self):
        client = NullMemoryClient()
        client.close()  # エラーなく完了


class TestCreateMemoryClient:
    def test_falls_back_on_import_error(self):
        """amygdala未インストール時にNullClientを返す"""
        with patch("src.clients.memory_client.MemoryClient", side_effect=ImportError):
            client = create_memory_client()
            assert isinstance(client, NullMemoryClient)

    def test_falls_back_on_exception(self):
        """初期化失敗時にNullClientを返す"""
        with patch("src.clients.memory_client.MemoryClient", side_effect=RuntimeError("DB error")):
            client = create_memory_client()
            assert isinstance(client, NullMemoryClient)


class TestMemoryClientIntegration:
    """amygdalaが利用可能な場合の統合テスト"""

    @pytest.fixture
    def memory_client(self, tmp_path):
        """テスト用インメモリDBのMemoryClient"""
        try:
            from src.clients.memory_client import MemoryClient
            client = MemoryClient(db_path=":memory:")
            yield client
            client.close()
        except ImportError:
            pytest.skip("amygdala not installed")

    def test_store_and_recall(self, memory_client):
        """保存→想起のラウンドトリップ"""
        mid = memory_client.store(
            text="臓器移植のアナロジーが有効だった",
            emotions={"trust": 0.8, "importance": 0.9},
            scenes=["work"],
        )
        assert mid >= 1

        results = memory_client.recall(
            "移植",
            emotions={"trust": 0.7, "importance": 0.8},
        )
        assert len(results) >= 1
        assert any("臓器移植" in r["content"] for r in results)

    def test_store_returns_id(self, memory_client):
        mid = memory_client.store("テスト記憶")
        assert isinstance(mid, int)
        assert mid >= 1

    def test_recall_empty_db(self, memory_client):
        results = memory_client.recall("何か")
        assert results == []

    def test_recall_with_default_emotions(self, memory_client):
        memory_client.store("記憶1", emotions={"joy": 0.5})
        results = memory_client.recall("記憶")
        assert isinstance(results, list)
