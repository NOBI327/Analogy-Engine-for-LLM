"""設定管理 — 環境変数から読み込み、デフォルト値を提供"""

import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("ANALOGY_ENGINE_MODEL", "claude-sonnet-4-6")
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Semantic SME thresholds
SME_SCORE_THRESHOLD = 0.3  # これ以下のマッチは除外
MAX_ANALOGIES = 10         # 近縁5 + 遠方5
