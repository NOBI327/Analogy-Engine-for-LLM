# Analogy Engine for LLM

> [日本語版はこちら / Japanese version below](#analogy-engine-for-llm-日本語)

**A pipeline that discovers structural correspondences invisible to humans and transforms them into solutions.**

What if you could derive solutions for "employee onboarding isn't sticking" from organ transplant rejection or invasive species ecology?

Analogy Engine implements cognitive scientist Gentner's Structure-Mapping Theory (1983) as an LLM pipeline. Give it a problem, and it discovers structurally similar patterns from domains humans wouldn't normally consider, then automatically generates actionable solution proposals.

---

## Getting Started

### Setup

```bash
git clone https://github.com/NOBI327/Analogy-Engine-for-LLM.git
cd Analogy-Engine-for-LLM

pip install -r requirements.txt
```

### Running (Two Modes)

**Mode A: Anthropic API Key (pay-per-token)**

```bash
cp .env.example .env
# Edit .env and set your ANTHROPIC_API_KEY

python run.py "Describe your problem here"
```

**Mode B: Claude Code Max Plan (flat-rate)**

```bash
# No API key needed. Just be logged into Claude Code.
python run.py --claude-code "Describe your problem here"
```

### Interactive Input

```bash
python run.py                    # API key mode
python run.py --claude-code      # Max plan mode
# Type your problem at the prompt → Ctrl+D to submit
```

---

## What Happens — Pipeline Flow

When you input a problem, five steps execute automatically:

```
Your problem (natural language)
    |
    v
[Step 1] Structure Extraction ——— Decompose causal/dependency relations
    |
    v
[Step 2] Similar Structure Search ——— Find 5 near-field + 5 far-field analogues
    |
    v
[Step 3] Semantic SME ——— Score and rank by structural similarity
    |
    v
[Step 4] Candidate Inference ——— Transfer principles from each domain into ideas
    |
    v
[Idea Bank] ——— Strip origin metadata, store all ideas flat
    |
    v
[Step 5] Cross-Planning ——— Compose final proposal judged purely on effectiveness
    |
    v
Final Proposal + Traceability (retrospectively check each idea's source domain)
```

### Why Strip the Origin?

Analogy is the most powerful discovery tool, but it becomes toxic when used for justification.

The moment you learn "this idea came from immunology," you become captivated by the metaphor and start applying the analogy beyond its effective range. Step 5 intentionally hides where ideas came from, forcing the LLM to evaluate purely on practical effectiveness. Origins are available after the final proposal for traceability.

---

## Use Cases

### 1. Organizational Problem Solving

```bash
python run.py "A 700-person IT company deployed an automated onboarding system, but field adoption is low. The system is becoming a formality."
```

**How the pipeline responds:**

- **Structure Extraction**: `[Automation System] → [Gap with tacit knowledge] → [Declining usage] → [Becoming a formality]`
- **Near-field discoveries**: ERP rollout failures, CRM adoption struggles, internal wiki ghost towns...
- **Far-field discoveries**: Organ transplant rejection, invasive species introduction, loanword adoption in linguistics...
- **Candidate Inferences**:
  - Organ transplant → "Immunosuppressants = Design a coexistence period with parallel operation of old and new processes"
  - Invasive species → "Create contact points with native species = Embed small automation steps within existing workflows"
- **Final Proposal**: "Instead of a big-bang rollout, embed automation steps one at a time within existing workflows (Redmine ticketing, SlackBot notifications). The first 3 months should be a parallel-operation period with the legacy process."

### 2. Product Design Stagnation

```bash
python run.py "Our SaaS free trial to paid conversion rate is stuck at 3%. Features are sufficient, but users churn before experiencing value."
```

Possible far-field discoveries:
- **Pharmacokinetics** → "Loading dose to reach therapeutic levels" = Front-load success experiences on trial day one
- **Ecological niche construction** → "Modify the environment to suit yourself" = Let the product reshape itself to match the user's workflow

### 3. Technical Architecture Challenges

```bash
python run.py "We migrated to microservices, but inter-service communication complexity exploded. Root cause analysis for incidents now takes hours."
```

Possible far-field discoveries:
- **Neuroscience (myelination)** → "Prioritize frequently used neural pathways for speed" = Re-merge high-traffic service pairs
- **Urban planning (zoning)** → "Zone by living area, not by function" = Reorganize services along domain boundaries

### 4. Training & Education Design

```bash
python run.py "We run technical training for junior engineers, but training content doesn't transfer to real work. They can do it in training but not on the job."
```

### 5. New Business Hypothesis Generation

```bash
python run.py "We want to offer DX solutions to small manufacturers in rural areas, but their IT investment appetite is low and sales are difficult."
```

---

## Example Output

```
============================================================
[Step 1] Structure Extraction
============================================================
{
  "domain": "IT Company Onboarding",
  "entities": [
    {"name": "Automation System", "attributes": ["newly introduced", "company-wide"]},
    {"name": "Field Employees", "attributes": ["proficient in existing workflows"]},
    {"name": "Tacit Knowledge", "attributes": ["unverbalized", "person-dependent"]}
  ],
  "first_order_relations": [
    {"source": "Automation System", "target": "Field Employees", "predicate": "attempts to replace"},
    {"source": "Field Employees", "target": "Tacit Knowledge", "predicate": "possess"},
    {"source": "Automation System", "target": "Tacit Knowledge", "predicate": "fails to capture"}
  ],
  "higher_order_relations": [
    {"type": "cause", "source": "fails to capture", "target": "attempts to replace"}
  ]
}

============================================================
[Step 3] Semantic SME (scoring 10 candidates)
============================================================
  Rank 1: Organ Transplant Rejection (score: 0.847)
  Rank 2: Invasive Species Introduction (score: 0.793)
  Rank 3: Loanword Adoption in Linguistics (score: 0.756)
  ...

============================================================
Final Proposal
============================================================
{
  "summary": "Rather than a big-bang rollout, embed automation steps incrementally within existing workflows. The first 3 months should be a parallel-operation period with legacy processes.",
  "actions": [
    "Inventory the field's core workflows (Redmine ticketing, SlackBot notifications, etc.)",
    "Replace only the most routine single step in each workflow with automation",
    "Run old and new procedures in parallel for 3 months; let field teams decide when to switch",
    "Track automation step usage weekly; proceed to the next step only after usage exceeds 80%"
  ],
  "combined_ideas": [
    "Transitional parallel operation design",
    "Incremental embedding into existing workflows",
    "Field-driven switchover decisions"
  ]
}

------------------------------------------------------------
Traceability (source domain of each idea)
------------------------------------------------------------
  [Organ Transplant] Transitional parallel operation design
  [Invasive Species] Incremental embedding into existing workflows
  [Loanword Adoption] Field-driven switchover decisions
```

---

## Project Structure

```
.
├── run.py                          # Entry point
├── src/
│   ├── config.py                   # Settings (env vars, thresholds)
│   ├── models.py                   # Data structures (TypedDict)
│   ├── pipeline.py                 # Pipeline orchestration
│   ├── idea_bank.py                # Idea Bank (origin stripping)
│   ├── clients/
│   │   ├── llm_client.py           # Claude API client (pay-per-token)
│   │   ├── claude_code_client.py   # Claude Code CLI client (Max plan flat-rate)
│   │   └── embedding_client.py     # sentence-transformers client
│   └── steps/
│       ├── step1_extract.py        # Structure Extraction
│       ├── step2_search.py         # Similar Structure Search (near + far)
│       ├── step3_sme.py            # Semantic SME (scoring)
│       ├── step4_infer.py          # Candidate Inference Generation
│       └── step5_plan.py           # Cross-Planning
├── tests/                          # 26 tests, 96% coverage
├── docs/
│   └── analogy_engine_proposal.md  # Design philosophy & theoretical background
├── requirements.txt
└── pyproject.toml
```

---

## Tech Stack

| Component | Technology | Role |
|---|---|---|
| LLM | Claude API / Claude Code CLI | Structure extraction, search, inference, planning |
| Embeddings | sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2) | Structural matching in Semantic SME |
| Testing | pytest + DI (Dependency Injection) | Tests run without any external API calls |

---

## Theoretical Background

Based on Dedre Gentner's **Structure-Mapping Theory** (1983, *Cognitive Science*).

Two core principles:

1. **Map relations, not attributes** — "The sun is large and hot" doesn't transfer. "The central body's gravity determines the orbiting body's trajectory" does.
2. **Systematicity** — Isolated relations are deprioritized. Systems of relations connected by higher-order relations (causation, dependency) are preferred.

What makes this engine different from standard LLM prompting:

- **Actively searches far-field domains, not just near-field** (Step 2 far search)
- **Ranks by structural similarity, not semantic similarity** (Step 3 Semantic SME)
- **Intentionally strips analogy origins before evaluation** (Idea Bank → Step 5)

For the full design philosophy, see [`docs/analogy_engine_proposal.md`](docs/analogy_engine_proposal.md).

---

## Testing

```bash
# Run all tests (no external API connections)
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=src --cov-report=term-missing
```

DI (Dependency Injection) design means all tests run without connecting to Claude API or sentence-transformers.

---

## Roadmap

- **Phase 1 (current)**: End-to-end pipeline construction and validation
- **Phase 2**: GNN-based structural similarity for higher precision, Hungarian algorithm for exclusive matching
- **Phase 3**: Integration with emotion memory system, external publication

---

## License

MIT

---

## References

- Gentner, D. (1983). "Structure-Mapping: A Theoretical Framework for Analogy." *Cognitive Science*, 7, 155-170.
- Falkenhainer, B., Forbus, K., & Gentner, D. (1989). "The Structure-Mapping Engine."
- Gentner, D. & Markman, A.B. (1997). "Structure Mapping in Analogy and Similarity." *American Psychologist*, 52.

---
---

# Analogy Engine for LLM (日本語)

**人間には見えない構造的対応を発見し、ソリューションに変換するパイプライン**

「オンボーディングが定着しない」という課題に対して、臓器移植の拒絶反応や外来種の生態系導入から解決策を引き出す。そんなことができたら？

Analogy Engineは、認知科学者Gentnerの構造写像理論（1983）をLLMパイプラインとして実装したツールです。課題を入力すると、人間が通常思いつかない分野から構造的に類似したパターンを発見し、実行可能なソリューション提案を自動生成します。

---

## どう使うか

### セットアップ

```bash
git clone https://github.com/NOBI327/Analogy-Engine-for-LLM.git
cd Analogy-Engine-for-LLM

pip install -r requirements.txt
```

### 実行（2つの方式）

**方式A: Anthropic APIキー（従量課金）**

```bash
cp .env.example .env
# .env を編集して ANTHROPIC_API_KEY を設定

python run.py "あなたの課題をここに書く"
```

**方式B: Claude Code Maxプラン（定額）**

```bash
# APIキー不要。Claude Codeにログイン済みであれば即使える
python run.py --claude-code "あなたの課題をここに書く"
```

### 対話的に入力する場合

```bash
python run.py                    # APIキー方式
python run.py --claude-code      # Maxプラン方式
# プロンプトが表示されるので課題を入力 → Ctrl+D で送信
```

---

## 何が起きるか — パイプラインの流れ

課題を入力すると、5つのステップが自動で実行されます。

```
あなたの課題（自然言語）
    |
    v
[Step 1] 構造抽出 ――― 課題の因果・依存関係を構造化
    |
    v
[Step 2] 類似構造探索 ――― 近縁分野5つ + 異分野5つを発見
    |
    v
[Step 3] Semantic SME ――― 構造的類似度でスコアリング・ランキング
    |
    v
[Step 4] 候補推論生成 ――― 各分野の法則を課題に転写してアイデア化
    |
    v
[アイデアバンク] ――― 全アイデアから出自情報を剥がしてフラットに格納
    |
    v
[Step 5] クロス立案 ――― 出自を知らない状態で、純粋に実効性だけで最終提案を構成
    |
    v
最終提案 + トレーサビリティ（事後的に各アイデアの発想元を確認可能）
```

### なぜ「出自を剥がす」のか？

アナロジーは発見のツールとしては最強ですが、正当化のツールとして使うと毒になります。

「このアイデアは免疫学から来た」と知った瞬間、人はそのメタファーに魅了され、アナロジーの有効射程を超えて適用し始めます。Step 5では意図的に出自を隠すことで、LLMにアイデアの実効性だけで判断させます。出自は最終提案の後で確認できます。

---

## ユースケース

### 1. 組織の課題解決

```bash
python run.py "700人規模のIT企業で、オンボーディング自動化システムを導入したが現場の定着率が低い。形骸化しつつある。"
```

**パイプラインの動き:**

- **構造抽出**: `[自動化システム] → [現場の暗黙知との乖離] → [利用率低下] → [形骸化]`
- **近縁発見**: ERP導入失敗、CRM定着化、社内Wiki過疎化...
- **異分野発見**: 臓器移植の拒絶反応、外来種の生態系導入、言語の借用語定着...
- **候補推論**:
  - 臓器移植 → 「免疫抑制剤 = 過渡期に既存プロセスと並行運用して共存を設計」
  - 外来種 → 「在来種との接点を作る = 既存業務フローの中に小さく埋め込む」
- **最終提案**: 「一括導入ではなく、既存フロー（Redmine起票、SlackBot通知）の中に自動化ステップを1つずつ埋め込む段階的導入。最初の3ヶ月は旧プロセスとの並行運用期間とする」

### 2. プロダクト設計の行き詰まり

```bash
python run.py "SaaSプロダクトの無料トライアルからの有料転換率が3%で停滞している。機能は十分だがユーザーが価値を実感する前に離脱する。"
```

想定される異分野からの発見:
- **薬物動態学** → 「治療域に達するまでの負荷投与（ローディングドーズ）」= トライアル初日に成功体験を集中投下
- **生態学のニッチ構築** → 「まず環境を自分に合わせる」= ユーザーのワークフローに合わせてプロダクト側が変形する

### 3. 技術的アーキテクチャの問題

```bash
python run.py "マイクロサービス化を進めたが、サービス間通信の複雑さが爆発し、障害の原因特定に時間がかかるようになった。"
```

想定される異分野からの発見:
- **神経科学の髄鞘形成** → 「頻繁に使う神経経路を優先的に高速化する」= 通信頻度の高いサービスペアを結合に戻す
- **都市計画のゾーニング** → 「機能別ではなく生活圏別に区画する」= ドメイン境界でサービスを再編成する

### 4. 教育・研修設計

```bash
python run.py "新人エンジニアの技術研修を実施しているが、研修内容が実務に活かされない。研修中はできるのに現場に出ると使えない。"
```

### 5. 新規事業の仮説構築

```bash
python run.py "地方の中小製造業向けにDXソリューションを提供したいが、顧客のIT投資意欲が低く、営業が困難。"
```

---

## 出力例

```
============================================================
[Step 1] 構造抽出
============================================================
{
  "domain": "IT企業オンボーディング",
  "entities": [
    {"name": "自動化システム", "attributes": ["新規導入", "全社展開"]},
    {"name": "現場社員", "attributes": ["既存業務に習熟"]},
    {"name": "暗黙知", "attributes": ["非言語化", "属人的"]}
  ],
  "first_order_relations": [
    {"source": "自動化システム", "target": "現場社員", "predicate": "置換しようとする"},
    {"source": "現場社員", "target": "暗黙知", "predicate": "保有する"},
    {"source": "自動化システム", "target": "暗黙知", "predicate": "捕捉できない"}
  ],
  "higher_order_relations": [
    {"type": "cause", "source": "捕捉できない", "target": "置換しようとする"}
  ]
}

============================================================
[Step 3] Semantic SME（10候補をスコアリング）
============================================================
  Rank 1: 臓器移植の拒絶反応 (score: 0.847)
  Rank 2: 外来種の生態系導入 (score: 0.793)
  Rank 3: 言語の借用語定着 (score: 0.756)
  ...

============================================================
最終提案
============================================================
{
  "summary": "オンボーディングシステムを一括導入するのではなく、既存の業務フローの中に自動化ステップを1つずつ埋め込む段階的導入を提案。最初の3ヶ月は旧プロセスとの並行運用期間とする。",
  "actions": [
    "現場の主要業務フロー（Redmine起票、SlackBot通知等）を棚卸しする",
    "各フローの中で最も定型的な1ステップだけを自動化に置換する",
    "3ヶ月間は旧手順と新手順を並行運用し、切替判断は現場チームに委ねる",
    "自動化ステップの利用率を週次で計測し、利用率が80%を超えたら次のステップに進む"
  ],
  "combined_ideas": [
    "過渡期の並行運用設計",
    "既存フローへの段階的埋め込み",
    "現場主導の切替判断"
  ]
}

------------------------------------------------------------
トレーサビリティ（各アイデアの出自）
------------------------------------------------------------
  [臓器移植] 過渡期の並行運用設計
  [外来種の生態系導入] 既存フローへの段階的埋め込み
  [言語の借用語定着] 現場主導の切替判断
```

---

## プロジェクト構成

```
.
├── run.py                          # エントリポイント
├── src/
│   ├── config.py                   # 設定（環境変数、閾値）
│   ├── models.py                   # データ構造（TypedDict）
│   ├── pipeline.py                 # パイプライン全体の実行制御
│   ├── idea_bank.py                # アイデアバンク（出自剥離）
│   ├── clients/
│   │   ├── llm_client.py           # Claude API クライアント（従量課金）
│   │   ├── claude_code_client.py   # Claude Code CLI クライアント（Maxプラン定額）
│   │   └── embedding_client.py     # sentence-transformers クライアント
│   └── steps/
│       ├── step1_extract.py        # 構造抽出
│       ├── step2_search.py         # 類似構造探索（近縁 + 遠方）
│       ├── step3_sme.py            # Semantic SME（スコアリング）
│       ├── step4_infer.py          # 候補推論生成
│       └── step5_plan.py           # クロス立案
├── tests/                          # 26テスト、カバレッジ96%
├── docs/
│   └── analogy_engine_proposal.md  # 設計思想・理論的背景の詳細
├── requirements.txt
└── pyproject.toml
```

---

## 技術スタック

| コンポーネント | 技術 | 役割 |
|---|---|---|
| LLM | Claude API / Claude Code CLI | 構造抽出、探索、推論生成、立案 |
| 埋め込み | sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2) | Semantic SMEでの構造マッチング |
| テスト | pytest + DI (依存性注入) | 外部API不要でテスト完結 |

---

## 理論的背景

Dedre Gentnerの**構造写像理論**（Structure-Mapping Theory, 1983）に基づいています。

核心は2つ:

1. **マッピング対象は「関係」であり「属性」ではない** — 「太陽は大きくて熱い」は転送されない。「中心天体の引力が周辺天体の軌道を決定する」が転送される。
2. **体系性原理** — 孤立した関係ではなく、因果・依存で結ばれた関係のシステムが優先される。

このエンジンが通常のLLMプロンプティングと異なるのは:

- **近縁だけでなく異分野を積極的に探索する**（Step 2の遠方探索）
- **意味的類似ではなく構造的類似でランキングする**（Step 3のSemantic SME）
- **アナロジーの出自を意図的に剥がして評価する**（アイデアバンク → Step 5）

詳しい設計思想は [`docs/analogy_engine_proposal.md`](docs/analogy_engine_proposal.md) を参照してください。

---

## テスト

```bash
# 全テスト実行（外部APIへの接続なし）
python -m pytest tests/ -v

# カバレッジ付き
python -m pytest tests/ --cov=src --cov-report=term-missing
```

DI（依存性注入）設計により、Claude APIやsentence-transformersに接続せずにテストが完結します。

---

## ロードマップ

- **Phase 1（現在）**: End-to-Endパイプラインの構築と検証
- **Phase 2**: GNN（グラフニューラルネットワーク）による構造的類似度の高精度化、ハンガリアン法による排他的マッチング
- **Phase 3**: 感情メモリシステムとの統合、外部発信

---

## ライセンス

MIT

---

## 参考文献

- Gentner, D. (1983). "Structure-Mapping: A Theoretical Framework for Analogy." *Cognitive Science*, 7, 155-170.
- Falkenhainer, B., Forbus, K., & Gentner, D. (1989). "The Structure-Mapping Engine."
- Gentner, D. & Markman, A.B. (1997). "Structure Mapping in Analogy and Similarity." *American Psychologist*, 52.
