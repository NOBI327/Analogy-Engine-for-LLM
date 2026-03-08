# Analogy Engine for LLM

**Discover Structural Analogies Invisible to Humans, Transform Them into Solutions**

[![日本語](https://img.shields.io/badge/lang-ja-blue)](README_ja.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-33%20passed-brightgreen)]()

---

What if you could derive solutions for "employee onboarding isn't sticking" from organ transplant rejection or invasive species ecology?

Analogy Engine implements cognitive scientist Gentner's Structure-Mapping Theory (1983) as an LLM pipeline. Give it a problem, and it discovers structurally similar patterns from domains humans wouldn't normally consider, then automatically generates actionable solution proposals.

---

## Getting Started

### Prerequisites

- **Python 3.12** (3.13+ is not yet supported by PyTorch)

### Setup

```bash
git clone https://github.com/NOBI327/Analogy-Engine-for-LLM.git
cd Analogy-Engine-for-LLM

# Create and activate virtual environment
python3.12 -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate

# Install PyTorch CPU version (lighter, recommended unless you need GPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt
```

### Emotion Memory Integration (Optional)

This project integrates with the [Emotion Memory System](https://github.com/NOBI327/amygdala) (MCP server). To allow all emotion memory features in Claude Code without per-call confirmation, run the setup script:

```bash
python setup_permissions.py
```

This displays all 6 features (store, recall, stats, pin, unpin, list) with descriptions, and registers them in `.claude/settings.local.json` with a single confirmation.

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

### Viewing Past Results

```bash
python run.py --history          # List all past runs
python run.py --history 3        # Show details of run ID 3
```

All pipeline results are automatically saved to SQLite (`data/runs.db`).

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
├── run.py                          # Entry point (--history for past results)
├── src/
│   ├── config.py                   # Settings (env vars, thresholds)
│   ├── models.py                   # Data structures (TypedDict)
│   ├── pipeline.py                 # Pipeline orchestration
│   ├── db.py                       # SQLite persistence (thin DAO)
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
| Persistence | SQLite (WAL mode) | Auto-save all pipeline results for review and feedback loops |
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

- **Phase 1** (done): End-to-end pipeline construction and validation
- **Phase 2** (in progress): Hungarian algorithm for exclusive matching (done), SQLite persistence (done), GNN-based structural similarity, A/B testing, feedback loops
- **Phase 3**: Integration with emotion memory system, external publication

---

## References

- Gentner, D. (1983). "Structure-Mapping: A Theoretical Framework for Analogy." *Cognitive Science*, 7, 155-170.
- Falkenhainer, B., Forbus, K., & Gentner, D. (1989). "The Structure-Mapping Engine."
- Gentner, D. & Markman, A.B. (1997). "Structure Mapping in Analogy and Similarity." *American Psychologist*, 52.
