# rag-eval-harness — Production-grade evals for RAG systems

> A small, focused evaluation harness for RAG pipelines. Measures **faithfulness**, **answer relevance**, and **retrieval recall@k** on a labelled question set. Runs locally, runs in CI, gives you a single number to decide whether a prompt or retrieval change is a regression.

**Why this exists.** The thing that kills production RAG is silent regressions — you tweak a prompt or swap an embedding model and the bot quietly gets dumber on 8% of questions. By the time the support team notices, you've shipped a worse system to thousands of users.

This harness is the gate you put in front of merging anything that touches the RAG pipeline.

---

## What it measures

| Metric | What it answers | Implementation |
|---|---|---|
| **Faithfulness** | Are the answer's claims supported by the retrieved context? | LLM-as-judge: for each claim sentence, retrieve supporting context, ask Claude/GPT to score 0–1 |
| **Answer relevance** | Does the answer actually address the question (vs evading it)? | LLM-as-judge: rate alignment of answer to question, 0–1 |
| **Retrieval recall@k** | Is the gold passage in the top-k retrieved chunks? | Exact match against labelled gold chunk IDs |
| **Latency p50 / p95** | How fast does end-to-end answer generation take? | Wall-clock per question |
| **Cost per question** | What does each answer cost to generate? | Token counts × model price |

You get one report card per run. Regressions vs the previous run are highlighted in red.

---

## How to use it

```bash
uv sync
cp .env.example .env   # add OPENAI_API_KEY or ANTHROPIC_API_KEY
uv run python harness/run_eval.py --questions data/golden.yml --pipeline examples/baseline_rag.py
```

Output:

```
RAG Eval Harness — run 2026-05-14 14:22
─────────────────────────────────────────
Pipeline:        examples/baseline_rag.py
Questions:       60
Model:           gpt-4o-mini  ($0.0012/q avg)

Faithfulness        0.94  ✓  (target ≥ 0.90)
Answer relevance    0.91  ✓  (target ≥ 0.85)
Retrieval recall@5  0.87  ✓  (target ≥ 0.80)

Latency p50         2.1s
Latency p95         4.8s
Total cost          $0.07

Regressions vs baseline_2026-05-08: none
```

---

## Repo structure

```
.
├── harness/
│   ├── run_eval.py          # entrypoint
│   ├── metrics.py           # faithfulness, relevance, recall implementations
│   ├── judge.py             # LLM-as-judge wrappers
│   └── report.py            # rendered comparison vs prior run
├── examples/
│   └── baseline_rag.py      # toy pipeline showing the contract
├── data/
│   └── golden.yml           # example labelled set
├── .github/workflows/
│   └── eval.yml             # CI workflow — fails the build on regression
└── pyproject.toml
```

## The contract for your pipeline

Your real RAG pipeline plugs in by implementing two functions:

```python
def retrieve(question: str) -> list[Chunk]: ...
def generate(question: str, chunks: list[Chunk]) -> str: ...
```

The harness handles the rest.

## Why LLM-as-judge instead of human eval

Pros: cheap, fast, repeatable.
Cons: judge biases, expensive at scale, sensitive to prompt phrasing.

The compromise here: LLM-as-judge for the iteration loop, **plus** a 10-question manually-labelled "anchor set" that's hand-graded and acts as a calibration check on the judge itself. If the judge's scores drift on the anchor set, the judge prompt itself is regressed.

## Status

- [x] Faithfulness scorer (Claude or GPT-4o)
- [x] Answer-relevance scorer
- [x] Retrieval recall@k
- [x] Baseline + regression comparison
- [x] GitHub Actions CI workflow
- [ ] HTML report output (nice-to-have)
- [ ] Hybrid retrieval evaluation (BM25 + dense)

## Author

Darrshan Govender · [Agulhas Code](https://agulhascode.co.za) · Durban GMT+2
