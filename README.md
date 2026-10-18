# riskcascade

**When should an expensive model overrule a cheap one?** A two-tier risk-defence framework, tested on two threats:

| Track | Team it targets | Threat | Tier 1 (every item, fast) | Tier 2 (uncertain items only, slow) |
|---|---|---|---|---|
| `content` | BRIC | Scam/gambling text smuggled into images so moderation models can't read it | DINOv3 + adapter "is there human-readable text here?" gate | OCR + MLLM policy judge |
| `transactions` | ECRC | E-commerce payment fraud | LightGBM on behavioural + entity-sharing features | LLM investigation agent with tools |

Both tracks share one core and answer the same question with the same metrics:

1. Route only the uncertain band to tier 2.
2. Measure whether tier 2 actually helps. Recent work found an LLM investigator *made decisions worse* (arXiv 2607.19266).
3. Train an **override gate** that decides when tier 2's disagreement is trustworthy.
4. Evaluate on **unseen threats**: held-out attack families (content) or later months (transactions).

## Layout

```
src/riskcascade/
  core/            shared by both tracks; runs today on synthetic data
    cascade.py     tier-1 → uncertainty band → tier-2 → override gate
    gate.py        learned override gate
    splits.py      held-out-family and temporal splits ("unseen threat" protocol)
    metrics.py     cost-weighted loss, recall @ fixed FPR, routing share, latency
  content/         BRIC track (stubs to fill)
    data.py        SmuggleBench + benign stylised captions + synthetic smuggling renderer
    gate_model.py  DINOv3 + adapter text-presence gate
    reader.py      restoration + OCR + MLLM judge
  transactions/    ECRC track (stubs to fill)
    data.py        IEEE-CIS loader
    features.py    behavioural + entity-sharing features
    scorer.py      LightGBM tier-1
    agent.py       tool-calling investigator
eval/run_eval.py   one results table for both tracks
configs/           one YAML per track
tests/             end-to-end check of the core on synthetic data
```

`core/` holds everything the two tracks have in common, so each track only implements three things: a tier-1 scorer, a tier-2 judge, and its data/split. Both plug into `Cascade`, so the evaluation is identical and directly comparable.

## Cut on purpose

- **Video extension** of SmuggleBench → optional stretch, after image results.
- **GNN / graph learning** → plain entity-sharing counts (cards, emails, devices shared across accounts). They capture most of the ring signal at a fraction of the effort.
- **LangGraph orchestration** → a plain tool-calling loop. The agent needs four tools, not a framework.
- **Mechanistic probing** → dropped (it was for the MSRA application).
- **Serving stack** → latency is measured in-process; no FastAPI/Redis.

## Milestones

| Week | Content track | Transaction track |
|---|---|---|
| 1 | SmuggleBench loaded; benign-caption negatives collected | IEEE-CIS loaded; time split; LightGBM baseline |
| 2 | Tier-1 gate trained; latency measured → **apply to BRIC** | Entity features; uncertainty band chosen → **apply to ECRC** |
| 3 | MLLM judge; first cascade table | Agent with 4 tools; agent-vs-classifier by score band and month |
| 4 | Override gate; held-out-family results | Override gate; drift results |
| 5 | (stretch) video clips | Write-up |

## Headline result to aim for

One table, two rows (one per track): tier-1 only vs. tier-2-always vs. cascade-with-gate, on unseen threats, reported as cost-weighted loss, recall @ 1% FPR, share of traffic sent to tier 2, and p99 tier-1 latency.

## Resume bullets (fill in numbers)

- **BRIC version:** "Two-stage defence against moderation-evading text: DINOv3-adapter gate (X ms/image) routes Y% of items to an OCR+MLLM judge; learned override gate cut attack success from A% to B% on held-out SmuggleBench attack families at C% FPR on benign stylised captions."
- **ECRC version:** "Two-tier fraud pipeline on 590K e-commerce transactions: LightGBM + entity-sharing scorer (p99 < X ms) with an LLM investigation agent on uncertain cases; a learned override gate cut cost-weighted loss Y% vs. classifier-only, with gains concentrated on drifted months."

## Setup

```bash
pip install -e .            # core only: numpy, scikit-learn
pip install -e ".[content]" # + torch, transformers, OCR
pip install -e ".[tx]"      # + lightgbm, pandas, shap
pytest tests/               # core runs end-to-end on synthetic data
```

## Data

- SmuggleBench (ACL 2026 Findings, arXiv 2604.06950): 1,700 adversarial smuggling instances. Get it from the paper's release.
- IEEE-CIS Fraud Detection (Kaggle): needs a Kaggle account; download manually into `data/raw/ieee-cis/`.
- Benign stylised text (hard negatives): memes and decorative captions; TextVQA or COCO-Text images work as a start.

Before starting each track, re-check arXiv for overlapping work from the last month.
