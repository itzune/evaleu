# Skill cards — "Best model per skill" proposal

## Goal

Add a compact visual section to the evaluation dashboard that answers the
question *"which model is best for what?"*  at a glance.  Each card shows a
human-readable skill, the winning model (highlighted), its absolute score,
and the margin over the runner-up.

```
┌─────────────────────────────────────────────────────────┐
│  KNOWLEDGE                                              │
│  ★ DeepSeek V4 Pro    75.8%   +2.1% over Latxa 70B    │
│  ────────────────────────────────────────────────────── │
│  What the model *knows* — facts, culture, exams         │
└─────────────────────────────────────────────────────────┘
```

The ★ star highlights the winner.  The runner-up name gives immediate
context — users instantly know which models are competitive.

---

## The five skill categories

Every benchmark that measures classification accuracy maps into **one**
of five skills.  Translation benchmarks remain in their own dedicated
table and chart (Table 2 / Figure 3), outside the skill cards.

### 1. KNOWLEDGE — factual, cultural, and academic recall

> *"How much does the model know?"*

Benchmarks:
- **EusTrivia** — Basque factual knowledge, 4-option MC
- **MMLU_eu** — Academic knowledge (MMLU-style in Basque)
- **BertaQA_eu** — Basque cultural + general knowledge trivia
- **LatxaEval_eusexams** — Professional and domain exams

### 2. READING — text comprehension and inference

> *"How well does the model understand a text?"*

Benchmarks:
- **XNLIeu** — Natural language inference (premise/hypothesis)
- **LatxaEval_eusreading** — Reading comprehension (passage + MC question)
- **BasqueGLUE_qnli** — Question-answer NLI (sentence-pair classification)

These all require reasoning over a text, not just recalling facts.

### 3. WRITING — Basque proficiency and usage

> *"How well does the model handle Basque as a language?"*

Benchmarks:
- **LatxaEval_eusproficiency** — Basque language proficiency (grammar, usage)

*Note:* This is currently a single-benchmark category.  Translation is
kept separate (it already has a dedicated heatmap table in the UI) and
is not mixed into this score.  Additional writing-oriented benchmarks
would strengthen this category over time.

### 4. REASONING — logic, math, and disambiguation

> *"How well does the model think through a problem?"*

Benchmarks:
- **MGSM_eu** — Grade-school math word problems (CoT + numeric answer)
- **BasqueGLUE_wic** — Word-in-Context disambiguation
- **BasqueGLUE_bec** — Sentiment classification (requires interpreting tone)

Reasoning is distinct from reading: these tasks need logical deduction
rather than surface-level comprehension.

### 5. PRAGMATICS — intent and real-world language use

> *"How well does the model understand what a speaker intends?"*

Benchmarks:
- **BasqueGLUE_intent** — 12-class intent classification

Intent classification measures a qualitatively different skill than
knowledge, reading, or reasoning — it captures whether the model
understands *why* someone is saying something, not just *what* they
are saying.  It is consistently the highest-accuracy benchmark across
all models (85–95%), making it a useful axis for non-technical users.

---

## Scoring methodology

### Per-benchmark normalisation

Every benchmark produces an accuracy score `s_b`:

```
s_b = accuracy_mean × 100       (range 0–100)
```

No translation metrics are mixed in — WRITING is purely accuracy-based.

### Per-skill aggregation

For skill `S` with benchmarks `{b₁, …, bₙ}`:

```
score_S(model) = (1/n) × Σ s_b(model)
```

Only benchmarks that have been evaluated for that model are included.
If a model has 0 evaluated benchmarks for a skill, it does not
participate in that skill's ranking.

### Winner and margin

```
winner  = argmax score_S(model)
margin  = score_S(winner) - score_S(second_best)
```

Displayed as `+X.X% over RunnerUp Name` on the card.

---

## Card layout specification

Each card contains, top to bottom:

| Element | Content | Style |
|---|---|---|
| Skill label | `KNOWLEDGE` | Uppercase, small, muted color |
| Winner row | `★ DeepSeek V4 Pro  75.8%  +2.1% over Latxa 70B` | Star highlights winner, model name in bold, score in mono |
| Divider | thin line | — |
| Description | `What the model *knows* — facts, culture, exams` | Italic, muted, one line |

### Visual treatment

- Winner model name is **bold** and preceded by a ★ star.
- Absolute score (`75.8%`) uses monospace font for alignment.
- Margin always includes the runner-up name: `+2.1% over Latxa 70B`.
- Runner-up name is de-emphasised (muted color).
- Cards have a subtle border and rounded corners, matching the existing
  dashboard aesthetic.

### Responsive grid

| Breakpoint | Cards per row |
|---|---|
| Desktop (≥1040px) | 5 |
| Tablet (≥700px) | 2–3 |
| Mobile (<700px) | 1 |

---

## Future: interactive skill view

Cards will eventually become clickable.  Clicking a skill card navigates
to a filtered "skill view" that shows:

- A ranking table of all models for that skill (sorted by score).
- Per-benchmark breakdown for the selected skill.
- The same chart types (bar, radar) filtered to the skill's benchmarks.

This is deferred to a follow-up iteration — v1 ships static cards.

---

## Implementation plan

### Data layer — `site/build_site_data.py`

Add a `skills` section to `site/data.json`:

```json
{
  "skills": [
    {
      "id": "knowledge",
      "label": "KNOWLEDGE",
      "description": "What the model *knows* — facts, culture, exams",
      "benchmarks": ["EusTrivia", "MMLU_eu", "BertaQA_eu", "LatxaEval_eusexams"],
      "winner": {
        "model_id": "deepseek-ai/DeepSeek-V4-Pro",
        "display_name": "DeepSeek V4 Pro",
        "score": 75.8
      },
      "runner_up": {
        "model_id": "latxa-70b",
        "display_name": "Latxa 70B",
        "score": 73.7
      },
      "margin": 2.1
    },
    …
  ]
}
```

The build script computes per-model skill scores, determines winners
and runner-ups, and writes the `skills` array.

### When benchmarks are missing

If a model hasn't been evaluated on some benchmarks in a skill, the
skill score is computed only from the available benchmarks.  If a
skill has 0 benchmarks with data for a model, that model is excluded
from that skill's ranking.

### Adding new models / benchmarks

Scoring is fully automatic — no manual curation needed.  As soon as
a benchmark is run for a model, its skill scores update on the next
`build` step.

---

## Alternatives considered

1. **Family-based cards** (Core / BasqueGLUE / LatxaEvalSuite).  Rejected
   because families are artifacts of benchmark provenance, not user-facing
   skills.  A user does not care whether a benchmark comes from GLUE or
   LatxaEval; they care whether it tests reading or writing.

2. **Per-benchmark cards** (9 cards).  Rejected because too many, and
   benchmarks within a family are highly correlated.  Skill grouping is
   more informative at a glance.

3. **Weighted aggregation** (e.g., MMLU gets ×2 weight because it's larger).
   Rejected for v1 — keep it simple with equal-weight averaging.  Weights
   can be added later if needed.

4. **Mixing chrF into WRITING**.  Rejected because chrF has no fixed
   ceiling and min-max normalisation across models would make scores
   unstable as new models are added.  Translation already has a dedicated
   heatmap table (Table 2); WRITING stays accuracy-only.

5. **Dropping PRAGMATICS to 4 categories**.  Rejected — intent
   classification measures a qualitatively different aspect of
   language understanding and is valuable to non-technical users
   who want to know which models handle real-world conversation best.
