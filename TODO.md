# TODO

## Basque Evaluation Expansion Backlog

- [x] Add support for **MMLU EU** using `orai-nlp/MMLU_HT_eu_sample` as the first integration target (MVP adapter + benchmark wiring).
- [ ] Decide and document whether `orai-nlp/MMLU_HT_eu_sample` is temporary (pilot) or long-term benchmark source.
- [ ] Evaluate optional multilingual alternatives with `eu` coverage (for larger-scale follow-up):
  - `alexandrainst/m_mmlu`
  - `jon-tow/okapi_mmlu`

## Already Implemented

- [x] **MMLU EU** — `orai-nlp/MMLU_HT_eu_sample`
- [x] **Math reasoning** — `MGSM_eu` (HiTZ/MGSM-eu, 3-shot CoT)
- [x] **Basque QA** — `BertaQA_eu` (combined local + global, HiTZ/BertaQA)
- [x] **Exam / proficiency / trivia**
  - [x] `EusTrivia` (HiTZ/EusTrivia)
  - [x] `LatxaEval_eusexams` (HiTZ/EusExams)
  - [x] `LatxaEval_eusproficiency` (HiTZ/EusProficiency)
- [x] **BasqueGLUE** — `BasqueGLUE_qnli`, `BasqueGLUE_bec`, `BasqueGLUE_wic`, `BasqueGLUE_intent`
- [x] **XNLI** — `XNLIeu`
- [x] **LatxaEval reading** — `LatxaEval_eusreading`
- [x] **Flores translation** — `eu↔en`, `eu↔es` (4 directions)

## Still Pending

- [ ] **Reading comprehension in Basque**
  - [x] `Belebele_eu` (facebook/belebele, eus_Latn, 4-way MC reading comprehension)
  - Candidate: `xstorycloze_eu`
- [ ] **Science / commonsense QA in Basque**
  - Candidate: `arc_eu_easy_mc`
  - Candidate: `arc_eu_challenge_mc`
  - Candidate: `piqa_eu_mc`
  - Candidate: `siqa_eu_mc`
- [ ] **Other candidates**
  - Candidate: `bl2mp`
  - Multilingual alternatives: `alexandrainst/m_mmlu`, `jon-tow/okapi_mmlu`

## Integration Planning (no implementation yet)

- [ ] Define per-benchmark adapter requirements (format, prompt style, scoring).
- [ ] Define evaluation order for incremental rollout (start with MMLU EU, then high-impact tasks).
- [ ] Add acceptance criteria for each new benchmark before publishing to site.
- [ ] Decide which benchmarks are shown in public leaderboard vs experimental section.
- [ ] Decide and document whether `orai-nlp/MMLU_HT_eu_sample` is temporary (pilot) or long-term benchmark source.
