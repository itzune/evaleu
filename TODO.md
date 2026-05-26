# TODO

## Basque Evaluation Expansion Backlog

- [x] Add support for **MMLU EU** using `orai-nlp/MMLU_HT_eu_sample` as the first integration target (MVP adapter + benchmark wiring).
- [ ] Decide and document whether `orai-nlp/MMLU_HT_eu_sample` is temporary (pilot) or long-term benchmark source.
- [ ] Evaluate optional multilingual alternatives with `eu` coverage (for larger-scale follow-up):
  - `alexandrainst/m_mmlu`
  - `jon-tow/okapi_mmlu`

## Additional Basque Benchmarks to Add (pending)

- [ ] **Math reasoning in Basque**
  - Candidate: `mgsm_native_cot_eu`
- [ ] **Reading comprehension in Basque**
  - Candidate: `xstorycloze_eu`
  - Candidate: `belebele_eus_Latn`
- [ ] **Science / commonsense QA in Basque**
  - Candidate: `arc_eu_easy_mc`
  - Candidate: `arc_eu_challenge_mc`
  - Candidate: `piqa_eu_mc`
  - Candidate: `siqa_eu_mc`
- [ ] **Exam / proficiency / trivia coverage alignment**
  - Candidate: `eus_exams_eu`
  - Candidate: `eus_proficiency`
  - Candidate: `eus_trivia`
- [ ] **Basque QA variants**
  - Candidate: `bertaqa_eu_local`
  - Candidate: `bertaqa_eu_global`
- [ ] **Other candidate benchmark from discussion**
  - Candidate: `bl2mp`

## Integration Planning (no implementation yet)

- [ ] Define per-benchmark adapter requirements (format, prompt style, scoring).
- [ ] Define evaluation order for incremental rollout (start with MMLU EU, then high-impact tasks).
- [ ] Add acceptance criteria for each new benchmark before publishing to site.
- [ ] Decide which benchmarks are shown in public leaderboard vs experimental section.
