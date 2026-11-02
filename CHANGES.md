# Changes in this follow-up

Academic, non-commercial follow-up to *AB-RAG: Adaptive Budgeted Retrieval* (arXiv:2606.29090). All work is on branch `phase0-fixes`. The paper's committed runs in `logs/` are only read, never rewritten; every new output goes to the git-ignored `results/`. Benchmarks and figures are in `results/report/REPORT.md`, rebuilt by `python experiments/12_report.py`.

## 1. Fixes so the repository runs as published

| Change | Files |
|---|---|
| Imports pointed at a missing `src/` package; now `experiments.` | `01`–`07` |
| Run commands and `reproduce_retrieval.sh` paths fixed to `experiments/` | `01`–`07`, `reproduce_retrieval.sh` |
| `gitignore` renamed to `.gitignore` (it was not active); `results/` ignored | `.gitignore` |
| New runs write to `results/`. Replay scripts read the paper's runs with `--runs_dir logs` and can no longer overwrite them | `01`–`07`, `configs/config.py` (`results_dir`) |
| Each adaptive record stores the weights, tau and budget it ran with | `04_adaptive_loop.py` |
| S3 docstring says "reward", matching the code | `confidence.py` |
| Missing dependencies (`ollama`, `openai`) | `requirements.txt` |

## 2. Data and evaluation protocol

| Change | Files |
|---|---|
| TriviaQA re-prepared with its answer **aliases**; first 600 questions identical to the paper's file | `00_prepare_data.py`, `00b_prepare_triviaqa.py --n 800`, `data/triviaqa.jsonl` (`00c` alias backfill, superseded) |
| Fixed **splits**: test = first 500 evaluable questions (contains the paper's 200), dev = next 300; tune only on dev | `00d_make_splits.py`, `data/splits_*.json` |
| Pooled corpus built from all 800 dev+test questions: 7,947 HotpotQA and 11,124 TriviaQA passages (paper: 4,963 and 2,582) | `09_full_traces.py` |
| Three correctness labels: EM, **EM+alias**, **contains** (gold appears as a whole phrase) | `experiments/metrics.py` |
| **Paired significance**: exact McNemar and paired bootstrap 95% CI | `experiments/metrics.py` |

## 3. New experiments

| Script | What it does |
|---|---|
| `08_rescore.py` | Re-scores the paper's runs under the three labels; recovers the weights each run really used |
| `09_full_traces.py` | Answers **every question at every round** (K = 5, 10, 15) with 5 samples per round and stores all signals; cached rankings, resumes after a crash, stops cleanly at an API daily quota or after `--max_hours`. `--retriever r1` answers with R1 retrieval into its own `-r1` traces file |
| `10_policies.py` | Replays static K, AB-RAG at any tau, AB-RAG + max-confidence, **budget-matched random stopping** and the oracle on identical answers; paired tests vs static K=10; gain decomposition; tau sweep with the random control's 95% band at every tau; N2 as on the released runs (S2, S3 held at their run mean); per-question correctness so the report can pair traces variants |
| `11_comparison_figures.py` | Earlier figures of the paper's numbers next to the re-scored runs (`results/figures/cmp*.png`); superseded by `12_report.py` |
| `11_value_policy.py` | N1, value-of-retrieval stopping. Built and checked on synthetic data only; **set aside, not run** |
| `12_report.py` | Every paper table recomputed next to the paper's numbers, the N2–N4 diagnostics, the improvements, and all figures |
| `13_answer_extract.py` | **I1** short-answer stage (local Qwen2.5-1.5B cuts a sentence answer to the answering phrase, copy-only, never sees the gold) and **I2** best-round answer, on the paper's saved answers; `--traces` writes an I1 copy (`-i1`) of any full-traces file |
| `14_retrieval_upgrade.py` | **R1** compares rerank pools (the paper's RRF top-20+20, or BM25 top-100 ∪ dense top-100) and rerankers (MiniLM, BGE base and large); picks one per dataset on dev questions, reports it on the paper's recall setting. The pick goes in `configs/config.py` (`r1_retrieval`), which `09 --retriever r1` uses |
| `paper_followup/` | Draft of the follow-up paper (LaTeX, `main.tex` + `refs.bib`); every number from `results/`; no PENDING items left |

## 4. Generation backends and compute

| Change | Files |
|---|---|
| Free-tier **Groq** and **OpenRouter** backends (OpenAI-compatible): request pacing, retry on 429/5xx, clean stop at the daily quota, per-model reasoning settings, usage counters. Keys only in the git-ignored `.env` (`.env.example` has placeholders) | `generation.py`, `configs/config.py` |
| Local model: batched sampling; `--device_map auto` (split across GPUs) and `--load_4bit` | `generation.py`, `09_full_traces.py` |
| **Colab** notebook runs everything on the 500 test questions: traces with the paper's retrieval and with R1, I1 on every traces file, replays, the report with paired tests; results on Drive | `notebooks/colab_phase1.ipynb` |
| Kaggle notebook (2× T4, one dataset per GPU) | `notebooks/kaggle_phase1.ipynb` |
| The test-split traces were produced on a university PBS cluster instead (3× H100, one run per GPU, about 20 minutes including I1), from the same code zip with the laptop's package versions; the results were copied back and replayed locally | — |
| One zip with code, data, cached rankings, locally computed results the report reads (R1 retrieval, recall, I1 cache), the paper's logs and figures; never `.env` or traces | `notebooks/make_code_zip.py` |
| Local environment only: scipy pinned to 1.15.3 because Windows Smart App Control blocks newer native DLLs | `.venv` |

## 5. Novelties

| | Status |
|---|---|
| **N1** value-of-retrieval stopping (stop / retrieve / abstain) | Code exists, set aside, no results |
| **N2** do S2 and S3 ever change a stop decision? | Measured on all four committed runs (report §7) and on the new traces (report §10): 3/602, 8/576, 3/627 executed decisions flip |
| **N3** answer format and verbosity confound confidence evaluation | Measured on all four committed runs (report §1, §2, §4, §8); brevity again predicts EM (AUROC 0.61) but not contains (0.46, 0.43) on the new traces |
| **N4** gain decomposition + budget-matched random stopping | Decomposition measured (report §9); random control on the new traces (report §9, §10): AB-RAG is no better than random stopping |

## 6. What re-checking the paper found (details in the report)

- Open-retrieval recall, re-run from scratch, reproduces the paper exactly. The QA tables match the paper's saved answers in `logs/` under EM. Of the models, only Qwen-1.5B was re-run (section 8); the Claude and Llama runs were not.
- **TriviaQA gain:** AB-RAG's +4.5 EM over static K=10 on Claude/TriviaQA is significant under EM (p=0.004) and EM+alias (p=0.012). It disappears under contains (80.5 vs 81.0, p=1.0): the gain is mostly answer format, not more correct answers.
- **Weights:** two of the four runs (Qwen and Claude on HotpotQA) used weights 0.5 / 0.4 / −0.1, not the reported 0.7 / 0.05 / 0.25.
- **S2 and S3 (N2):** holding S2 and S3 at their run means changes 0 stop decisions in three runs and 8 of 218 in the fourth.
- **Brevity (N3):** answer length alone predicts EM-correctness better than any confidence signal for Claude: AUROC 0.85 and 0.91 on first-round answers, 0.83 and 0.90 on final answers.
- **Low-confidence answers:** under contains, 53–61% of Claude's low-confidence answers are correct, not 0%.
- **Paper's S3 numbers:** the S3 AUROC for TriviaQA (0.594) is a different quantity from the S3 the loop used, which scores 0.456. Qwen's S3 row is sign-flipped.
- **tau sweep:** only tau ≤ 0.6 can be replayed from the committed runs. The paper's tau = 0.30 row equals static K=10, not the K=5 answers that tau implies.
- **Gain decomposition (N4):** on Claude/TriviaQA under EM, +3.0 of the +4.5 comes from starting at K=5 rather than K=10, and +1.5 from iterating.

## 7. Improvements over the paper's numbers (report §10)

| | Paper AB-RAG EM | With I1 + I2 | Paired vs paper |
|---|---|---|---|
| Qwen-1.5B / HotpotQA | 34.5 | 38.5 | +8/−0, p=0.008 |
| Llama-3.2-3B / HotpotQA | 45.0 | 48.0 | +7/−1, p=0.07 |
| Claude Haiku / HotpotQA | 33.0 | 45.5 | +25/−0, p<0.001 |
| Claude Haiku / TriviaQA | 40.0 | 46.5 | +13/−0, p<0.001 |

These re-use the paper's answers: I1 fixes answer format, it adds no new correct content (contains does not rise). I1's instruction was revised once after reading errors on these 200 questions; the frozen version gets its clean test on the new runs.

**R1 (retrieval, report §10):** on HotpotQA the dev pick is the paper's pool reranked by bge-reranker-base: recall@5 86.6 → 92.9 and recall@10 94.0 → 96.4 on the paper's 500 questions. On TriviaQA every BGE reranker lost recall on dev, so the pick keeps MiniLM (wider pool) and matches the paper within 0.4 points at @5 and @10; R1 is not run there.

## 8. New runs on the 500-question test split (report §10, paper Section 8)

Qwen2.5-1.5B, the paper's settings, every round answered for every question. On the paper's 200 HotpotQA questions the regenerated pipeline scores 34.0 static / 36.0 AB-RAG EM (paper: 34.5 / 34.5, smaller corpus).

| EM, 500 questions | Static K=10 | AB-RAG | Random stopping, same budget [95%] | Oracle |
|---|---|---|---|---|
| HotpotQA, paper retrieval | 36.2 | 38.0 | 37.9 [37.0, 38.8] | 45.4 |
| HotpotQA, R1 + I1 | 44.8 | 44.2 | 44.3 [43.4, 45.2] | 52.0 |
| TriviaQA, paper retrieval | 47.4 | 47.8 | 47.9 [46.8, 49.0] | 55.0 |
| TriviaQA, + I1 | 50.2 | 50.0 | 50.3 [49.2, 51.2] | 57.4 |

- **Stopping rule:** AB-RAG is inside the random control's interval in 5 of 6 pipelines at tau = 0.6 (below it with R1), and within 0.6 points of the interval at every tau from 0.30 to 0.95. It stops after round 1 for 85–91% of questions, so it behaves like static K=5. No pipeline shows a significant AB-RAG vs static K=10 difference (p > 0.18).
- **I1 on unseen questions:** +4.2 (HotpotQA, +21/−0) and +2.2 (TriviaQA, +11/−0) EM; on questions 201–500, never used to revise the instruction, +4.7 (p<0.001) and +2.4 (p=0.016). Contains −1.0 and −0.2.
- **I2:** no effect (at most two answers change), since AB-RAG rarely iterates.
- **R1:** static K=10 EM +4.4 (36.2 → 40.6, p=0.009), but AB-RAG only +0.8 (p=0.72). R1 + I1: AB-RAG 38.0 → 44.2 (p<0.001; +6.0 on questions 201–500, p=0.013); static K=10 reaches 44.8. Contains rises by at most 2.6 points, not significantly.
