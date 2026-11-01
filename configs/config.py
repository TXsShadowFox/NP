"""
Central configuration. Everything tunable lives here so the report's
hyperparameter table matches the code exactly. Import CONFIG everywhere.
"""

CONFIG = {
    # ---- retrieval ----
    "embed_model": "BAAI/bge-large-en-v1.5",
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L12-v2",
    # R1 (09 --retriever r1): (rerank pool, reranker) per dataset, picked on dev questions by
    # 14_retrieval_upgrade.py (best mean recall@5/@10). rrf20 = the paper's pool (BM25 top-20 +
    # dense top-20, RRF); union100 = BM25 top-100 + dense top-100. On TriviaQA no BGE reranker
    # beat MiniLM on dev, so its pick keeps the paper's reranker.
    "r1_retrieval": {"hotpotqa": ("rrf20", "BAAI/bge-reranker-base"),
                     "triviaqa": ("union100", "cross-encoder/ms-marco-MiniLM-L12-v2")},
    "bm25_topk": 20,          # top-k from sparse retriever
    "dense_topk": 20,         # top-k from dense retriever
    "rrf_k": 60,              # RRF smoothing constant (standard value)
    "rerank_topk": 5,         # docs kept after reranking as evidence

    # ---- BM25 params (standard Okapi values, reported in your paper) ----
    "bm25_k1": 1.5,
    "bm25_b": 0.75,

    # ---- confidence estimation weights ----
    # Conf = alpha*token + beta*consistency + gamma*Var(R)
    # NOTE: gamma is now a REWARD (sign flipped from penalty) per signal diagnostics:
    # high rerank-score variance => reranker cleanly separated relevant/irrelevant => good.
    # beta kept small: S2 (evidence consistency) found non-predictive on short-answer QA
    # (retained in framework for completeness; reported in ablation).
    "alpha": 0.7,             # token-probability / self-consistency weight (primary signal)
    "beta": 0.05,             # evidence-consistency weight (near-zero; non-predictive)
    "gamma": 0.25,            # retrieval-variance REWARD weight (now positive contribution)

    # ---- adaptive policy ----
    "tau": 0.6,               # confidence threshold to STOP
    "T_max": 3,               # retrieval budget (max iterations)
    "k_step": 5,              # extra docs pulled in per RETRIEVE_MORE round

    # ---- generation ----
    "local_model": "Qwen/Qwen2.5-1.5B-Instruct",  # fits 4GB VRAM in fp16, exposes token logprobs
    "local_device_map": None,  # "auto" = split the model over all GPUs (09's --device_map)
    "local_4bit": False,       # bitsandbytes 4-bit weights for 7-8B on one T4 (09's --load_4bit)
    "claude_model": "claude-haiku-4-5-20251001",   # secondary backend (no logprobs -> self-consistency)
    "ollama_model": "llama3.2:3b",                  # open-source tier via Ollama (self-consistency)
    # free-tier API backends (OpenAI-compatible); keys go in the git-ignored .env
    # Groq free plan (checked 2026-11-01), quotas are PER MODEL, so models can run
    # in parallel: gpt-oss-120b / gpt-oss-20b / qwen3.8-27b 1K req/day, 8K tokens/min,
    # 200K tokens/day; allam-2-7b 7K req/day, 6K tokens/min. Cached prompt tokens
    # do not count toward limits. Pick a model per run with 09's --model.
    "groq_model": "openai/gpt-oss-120b",   # same free quota as gpt-oss-20b, stronger model
    "groq_rpm": 30,
    "groq_extra_body": {   # by model-id prefix; models not listed (allam) get no extras
        "openai/gpt-oss": {"reasoning_effort": "low"},
        "qwen/": {"reasoning_effort": "low", "reasoning_format": "hidden"},   # keep <think> out of answers
    },
    # OpenRouter ":free" models (checked 2026-11-01): 20 RPM; 50 req/day per
    # account, or 1000/day once $10+ of credits were ever bought.
    "openrouter_model": "google/gemma-4-31b-it:free",
    "openrouter_rpm": 20,
    "openrouter_extra_body": {"": {"reasoning": {"effort": "low", "exclude": True}}},
    "api_max_tokens": 512,    # reasoning models spend completion tokens thinking before answering
    "n_consistency_samples": 3,    # for Claude/Ollama paths: samples to estimate self-consistency confidence
    "max_new_tokens": 64,     # short-answer QA needs very little
    "gen_temperature": 0.0,   # deterministic answers for reproducibility

    # ---- full traces (09_full_traces.py) ----
    # Every question is answered at ALL T_max rounds (K = 5, 10, 15) so that static
    # baselines, AB-RAG at any tau, random stopping and new policies can all be
    # replayed offline from one generation run.
    "n_trace_samples": 5,     # sampled answers per round (temp 0.7); the paper's S1 uses the first 3
    "sample_temperature": 0.7,
    "trace_pool_depth": 15,   # ranked passages stored per question (= largest K)

    # ---- N1: value-of-retrieval stopping (11_value_policy.py) ----
    "n1_label": "contains",   # correctness label the policy learns (EM rewards terse format, see N3)
    "n1_lambda": 0.002,       # cost per passage read, in accuracy units (10 passages = 2 points)
    "n1_risk": 0.0,           # penalty for a wrong answer vs abstaining (0 = plain accuracy)
    "n1_folds": 5,            # cross-fitting folds for out-of-fold training targets

    # ---- eval ----
    "seed": 42,
    "results_dir": "results",   # new runs land here; the paper's committed runs are in logs/
    # fixed splits (00d_make_splits.py): test = first 500 evaluable questions (includes
    # the paper's 200, so results stay comparable), dev = next 300. Tune ONLY on dev.
    "n_test": 500,
    "n_dev": 300,
}
