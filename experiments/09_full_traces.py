"""
09_full_traces.py
-----------------
Phase 1 data generation. Unlike 04_adaptive_loop.py, which stops as soon as it
is confident, this answers EVERY question at EVERY round (K = 5, 10, 15) and
stores everything the policies could look at:

  per question : ranked passage ids + reranker scores (top 15), which of them
                 are gold, the gold answer and its aliases
  per round    : the paper-style answer and S1/S2/S3/confidence, plus
                 n_trace_samples sampled answers (and token logprobs on the
                 local backend) for new confidence signals

From one run, 10_policies.py can then replay static RAG at any K, AB-RAG at
any tau, budget-matched random stopping, the oracle, and new stopping
policies, all offline and on identical answers.

The pooled corpus is built from ALL dev+test questions, so it is the same for
both splits and every backend. Retrieval runs once per dataset and is cached.
Output is appended per question, so a crash or rate limit loses nothing.

--retriever r1 swaps in improvement R1 (14_retrieval_upgrade.py): the dataset's
dev-picked rerank pool and reranker from config r1_retrieval (HotpotQA: the
paper's RRF top-20+20 pool reranked by bge-reranker-base instead of MiniLM).
It has its own ranking cache and its own traces file (label suffix -r1), so
paper and R1 traces sit side by side.

Run:
    python experiments/09_full_traces.py --dataset hotpotqa --split dev --backend local
    python experiments/09_full_traces.py --dataset triviaqa --split test --backend claude --confirm_paid
    python experiments/09_full_traces.py --dataset hotpotqa --split all --backend local --max_hours 10.5
        (the last form is what notebooks/kaggle_phase1.ipynb runs, one dataset per GPU)
    python experiments/09_full_traces.py --dataset hotpotqa --split test --backend local --retriever r1
Output: results/traces_{dataset}_{backend}[-r1].jsonl
"""
import argparse, importlib.util, json, os, sys, time
from collections import Counter
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from tqdm import tqdm
from configs.config import CONFIG
from experiments.confidence import token_probability_confidence, retrieval_score_variance
from experiments.generation import get_generator, ClaudeGenerator, DailyLimitReached

_spec = importlib.util.spec_from_file_location(
    "adaptive_mod", os.path.join(os.path.dirname(__file__), "04_adaptive_loop.py"))
adaptive_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adaptive_mod)

RESULTS = CONFIG["results_dir"]
os.makedirs(RESULTS, exist_ok=True)


def load_split_rows(dataset, split, limit):
    """Returns (rows to answer, rows that make up the shared corpus)."""
    rows = [json.loads(l) for l in open(f"data/{dataset}.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["passages"] and r["gold_pids"]]
    split_path = f"data/splits_{dataset}.json"
    if os.path.exists(split_path):
        splits = json.load(open(split_path))
        pool_ids = set(splits["dev"]) | set(splits["test"])
        want = pool_ids if split == "all" else set(splits[split])
    else:                                   # e.g. the 5-question sample set
        print(f"[!] no {split_path}; using every row as one split")
        pool_ids = want = {r["qid"] for r in rows}
    pool = [r for r in rows if r["qid"] in pool_ids]
    todo = [r for r in rows if r["qid"] in want]
    return (todo[:limit] if limit else todo), pool


def r1_ranked_pool(retr, reranker, text_of, question, depth=100):
    """R1 union100 pool: rerank the union of BM25 top-100 and dense top-100 (the reranker sets the order)."""
    bm = retr.bm25.get_scores(adaptive_mod.tok(question))
    bm_pids = [retr.cpids[i] for i in np.argsort(bm)[::-1][:depth]]
    q = retr.embedder.encode(["Represent this sentence for searching relevant passages: " + question],
                             normalize_embeddings=True, show_progress_bar=False)
    dn_pids = [retr.cpids[i] for i in retr.index.search(q.astype(np.float32), depth)[1][0]]
    pool = list(dict.fromkeys(bm_pids + dn_pids))
    ce = reranker.predict([[question, text_of[p]] for p in pool], batch_size=32, show_progress_bar=False)
    order = np.argsort(ce)[::-1]
    return [pool[i] for i in order], [float(ce[i]) for i in order]


def rank_all(dataset, pool, dev, retriever="paper", r1=None):
    """Rank every pooled question once and cache rankings + corpus embeddings.
    r1 = (rerank pool, reranker) for --retriever r1, pool "rrf20" (the paper's) or "union100"."""
    tag = "" if retriever == "paper" else f"_r1-{r1[0]}-{r1[1].split('/')[-1]}"
    rank_path = f"{RESULTS}/rankings_{dataset}{tag}.json"
    emb_path = f"{RESULTS}/corpus_emb_{dataset}.npy"
    from sentence_transformers import SentenceTransformer, CrossEncoder
    embedder = SentenceTransformer(CONFIG["embed_model"], device=dev)
    if os.path.exists(rank_path) and os.path.exists(emb_path):
        cache = json.load(open(rank_path))
        if cache["n_pool"] == len(pool):
            print(f"[cache] rankings for {len(cache['ranked'])} questions from {rank_path}")
            return cache, np.load(emb_path), embedder.to("cpu")
    if retriever == "paper":
        reranker = CrossEncoder(CONFIG["reranker_model"], device=dev)
    else:
        reranker = CrossEncoder(r1[1], device=dev, max_length=512)
        if dev == "cuda":
            reranker.model.half()
    corpus = adaptive_mod.build_corpus(pool)
    retr = adaptive_mod.Retriever(corpus, embedder, reranker, dev)
    text_of = dict(zip(retr.cpids, retr.ctexts))
    depth = CONFIG["trace_pool_depth"]
    ranked = {}
    for r in tqdm(pool, desc=f"retrieve+rerank ({retriever})", unit="q"):
        if retriever == "paper" or r1[0] == "rrf20":     # the paper's pool, ordered by this reranker
            pids, _, scores = retr.ranked_pool(r["question"], depth=100)
        else:
            pids, scores = r1_ranked_pool(retr, reranker, text_of, r["question"])
        ranked[r["qid"]] = {"pids": pids[:depth], "scores": scores[:depth]}
    emb = retr.index.reconstruct_n(0, retr.index.ntotal)     # normalized passage vectors
    cache = {"n_pool": len(pool), "corpus_pids": retr.cpids, "ranked": ranked}
    json.dump(cache, open(rank_path, "w"))
    np.save(emb_path, emb)
    # free the GPU for the generator: keep only a CPU embedder for short answers
    del retr, reranker
    import torch
    torch.cuda.empty_cache()
    return cache, emb, embedder.to("cpu")


def consistency_s2(answer, ev_idx, emb, embedder):
    """S2 exactly as confidence.evidence_answer_consistency, but reusing the
    cached passage embeddings instead of re-encoding the evidence."""
    a = embedder.encode([answer], normalize_embeddings=True, show_progress_bar=False)[0]
    e = emb[ev_idx].mean(axis=0)
    e = e / (np.linalg.norm(e) + 1e-9)
    return (float(np.dot(a, e)) + 1.0) / 2.0


def paper_s1_from_samples(samples, k=3):
    """The paper's self-consistency S1 and answer: modal answer of the first k
    samples under the paper's light normalization (lowercase + whitespace)."""
    norms = [ClaudeGenerator._norm(s) for s in samples[:k]]
    modal, count = Counter(norms).most_common(1)[0]
    answer = next(s for s in samples[:k] if ClaudeGenerator._norm(s) == modal)
    return answer, count / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--backend", choices=["local", "claude", "ollama", "groq", "openrouter"], default="local")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions (smoke tests)")
    ap.add_argument("--confirm_paid", action="store_true",
                    help="required for the claude backend, which spends API credit")
    ap.add_argument("--n_samples", type=int, default=CONFIG["n_trace_samples"],
                    help="sampled answers per round (min 3: the paper's S1 uses the first 3); "
                         "3 stretches a free-tier daily quota furthest")
    ap.add_argument("--model", default=None,
                    help="override the backend's model from config (e.g. --backend groq "
                         "--model qwen/qwen3.8-27b); each model gets its own traces file")
    ap.add_argument("--rank_only", action="store_true",
                    help="only build the cached retrieval rankings for the dataset, then exit")
    ap.add_argument("--device_map", choices=["auto"], default=None,
                    help="local backend: split the model over all visible GPUs (e.g. 7-8B on 2x T4)")
    ap.add_argument("--load_4bit", action="store_true",
                    help="local backend: 4-bit weights via bitsandbytes; this changes S1, so the "
                         "traces get their own -4bit file")
    ap.add_argument("--max_hours", type=float, default=0,
                    help="stop cleanly after this many hours, e.g. to finish inside a Kaggle session")
    ap.add_argument("--retriever", choices=["paper", "r1"], default="paper",
                    help="r1: the dev-picked rerank pool + reranker (improvement R1); own -r1 traces file")
    ap.add_argument("--r1_pool", choices=["rrf20", "union100"], default=None,
                    help="override the R1 rerank pool (default: the dataset's pick in config r1_retrieval)")
    ap.add_argument("--r1_reranker", default=None,
                    help="override the R1 cross-encoder (default: the dataset's pick in config r1_retrieval)")
    args = ap.parse_args()
    t0 = time.time()
    if args.n_samples < 3:
        ap.error("--n_samples must be at least 3")
    if (args.device_map or args.load_4bit) and args.backend != "local":
        ap.error("--device_map / --load_4bit only apply to the local backend")
    pick = CONFIG["r1_retrieval"].get(args.dataset, (None, None))
    r1 = (args.r1_pool or pick[0], args.r1_reranker or pick[1])
    if args.retriever == "r1" and None in r1:
        ap.error(f"no R1 pick for {args.dataset} in config r1_retrieval; pass --r1_pool and --r1_reranker")

    todo, pool = load_split_rows(args.dataset, args.split, args.limit)
    model = args.model or CONFIG[f"{args.backend}_model"]
    # API backends can switch models, so their files are tagged with the model
    label = (f"{args.backend}-{model.split('/')[-1].replace(':', '-')}"
             if args.model or args.backend in ("groq", "openrouter") else args.backend)
    label += "-4bit" if args.load_4bit else ""
    label += "-r1" if args.retriever == "r1" else ""
    out_path = f"{RESULTS}/traces_{args.dataset}_{label}.jsonl"
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["qid"] for l in open(out_path, encoding="utf-8")}
    todo = [r for r in todo if r["qid"] not in done]
    Ks = [CONFIG["rerank_topk"] + i * CONFIG["k_step"] for i in range(CONFIG["T_max"])]
    n_s = args.n_samples
    calls = len(todo) * len(Ks) * (n_s if args.backend != "local" else 2)
    print(f"{len(todo)} questions to answer ({len(done)} already done); rounds K={Ks}; "
          f"~{calls} generation calls; corpus pooled from {len(pool)} questions")
    if args.rank_only:
        import torch
        rank_all(args.dataset, pool, "cuda" if torch.cuda.is_available() else "cpu",
                 args.retriever, r1)
        print(f"rankings cached for {args.dataset} ({args.retriever})"); return
    if not todo:
        return
    if args.backend == "claude" and not args.confirm_paid:
        print("[!] the claude backend spends API credit; re-run with --confirm_paid"); return
    if args.backend in ("groq", "openrouter"):
        print(f"[free tier] {len(Ks) * n_s} requests per question; the run stops cleanly at the "
              f"daily quota and resumes from the next unanswered question when re-run")

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev.upper()}")
    cache, emb, embedder = rank_all(args.dataset, pool, dev, args.retriever, r1)
    pid2idx = {pid: i for i, pid in enumerate(cache["corpus_pids"])}
    texts = {p["pid"]: p["text"] for r in pool for p in r["passages"]}
    gen = get_generator(args.backend, {**CONFIG, "n_consistency_samples": n_s,
                                       f"{args.backend}_model": model,
                                       "local_device_map": args.device_map,
                                       "local_4bit": args.load_4bit}, device=dev)
    load = {"device_map": args.device_map, "load_4bit": args.load_4bit} if args.backend == "local" else {}
    load["retriever"] = args.retriever
    if args.retriever == "r1":
        load.update(reranker_model=r1[1], rerank_pool={"rrf20": "BM25 top-20 + dense top-20, RRF (the paper's)",
                                                       "union100": "BM25 top-100 + dense top-100"}[r1[0]])
    deadline = t0 + 3600 * args.max_hours if args.max_hours else None

    try:
        write_traces(todo, Ks, n_s, cache, emb, embedder, pid2idx, texts, gen, args.backend, model, out_path,
                     deadline=deadline, extra_settings=load)
    except DailyLimitReached as e:
        print(f"\n[stopped] {e}\nEverything answered so far is saved; re-run the same command "
              f"after the quota resets to continue.")
    if getattr(gen, "usage", None):
        print(f"API usage this run: {dict(gen.usage)}")
    print(f"saved -> {out_path}")


def write_traces(todo, Ks, n_s, cache, emb, embedder, pid2idx, texts, gen, backend, model, out_path,
                 deadline=None, extra_settings=None):
    a, b, g = CONFIG["alpha"], CONFIG["beta"], CONFIG["gamma"]
    with open(out_path, "a", encoding="utf-8") as fout:
        for r in tqdm(todo, desc=f"full traces ({backend})", unit="q"):
            if deadline and time.time() > deadline:
                print("\n[stopped] --max_hours reached; re-run the same command to continue "
                      "from the next unanswered question")
                break
            rk = cache["ranked"][r["qid"]]
            rounds = []
            for K in Ks:
                ev_pids = rk["pids"][:K]
                ev_texts = [texts[p] for p in ev_pids]
                if backend == "local":
                    out = gen.generate(r["question"], ev_texts)
                    answer, logprobs = out["answer"], out["token_logprobs"]
                    c_tok = token_probability_confidence(logprobs)
                    samples = gen.sample(r["question"], ev_texts, n_s, CONFIG["sample_temperature"])
                else:
                    samples = gen.generate(r["question"], ev_texts)["samples"]
                    answer, c_tok = paper_s1_from_samples(samples)
                    logprobs = None
                c_con = consistency_s2(answer, [pid2idx[p] for p in ev_pids], emb, embedder)
                v_ret = retrieval_score_variance(rk["scores"][:K])
                rounds.append({
                    "K": K, "answer": answer, "samples": samples, "token_logprobs": logprobs,
                    "c_tok": round(c_tok, 4), "c_con": round(c_con, 4), "v_ret": round(v_ret, 4),
                    "conf": round(float(np.clip(a * c_tok + b * c_con + g * v_ret, 0, 1)), 4),
                })
            gold = set(r["gold_pids"])
            fout.write(json.dumps({
                "qid": r["qid"], "question": r["question"],
                "gold_answer": r["answer"], "aliases": r.get("aliases", []),
                "pids": rk["pids"], "rerank_scores": [round(s, 4) for s in rk["scores"]],
                "gold_flags": [int(p in gold) for p in rk["pids"]],
                "rounds": rounds,
                "backend": backend, "model": model, "n_samples": n_s,
                "settings": {**{k: CONFIG[k] for k in ("alpha", "beta", "gamma", "tau", "T_max", "k_step",
                                                       "rerank_topk", "n_trace_samples", "sample_temperature",
                                                       "embed_model", "reranker_model")},
                             **(extra_settings or {})},
            }) + "\n")
            fout.flush()


if __name__ == "__main__":
    main()
