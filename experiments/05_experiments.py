"""
05_experiments.py
-----------------
The headline experiment. Compares STATIC RAG (always uses a fixed K, single
pass) against AB-RAG (adaptive budgeted loop) on the SAME questions, and runs
the confidence-vs-correctness analysis that validates the core idea.

Metrics:
  - Exact Match (EM) and token-F1 (SQuAD-style) vs gold answers
  - Average retrieval iterations / query (efficiency)
  - Confidence-correctness analysis: accuracy of high-confidence vs
    low-confidence answers, and AUROC of confidence predicting correctness.

It reuses results/adaptive_{dataset}_{backend}.jsonl (from 04) for AB-RAG,
and runs a static baseline (fixed K, one pass) inline for fair comparison.

Run:
    python experiments/05_experiments.py --dataset hotpotqa --n 100 --backend local
"""
import argparse, json, os, re, sys, string
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder
from configs.config import CONFIG
from experiments.generation import get_generator
from experiments.confidence import (token_probability_confidence, evidence_answer_consistency,
                            retrieval_score_variance)

# reuse the Retriever from the adaptive module
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "adaptive_mod", os.path.join(os.path.dirname(__file__), "04_adaptive_loop.py"))
adaptive_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adaptive_mod)
Retriever = adaptive_mod.Retriever
build_corpus = adaptive_mod.build_corpus
load_rows = adaptive_mod.load_rows
confidence_from = adaptive_mod.confidence_from


# ---------- SQuAD-style EM / F1 ----------
def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred, gold):
    return float(normalize(pred) == normalize(gold))


def token_f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = {}
    for t in p:
        if t in g:
            common[t] = min(p.count(t), g.count(t))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p)
    recall = num_same / len(g)
    return 2 * precision * recall / (precision + recall)


def auroc(scores, labels):
    """AUROC of `scores` predicting binary `labels` (1=correct). Rank-based."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    # probability a random correct answer has higher confidence than a random wrong one
    wins = 0.0
    for sp in pos:
        for sn in neg:
            wins += 1.0 if sp > sn else (0.5 if sp == sn else 0.0)
    return wins / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--backend", choices=["local", "claude", "ollama", "groq", "openrouter"], default="local")
    ap.add_argument("--static_k", type=int, default=10,
                    help="fixed K for the static RAG baseline")
    ap.add_argument("--runs_dir", default=CONFIG["results_dir"],
                    help="where to read run outputs from; use logs to replay the paper's committed runs")
    args = ap.parse_args()
    RUNS, RESULTS = args.runs_dir, CONFIG["results_dir"]   # read from RUNS, always write to RESULTS
    os.makedirs(RESULTS, exist_ok=True)

    rows = load_rows(f"data/{args.dataset}.jsonl", args.n)
    corpus = build_corpus(rows)
    print(f"{len(rows)} questions; corpus={len(corpus)}; backend={args.backend}")

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(CONFIG["embed_model"], device=dev)
    reranker = CrossEncoder(CONFIG["reranker_model"], device=dev)
    gen = get_generator(args.backend, CONFIG, device=dev)
    retr = Retriever(corpus, embedder, reranker, dev)

    # ---- load AB-RAG results from step 04 ----
    adaptive_path = f"{RUNS}/adaptive_{args.dataset}_{args.backend}.jsonl"
    if not os.path.exists(adaptive_path):
        print(f"[!] missing {adaptive_path}. Run 04_adaptive_loop.py first."); return
    adaptive = {json.loads(l)["qid"]: json.loads(l) for l in open(adaptive_path)}

    # ---- CHECKPOINT: cache each static answer as we go, so a crash or
    #      rate-limit never wastes API credit. On restart, skip done qids. ----
    ckpt_path = f"{RESULTS}/static_cache_{args.dataset}_{args.backend}.jsonl"
    done = {}
    seed_path = f"{RUNS}/static_cache_{args.dataset}_{args.backend}.jsonl"
    for p in dict.fromkeys([seed_path, ckpt_path]):   # RUNS copy is only read, never appended to
        if os.path.exists(p):
            for l in open(p):
                try:
                    d = json.loads(l); done[d["qid"]] = d
                except Exception:
                    continue
    if done:
        print(f"[checkpoint] resuming: {len(done)} static answers already cached")

    static_rows, abrag_rows = [], []
    ckpt_fout = open(ckpt_path, "a")   # append preserves prior work

    for r in tqdm(rows, desc="static baseline", unit="q"):
        qid, q, gold = r["qid"], r["question"], r["answer"]
        if qid not in adaptive:
            continue
        # STATIC: fixed K, single pass (skip -> $0 if already cached)
        if qid in done:
            d = done[qid]
            static_rows.append({"qid": qid, "pred": d["pred"], "gold": gold,
                                "conf": d["conf"], "iters": 1})
        else:
            pids, texts, scores = retr.ranked_pool(q, depth=100)
            ev_texts, ev_scores = texts[:args.static_k], scores[:args.static_k]
            g = gen.generate(q, ev_texts)
            conf, *_ = confidence_from(g, g["answer"], ev_texts, ev_scores, embedder)
            static_rows.append({"qid": qid, "pred": g["answer"], "gold": gold,
                                "conf": conf, "iters": 1})
            ckpt_fout.write(json.dumps({"qid": qid, "pred": g["answer"], "conf": conf}) + "\n")
            ckpt_fout.flush()
        # ABRAG from saved trace
        a = adaptive[qid]
        abrag_rows.append({"qid": qid, "pred": a["pred_answer"], "gold": gold,
                           "conf": a["final_confidence"], "iters": a["iterations_used"]})

    ckpt_fout.close()

    def summarize(name, rs):
        em = np.mean([exact_match(x["pred"], x["gold"]) for x in rs])
        f1 = np.mean([token_f1(x["pred"], x["gold"]) for x in rs])
        iters = np.mean([x["iters"] for x in rs])
        labels = [int(exact_match(x["pred"], x["gold"])) for x in rs]
        confs = [x["conf"] for x in rs]
        au = auroc(confs, labels)
        return {"EM": round(100*em, 1), "F1": round(100*f1, 1),
                "avg_iters": round(float(iters), 2),
                "conf_auroc": round(au, 3) if au == au else None}

    static_sum = summarize("static", static_rows)
    abrag_sum = summarize("abrag", abrag_rows)

    # confidence-correctness: accuracy of high vs low confidence (AB-RAG)
    tau = CONFIG["tau"]
    hi = [x for x in abrag_rows if x["conf"] >= tau]
    lo = [x for x in abrag_rows if x["conf"] < tau]
    def acc(rs): return round(100*np.mean([exact_match(x["pred"], x["gold"]) for x in rs]), 1) if rs else None
    conf_analysis = {
        "tau": tau,
        "high_conf_n": len(hi), "high_conf_EM": acc(hi),
        "low_conf_n": len(lo), "low_conf_EM": acc(lo),
    }

    out = {
        "n": len(static_rows), "static_k": args.static_k,
        "static": static_sum, "abrag": abrag_sum,
        "confidence_analysis": conf_analysis,
    }
    op = f"{RESULTS}/experiment_{args.dataset}_{args.backend}.json"
    json.dump(out, open(op, "w"), indent=2)

    print("\n================ HEADLINE RESULTS ================")
    print(f"{'metric':<16}{'Static RAG':>14}{'AB-RAG':>14}")
    for m, label in [("EM", "EM (%)"), ("F1", "F1 (%)"),
                     ("avg_iters", "avg retrievals"), ("conf_auroc", "conf AUROC")]:
        print(f"{label:<16}{str(static_sum[m]):>14}{str(abrag_sum[m]):>14}")
    print("\n--- Confidence validates correctness? (AB-RAG) ---")
    print(f"  high-conf (>= {tau}): EM={conf_analysis['high_conf_EM']}%  (n={conf_analysis['high_conf_n']})")
    print(f"  low-conf  (<  {tau}): EM={conf_analysis['low_conf_EM']}%  (n={conf_analysis['low_conf_n']})")
    print(f"\nsaved -> {op}")


if __name__ == "__main__":
    main()
