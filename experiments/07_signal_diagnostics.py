"""
07_signal_diagnostics.py
------------------------
Investigates WHY signals S2 (evidence consistency) and S3 (retrieval variance)
are anti-predictive (AUROC < 0.5), and tests improved formulations.

Key idea: the saved adaptive results already contain the QUESTION, the chosen
ANSWER, and gold answer. S2 and S3 depend only on (answer, evidence, rerank
scores) — NOT on the LLM. So we can re-derive evidence for each question via
retrieval (GPU, fast, no API calls) and recompute S2/S3 under several
formulations, then measure AUROC vs correctness for each.

Formulations tested:
  S2 (evidence-answer consistency):
    - mean   : sim(answer, mean(evidence))          [original]
    - max    : max_i sim(answer, evidence_i)        [does answer match the BEST passage?]
    - top1   : sim(answer, top_reranked_passage)
  S3 (retrieval score variance):
    - penalty(+): -var   (original: high var = bad)
    - reward(-) : +var   (test inverted: high var = reranker separated good/bad = good)
    - mean_score: mean rerank score (alternative: high mean = confident retrieval)

Run:
    python experiments/07_signal_diagnostics.py --dataset hotpotqa --backend claude --n 200
"""
import argparse, json, os, re, string, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder
from configs.config import CONFIG

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "adaptive_mod", os.path.join(os.path.dirname(__file__), "04_adaptive_loop.py"))
adaptive_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adaptive_mod)
Retriever, build_corpus, load_rows = adaptive_mod.Retriever, adaptive_mod.build_corpus, adaptive_mod.load_rows


def normalize(s):
    s = s.lower(); s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())

def em(p, g):
    return int(normalize(p) == normalize(g))

def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg: return float("nan")
    w = sum((1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg)
    return w/(len(pos)*len(neg))

def cos01(a, b):
    return (float(np.dot(a, b)) + 1.0)/2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--backend", default="claude")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--runs_dir", default=CONFIG["results_dir"],
                    help="where to read run outputs from; use logs to replay the paper's committed runs")
    args = ap.parse_args()
    RUNS, RESULTS = args.runs_dir, CONFIG["results_dir"]   # read from RUNS, always write to RESULTS
    os.makedirs(RESULTS, exist_ok=True)

    adaptive_path = f"{RUNS}/adaptive_{args.dataset}_{args.backend}.jsonl"
    saved = {json.loads(l)["qid"]: json.loads(l) for l in open(adaptive_path)}

    rows = load_rows(f"data/{args.dataset}.jsonl", args.n)
    rows = [r for r in rows if r["qid"] in saved]
    corpus = build_corpus(rows)

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(CONFIG["embed_model"], device=dev)
    reranker = CrossEncoder(CONFIG["reranker_model"], device=dev)
    retr = Retriever(corpus, embedder, reranker, dev)

    K = CONFIG["rerank_topk"]
    # collect per-question signal variants + correctness label
    s2_mean, s2_max, s2_top1 = [], [], []
    s3_negvar, s3_posvar, s3_meanscore = [], [], []
    labels = []

    for r in tqdm(rows, desc="re-deriving signals", unit="q"):
        ans = saved[r["qid"]]["pred_answer"]
        labels.append(em(ans, r["answer"]))
        pids, texts, scores = retr.ranked_pool(r["question"], depth=100)
        ev_texts, ev_scores = texts[:K], scores[:K]

        a_emb = embedder.encode([ans], normalize_embeddings=True)[0]
        E = embedder.encode(ev_texts, normalize_embeddings=True)
        # S2 variants
        e_mean = E.mean(axis=0); e_mean /= (np.linalg.norm(e_mean)+1e-9)
        s2_mean.append(cos01(a_emb, e_mean))
        sims = [cos01(a_emb, e) for e in E]
        s2_max.append(max(sims))
        s2_top1.append(sims[0])           # top reranked passage
        # S3 variants
        v = float(np.var(ev_scores))
        s3_negvar.append(-v)
        s3_posvar.append(+v)
        s3_meanscore.append(float(np.mean(ev_scores)))

    print(f"\nn={len(labels)}, positives(correct)={sum(labels)}")
    print("\n=== S2 (evidence-answer consistency) formulations — AUROC vs correctness ===")
    for name, vals in [("mean (original)", s2_mean), ("max-passage", s2_max), ("top1-passage", s2_top1)]:
        print(f"  {name:<18} AUROC = {auroc(vals, labels):.3f}")
    print("\n=== S3 (retrieval signal) formulations — AUROC vs correctness ===")
    for name, vals in [("-variance (original)", s3_negvar), ("+variance (inverted)", s3_posvar),
                       ("mean rerank score", s3_meanscore)]:
        print(f"  {name:<22} AUROC = {auroc(vals, labels):.3f}")

    out = {
        "n": len(labels),
        "S2": {"mean": auroc(s2_mean, labels), "max": auroc(s2_max, labels), "top1": auroc(s2_top1, labels)},
        "S3": {"neg_var": auroc(s3_negvar, labels), "pos_var": auroc(s3_posvar, labels),
               "mean_score": auroc(s3_meanscore, labels)},
    }
    json.dump(out, open(f"{RESULTS}/signal_diag_{args.dataset}_{args.backend}.json", "w"), indent=2)
    print(f"\nsaved -> results/signal_diag_{args.dataset}_{args.backend}.json")
    print("\nInterpretation: any formulation with AUROC > 0.55 is a usable signal.")
    print("If max/top1 S2 or mean-score S3 beats 0.55, we adopt it and all 3 signals contribute.")


if __name__ == "__main__":
    main()
