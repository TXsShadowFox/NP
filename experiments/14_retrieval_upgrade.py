"""
14_retrieval_upgrade.py
-----------------------
Improvement R1: a wider rerank pool and a stronger cross-encoder, measured on
the paper's open-retrieval benchmark (Table tab:recall).

The paper reranks only the RRF fusion of BM25 top-20 and dense top-20 (at most
40 passages) with ms-marco-MiniLM-L12-v2, so reranked recall@50 can never pass
hybrid recall@50, and at @5 the reranker trails plain dense retrieval. Here:

  pool      rrf20     the paper's pool: BM25 top-20 + dense top-20, RRF
            union100  BM25 top-100 + dense top-100 (the reranker sees ~150-200)
  reranker  the paper's MiniLM-L12, BAAI/bge-reranker-base, BAAI/bge-reranker-large

The rrf20 pool is a subset of union100, so each reranker scores union100 once
and rrf20 reuses those scores.

Protocol: pick the (pool, reranker) pair on DEV questions (rows 500-800, disjoint
from every question the paper evaluated) by mean recall@5/@10, the evidence
sizes of AB-RAG's first two rounds; then report it on the paper's setting
(HotpotQA first 500, TriviaQA first 200), where the paper's numbers apply.

Run:
    python experiments/14_retrieval_upgrade.py --dataset hotpotqa --eval dev
    python experiments/14_retrieval_upgrade.py --dataset hotpotqa --eval paper
Output: results/retrieval_upgrade_{dataset}_{eval}.json
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from configs.config import CONFIG

RERANKERS = [CONFIG["reranker_model"], "BAAI/bge-reranker-base", "BAAI/bge-reranker-large"]
PAPER_N = {"hotpotqa": 500, "triviaqa": 200}
DEV = (500, 800)
KS = [5, 10, 15, 20, 50]
DEPTH = 100


def load_rows(path, start, stop):
    rows = [json.loads(l) for i, l in enumerate(open(path, encoding="utf-8")) if start <= i < stop]
    return [r for r in rows if r["passages"] and r["gold_pids"]]


def tok(s):
    return s.lower().split()


def recall(ranked, gold, k):
    return len(set(ranked[:k]) & set(gold)) / len(gold)


def rrf(lists):
    fused = {}
    for pids in lists:
        for rank, pid in enumerate(pids):
            fused[pid] = fused.get(pid, 0.0) + 1.0 / (CONFIG["rrf_k"] + rank + 1)
    return sorted(fused, key=fused.get, reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--eval", choices=["dev", "paper"], default="dev")
    ap.add_argument("--rerankers", nargs="+", default=RERANKERS)
    args = ap.parse_args()

    start, stop = DEV if args.eval == "dev" else (0, PAPER_N[args.dataset])
    rows = load_rows(f"data/{args.dataset}.jsonl", start, stop)
    corpus = list({p["pid"]: p for r in rows for p in r["passages"]}.values())
    cpids, ctexts = [p["pid"] for p in corpus], [p["text"] for p in corpus]
    text_of = dict(zip(cpids, ctexts))
    print(f"{args.dataset} {args.eval}: {len(rows)} questions, corpus {len(corpus)} passages")

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bm25 = BM25Okapi([tok(t) for t in ctexts], k1=CONFIG["bm25_k1"], b=CONFIG["bm25_b"])
    embedder = SentenceTransformer(CONFIG["embed_model"], device=dev)
    emb = embedder.encode(ctexts, normalize_embeddings=True, show_progress_bar=True, batch_size=64)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb.astype(np.float32))

    cands = []
    for r in tqdm(rows, desc="first stage", unit="q"):
        q = r["question"]
        bm = [cpids[i] for i in np.argsort(bm25.get_scores(tok(q)))[::-1][:DEPTH]]
        qe = embedder.encode(["Represent this sentence for searching relevant passages: " + q],
                             normalize_embeddings=True, show_progress_bar=False)
        dn = [cpids[i] for i in index.search(qe.astype(np.float32), DEPTH)[1][0]]
        small = rrf([bm[:CONFIG["bm25_topk"]], dn[:CONFIG["dense_topk"]]])
        cands.append({"bm25": bm, "dense": dn, "hybrid": small, "union100": rrf([bm, dn])})
    del embedder
    torch.cuda.empty_cache() if dev == "cuda" else None

    agg = {}

    def add(name, ranked, gold):
        for k in KS:
            agg.setdefault(name, {}).setdefault(k, []).append(recall(ranked, gold, k))

    for r, c in zip(rows, cands):
        for m in ("bm25", "dense", "hybrid"):
            add(m, c[m], r["gold_pids"])

    for model in args.rerankers:
        ce = CrossEncoder(model, device=dev, max_length=512)
        if dev == "cuda":
            ce.model.half()
        short = model.split("/")[-1]
        for r, c in tqdm(list(zip(rows, cands)), desc=short, unit="q"):
            pool = c["union100"]
            scores = dict(zip(pool, ce.predict([[r["question"], text_of[p]] for p in pool],
                                               batch_size=32, show_progress_bar=False)))
            for name, sub in (("rrf20", c["hybrid"]), ("union100", pool)):
                add(f"{name} + {short}", sorted(sub, key=scores.get, reverse=True), r["gold_pids"])
        del ce
        torch.cuda.empty_cache() if dev == "cuda" else None

    summary = {m: {f"recall@{k}": round(100 * float(np.mean(v[k])), 1) for k in KS} for m, v in agg.items()}
    out = f"{CONFIG['results_dir']}/retrieval_upgrade_{args.dataset}_{args.eval}.json"
    json.dump({"dataset": args.dataset, "eval": args.eval, "rows": [start, stop], "n": len(rows),
               "corpus_size": len(corpus), "summary": summary}, open(out, "w"), indent=2)
    print(f"\n{'method':38s}" + "".join(f"{'@' + str(k):>7}" for k in KS))
    for m, s in summary.items():
        print(f"{m:38s}" + "".join(f"{s[f'recall@{k}']:>7}" for k in KS))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
