"""
01b_open_retrieval.py
---------------------
HARDER retrieval setting. Instead of each question having its own ~10 passages
(which saturates at Recall@10), we POOL every passage from every question into
ONE shared corpus of several thousand passages, build a single BM25 + FAISS
index over the whole thing, and make each question find its 2 gold passages
among the entire pool.

This is "open retrieval" and it is where:
  - Recall@k stays meaningful across all k (no saturation),
  - hybrid retrieval can beat single retrievers (lexical rescues dense misses),
  - the cross-encoder reranker shows a real boost.

Run:
    python experiments/01b_open_retrieval.py --dataset hotpotqa --n 500
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from configs.config import CONFIG

RESULTS = CONFIG["results_dir"]
os.makedirs(RESULTS, exist_ok=True)


def load_rows(path, n):
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            rows.append(json.loads(line))
    return [r for r in rows if r["passages"] and r["gold_pids"]]


def tok(s):
    return s.lower().split()


def build_corpus(rows):
    """Pool all unique passages into one shared corpus."""
    pid2passage = {}
    for r in rows:
        for p in r["passages"]:
            pid2passage[p["pid"]] = p
    corpus = list(pid2passage.values())
    return corpus


def recall_at_k(ranked_pids, gold_pids, k):
    return len(set(ranked_pids[:k]) & set(gold_pids)) / len(gold_pids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--rerank", action="store_true",
                    help="also evaluate cross-encoder reranking on the pool")
    args = ap.parse_args()

    rows = load_rows(f"data/{args.dataset}.jsonl", args.n)
    if not rows:
        print(f"[!] no data in data/{args.dataset}.jsonl — run 00_prepare_data.py first")
        return

    corpus = build_corpus(rows)
    cpids = [p["pid"] for p in corpus]
    ctexts = [p["text"] for p in corpus]
    cpid_to_idx = {pid: i for i, pid in enumerate(cpids)}
    print(f"loaded {len(rows)} questions; pooled corpus = {len(corpus)} passages")

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev.upper()}")

    # ---- build indexes ONCE over the whole corpus ----
    print("building BM25 index over corpus ...")
    bm25 = BM25Okapi([tok(t) for t in ctexts], k1=CONFIG["bm25_k1"], b=CONFIG["bm25_b"])

    print(f"embedding corpus with {CONFIG['embed_model']} (one-time) ...")
    embedder = SentenceTransformer(CONFIG["embed_model"], device=dev)
    corpus_emb = embedder.encode(ctexts, normalize_embeddings=True,
                                 show_progress_bar=True, batch_size=64)
    index = faiss.IndexFlatIP(corpus_emb.shape[1])
    index.add(corpus_emb.astype(np.float32))

    reranker = None
    if args.rerank:
        print("loading cross-encoder for reranking ...")
        reranker = CrossEncoder(CONFIG["reranker_model"], device=dev)

    ks = [5, 10, 20, 50]
    methods = ["bm25", "dense", "hybrid"] + (["reranked"] if args.rerank else [])
    agg = {m: {k: [] for k in ks} for m in methods}

    TOPN = 100  # candidate depth pulled from each retriever before fusion/rerank

    for r in tqdm(rows, desc="open-retrieval", unit="q"):
        q, gold = r["question"], r["gold_pids"]

        # BM25 over full corpus
        bm_scores = bm25.get_scores(tok(q))
        bm_top = np.argsort(bm_scores)[::-1][:TOPN]
        bm_pids = [cpids[i] for i in bm_top]

        # dense over full corpus
        q_emb = embedder.encode(
            ["Represent this sentence for searching relevant passages: " + q],
            normalize_embeddings=True, show_progress_bar=False)
        dn_scores, dn_idx = index.search(q_emb.astype(np.float32), TOPN)
        dn_pids = [cpids[i] for i in dn_idx[0]]

        # hybrid RRF
        fused = {}
        for pids in [bm_pids[:CONFIG["bm25_topk"]], dn_pids[:CONFIG["dense_topk"]]]:
            for rank, pid in enumerate(pids):
                fused[pid] = fused.get(pid, 0.0) + 1.0 / (CONFIG["rrf_k"] + rank + 1)
        hy_pids = sorted(fused, key=fused.get, reverse=True)

        for k in ks:
            agg["bm25"][k].append(recall_at_k(bm_pids, gold, k))
            agg["dense"][k].append(recall_at_k(dn_pids, gold, k))
            agg["hybrid"][k].append(recall_at_k(hy_pids, gold, k))

        if reranker is not None:
            # rerank the fused candidate pool
            pool = hy_pids[:TOPN]
            pairs = [[q, ctexts[cpid_to_idx[pid]]] for pid in pool]
            ce = reranker.predict(pairs, show_progress_bar=False)
            order = np.argsort(ce)[::-1]
            rr_pids = [pool[i] for i in order]
            for k in ks:
                agg["reranked"][k].append(recall_at_k(rr_pids, gold, k))

    summary = {m: {f"recall@{k}": round(100*float(np.mean(agg[m][k])), 1) for k in ks}
               for m in methods}
    out = f"{RESULTS}/open_retrieval_{args.dataset}.json"
    with open(out, "w") as f:
        json.dump({"n": len(rows), "corpus_size": len(corpus), "summary": summary}, f, indent=2)

    print(f"\n=== Open-retrieval Recall (%) | corpus={len(corpus)} passages ===")
    header = f"{'method':<12}" + "".join(f"{'@'+str(k):>8}" for k in ks)
    print(header)
    for m in methods:
        print(f"{m:<12}" + "".join(f"{summary[m]['recall@'+str(k)]:>8}" for k in ks))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
