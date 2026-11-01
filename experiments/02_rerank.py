"""
02_rerank.py
------------
Adds cross-encoder reranking on top of hybrid retrieval and measures the
recall improvement. This is Figure 3.3 / the reranking stage of your paper.

Why a cross-encoder: BM25 and dense retrieval score the query and document
SEPARATELY (bi-encoder style), so they miss fine-grained query-document
interaction. A cross-encoder feeds (query, doc) together through one model
and outputs a single relevance score, modelling full cross-attention. It is
slower, so we only apply it to the small candidate pool (~10-40 docs), never
the whole corpus. Fast even on CPU because the pool is tiny.

Model: cross-encoder/ms-marco-MiniLM-L12-v2  (trained on MS MARCO ranking).

Run:
    python experiments/02_rerank.py --dataset hotpotqa --n 200
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from rank_bm25 import BM25Okapi
from configs.config import CONFIG

RESULTS = CONFIG["results_dir"]
os.makedirs(RESULTS, exist_ok=True)


def load_data(path, n):
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            rows.append(json.loads(line))
    return [r for r in rows if r["passages"] and r["gold_pids"]]


def tok(s):
    return s.lower().split()


def hybrid_candidate_pool(question, passages, embedder):
    """Return the fused candidate pool (list of passage dicts) via BM25+dense+RRF."""
    # BM25
    bm = BM25Okapi([tok(p["text"]) for p in passages],
                   k1=CONFIG["bm25_k1"], b=CONFIG["bm25_b"])
    bm_scores = bm.get_scores(tok(question))
    bm_order = np.argsort(bm_scores)[::-1]
    bm_pids = [passages[i]["pid"] for i in bm_order]
    # dense
    texts = [p["text"] for p in passages]
    q_emb = embedder.encode(["Represent this sentence for searching relevant passages: " + question],
                            normalize_embeddings=True, show_progress_bar=False)
    d_emb = embedder.encode(texts, normalize_embeddings=True,
                            show_progress_bar=False, batch_size=32)
    index = faiss.IndexFlatIP(d_emb.shape[1])
    index.add(d_emb.astype(np.float32))
    _, idx = index.search(q_emb.astype(np.float32), len(passages))
    dn_pids = [passages[i]["pid"] for i in idx[0]]
    # RRF fuse
    fused = {}
    for pids in [bm_pids[:CONFIG["bm25_topk"]], dn_pids[:CONFIG["dense_topk"]]]:
        for rank, pid in enumerate(pids):
            fused[pid] = fused.get(pid, 0.0) + 1.0 / (CONFIG["rrf_k"] + rank + 1)
    pid2passage = {p["pid"]: p for p in passages}
    fused_pids = sorted(fused, key=fused.get, reverse=True)
    return [pid2passage[pid] for pid in fused_pids]


def recall_at_k(ranked_pids, gold_pids, k):
    return len(set(ranked_pids[:k]) & set(gold_pids)) / len(gold_pids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    rows = load_data(f"data/{args.dataset}.jsonl", args.n)
    if not rows:
        print(f"[!] no data in data/{args.dataset}.jsonl — run 00_prepare_data.py first")
        return
    print(f"loaded {len(rows)} questions from {args.dataset}")

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev.upper()}")
    print("loading embedder + cross-encoder ...")
    embedder = SentenceTransformer(CONFIG["embed_model"], device=dev)
    reranker = CrossEncoder(CONFIG["reranker_model"], device=dev)

    ks = [5, 10]
    hybrid_recall = {k: [] for k in ks}
    rerank_recall = {k: [] for k in ks}

    for r in tqdm(rows, desc="rerank", unit="q"):
        q, gold = r["question"], r["gold_pids"]
        pool = hybrid_candidate_pool(q, r["passages"], embedder)
        pool_pids = [p["pid"] for p in pool]

        # cross-encoder scores every (query, doc) in the pool jointly
        pairs = [[q, p["text"]] for p in pool]
        ce_scores = reranker.predict(pairs, show_progress_bar=False)
        order = np.argsort(ce_scores)[::-1]
        reranked_pids = [pool[i]["pid"] for i in order]

        for k in ks:
            hybrid_recall[k].append(recall_at_k(pool_pids, gold, k))
            rerank_recall[k].append(recall_at_k(reranked_pids, gold, k))

    summary = {
        "hybrid":          {f"recall@{k}": round(100*float(np.mean(hybrid_recall[k])), 1) for k in ks},
        "hybrid_reranked": {f"recall@{k}": round(100*float(np.mean(rerank_recall[k])), 1) for k in ks},
    }
    out = f"{RESULTS}/rerank_{args.dataset}.json"
    with open(out, "w") as f:
        json.dump({"n": len(rows), "summary": summary}, f, indent=2)

    print("\n=== Reranking effect (Recall %) ===")
    print(f"{'method':<18}{'@5':>8}{'@10':>8}")
    for m in ["hybrid", "hybrid_reranked"]:
        s = summary[m]
        print(f"{m:<18}{s['recall@5']:>8}{s['recall@10']:>8}")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
