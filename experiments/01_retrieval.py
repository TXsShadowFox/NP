"""
01_retrieval.py
---------------
Hybrid retrieval = BM25 (sparse) UNION FAISS-dense, fused with Reciprocal
Rank Fusion (RRF). Produces the first REAL numbers of the project:
Recall@{5,10,20} for BM25-only, Dense-only, and Hybrid.

Per-question corpus: each HotpotQA question carries its own ~10 paragraphs,
so we build a fresh tiny BM25 + FAISS index per question. This is fast and
gives an honest retrieval task (find the 2 gold paragraphs among ~10).

Run:
    python experiments/01_retrieval.py --dataset hotpotqa --n 500
"""
import argparse, json, os, sys
# --- make project root importable no matter where we run from ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
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
    # only keep questions that actually have passages + gold (retrieval-evaluable)
    rows = [r for r in rows if r["passages"] and r["gold_pids"]]
    return rows


def tok(s):
    return s.lower().split()


def bm25_rank(question, passages):
    corpus = [tok(p["text"]) for p in passages]
    bm25 = BM25Okapi(corpus, k1=CONFIG["bm25_k1"], b=CONFIG["bm25_b"])
    scores = bm25.get_scores(tok(question))
    order = np.argsort(scores)[::-1]
    return [passages[i]["pid"] for i in order], scores[order]


def dense_rank(question, passages, embedder):
    texts = [p["text"] for p in passages]
    # bge wants an instruction prefix on the QUERY only
    q_emb = embedder.encode(["Represent this sentence for searching relevant passages: " + question],
                            normalize_embeddings=True, show_progress_bar=False)
    d_emb = embedder.encode(texts, normalize_embeddings=True,
                            show_progress_bar=False, batch_size=32)
    dim = d_emb.shape[1]
    index = faiss.IndexFlatIP(dim)        # inner product on normalized = cosine
    index.add(d_emb.astype(np.float32))
    scores, idx = index.search(q_emb.astype(np.float32), len(passages))
    idx, scores = idx[0], scores[0]
    return [passages[i]["pid"] for i in idx], scores


def rrf_fuse(ranked_lists, k=60):
    """ranked_lists: list of (pid_list). Returns fused pid list by RRF score."""
    scores = {}
    for pids in ranked_lists:
        for rank, pid in enumerate(pids):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def recall_at_k(ranked_pids, gold_pids, k):
    topk = set(ranked_pids[:k])
    hit = len(topk & set(gold_pids))
    return hit / len(gold_pids)      # fraction of gold paragraphs found


def evaluate(rows, embedder):
    from tqdm import tqdm
    ks = [5, 10, 20]
    agg = {m: {k: [] for k in ks} for m in ["bm25", "dense", "hybrid"]}
    for r in tqdm(rows, desc="retrieving", unit="q"):
        q, ps, gold = r["question"], r["passages"], r["gold_pids"]
        bm_pids, _ = bm25_rank(q, ps)
        dn_pids, _ = dense_rank(q, ps, embedder)
        hy_pids = rrf_fuse([bm_pids[:CONFIG["bm25_topk"]],
                            dn_pids[:CONFIG["dense_topk"]]], CONFIG["rrf_k"])
        for k in ks:
            agg["bm25"][k].append(recall_at_k(bm_pids, gold, k))
            agg["dense"][k].append(recall_at_k(dn_pids, gold, k))
            agg["hybrid"][k].append(recall_at_k(hy_pids, gold, k))
    # mean
    summary = {m: {f"recall@{k}": round(100 * float(np.mean(v[k])), 1) for k in ks}
               for m, v in agg.items()}
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()

    rows = load_data(f"data/{args.dataset}.jsonl", args.n)
    if not rows:
        print(f"\n[!] No retrieval-evaluable rows found in data/{args.dataset}.jsonl")
        print("    If you haven't run 00_prepare_data.py yet, or HuggingFace is")
        print("    blocked, test the pipeline first with the bundled sample:")
        print("        python experiments/01_retrieval.py --dataset sample --n 5\n")
        return
    print(f"loaded {len(rows)} retrieval-evaluable questions from {args.dataset}")

    print(f"loading embedder {CONFIG['embed_model']} ...")
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(CONFIG["embed_model"], device=dev)
    print(f"  -> embedder running on: {dev.upper()}"
          + ("  (GPU - fast)" if dev == "cuda" else "  (CPU - slower; this is why it may lag)"))

    summary = evaluate(rows, embedder)
    out_path = f"{RESULTS}/retrieval_{args.dataset}.json"
    with open(out_path, "w") as f:
        json.dump({"n": len(rows), "summary": summary}, f, indent=2)

    print("\n=== Recall (%) ===")
    print(f"{'method':<12}{'@5':>8}{'@10':>8}{'@20':>8}")
    for m in ["bm25", "dense", "hybrid"]:
        s = summary[m]
        print(f"{m:<12}{s['recall@5']:>8}{s['recall@10']:>8}{s['recall@20']:>8}")
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()