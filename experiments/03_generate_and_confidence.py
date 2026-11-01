"""
03_generate_and_confidence.py
-----------------------------
For each question: retrieve evidence (hybrid + rerank), generate an answer,
and compute the THREE confidence signals + the combined confidence score.
Saves a per-question record so later scripts (adaptive loop, experiments,
plots) can reuse it without re-running the LLM.

Run (local, free):
    python experiments/03_generate_and_confidence.py --dataset hotpotqa --n 100 --backend local
Run (Claude, needs ANTHROPIC_API_KEY):
    python experiments/03_generate_and_confidence.py --dataset hotpotqa --n 100 --backend claude

Output: results/gen_conf_{dataset}_{backend}.jsonl
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from configs.config import CONFIG
from experiments.confidence import (token_probability_confidence, evidence_answer_consistency,
                            retrieval_score_variance)
from experiments.generation import get_generator

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


def retrieve_and_rerank(question, passages, embedder, reranker):
    """Hybrid retrieve + cross-encoder rerank. Returns (evidence_texts, rerank_scores)."""
    # BM25
    bm = BM25Okapi([tok(p["text"]) for p in passages], k1=CONFIG["bm25_k1"], b=CONFIG["bm25_b"])
    bm_scores = bm.get_scores(tok(question))
    bm_pids = [passages[i]["pid"] for i in np.argsort(bm_scores)[::-1]]
    # dense
    texts = [p["text"] for p in passages]
    q_emb = embedder.encode(["Represent this sentence for searching relevant passages: " + question],
                            normalize_embeddings=True, show_progress_bar=False)
    d_emb = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32)
    index = faiss.IndexFlatIP(d_emb.shape[1]); index.add(d_emb.astype(np.float32))
    _, idx = index.search(q_emb.astype(np.float32), len(passages))
    dn_pids = [passages[i]["pid"] for i in idx[0]]
    # RRF
    fused = {}
    for pids in [bm_pids[:CONFIG["bm25_topk"]], dn_pids[:CONFIG["dense_topk"]]]:
        for rank, pid in enumerate(pids):
            fused[pid] = fused.get(pid, 0.0) + 1.0 / (CONFIG["rrf_k"] + rank + 1)
    pid2p = {p["pid"]: p for p in passages}
    pool = [pid2p[pid] for pid in sorted(fused, key=fused.get, reverse=True)]
    # rerank
    pairs = [[question, p["text"]] for p in pool]
    ce = reranker.predict(pairs, show_progress_bar=False)
    order = np.argsort(ce)[::-1]
    top = order[:CONFIG["rerank_topk"]]
    evidence_texts = [pool[i]["text"] for i in top]
    rerank_scores = [float(ce[i]) for i in top]
    return evidence_texts, rerank_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--backend", choices=["local", "claude", "ollama", "groq", "openrouter"], default="local")
    args = ap.parse_args()

    rows = load_rows(f"data/{args.dataset}.jsonl", args.n)
    if not rows:
        print(f"[!] no data in data/{args.dataset}.jsonl"); return
    print(f"loaded {len(rows)} questions; backend={args.backend}")

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev.upper()}")
    embedder = SentenceTransformer(CONFIG["embed_model"], device=dev)
    reranker = CrossEncoder(CONFIG["reranker_model"], device=dev)
    gen = get_generator(args.backend, CONFIG, device=dev)

    out_path = f"{RESULTS}/gen_conf_{args.dataset}_{args.backend}.jsonl"
    fout = open(out_path, "w")

    for r in tqdm(rows, desc=f"gen+conf ({args.backend})", unit="q"):
        q, gold_answer = r["question"], r["answer"]
        evidence, rerank_scores = retrieve_and_rerank(q, r["passages"], embedder, reranker)
        g = gen.generate(q, evidence)
        answer = g["answer"]

        # Signal 1 (backend-dependent)
        if g["signal1_kind"] == "token_logprob":
            c_tok = token_probability_confidence(g["token_logprobs"])
        else:  # self_consistency proxy for closed API
            c_tok = float(g["self_consistency"])

        # Signals 2 and 3 (same for both backends)
        c_con = evidence_answer_consistency(answer, evidence, embedder)
        v_ret = retrieval_score_variance(rerank_scores)

        conf = (CONFIG["alpha"] * c_tok + CONFIG["beta"] * c_con
                + CONFIG["gamma"] * v_ret)
        conf = float(np.clip(conf, 0.0, 1.0))

        rec = {
            "qid": r["qid"], "question": q,
            "gold_answer": gold_answer, "pred_answer": answer,
            "signal1_kind": g["signal1_kind"],
            "conf_token": round(c_tok, 4),
            "consistency": round(c_con, 4),
            "retrieval_var": round(v_ret, 4),
            "confidence": round(conf, 4),
            "n_evidence": len(evidence),
        }
        fout.write(json.dumps(rec) + "\n")
        fout.flush()

    fout.close()
    print(f"\nsaved per-question records -> {out_path}")
    print("preview of first 3 records:")
    for line in open(out_path).readlines()[:3]:
        d = json.loads(line)
        print(f"  Q: {d['question'][:50]}... | pred: {d['pred_answer'][:30]} "
              f"| conf={d['confidence']}")


if __name__ == "__main__":
    main()
