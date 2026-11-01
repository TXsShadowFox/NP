"""
04_adaptive_loop.py
-------------------
THE CORE AB-RAG ALGORITHM. For each question, runs the budgeted adaptive
retrieval loop:

  t = 1
  K = rerank_topk                       # start with a small evidence set
  loop:
      evidence = top-K reranked passages from the shared corpus
      answer   = LLM(question, evidence)
      conf     = alpha*S1 + beta*S2 + gamma*S3   # S3 (variance) is a REWARD (see diagnostics)
      if conf >= tau   -> STOP (confident)
      elif t >= T_max  -> STOP (budget exhausted)
      else             -> K += k_step ; t += 1 ; RETRIEVE MORE

Logs EVERY iteration so we can show confidence rising across rounds and
count how many retrievals each question actually used (the efficiency story).

This runs on the OPEN-RETRIEVAL corpus (pooled passages) because that is the
only setting where "retrieve more" is meaningful.

Run:
    python experiments/04_adaptive_loop.py --dataset hotpotqa --n 100 --backend local

Output: results/adaptive_{dataset}_{backend}.jsonl  (one record per question,
        including the full per-iteration trace)
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


def build_corpus(rows):
    pid2p = {}
    for r in rows:
        for p in r["passages"]:
            pid2p[p["pid"]] = p
    corpus = list(pid2p.values())
    return corpus


class Retriever:
    """Builds shared BM25+FAISS indexes ONCE, then ranks+reranks per query."""
    def __init__(self, corpus, embedder, reranker, device):
        self.corpus = corpus
        self.cpids = [p["pid"] for p in corpus]
        self.ctexts = [p["text"] for p in corpus]
        self.embedder = embedder
        self.reranker = reranker
        print("building shared BM25 index ...")
        self.bm25 = BM25Okapi([tok(t) for t in self.ctexts],
                              k1=CONFIG["bm25_k1"], b=CONFIG["bm25_b"])
        print("embedding shared corpus (one-time) ...")
        emb = embedder.encode(self.ctexts, normalize_embeddings=True,
                              show_progress_bar=True, batch_size=64)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb.astype(np.float32))

    def ranked_pool(self, question, depth=100):
        """Return reranked (pids, texts, ce_scores) for the top `depth` candidates."""
        bm_scores = self.bm25.get_scores(tok(question))
        bm_pids = [self.cpids[i] for i in np.argsort(bm_scores)[::-1][:depth]]
        q_emb = self.embedder.encode(
            ["Represent this sentence for searching relevant passages: " + question],
            normalize_embeddings=True, show_progress_bar=False)
        _, idx = self.index.search(q_emb.astype(np.float32), depth)
        dn_pids = [self.cpids[i] for i in idx[0]]
        fused = {}
        for pids in [bm_pids[:CONFIG["bm25_topk"]], dn_pids[:CONFIG["dense_topk"]]]:
            for rank, pid in enumerate(pids):
                fused[pid] = fused.get(pid, 0.0) + 1.0 / (CONFIG["rrf_k"] + rank + 1)
        pool_pids = sorted(fused, key=fused.get, reverse=True)
        pid2idx = {pid: i for i, pid in enumerate(self.cpids)}
        pool_texts = [self.ctexts[pid2idx[pid]] for pid in pool_pids]
        pairs = [[question, t] for t in pool_texts]
        ce = self.reranker.predict(pairs, show_progress_bar=False)
        order = np.argsort(ce)[::-1]
        pids = [pool_pids[i] for i in order]
        texts = [pool_texts[i] for i in order]
        scores = [float(ce[i]) for i in order]
        return pids, texts, scores


def confidence_from(gen_out, answer, evidence_texts, rerank_scores, embedder):
    if gen_out["signal1_kind"] == "token_logprob":
        c_tok = token_probability_confidence(gen_out["token_logprobs"])
    else:
        c_tok = float(gen_out["self_consistency"])
    c_con = evidence_answer_consistency(answer, evidence_texts, embedder)
    v_ret = retrieval_score_variance(rerank_scores)
    conf = CONFIG["alpha"]*c_tok + CONFIG["beta"]*c_con + CONFIG["gamma"]*v_ret
    return float(np.clip(conf, 0.0, 1.0)), c_tok, c_con, v_ret


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--backend", choices=["local", "claude", "ollama", "groq", "openrouter"], default="local")
    args = ap.parse_args()

    rows = load_rows(f"data/{args.dataset}.jsonl", args.n)
    if not rows:
        print(f"[!] no data in data/{args.dataset}.jsonl"); return
    corpus = build_corpus(rows)
    print(f"loaded {len(rows)} questions; corpus={len(corpus)} passages; backend={args.backend}")

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev.upper()}")
    embedder = SentenceTransformer(CONFIG["embed_model"], device=dev)
    reranker = CrossEncoder(CONFIG["reranker_model"], device=dev)
    gen = get_generator(args.backend, CONFIG, device=dev)
    retr = Retriever(corpus, embedder, reranker, dev)

    tau, T_max, k0, kstep = CONFIG["tau"], CONFIG["T_max"], CONFIG["rerank_topk"], CONFIG["k_step"]
    out_path = f"{RESULTS}/adaptive_{args.dataset}_{args.backend}.jsonl"
    fout = open(out_path, "w")

    for r in tqdm(rows, desc=f"adaptive ({args.backend})", unit="q"):
        q = r["question"]
        # rank the pool once (deep), then just take more of it each iteration
        pids, texts, scores = retr.ranked_pool(q, depth=100)

        trace = []
        t = 1
        K = k0
        final = None
        while True:
            ev_texts = texts[:K]
            ev_scores = scores[:K]
            g = gen.generate(q, ev_texts)
            conf, c_tok, c_con, v_ret = confidence_from(g, g["answer"], ev_texts, ev_scores, embedder)
            trace.append({"t": t, "K": K, "answer": g["answer"],
                          "conf": round(conf, 4), "c_tok": round(c_tok, 4),
                          "c_con": round(c_con, 4), "v_ret": round(v_ret, 4)})
            final = {"answer": g["answer"], "conf": conf, "K": K, "iters": t}
            if conf >= tau:
                final["stop_reason"] = "confident"
                break
            if t >= T_max:
                final["stop_reason"] = "budget_exhausted"
                break
            t += 1
            K += kstep

        rec = {
            "qid": r["qid"], "question": q,
            "gold_answer": r["answer"],
            "pred_answer": final["answer"],
            "final_confidence": round(final["conf"], 4),
            "iterations_used": final["iters"],
            "final_K": final["K"],
            "stop_reason": final["stop_reason"],
            "trace": trace,
            # the exact policy settings behind this record, so replays and
            # the paper's tables can always be checked against what really ran
            "settings": {k: CONFIG[k] for k in ("alpha", "beta", "gamma", "tau", "T_max",
                                                "k_step", "rerank_topk", "n_consistency_samples")},
        }
        fout.write(json.dumps(rec) + "\n")
        fout.flush()

    fout.close()
    # quick summary
    recs = [json.loads(l) for l in open(out_path)]
    avg_iters = np.mean([r["iterations_used"] for r in recs])
    stop_conf = sum(1 for r in recs if r["stop_reason"] == "confident")
    print(f"\nsaved -> {out_path}")
    print(f"avg retrieval iterations/query: {avg_iters:.2f}  (max budget {T_max})")
    print(f"stopped early (confident): {stop_conf}/{len(recs)} questions")
    print("example trace (first question):")
    for step in recs[0]["trace"]:
        print(f"  t={step['t']} K={step['K']} conf={step['conf']} ans={step['answer'][:30]}")


if __name__ == "__main__":
    main()
