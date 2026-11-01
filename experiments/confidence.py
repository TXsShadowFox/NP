"""
confidence.py
-------------
THE NOVEL CORE of AB-RAG. After the LLM produces an answer, we estimate how
trustworthy it is using THREE complementary signals, then combine them.

  Conf = alpha * Conf_token + beta * Consistency + gamma * Var(R)

Signal 1 - Token Probability Confidence
    Mean per-token probability of the generated answer.
    High  -> model internally certain.   Source: generation logprobs.

Signal 2 - Evidence-Answer Consistency
    Cosine similarity between the answer embedding and the (mean) evidence
    embedding. High -> answer is grounded in retrieved text.

Signal 3 - Retrieval Score Variance  (a REWARD)
    Variance of the reranker relevance scores of the evidence set.
    High variance -> reranker cleanly separated relevant from irrelevant
    passages. Originally a penalty; 07_signal_diagnostics showed the sign
    was backwards (AUROC 0.427 as penalty, 0.573 as reward).

All three are normalized to [0,1] so the weighted sum is interpretable as a
confidence in [0,1] (after clipping). gamma adds the variance reward.

This module is pure numpy + an embedder you pass in. No model training.
"""
import numpy as np


def token_probability_confidence(token_logprobs):
    """
    token_logprobs: list[float] natural-log probs of each generated token.
    Returns mean token PROBABILITY in [0,1].
    """
    if not token_logprobs:
        return 0.0
    probs = np.exp(np.array(token_logprobs, dtype=np.float64))
    return float(np.clip(probs.mean(), 0.0, 1.0))


def evidence_answer_consistency(answer_text, evidence_texts, embedder):
    """
    Cosine similarity between answer embedding and the mean evidence embedding.
    embedder: a SentenceTransformer-like object with .encode(..., normalize_embeddings=True)
    Returns value in [0,1] (cosine of normalized vecs mapped from [-1,1] -> [0,1]).
    """
    if not evidence_texts:
        return 0.0
    a = embedder.encode([answer_text], normalize_embeddings=True)[0]
    E = embedder.encode(evidence_texts, normalize_embeddings=True)
    e_mean = E.mean(axis=0)
    e_mean = e_mean / (np.linalg.norm(e_mean) + 1e-9)
    cos = float(np.dot(a, e_mean))           # in [-1, 1]
    return (cos + 1.0) / 2.0                  # map to [0, 1]


def retrieval_score_variance(rerank_scores):
    """
    Variance of evidence relevance scores, normalized to [0,1] (used as a reward).
    We min-max the scores first so variance is scale-free, then return the
    variance of the normalized scores (already in [0,1] range-ish, clipped).
    """
    s = np.array(rerank_scores, dtype=np.float64)
    if s.size <= 1:
        return 0.0
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-9:
        return 0.0
    s_norm = (s - lo) / (hi - lo)             # -> [0,1]
    return float(np.clip(s_norm.var(), 0.0, 1.0))


def combined_confidence(token_logprobs, answer_text, evidence_texts,
                        rerank_scores, embedder, alpha, beta, gamma):
    """
    Returns (conf, breakdown_dict). conf clipped to [0,1].
    """
    c_tok = token_probability_confidence(token_logprobs)
    c_con = evidence_answer_consistency(answer_text, evidence_texts, embedder)
    v_ret = retrieval_score_variance(rerank_scores)
    conf = alpha * c_tok + beta * c_con + gamma * v_ret
    conf = float(np.clip(conf, 0.0, 1.0))
    return conf, {
        "token_conf": round(c_tok, 4),
        "consistency": round(c_con, 4),
        "retrieval_var": round(v_ret, 4),
        "combined": round(conf, 4),
    }


# --- self-test with hand numbers so you can see it behaves sensibly ---
if __name__ == "__main__":
    import numpy as np
    # high-confidence case: peaked token probs
    high = token_probability_confidence([np.log(0.95), np.log(0.92), np.log(0.97)])
    # low-confidence case: flat token probs
    low = token_probability_confidence([np.log(0.4), np.log(0.3), np.log(0.5)])
    print(f"token_conf high={high:.3f}  low={low:.3f}  (expect high>low)")

    var_consistent = retrieval_score_variance([0.9, 0.88, 0.91, 0.89])
    var_shaky = retrieval_score_variance([0.9, 0.1, 0.85, 0.2])
    print(f"retrieval_var consistent={var_consistent:.3f}  shaky={var_shaky:.3f}  "
          f"(expect consistent<shaky)")
    assert high > low and var_consistent < var_shaky
    print("confidence logic OK")
