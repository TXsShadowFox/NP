"""
metrics.py
----------
Answer-quality, confidence and significance metrics shared by the analysis
scripts. Pure python + numpy, no models.

  - exact_match / token_f1  : SQuAD-style, scored against EVERY accepted gold
                              answer (TriviaQA ships aliases) and maxed.
  - contains                : normalized gold appears as a whole phrase inside
                              the prediction. Lenient: credits verbose answers
                              ("Wichita, Kansas" for "Wichita") that EM rejects.
  - auroc                   : P(conf of a correct answer > conf of a wrong one).
  - mcnemar / paired_bootstrap : paired tests for "system B beats system A"
                              on the SAME questions.
"""
import math, re, string
import numpy as np

_PUNCT = set(string.punctuation)


def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def golds_of(row):
    """All accepted answers for a data row / record: the answer plus any aliases."""
    gold = row.get("answer", row.get("gold_answer"))
    return [gold] + [a for a in row.get("aliases", []) if a and a != gold]


def _as_list(golds):
    return [golds] if isinstance(golds, str) else list(golds)


def exact_match(pred, golds):
    p = normalize(pred)
    return float(any(p == normalize(g) for g in _as_list(golds)))


def _f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum(min(p.count(t), g.count(t)) for t in set(p) if t in g)
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def token_f1(pred, golds):
    return max(_f1(pred, g) for g in _as_list(golds))


def contains(pred, golds):
    p = f" {normalize(pred)} "
    return float(any(normalize(g) and f" {normalize(g)} " in p for g in _as_list(golds)))


def auroc(scores, labels):
    """Rank-based AUROC of `scores` predicting binary `labels` (1 = correct)."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    w = sum((1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg)
    return w / (len(pos) * len(neg))


def mcnemar(correct_a, correct_b):
    """Exact two-sided McNemar test on paired 0/1 correctness.
    Returns (n_only_b_right, n_only_a_right, p_value)."""
    b_wins = sum(1 for a, b in zip(correct_a, correct_b) if not a and b)
    a_wins = sum(1 for a, b in zip(correct_a, correct_b) if a and not b)
    n = a_wins + b_wins
    if n == 0:
        return b_wins, a_wins, 1.0
    tail = sum(math.comb(n, i) for i in range(min(a_wins, b_wins) + 1)) / 2 ** n
    return b_wins, a_wins, min(1.0, 2 * tail)


def paired_bootstrap(correct_a, correct_b, iters=10000, seed=42):
    """95% CI (in percentage points) of mean(correct_b) - mean(correct_a),
    resampling questions so each draw keeps the A/B pairing."""
    a, b = np.asarray(correct_a, float), np.asarray(correct_b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(iters, len(a)))
    diffs = 100 * (b[idx].mean(axis=1) - a[idx].mean(axis=1))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
