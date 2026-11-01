"""
11_value_policy.py
------------------
N1: value-of-retrieval stopping. AB-RAG retrieves more whenever confidence is
below tau, i.e. it treats "unsure now" as "more passages will help". This
policy instead estimates, at every round, the expected utility of each action

  stop      answer now                 +1 if correct, -r if wrong
  abstain   give no answer              0
  retrieve  go on to the next round     expected utility of continuing,
                                        minus lambda per extra passage read

and takes the best one. An "unknown" answer counts as abstaining, so a
confidently repeated "unknown" no longer ends the search on its own.

Values are learned from the full traces of 09_full_traces.py by backward
induction on realized returns (Longstaff-Schwartz style):
  1. p_t = P(correct | features at round t), logistic regression pooled over rounds
  2. last round: answer if that beats abstaining
  3. t = T-1 .. 1: regress the utility actually obtained from round t+1 on
     (under the policy already fitted for the later rounds) on the round-t
     features, and retrieve only if that, minus lambda * K_{t+1}, beats both
     stopping and abstaining
Training targets come from out-of-fold predictions (cross-fitting), so no
regression is fitted on decisions made by a model that saw the same question.

Features at round t use only what the system has seen by then (answers,
samples, S1/S2/S3, reranker scores of the passages read) plus the reranker
scores of the next block, which exist before any generation. Never gold.

Default: fit on dev traces, evaluate on test traces. With --cv K the policy is
cross-fitted on one split instead (for when only that split has traces).

Compared with static K, AB-RAG (tau), AB-RAG + abstain at budget, random
stopping budget-matched to N1, and the oracle:
  - operating point (lambda, r from config)
  - lambda sweep: N1's cost-accuracy frontier next to AB-RAG's tau sweep
  - matched cost: the lambda at which N1 reads as many passages as AB-RAG,
    picked from costs alone (no labels), with McNemar / bootstrap tests
  - r sweep: coverage, selective accuracy, utility; AURC of each policy's score
  - fixability: AUROC for "wrong now, right next round" of 1 - conf (when
    AB-RAG retrieves) vs N1's estimated gain from retrieving
  - ablation: the same policy on the paper's three signals only, separating
    the decision rule from the richer features
  - standardized logistic coefficients (which signals matter, see N2)

Run:
    python experiments/11_value_policy.py --dataset triviaqa --backend groq-gpt-oss-120b
    python experiments/11_value_policy.py --dataset triviaqa --backend groq-gpt-oss-120b --cv 5
Output: results/n1_{dataset}_{backend}_{label}_{mode}.json
"""
import argparse, importlib.util, json, math, os, re, sys
from collections import Counter
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from configs.config import CONFIG
from experiments.metrics import normalize, auroc, mcnemar, paired_bootstrap

_spec = importlib.util.spec_from_file_location(
    "policies_mod", os.path.join(os.path.dirname(__file__), "10_policies.py"))
policies_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(policies_mod)
LABELS, TAUS, stop_rounds, pct = (policies_mod.LABELS, policies_mod.TAUS,
                                  policies_mod.stop_rounds, policies_mod.pct)

LAMBDAS = [0.0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05]
RISKS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
ALL_FEATURES = ["c_tok", "c_con", "v_ret", "agree", "distinct_frac", "entropy",
                "unknown", "unknown_frac", "log_len", "ans_in_evidence",
                "top1", "margin12", "mean_read", "next_max", "next_mean", "changed", "d_ctok"]
PAPER_FEATURES = ["c_tok", "c_con", "v_ret"]
STOP, ABSTAIN, RETRIEVE = 0, 1, 2
_UNKNOWN = re.compile(r"^(unknown|insufficient|not enough information|cannot be determined|evidence is insufficient)")


def is_unknown(answer):
    """The prompt asks for 'unknown' when the evidence is insufficient."""
    n = normalize(answer)
    return not n or bool(_UNKNOWN.match(n))


# ---------------------------------------------------------------- features

def round_features(rec, t, texts_norm):
    rounds, scores = rec["rounds"], rec["rerank_scores"]
    rd = rounds[t]
    K = rd["K"]
    ans = normalize(rd["answer"])
    samp = [normalize(s) for s in rd["samples"]] or [ans]
    counts = Counter(samp)
    probs = np.array(list(counts.values()), float) / len(samp)
    nxt = scores[K:rounds[t + 1]["K"]] if t + 1 < len(rounds) else []
    ev = " " + " ".join(texts_norm[p] for p in rec["pids"][:K]) + " "
    prev = rounds[t - 1] if t else None
    return {
        "c_tok": rd["c_tok"], "c_con": rd["c_con"], "v_ret": rd["v_ret"],
        "agree": counts[ans] / len(samp),               # samples that match the answer
        "distinct_frac": len(counts) / len(samp),
        "entropy": float(-(probs * np.log(probs)).sum()),
        "unknown": float(is_unknown(rd["answer"])),
        "unknown_frac": float(np.mean([is_unknown(s) for s in rd["samples"] or [rd["answer"]]])),
        "log_len": math.log1p(len(ans.split())),
        "ans_in_evidence": float(bool(ans) and f" {ans} " in ev),
        "top1": scores[0],
        "margin12": scores[0] - scores[1] if len(scores) > 1 else 0.0,
        "mean_read": float(np.mean(scores[:K])),
        "next_max": float(max(nxt)) if nxt else 0.0,    # reranker scores of the passages
        "next_mean": float(np.mean(nxt)) if nxt else 0.0,   # the next round would add
        "changed": float(prev is not None and normalize(prev["answer"]) != ans),
        "d_ctok": rd["c_tok"] - prev["c_tok"] if prev else 0.0,
    }


def arrays(recs, label_fn, texts_norm):
    """Per-round features, correctness Y, 'unknown' mask U and paper confidence, all (n, T)."""
    T = len(recs[0]["rounds"])
    F = [[round_features(r, t, texts_norm) for t in range(T)] for r in recs]
    Y = np.array([[label_fn(rd["answer"], r) for rd in r["rounds"]] for r in recs], float)
    U = np.array([[is_unknown(rd["answer"]) for rd in r["rounds"]] for r in recs])
    Y[U] = 0.0                                          # "unknown" is never credited
    conf = np.array([[rd["conf"] for rd in r["rounds"]] for r in recs])
    return F, Y, U, conf


def design(F, names):
    """(n, T, T + len(names)): round one-hot + the named features."""
    T = len(F[0])
    return np.array([[[float(t == j) for j in range(T)] + [f[t][k] for k in names]
                      for t in range(T)] for f in F])


# ------------------------------------------------------------------ policy

class _Const:
    """Stands in for the classifier when a training fold has a single class."""
    def __init__(self, v):
        self.v = v

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.v), np.full(len(X), self.v)])


def fit_p(X, y):
    if y.min() == y.max():
        return _Const(float(y[0]))
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X, y)


def fit_m(X, y):
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12))).fit(X, y)


def predict_p(model, X):
    n, T, d = X.shape
    return model.predict_proba(X.reshape(-1, d))[:, 1].reshape(n, T)


def stop_values(P, U, risk):
    return np.where(U, 0.0, P * (1 + risk) - risk)


def payoff(Y, U, risk):
    return np.where(U, 0.0, Y * (1 + risk) - risk)


def decide(S, Q, U):
    """RETRIEVE if continuing beats both stopping and abstaining, else STOP if
    answering beats abstaining (and the answer is not 'unknown'), else ABSTAIN."""
    act = np.where((S >= 0) & ~U, STOP, ABSTAIN)
    return np.where(Q > np.maximum(S, 0), RETRIEVE, act)


def fit_policy(X, Y, U, Ks, lam, risk, folds, seed):
    n, T, d = X.shape
    kf = list(KFold(min(folds, n), shuffle=True, random_state=seed).split(np.arange(n)))
    P = np.zeros((n, T))
    for tr, te in kf:
        P[te] = predict_p(fit_p(X[tr].reshape(-1, d), Y[tr].ravel()), X[te])
    S = stop_values(P, U, risk)
    # realized utility from the last round on, under out-of-fold decisions
    G = np.where(decide(S[:, -1], -np.inf, U[:, -1]) == STOP, payoff(Y[:, -1], U[:, -1], risk), 0.0)
    regs = [None] * (T - 1)
    for t in range(T - 2, -1, -1):
        Z = np.column_stack([X[:, t], P[:, t]])
        m = np.zeros(n)
        for tr, te in kf:
            m[te] = fit_m(Z[tr], G[tr]).predict(Z[te])
        regs[t] = fit_m(Z, G)
        act = decide(S[:, t], m - lam * Ks[t + 1], U[:, t])
        G = np.where(act == RETRIEVE, G - lam * Ks[t + 1],
                     np.where(act == STOP, payoff(Y[:, t], U[:, t], risk), 0.0))
    return {"p": fit_p(X.reshape(-1, d), Y.ravel()), "m": regs, "lam": lam, "risk": risk}


def run_policy(pol, X, U, Ks):
    """Forward pass over full traces. Returns the stop round, an abstain flag,
    P(correct) at every round and the estimated gain of retrieving
    (continuation value minus the best of stop/abstain; -inf at the last round)."""
    n, T, _ = X.shape
    P = predict_p(pol["p"], X)
    S = stop_values(P, U, pol["risk"])
    Q = np.full((n, T), -np.inf)
    for t in range(T - 1):
        Q[:, t] = pol["m"][t].predict(np.column_stack([X[:, t], P[:, t]])) - pol["lam"] * Ks[t + 1]
    act = decide(S, Q, U)
    s = (act != RETRIEVE).argmax(axis=1)                # the last round never retrieves
    return s, act[np.arange(n), s] == ABSTAIN, P, Q - np.maximum(S, 0)


# -------------------------------------------------------------- evaluation

def score(Y, U, s, abstain, Ks, risk, lam, cost=None):
    rows = np.arange(len(s))
    answered = ~abstain & ~U[rows, s]
    correct = Y[rows, s] * answered
    passages = np.cumsum(Ks)[s] if cost is None else np.full(len(s), float(cost))
    util = np.where(answered, Y[rows, s] * (1 + risk) - risk, 0.0) - lam * passages
    return {"acc": pct(correct), "coverage": pct(answered),
            "selective_acc": round(100 * float(correct.sum()) / max(int(answered.sum()), 1), 1),
            "avg_rounds": round(float(s.mean() + 1), 2),
            "passages_read": round(float(passages.mean()), 1),
            "utility": round(float(util.mean()), 4)}, correct


def baselines(conf, Ks, tau):
    """name -> (stop round, abstain flag, fixed cost or None for cumulative)."""
    n = len(conf)
    out = {f"static K={K}": (np.full(n, t), np.zeros(n, bool), K) for t, K in enumerate(Ks)}
    s = stop_rounds(conf, tau)
    out["AB-RAG"] = (s, np.zeros(n, bool), None)
    out["AB-RAG + abstain at budget"] = (s, conf[np.arange(n), s] < tau, None)
    return out


def random_matched(Y, U, s, abstain, n_perm, rng):
    """N1's (stop round, abstain) pairs shuffled across questions: same cost and
    coverage, no information about the question."""
    rows = np.arange(len(s))
    accs = []
    for _ in range(n_perm):
        p = rng.permutation(len(s))
        accs.append(float((Y[rows, s[p]] * (~abstain[p] & ~U[rows, s[p]])).mean()))
    return np.array(accs)


def aurc(scores, correct):
    """Area under the risk-coverage curve (lower is better): answer questions in
    order of decreasing score, average the error rate over every coverage."""
    order = np.argsort(-np.asarray(scores, float), kind="stable")
    err = 1 - np.asarray(correct, float)[order]
    return round(float(np.mean(np.cumsum(err) / np.arange(1, len(err) + 1))), 4)


def paired(c_ref, c_new):
    plus, minus, p = mcnemar(c_ref, c_new)
    lo, hi = paired_bootstrap(c_ref, c_new)
    return {"plus": plus, "minus": minus, "mcnemar_p": round(p, 4), "ci95_pts": [round(lo, 1), round(hi, 1)]}


def analyze(train_recs, eval_recs, texts_norm, a):
    label_fn = LABELS[a.label]
    Ks = [rd["K"] for rd in eval_recs[0]["rounds"]]
    Fe, Ye, Ue, conf = arrays(eval_recs, label_fn, texts_norm)
    if a.cv:
        Ft, Yt, Ut = Fe, Ye, Ue
    else:
        assert [rd["K"] for rd in train_recs[0]["rounds"]] == Ks, "train/eval rounds differ"
        Ft, Yt, Ut, _ = arrays(train_recs, label_fn, texts_norm)
    n = len(eval_recs)
    rows = np.arange(n)
    rng = np.random.default_rng(a.seed)
    feature_sets = {"N1": ALL_FEATURES, "N1 (paper signals)": PAPER_FEATURES}

    def solve(names, lam, risk):
        Xe = design(Fe, names)
        if not a.cv:
            pol = fit_policy(design(Ft, names), Yt, Ut, Ks, lam, risk, a.folds, a.seed)
            return run_policy(pol, Xe, Ue, Ks)
        s, ab = np.zeros(n, int), np.zeros(n, bool)
        P, gain = np.zeros(Ye.shape), np.zeros(Ye.shape)
        for tr, te in KFold(a.cv, shuffle=True, random_state=a.seed).split(rows):
            pol = fit_policy(Xe[tr], Ye[tr], Ue[tr], Ks, lam, risk, a.folds, a.seed)
            s[te], ab[te], P[te], gain[te] = run_policy(pol, Xe[te], Ue[te], Ks)
        return s, ab, P, gain

    out = {"n_train": len(Yt), "n_eval": n, "Ks": Ks, "label": a.label, "lambda": a.lam,
           "risk": a.risk, "tau": a.tau, "mode": a.mode}

    # ---- operating point
    base = baselines(conf, Ks, a.tau)
    table, correct = {}, {}
    for name, (s, ab, cost) in base.items():
        table[name], correct[name] = score(Ye, Ue, s, ab, Ks, a.risk, a.lam, cost)
    runs = {fs: solve(names, a.lam, a.risk) for fs, names in feature_sets.items()}
    for fs, (s, ab, _, _) in runs.items():
        table[fs], correct[fs] = score(Ye, Ue, s, ab, Ks, a.risk, a.lam)
    s1, ab1, P1, gain1 = runs["N1"]
    perm = random_matched(Ye, Ue, s1, ab1, a.n_perm, rng)
    table["random (budget-matched to N1)"] = {
        "acc": pct(perm), "acc_95": [round(100 * float(np.percentile(perm, q)), 1) for q in (2.5, 97.5)],
        "p_random_ge_n1": round(float((perm >= correct["N1"].mean()).mean()), 4),
        "avg_rounds": table["N1"]["avg_rounds"], "passages_read": table["N1"]["passages_read"]}
    table["oracle (any round)"] = {"acc": pct(Ye.max(axis=1)), "avg_rounds": float(len(Ks)),
                                   "passages_read": float(sum(Ks))}
    out["policies"] = table
    out["n1_vs"] = {ref: paired(correct[ref], correct["N1"]) for ref in ("AB-RAG", f"static K={Ks[1]}")}

    # ---- lambda sweep vs the AB-RAG tau sweep
    cum = np.cumsum(Ks)
    tau_pts = sorted((float(cum[st].mean()), pct(Ye[rows, st]))
                     for st in (stop_rounds(conf, t_) for t_ in TAUS))
    xs, ys = np.array([p[0] for p in tau_pts]), np.array([p[1] for p in tau_pts])
    out["abrag_tau_sweep"] = [{"tau": t_, "passages_read": round(float(cum[st].mean()), 1),
                               "acc": pct(Ye[rows, st])}
                              for t_, st in ((t_, stop_rounds(conf, t_)) for t_ in TAUS)]
    sweep, kept = {}, {}
    for fs, names in feature_sets.items():
        sweep[fs], kept[fs] = [], []
        for lam in LAMBDAS:
            s, ab = solve(names, lam, a.risk)[:2]
            m, c = score(Ye, Ue, s, ab, Ks, a.risk, lam)
            c_ab = m["passages_read"]
            sweep[fs].append({"lambda": lam, **m, "abrag_acc_same_cost":
                              round(float(np.interp(c_ab, xs, ys)), 1) if xs[0] <= c_ab <= xs[-1] else None})
            kept[fs].append((c, s, ab))
    out["lambda_sweep"] = sweep

    # ---- matched cost: lambda chosen by passages read only (label-free)
    target = table["AB-RAG"]["passages_read"]
    out["matched_cost"] = {"abrag_passages_read": target}
    for fs in feature_sets:
        i = int(np.argmin([abs(p["passages_read"] - target) for p in sweep[fs]]))
        c, s, ab = kept[fs][i]
        pm = random_matched(Ye, Ue, s, ab, a.n_perm, rng)
        out["matched_cost"][fs] = {
            "lambda": LAMBDAS[i], "acc": sweep[fs][i]["acc"], "passages_read": sweep[fs][i]["passages_read"],
            "vs_abrag": paired(correct["AB-RAG"], c),
            "random_same_stops_acc": pct(pm),
            "p_random_ge": round(float((pm >= c.mean()).mean()), 4)}

    # ---- selective prediction: r sweep and AURC
    out["risk_sweep"] = []
    for r in RISKS:
        s, ab = solve(ALL_FEATURES, a.lam, r)[:2]
        m, _ = score(Ye, Ue, s, ab, Ks, r, a.lam)
        ref, _ = score(Ye, Ue, *base["AB-RAG + abstain at budget"][:2], Ks, r, a.lam)
        out["risk_sweep"].append({"risk": r, **{k: m[k] for k in ("acc", "coverage", "selective_acc",
                                                                 "passages_read", "utility")},
                                  "abrag_abstain_utility": ref["utility"]})
    sab = base["AB-RAG"][0]
    out["aurc"] = {"N1, P(correct) at its stop": aurc(P1[rows, s1], Ye[rows, s1]),
                   "AB-RAG, conf at its stop": aurc(conf[rows, sab], Ye[rows, sab]),
                   "AB-RAG answers ranked by N1's P(correct)": aurc(P1[rows, sab], Ye[rows, sab])}

    # ---- fixability: does the policy know WHEN another round helps?
    wrong = Ye[:, :-1] == 0
    nxt = (wrong & (Ye[:, 1:] == 1)).ravel()
    later = (wrong & (np.maximum.accumulate(Ye[:, ::-1], axis=1)[:, ::-1][:, 1:] == 1)).ravel()
    out["fixability_auroc"] = {
        target_name: {"1 - conf (AB-RAG)": round(auroc(1 - conf[:, :-1].ravel(), lab), 3),
                      "N1 estimated gain": round(auroc(gain1[:, :-1].ravel(), lab), 3),
                      "positives": int(lab.sum())}
        for target_name, lab in (("right at next round", nxt), ("right at any later round", later))}

    # ---- which signals matter (descriptive: refit on all training data)
    Xc = design(Ft, ALL_FEATURES)
    pol = fit_policy(Xc, Yt, Ut, Ks, a.lam, a.risk, a.folds, a.seed)
    if not isinstance(pol["p"], _Const):
        names = [f"round_{t + 1}" for t in range(len(Ks))] + ALL_FEATURES
        coef = pol["p"][-1].coef_[0]
        out["p_correct_coefs_std"] = {k: round(float(v), 3) for k, v in
                                      sorted(zip(names, coef), key=lambda kv: -abs(kv[1]))}
    return out


def print_report(out):
    print(f"\nlabel={out['label']}  lambda={out['lambda']}  r={out['risk']}  tau={out['tau']}  "
          f"rounds K={out['Ks']}  train n={out['n_train']}  eval n={out['n_eval']}  ({out['mode']})")
    print(f"{'policy':<32}{'acc':>7}{'cover':>7}{'sel.acc':>8}{'rounds':>8}{'passages':>10}{'utility':>9}")
    for k, v in out["policies"].items():
        cells = "".join(f"{v[c]:>{w}}" if c in v else f"{'':>{w}}" for c, w in
                        (("acc", 7), ("coverage", 7), ("selective_acc", 8), ("avg_rounds", 8),
                         ("passages_read", 10), ("utility", 9)))
        extra = f"  95% [{v['acc_95'][0]}, {v['acc_95'][1]}], P(random>=N1)={v['p_random_ge_n1']}" \
            if "acc_95" in v else ""
        print(f"{k:<32}{cells}{extra}")
    for ref, d in out["n1_vs"].items():
        print(f"N1 vs {ref}: +{d['plus']}/-{d['minus']}, McNemar p={d['mcnemar_p']}, 95% CI {d['ci95_pts']} pts")

    print("\nlambda sweep (N1 full features | AB-RAG at the same passages read)")
    for p in out["lambda_sweep"]["N1"]:
        print(f"  lambda={p['lambda']:<7} acc {p['acc']:>5}  passages {p['passages_read']:>5}  "
              f"AB-RAG {p['abrag_acc_same_cost'] if p['abrag_acc_same_cost'] is not None else '-':>5}")
    mc = out["matched_cost"]
    print(f"\nmatched cost (AB-RAG reads {mc['abrag_passages_read']} passages; lambda picked without labels)")
    for fs in ("N1", "N1 (paper signals)"):
        d = mc[fs]
        v = d["vs_abrag"]
        print(f"  {fs:<20} lambda={d['lambda']:<7} acc {d['acc']} at {d['passages_read']} passages; vs AB-RAG "
              f"+{v['plus']}/-{v['minus']} p={v['mcnemar_p']} CI {v['ci95_pts']}; "
              f"random same stops {d['random_same_stops_acc']} (P>= {d['p_random_ge']})")

    print("\nrisk sweep (N1, abstain allowed)")
    for p in out["risk_sweep"]:
        print(f"  r={p['risk']:<5} coverage {p['coverage']:>5}  selective acc {p['selective_acc']:>5}  "
              f"utility {p['utility']:>8}  (AB-RAG + abstain {p['abrag_abstain_utility']})")
    print("AURC (lower is better): " + ", ".join(f"{k} {v}" for k, v in out["aurc"].items()))
    for k, d in out["fixability_auroc"].items():
        print(f"fixability AUROC, {k} ({d['positives']} cases): 1-conf {d['1 - conf (AB-RAG)']}, "
              f"N1 gain {d['N1 estimated gain']}")
    if "p_correct_coefs_std" in out:
        top = list(out["p_correct_coefs_std"].items())[:8]
        print("largest standardized P(correct) coefficients: " + ", ".join(f"{k} {v:+}" for k, v in top))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--backend", default="local",
                    help="label from the traces file name, e.g. local or groq-gpt-oss-120b")
    ap.add_argument("--train_split", choices=["dev", "test"], default="dev")
    ap.add_argument("--eval_split", choices=["dev", "test", "all"], default="test")
    ap.add_argument("--cv", type=int, default=0,
                    help="cross-fit on --eval_split with this many folds instead of training on --train_split")
    ap.add_argument("--label", choices=list(LABELS), default=CONFIG["n1_label"])
    ap.add_argument("--lam", type=float, default=CONFIG["n1_lambda"])
    ap.add_argument("--risk", type=float, default=CONFIG["n1_risk"])
    ap.add_argument("--tau", type=float, default=CONFIG["tau"])
    ap.add_argument("--folds", type=int, default=CONFIG["n1_folds"])
    ap.add_argument("--n_perm", type=int, default=2000)
    a = ap.parse_args()
    a.seed = CONFIG["seed"]
    if a.cv == 1:
        ap.error("--cv needs at least 2 folds")
    if not a.cv and (a.eval_split == "all" or a.train_split == a.eval_split):
        ap.error("train and eval splits must differ; use --cv K to cross-fit on one split")
    a.mode = f"cv{a.cv}-{a.eval_split}" if a.cv else f"{a.train_split}-to-{a.eval_split}"
    RESULTS = CONFIG["results_dir"]

    path = f"{RESULTS}/traces_{a.dataset}_{a.backend}.jsonl"
    if not os.path.exists(path):
        print(f"[!] missing {path}; run 09_full_traces.py first"); return
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    splits = json.load(open(f"data/splits_{a.dataset}.json"))
    pick = lambda split: recs if split == "all" else [r for r in recs if r["qid"] in set(splits[split])]
    eval_recs = pick(a.eval_split)
    train_recs = eval_recs if a.cv else pick(a.train_split)
    need = 20 if not a.cv else 20 * a.cv // (a.cv - 1)
    if len(train_recs) < need or not eval_recs:
        print(f"[!] too few traces: {len(train_recs)} to train on ({a.train_split if not a.cv else a.eval_split}), "
              f"{len(eval_recs)} to evaluate; need at least {need} to fit"); return

    want = {p for r in recs for p in r["pids"]}
    texts_norm = {}
    for l in open(f"data/{a.dataset}.jsonl", encoding="utf-8"):
        for p in json.loads(l)["passages"]:
            if p["pid"] in want:
                texts_norm[p["pid"]] = normalize(p["text"])

    out = {"dataset": a.dataset, "backend": a.backend, **analyze(train_recs, eval_recs, texts_norm, a)}
    print_report(out)
    op = f"{RESULTS}/n1_{a.dataset}_{a.backend}_{a.label}_{a.mode}.json"
    json.dump(out, open(op, "w"), indent=2)
    print(f"\nsaved -> {op}")


if __name__ == "__main__":
    main()
