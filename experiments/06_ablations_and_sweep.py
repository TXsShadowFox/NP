"""
06_ablations_and_sweep.py
-------------------------
Tuning + ablation experiments. Reuses the per-iteration traces already saved
by 04_adaptive_loop.py, so NO LLM re-runs — runs in seconds.

Because every trace stored c_tok, c_con, v_ret at each iteration, we can
REPLAY the adaptive loop under any (alpha, beta, gamma, tau) and recompute:
  - final answer chosen
  - avg retrieval iterations
  - EM / F1
  - whether confidence separates correct from incorrect (AUROC)

Experiments:
  A. Threshold sweep: vary tau, show accuracy vs avg-retrievals tradeoff.
  B. Signal ablation: token-only, consistency-only, no-variance, full.
  C. Pick the tau that best separates correct vs incorrect answers.

Run:
    python experiments/06_ablations_and_sweep.py --dataset hotpotqa --backend local
"""
import argparse, json, os, re, sys, string
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from configs.config import CONFIG


def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em(pred, gold):
    return float(normalize(pred) == normalize(gold))


def token_f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum(min(p.count(t), g.count(t)) for t in set(p) if t in g)
    if common == 0:
        return 0.0
    prec, rec = common/len(p), common/len(g)
    return 2*prec*rec/(prec+rec)


def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    w = sum((1.0 if sp > sn else 0.5 if sp == sn else 0.0) for sp in pos for sn in neg)
    return w/(len(pos)*len(neg))


def conf_of(step, a, b, g):
    """recompute confidence for one trace step under given weights.
    S3 (v_ret) is a REWARD (+) per signal diagnostics, not a penalty."""
    c = a*step["c_tok"] + b*step["c_con"] + g*step["v_ret"]
    return float(np.clip(c, 0.0, 1.0))


def replay(records, alpha, beta, gamma, tau, T_max):
    """Replay the adaptive loop under given weights+threshold using saved traces."""
    out = []
    for r in records:
        trace = r["trace"]
        chosen = trace[-1]      # default: last step (budget exhausted)
        iters = len(trace)
        for i, step in enumerate(trace):
            c = conf_of(step, alpha, beta, gamma)
            if c >= tau:
                chosen = step
                iters = i + 1
                break
            if i + 1 >= T_max:
                chosen = step
                iters = i + 1
                break
        final_conf = conf_of(chosen, alpha, beta, gamma)
        out.append({
            "pred": chosen["answer"], "gold": r["gold_answer"],
            "conf": final_conf, "iters": iters,
            "correct": int(em(chosen["answer"], r["gold_answer"])),
        })
    return out


def metrics(rows):
    EM = 100*np.mean([em(x["pred"], x["gold"]) for x in rows])
    F1 = 100*np.mean([token_f1(x["pred"], x["gold"]) for x in rows])
    it = np.mean([x["iters"] for x in rows])
    au = auroc([x["conf"] for x in rows], [x["correct"] for x in rows])
    return EM, F1, it, au


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--backend", default="local")
    ap.add_argument("--runs_dir", default=CONFIG["results_dir"],
                    help="where to read run outputs from; use logs to replay the paper's committed runs")
    args = ap.parse_args()
    RUNS, RESULTS = args.runs_dir, CONFIG["results_dir"]   # read from RUNS, always write to RESULTS
    os.makedirs(RESULTS, exist_ok=True)

    path = f"{RUNS}/adaptive_{args.dataset}_{args.backend}.jsonl"
    if not os.path.exists(path):
        print(f"[!] missing {path}; run 04_adaptive_loop.py first"); return
    records = [json.loads(l) for l in open(path)]
    print(f"replaying {len(records)} saved traces (no LLM calls)\n")

    a, b, g = CONFIG["alpha"], CONFIG["beta"], CONFIG["gamma"]
    T_max = CONFIG["T_max"]

    # ---- A. threshold sweep ----
    print("=== A. Threshold sweep (full signals) ===")
    print(f"{'tau':>6}{'EM':>8}{'F1':>8}{'avg_it':>9}{'AUROC':>8}")
    best = None
    for tau in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        rows = replay(records, a, b, g, tau, T_max)
        EM, F1, it, au = metrics(rows)
        print(f"{tau:>6.2f}{EM:>8.1f}{F1:>8.1f}{it:>9.2f}{au:>8.3f}")
        # track tau giving best EM-per-retrieval, say
        if best is None or EM > best[1]:
            best = (tau, EM, F1, it, au)

    # ---- B. signal ablation (at default tau) ----
    print("\n=== B. Signal ablation (tau=%.2f) ===" % CONFIG["tau"])
    print(f"{'config':<22}{'EM':>8}{'F1':>8}{'avg_it':>9}{'AUROC':>8}")
    ablations = {
        "full (a,b,g)":        (a, b, g),
        "token only":          (1.0, 0.0, 0.0),
        "consistency only":    (0.0, 1.0, 0.0),
        "variance only":       (0.0, 0.0, 1.0),
        "token+variance":      (0.6, 0.0, 0.4),
        "no consistency":      (a, 0.0, g),
    }
    for name, (aa, bb, gg) in ablations.items():
        rows = replay(records, aa, bb, gg, CONFIG["tau"], T_max)
        EM, F1, it, au = metrics(rows)
        print(f"{name:<22}{EM:>8.1f}{F1:>8.1f}{it:>9.2f}{au:>8.3f}")

    # ---- C. which signal best predicts correctness (AUROC, single signal) ----
    print("\n=== C. Single-signal AUROC (does each signal predict correctness?) ===")
    # use first-iteration values for each question
    labels = [int(em(r["trace"][0]["answer"], r["gold_answer"])) for r in records]
    for sig in ["c_tok", "c_con", "v_ret"]:
        scores = [r["trace"][0][sig] for r in records]
        # v_ret is now a REWARD (higher variance = better retrieval separation),
        # so no inversion needed; higher = more confident for all three.
        au = auroc(scores, labels)
        print(f"  {sig:<8} AUROC = {au:.3f}")

    print(f"\nbest tau by EM: {best[0]:.2f} (EM={best[1]:.1f}, F1={best[2]:.1f}, "
          f"avg_it={best[3]:.2f}, AUROC={best[4]:.3f})")

    out = {"n": len(records), "best_tau_by_EM": best[0]}
    json.dump(out, open(f"{RESULTS}/sweep_{args.dataset}_{args.backend}.json", "w"), indent=2)
    print(f"saved -> results/sweep_{args.dataset}_{args.backend}.json")


if __name__ == "__main__":
    main()
