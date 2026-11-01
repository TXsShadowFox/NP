"""
08_rescore.py
-------------
Re-scores saved adaptive runs WITHOUT any model calls, to check how much the
headline numbers depend on the answer-matching rule and on the run setup.

For every run it reports, under three correctness labels
  EM        exact match against the single stored answer (what the paper used)
  EM+alias  exact match against the answer and its aliases (TriviaQA)
  contains  gold appears as a phrase inside the prediction (lenient)
the following:
  - accuracy of AB-RAG, and of the first round alone (K=5, one pass)
  - high- vs low-confidence accuracy at tau
  - AUROC of the final confidence, and of answer brevity as a "confidence"
    (if brevity rivals the confidence, the confidence/EM link is a format effect)
  - paired McNemar + bootstrap CI vs the static baseline, when its cache exists
plus two run-integrity checks:
  - the (alpha, beta, gamma) that actually produced the stored confidences
  - how often S2+S3 change the stop decision vs an S1-only rule

Run (replaying the paper's committed runs):
    python experiments/08_rescore.py --runs_dir logs
Output: results/rescore_{run}.json
"""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from configs.config import CONFIG
from experiments.metrics import (exact_match, contains, golds_of, auroc,
                                 mcnemar, paired_bootstrap)

LABELS = {
    "EM": lambda pred, rec: exact_match(pred, rec["gold_answer"]),
    "EM+alias": lambda pred, rec: exact_match(pred, golds_of(rec)),
    "contains": lambda pred, rec: contains(pred, golds_of(rec)),
}


def load_aliases(dataset):
    path = f"data/{dataset}.jsonl"
    if not os.path.exists(path):
        return {}
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        out[r["qid"]] = r.get("aliases", [])
    return out


def fitted_weights(records):
    steps = [s for r in records for s in r["trace"] if 0.0 < s["conf"] < 1.0]   # skip clipped
    X = np.array([[s["c_tok"], s["c_con"], s["v_ret"]] for s in steps])
    y = np.array([s["conf"] for s in steps])
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    return [round(float(v), 3) for v in w], float(np.abs(X @ w - y).max())


def pct(xs):
    return round(100 * float(np.mean(xs)), 1) if len(xs) else None


def rescore(path, runs_dir, tau):
    run = os.path.basename(path)[len("adaptive_"):-len(".jsonl")]
    dataset = run.split("_")[0]
    records = [json.loads(l) for l in open(path, encoding="utf-8")]
    aliases = load_aliases(dataset)
    for r in records:
        r["aliases"] = aliases.get(r["qid"], [])

    out = {"run": run, "n": len(records),
           "n_with_aliases": sum(1 for r in records if r["aliases"])}
    w, err = fitted_weights(records)
    out["weights_used"] = {"alpha_beta_gamma": w, "max_fit_error": round(err, 4),
                           "config": [CONFIG["alpha"], CONFIG["beta"], CONFIG["gamma"]]}
    out["avg_iterations"] = round(float(np.mean([r["iterations_used"] for r in records])), 2)
    steps = [s for r in records for s in r["trace"]]
    a, b, g = CONFIG["alpha"], CONFIG["beta"], CONFIG["gamma"]
    full = lambda s: min(1.0, a * s["c_tok"] + b * s["c_con"] + g * s["v_ret"]) >= tau
    out["s2s3_decision_changes"] = f"{sum(1 for s in steps if full(s) != (a * s['c_tok'] >= tau))}/{len(steps)}"

    static_path = f"{runs_dir}/static_cache_{run}.jsonl"
    static = ({json.loads(l)["qid"]: json.loads(l) for l in open(static_path, encoding="utf-8")}
              if os.path.exists(static_path) else {})

    conf = [r["final_confidence"] for r in records]
    brevity = [-len(r["pred_answer"].split()) for r in records]
    for name, fn in LABELS.items():
        final = [fn(r["pred_answer"], r) for r in records]
        first = [fn(r["trace"][0]["answer"], r) for r in records]
        hi = [c for c, x in zip(final, conf) if x >= tau]
        lo = [c for c, x in zip(final, conf) if x < tau]
        row = {"abrag": pct(final), "first_round_K5": pct(first),
               "high_conf": f"{pct(hi)} (n={len(hi)})", "low_conf": f"{pct(lo)} (n={len(lo)})",
               "auroc_conf": round(auroc(conf, final), 3),
               "auroc_brevity": round(auroc(brevity, final), 3)}
        paired = [(fn(static[r["qid"]]["pred"], r), c) for r, c in zip(records, final)
                  if r["qid"] in static]
        if paired:
            sa, sb = zip(*paired)
            only_ab, only_static, p = mcnemar(sa, sb)
            lo_ci, hi_ci = paired_bootstrap(sa, sb)
            row["static_K10"] = pct(sa)
            row["vs_static"] = (f"+{only_ab}/-{only_static} questions, McNemar p={p:.3f}, "
                                f"95% CI [{lo_ci:+.1f}, {hi_ci:+.1f}] pts")
        out[name] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default=CONFIG["results_dir"],
                    help="where to read run outputs from; use logs to replay the paper's committed runs")
    args = ap.parse_args()
    RESULTS = CONFIG["results_dir"]
    os.makedirs(RESULTS, exist_ok=True)

    paths = sorted(glob.glob(f"{args.runs_dir}/adaptive_*.jsonl"))
    if not paths:
        print(f"[!] no adaptive_*.jsonl in {args.runs_dir}"); return
    for path in paths:
        res = rescore(path, args.runs_dir, CONFIG["tau"])
        json.dump(res, open(f"{RESULTS}/rescore_{res['run']}.json", "w"), indent=2)
        wu = res["weights_used"]
        print(f"\n=== {res['run']}  n={res['n']}  (aliases for {res['n_with_aliases']}) ===")
        print(f"weights that produced the stored confidences: {wu['alpha_beta_gamma']} "
              f"(fit err {wu['max_fit_error']}; config says {wu['config']})")
        print(f"avg iterations {res['avg_iterations']};  under the config weights, S2+S3 change the "
              f"stop decision in {res['s2s3_decision_changes']} steps")
        print(f"{'label':<10}{'AB-RAG':>8}{'K5 1st':>8}{'high-conf':>16}{'low-conf':>15}"
              f"{'AUROC':>7}{'brev.':>7}")
        for name in LABELS:
            r = res[name]
            print(f"{name:<10}{r['abrag']:>8}{r['first_round_K5']:>8}{r['high_conf']:>16}"
                  f"{r['low_conf']:>15}{r['auroc_conf']:>7}{r['auroc_brevity']:>7}")
            if "vs_static" in r:
                print(f"{'':<10}static K=10 {r['static_K10']}  ->  AB-RAG {r['vs_static']}")
    print(f"\nsaved -> {RESULTS}/rescore_*.json")


if __name__ == "__main__":
    main()
