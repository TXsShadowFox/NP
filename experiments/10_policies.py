"""
10_policies.py
--------------
Phase 1 analysis. Replays stopping policies on the full traces from
09_full_traces.py. No model calls, runs in seconds.

Policies (same answers for all of them, so differences are the policy alone):
  static K=5 / 10 / 15   one pass with a fixed evidence size
  AB-RAG (tau)           the paper's rule: stop at the first round with
                         conf >= tau, else after T_max; return the LAST answer
  AB-RAG + max-conf      same stopping, return the most confident round
  random (budget-matched) AB-RAG's stop rounds shuffled across questions:
                         identical cost, no confidence. The key control.
  oracle                 correct if ANY round is correct (ceiling)

Cost is "passages read": every executed round re-reads its K passages, so
AB-RAG stopping at round 2 costs 5 + 10 = 15, static K=10 costs 10.

Every policy is scored under three correctness labels (EM, EM+alias,
contains; see metrics.py), with paired tests against static K=10 and a gain
decomposition: static K=10 -> first round (K=5) -> AB-RAG -> + max-conf.
Also N2 (S2 and S3 held at their run mean, as for the released runs) and a
tau sweep (0.30-0.95) with the random control's 95% band at every tau.

Run:
    python experiments/10_policies.py --dataset hotpotqa --backend local --split dev
Output: results/policies_{dataset}_{backend}_{split}.json
"""
import argparse, json, os, sys
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
TAUS = [round(t, 2) for t in np.arange(0.30, 0.951, 0.05)]


def stop_rounds(conf, tau):
    """Paper rule: index of the first round with conf >= tau, else the last round."""
    hit = conf >= tau
    return np.where(hit.any(axis=1), hit.argmax(axis=1), conf.shape[1] - 1)


def evaluate(C, conf, Ks, tau, n_perm, rng):
    """C: (n, T) 0/1 correctness per round; conf: (n, T) paper confidence."""
    n, T = C.shape
    rows = np.arange(n)
    cum_cost = np.cumsum(Ks)
    res = {}
    for t, K in enumerate(Ks):
        res[f"static K={K}"] = (C[:, t], 1.0, float(K))
    s = stop_rounds(conf, tau)
    cost = float(cum_cost[s].mean())
    res["AB-RAG"] = (C[rows, s], float(s.mean() + 1), cost)
    masked = np.where(np.arange(T)[None, :] <= s[:, None], conf, -1.0)
    res["AB-RAG + max-conf"] = (C[rows, masked.argmax(axis=1)], float(s.mean() + 1), cost)
    res["oracle (any round)"] = (C.max(axis=1), float(T), float(cum_cost[-1]))
    perm_acc = np.array([C[rows, rng.permutation(s)].mean() for _ in range(n_perm)])
    return res, s, perm_acc


def pct(x):
    return round(100 * float(np.mean(x)), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--backend", default="local",
                    help="label from the traces file name, e.g. local or groq-gpt-oss-120b")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--n_perm", type=int, default=2000)
    args = ap.parse_args()
    RESULTS = CONFIG["results_dir"]

    path = f"{RESULTS}/traces_{args.dataset}_{args.backend}.jsonl"
    if not os.path.exists(path):
        print(f"[!] missing {path}; run 09_full_traces.py first"); return
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    split_path = f"data/splits_{args.dataset}.json"
    if args.split != "all" and os.path.exists(split_path):
        keep = set(json.load(open(split_path))[args.split])
        recs = [r for r in recs if r["qid"] in keep]
    if not recs:
        print(f"[!] no traces for split={args.split}"); return
    Ks = [rd["K"] for rd in recs[0]["rounds"]]
    conf = np.array([[rd["conf"] for rd in r["rounds"]] for r in recs])
    tau = CONFIG["tau"]
    rng = np.random.default_rng(CONFIG["seed"])
    out = {"dataset": args.dataset, "backend": args.backend, "split": args.split,
           "n": len(recs), "Ks": Ks, "tau": tau, "qids": [r["qid"] for r in recs]}

    # N2: can S2+S3 ever change the stop decision at tau?
    a, b, g = CONFIG["alpha"], CONFIG["beta"], CONFIG["gamma"]
    steps = [rd for r in recs for rd in r["rounds"]]
    out["s2s3_decision_changes"] = sum(1 for rd in steps if (rd["conf"] >= tau) != (a * rd["c_tok"] >= tau))
    # N2 as 12_report.py measures it on the released runs: hold S2 and S3 at their run mean and count
    # flipped stop decisions, over the rounds AB-RAG executes (the paper's count) and over all rounds
    X = np.array([[[rd["c_tok"], rd["c_con"], rd["v_ret"]] for rd in r["rounds"]] for r in recs])
    flat = np.clip(a * X[..., 0] + b * X[..., 1].mean() + g * X[..., 2].mean(), 0, 1)
    flip = (conf >= tau) != (flat >= tau)
    executed = np.arange(len(Ks))[None, :] <= stop_rounds(conf, tau)[:, None]
    s2s3 = b * (X[..., 1] - X[..., 1].mean()) + g * (X[..., 2] - X[..., 2].mean())
    out["n2_held_at_mean"] = {"executed_rounds": int(executed.sum()), "flips_executed": int((flip & executed).sum()),
                              "all_rounds": int(flip.size), "flips_all": int(flip.sum()),
                              "sd_s2s3_contribution": round(float(s2s3.std()), 3),
                              "sd_s1_contribution": round(float((a * X[..., 0]).std()), 3),
                              "round1_stop_rate": round(float((conf[:, 0] >= tau).mean()), 3)}
    print(f"{args.dataset} / {args.backend} / {args.split}: n={len(recs)}, rounds K={Ks}, tau={tau}")
    print(f"S2+S3 changed the stop decision in {out['s2s3_decision_changes']}/{len(steps)} rounds")
    n2 = out["n2_held_at_mean"]
    print(f"N2, S2+S3 held at their mean: {n2['flips_executed']}/{n2['executed_rounds']} executed rounds flip "
          f"({n2['flips_all']}/{n2['all_rounds']} of all rounds); round-1 stop rate {n2['round1_stop_rate']}")
    sweep_rng = np.random.default_rng(CONFIG["seed"] + 1)   # own stream: keeps the tau-level control unchanged

    for name, fn in LABELS.items():
        C = np.array([[fn(rd["answer"], r) for rd in r["rounds"]] for r in recs], dtype=float)
        res, s, perm = evaluate(C, conf, Ks, tau, args.n_perm, rng)
        ab = res["AB-RAG"][0]
        block = {"policies": {k: {"acc": pct(v[0]), "avg_rounds": round(v[1], 2),
                                  "passages_read": round(v[2], 1)} for k, v in res.items()}}
        block["policies"]["random (budget-matched)"] = {
            "acc": pct(perm), "acc_95": [round(100 * float(np.percentile(perm, q)), 1) for q in (2.5, 97.5)],
            "p_random_ge_abrag": round(float((perm >= ab.mean()).mean()), 4),
            "avg_rounds": round(float(s.mean() + 1), 2),
            "passages_read": block["policies"]["AB-RAG"]["passages_read"]}
        # per-question 0/1, in qid order, so 12_report.py can pair traces variants question by question
        block["per_question"] = {k: "".join("1" if x else "0" for x in res[k][0])
                                 for k in ("static K=10", "AB-RAG", "AB-RAG + max-conf")}
        k10 = res["static K=10"][0]
        only_ab, only_st, p = mcnemar(k10, ab)
        lo, hi = paired_bootstrap(k10, ab)
        block["abrag_vs_static10"] = {"plus": only_ab, "minus": only_st, "mcnemar_p": round(p, 4),
                                      "ci95_pts": [round(lo, 1), round(hi, 1)]}
        block["gain_decomposition_pts"] = {   # N4
            "start_K5_vs_K10": round(pct(res["static K=5"][0]) - pct(k10), 1),
            "iteration": round(pct(ab) - pct(res["static K=5"][0]), 1),
            "selection_max_conf": round(pct(res["AB-RAG + max-conf"][0]) - pct(ab), 1)}
        brevity = [-len(r["rounds"][0]["answer"].split()) for r in recs]
        block["auroc_round1"] = {"conf": round(auroc(conf[:, 0], C[:, 0]), 3),
                                 "S1": round(auroc([r["rounds"][0]["c_tok"] for r in recs], C[:, 0]), 3),
                                 "brevity": round(auroc(brevity, C[:, 0]), 3)}   # N3
        block["tau_sweep"] = []
        rows = np.arange(len(recs))
        for t_ in TAUS:
            st = stop_rounds(conf, t_)
            rnd = np.array([C[rows, sweep_rng.permutation(st)].mean() for _ in range(1000)])
            block["tau_sweep"].append({"tau": t_, "avg_rounds": round(float(st.mean() + 1), 2),
                                       "passages_read": round(float(np.cumsum(Ks)[st].mean()), 2),
                                       "acc": pct(C[rows, st]), "acc_max_conf": pct(C[rows, np.where(
                                           np.arange(len(Ks))[None, :] <= st[:, None], conf, -1.0).argmax(axis=1)]),
                                       "random_acc": pct(rnd),
                                       "random_95": [round(100 * float(np.percentile(rnd, q)), 1) for q in (2.5, 97.5)]})
        out[name] = block

        print(f"\n--- label: {name} ---")
        print(f"{'policy':<26}{'acc':>7}{'rounds':>8}{'passages':>10}")
        for k, v in block["policies"].items():
            extra = f"   95% [{v['acc_95'][0]}, {v['acc_95'][1]}], P(random>=AB-RAG)={v['p_random_ge_abrag']}" \
                if "acc_95" in v else ""
            print(f"{k:<26}{v['acc']:>7}{v['avg_rounds']:>8}{v['passages_read']:>10}{extra}")
        d, gd = block["abrag_vs_static10"], block["gain_decomposition_pts"]
        print(f"AB-RAG vs static K=10: +{d['plus']}/-{d['minus']} questions, McNemar p={d['mcnemar_p']}, "
              f"95% CI {d['ci95_pts']} pts")
        print(f"gain decomposition (pts): start K5 {gd['start_K5_vs_K10']:+}, iteration {gd['iteration']:+}, "
              f"max-conf selection {gd['selection_max_conf']:+}")
        au = block["auroc_round1"]
        print(f"round-1 AUROC: conf {au['conf']}, S1 {au['S1']}, answer brevity {au['brevity']}")

    op = f"{RESULTS}/policies_{args.dataset}_{args.backend}_{args.split}.json"
    json.dump(out, open(op, "w"), indent=2)
    print(f"\nsaved -> {op}")


if __name__ == "__main__":
    main()
