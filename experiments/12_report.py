"""
12_report.py
------------
Benchmark report for N2-N4. Every results table of the paper (arXiv:2606.29090)
is recomputed from its committed runs in logs/ with the corrected evaluation and
set next to the numbers the paper prints, together with the new diagnostics and
all figures. Reads saved outputs only: no model or API calls, runs in seconds.

  tab:static    static RAG vs AB-RAG under EM / EM+alias / contains, paired tests
  tab:confsep   high- vs low-confidence accuracy under each label
  tab:sweep     tau sweep replayed from the committed traces (tau <= 0.6 only:
                the runs stopped at 0.6, so higher tau needs new full traces)
  tab:auroc     single-signal AUROC of the signals the loop actually used
  tab:recall    open-retrieval recall: paper / our rerun / 800-question pool
  tab:adaptive  the paper's worked examples, looked up in the committed traces
  N2            weights the runs really used; how often S2+S3 changed a stop decision
  N3            correct answers EM rejects for format; brevity as a "confidence"
  N4            gain decomposition (static K=10 -> first round K=5 -> AB-RAG), and
                the budget-matched random-stopping control once full traces from
                09_full_traces.py have been replayed by 10_policies.py

Run:
    python experiments/12_report.py
Output: results/report/REPORT.md, report.json, figures/*.png, paper_figures/*.png
"""
import glob, importlib.util, json, os, sys, zipfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from configs.config import CONFIG
from experiments.metrics import normalize, golds_of, token_f1, auroc, mcnemar, paired_bootstrap

_spec = importlib.util.spec_from_file_location(
    "policies_mod", os.path.join(os.path.dirname(__file__), "10_policies.py"))
policies_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(policies_mod)
LABELS, pct = policies_mod.LABELS, policies_mod.pct

OUT = os.path.join(CONFIG["results_dir"], "report")
FIG = os.path.join(OUT, "figures")
TAU = CONFIG["tau"]
RUNS = [("hotpotqa", "local", "Qwen-1.5B / HotpotQA"),
        ("hotpotqa", "ollama", "Llama-3.2-3B / HotpotQA"),
        ("hotpotqa", "claude", "Claude Haiku / HotpotQA"),
        ("triviaqa", "claude", "Claude Haiku / TriviaQA")]

# ---- numbers as printed in the paper (abrag.tex) ----
PAPER_STATIC = {   # static EM, AB-RAG EM, static F1, AB-RAG F1, avg iterations, conf AUROC
    "hotpotqa_local": (34.5, 34.5, 47.3, 46.2, 1.09, 0.600),
    "hotpotqa_ollama": (39.5, 45.0, 56.3, 58.3, 1.90, 0.644),
    "hotpotqa_claude": (32.5, 33.0, 51.5, 50.7, 1.36, 0.682),
    "triviaqa_claude": (35.5, 40.0, 50.8, 53.5, 1.81, 0.733)}
PAPER_CONFSEP = {"hotpotqa_local": ((34.7, 199), (0.0, 1)), "hotpotqa_ollama": ((51.6, 155), (22.2, 45)),
                 "hotpotqa_claude": ((36.5, 181), (0.0, 19)), "triviaqa_claude": ((57.6, 139), (0.0, 61))}
PAPER_SWEEP = {   # tau -> (avg iterations, EM)
    "hotpotqa_claude": {0.30: (1.00, 32.5), 0.45: (1.14, 32.5), 0.60: (1.36, 33.0), 0.75: (1.98, 34.5), 0.90: (2.61, 35.5)},
    "triviaqa_claude": {0.30: (1.00, 35.5), 0.45: (1.30, 37.0), 0.60: (1.81, 40.0), 0.75: (2.42, 40.0), 0.90: (2.88, 40.0)},
    "hotpotqa_local": {0.30: (1.00, 34.5), 0.45: (1.02, 34.5), 0.60: (1.09, 34.5), 0.75: (1.18, 34.0), 0.90: (1.27, 33.5)}}
PAPER_AUROC = {"hotpotqa_local": (0.607, 0.517, 0.429), "hotpotqa_ollama": (0.651, 0.568, 0.550),
               "hotpotqa_claude": (0.769, 0.353, 0.572), "triviaqa_claude": (0.776, 0.374, 0.594)}
PAPER_RECALL = {"hotpotqa": {"n": 500, "corpus": 4963, "bm25": [58.7, 72.8, 80.9, 86.2], "dense": [89.2, 95.3, 97.2, 98.3],
                             "hybrid": [79.8, 93.3, 96.9, 97.6], "reranked": [86.6, 94.0, 97.3, 97.6]},
                "triviaqa": {"n": 200, "corpus": 2582, "bm25": [43.6, 54.1, 63.7, 69.6], "dense": [66.9, 80.6, 89.1, 95.3],
                             "hybrid": [59.4, 75.8, 86.2, 90.8], "reranked": [67.0, 80.3, 88.4, 90.8]}}
PAPER_ADAPTIVE = [(0.83, None, "stop at round 1, correct"), (0.30, 0.77, "retrieve more, fixed in round 2"),
                  (0.41, 0.79, "retrieve more, fixed in round 2"), (0.22, None, "stays low, stop at budget")]
PAPER_VERBOSITY = [("Wichita", "Wichita, Kansas"), ("French", "the French language"), ("1912", "in the year 1912")]
PAPER_WEIGHTS = (CONFIG["alpha"], CONFIG["beta"], CONFIG["gamma"])
PAPER_FIGS = {"fig1": "concept: fixed-depth vs adaptive retrieval", "fig2": "architecture",
              "fig3": "the three confidence signals", "fig4": "worked example of the loop",
              "fig5": "open-retrieval recall", "fig6": "static vs AB-RAG EM across backbones",
              "fig7": "high- vs low-confidence EM", "fig8": "cost-accuracy tradeoff (tau sweep)",
              "fig9": "single-signal AUROC"}
MIN_TRACES = 50   # full-trace replays on fewer questions are not reported

# ---- figure style: reference palette (dataviz skill), light surface for print ----
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "legend.fontsize": 8.5, "figure.dpi": 110, "savefig.dpi": 220})


# ------------------------------------------------------------------ data

def load_run(ds, be):
    recs = [json.loads(l) for l in open(f"logs/adaptive_{ds}_{be}.jsonl", encoding="utf-8")]
    aliases = {}
    for l in open(f"data/{ds}.jsonl", encoding="utf-8"):
        r = json.loads(l)
        aliases[r["qid"]] = r.get("aliases", [])
    for r in recs:
        r["aliases"] = aliases.get(r["qid"], [])
    static = {}
    p = f"logs/static_cache_{ds}_{be}.jsonl"
    if os.path.exists(p):                                    # per-question static K=10 answers
        static = {json.loads(l)["qid"]: json.loads(l)["pred"] for l in open(p, encoding="utf-8")}
    exp = json.load(open(f"logs/experiment_{ds}_{be}.json"))
    return recs, static, exp


def f1_of(pred, rec, label):
    return token_f1(pred, rec["gold_answer"] if label == "EM" else golds_of(rec))


def paired(a, b):
    plus, minus, p = mcnemar(a, b)
    lo, hi = paired_bootstrap(a, b)
    return {"plus": plus, "minus": minus, "mcnemar_p": round(p, 4), "ci95_pts": [round(lo, 1), round(hi, 1)]}


def static_rows(recs, static, exp):
    has_alias = any(r["aliases"] for r in recs)
    out = {}
    for lab, fn in LABELS.items():
        ab = np.array([fn(r["pred_answer"], r) for r in recs])
        k5 = np.array([fn(r["trace"][0]["answer"], r) for r in recs])
        row = {"abrag": pct(ab), "first_round_K5": pct(k5),
               "conf_auroc": round(auroc([r["final_confidence"] for r in recs], ab), 3)}
        if lab != "contains":
            row["abrag_f1"] = pct([f1_of(r["pred_answer"], r, lab) for r in recs])
        if static:
            st = np.array([fn(static[r["qid"]], r) for r in recs])
            row["static_K10"] = pct(st)
            if lab != "contains":
                row["static_f1"] = pct([f1_of(static[r["qid"]], r, lab) for r in recs])
            row["vs_static"] = paired(st, ab)
        elif lab == "EM" or (lab == "EM+alias" and not has_alias):
            row["static_K10"] = exp["static"]["EM"]          # aggregate only: no per-question answers
            row["static_f1"] = exp["static"]["F1"]
        out[lab] = row
    return out


def confsep(recs):
    out = {}
    for lab, fn in LABELS.items():
        hi = [fn(r["pred_answer"], r) for r in recs if r["final_confidence"] >= TAU]
        lo = [fn(r["pred_answer"], r) for r in recs if r["final_confidence"] < TAU]
        out[lab] = {"high": [pct(hi), len(hi)] if hi else [None, 0], "low": [pct(lo), len(lo)] if lo else [None, 0]}
    return out


def replay_sweep(recs, fn):
    """AB-RAG at tau <= the run's tau, replayed exactly from the saved traces."""
    out = {}
    for tau in (0.30, 0.45, 0.60):
        stops = [next((i for i, st in enumerate(r["trace"]) if st["conf"] >= tau), len(r["trace"]) - 1)
                 for r in recs]
        out[tau] = (round(float(np.mean(stops)) + 1, 2),
                    pct([fn(r["trace"][s]["answer"], r) for r, s in zip(recs, stops)]))
    return out


def signal_auroc(recs):
    """First-round signals vs first-round correctness (the protocol the paper's S1/S2 match)."""
    out = {}
    for lab, fn in LABELS.items():
        y = [fn(r["trace"][0]["answer"], r) for r in recs]
        sig = {"S1": [r["trace"][0]["c_tok"] for r in recs], "S2": [r["trace"][0]["c_con"] for r in recs],
               "S3": [r["trace"][0]["v_ret"] for r in recs], "conf": [r["trace"][0]["conf"] for r in recs],
               "brevity": [-len(r["trace"][0]["answer"].split()) for r in recs]}
        out[lab] = {k: round(auroc(v, y), 3) for k, v in sig.items()}
    return out


def n2_influence(recs):
    steps = [st for r in recs for st in r["trace"]]
    X = np.array([[st["c_tok"], st["c_con"], st["v_ret"]] for st in steps])
    c = np.array([st["conf"] for st in steps])
    inside = (c > 0) & (c < 1)                       # unclipped rounds identify the weights
    w = np.linalg.lstsq(X[inside], c[inside], rcond=None)[0]
    err = float(np.abs(X[inside] @ w - c[inside]).max())
    # (a) under the weights the run really used: does the per-question VARIATION of S2/S3
    #     change any stop decision? Replace both by their run means and compare.
    flat = np.clip(w[0] * X[:, 0] + w[1] * X[:, 1].mean() + w[2] * X[:, 2].mean(), 0, 1)
    changes = int(((c >= TAU) != (flat >= TAU)).sum())
    # (b) as 08_rescore.py: paper weights, full confidence vs an S1-only rule
    a, b, g = PAPER_WEIGHTS
    paper_full = np.minimum(1.0, a * X[:, 0] + b * X[:, 1] + g * X[:, 2]) >= TAU
    changes_paper = int((paper_full != (a * X[:, 0] >= TAU)).sum())
    contrib = (X[:, 1] - X[:, 1].mean()) * w[1] + (X[:, 2] - X[:, 2].mean()) * w[2]
    return {"weights_used": [round(float(x), 3) for x in w], "fit_max_error": round(err, 4),
            "weights_paper": list(PAPER_WEIGHTS), "rounds": len(steps), "decision_changes": changes,
            "decision_changes_paper_weights_vs_s1_only": changes_paper,
            "sd_s2s3_contribution": round(float(contrib.std()), 3),
            "sd_s1_contribution": round(float((w[0] * X[:, 0]).std()), 3)}


def n3_verbosity(recs):
    em = [LABELS["EM"](r["pred_answer"], r) for r in recs]
    ct = [LABELS["contains"](r["pred_answer"], r) for r in recs]
    lens = np.array([len(r["pred_answer"].split()) for r in recs])
    em_a = np.array(em, bool)
    found = []
    for g, m in PAPER_VERBOSITY:
        hit = [r["qid"] for r in recs for st in r["trace"]
               if normalize(r["gold_answer"]) == normalize(g) and normalize(st["answer"]) == normalize(m)]
        found.append({"gold": g, "model": m, "found_in_logs": bool(hit)})
    return {"em_wrong_but_contains_gold": int(sum(1 for e, c in zip(em, ct) if not e and c)), "n": len(recs),
            "mean_words_em_correct": round(float(lens[em_a].mean()), 2) if em_a.any() else None,
            "mean_words_em_wrong": round(float(lens[~em_a].mean()), 2) if (~em_a).any() else None,
            "auroc_conf_final": {lab: round(auroc([r["final_confidence"] for r in recs],
                                                  [fn(r["pred_answer"], r) for r in recs]), 3) for lab, fn in LABELS.items()},
            "auroc_brevity_final": {lab: round(auroc([-len(r["pred_answer"].split()) for r in recs],
                                                     [fn(r["pred_answer"], r) for r in recs]), 3) for lab, fn in LABELS.items()},
            "paper_examples": found}


def n4_decomposition(srows):
    out = {}
    for lab, row in srows.items():
        st = row.get("static_K10")
        out[lab] = {"static_K10": st, "first_round_K5": row["first_round_K5"], "abrag": row["abrag"],
                    "start_K5_vs_K10": round(row["first_round_K5"] - st, 1) if st is not None else None,
                    "iteration": round(row["abrag"] - row["first_round_K5"], 1)}
    return out


def adaptive_examples(recs):
    out = []
    for c1, c2, claim in PAPER_ADAPTIVE:
        hits = [r for r in recs if round(r["trace"][0]["conf"], 2) == c1 and
                (c2 is None or (len(r["trace"]) > 1 and round(r["trace"][1]["conf"], 2) == c2))]
        out.append({"claim": claim, "first_conf": c1, "second_conf": c2, "matches": len(hits),
                    "examples": [{"qid": r["qid"], "confs": [round(s["conf"], 2) for s in r["trace"]],
                                  "EM": LABELS["EM+alias"](r["pred_answer"], r),
                                  "contains": LABELS["contains"](r["pred_answer"], r)} for r in hits[:3]]})
    return out


def retrieval():
    out = {}
    for ds in ("hotpotqa", "triviaqa"):
        out[ds] = {"paper": PAPER_RECALL[ds]}
        for tag, n in (("rerun_paper_setting", PAPER_RECALL[ds]["n"]), ("pool_800", 800)):
            p = f"{CONFIG['results_dir']}/open_retrieval_{ds}_n{n}.json"
            if os.path.exists(p):
                d = json.load(open(p))
                out[ds][tag] = {"n": d["n"], "corpus": d["corpus_size"],
                                **{m: [v[f"recall@{k}"] for k in (5, 10, 20, 50)] for m, v in d["summary"].items()}}
    return out


def improvements():
    """13_answer_extract.py (I1, I2) and 14_retrieval_upgrade.py (R1) outputs, when present.
    R1's (pool, reranker) is picked on dev by mean recall@5/@10, then read on the paper setting."""
    out = {}
    p = f"{CONFIG['results_dir']}/improved_answers.json"
    if os.path.exists(p):
        out["answers"] = json.load(open(p, encoding="utf-8"))
    for ds in ("hotpotqa", "triviaqa"):
        d = {}
        for split in ("dev", "paper"):
            f = f"{CONFIG['results_dir']}/retrieval_upgrade_{ds}_{split}.json"
            if os.path.exists(f):
                d[split] = json.load(open(f))
        if len(d) == 2:
            rr = {m: s for m, s in d["dev"]["summary"].items() if " + " in m}
            pick = max(rr, key=lambda m: rr[m]["recall@5"] + rr[m]["recall@10"])
            out.setdefault("retrieval", {})[ds] = {"pick": pick, "dev": d["dev"], "paper": d["paper"]}
    return out


def full_trace_replays():
    """10_policies.py outputs on full traces (random control, all-tau sweep)."""
    out = {}
    for p in sorted(glob.glob(f"{CONFIG['results_dir']}/policies_*.json")):
        d = json.load(open(p))
        if d.get("n", 0) >= MIN_TRACES:
            out[f"{d['dataset']} / {d['backend']} / {d['split']}"] = d
    return out


# --------------------------------------------------------------- figures

def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    return f"figures/{name}"


def short(name):
    model, ds = name.split(" / ")
    return f"{model.replace(' Haiku', '').replace('-3.2', '')}\n{ds}"


def fig_labels(R):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
    x = np.arange(len(RUNS))
    w = 0.26
    for i, lab in enumerate(LABELS):
        vals = [R[f"{ds}_{be}"]["static"][lab]["abrag"] for ds, be, _ in RUNS]
        a.bar(x + (i - 1) * w, vals, w * 0.92, color=SERIES[i], label=lab, zorder=2)
    a.set_xticks(x, [short(n) for _, _, n in RUNS])
    a.set_ylabel("AB-RAG accuracy (%)")
    a.set_ylim(0, 100)
    a.set_title("Same answers, three correctness labels")
    a.legend(loc="upper left", ncol=3)
    vs = R["triviaqa_claude"]["static"]
    labs = list(LABELS)
    for i, lab in enumerate(labs):
        d = vs[lab]["vs_static"]
        mid = vs[lab]["abrag"] - vs[lab]["static_K10"]
        b.errorbar(mid, i, xerr=[[mid - d["ci95_pts"][0]], [d["ci95_pts"][1] - mid]], fmt="o",
                   color=SERIES[i], ms=7, capsize=4, lw=2, zorder=3)
        b.annotate(f"{mid:+.1f} pts, p={d['mcnemar_p']:.3f}", (d["ci95_pts"][1], i), xytext=(6, 0),
                   textcoords="offset points", va="center", fontsize=8.5, color=INK2)
    b.axvline(0, color=MUTED, lw=1)
    b.set_yticks(range(len(labs)), labs)
    b.set_ylim(-0.6, len(labs) - 0.4)
    b.invert_yaxis()
    b.set_xlim(-5, 14)
    b.set_xlabel("AB-RAG minus static K=10 (points, 95% CI)")
    b.set_title("Claude / TriviaQA gain")
    fig.tight_layout()
    return save(fig, "fig_a_labels_static_vs_abrag.png")


def fig_confsep(R):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, lab in zip(axes, ("EM", "contains")):
        for j, (ds, be, name) in enumerate(RUNS):
            c = R[f"{ds}_{be}"]["confsep"][lab]
            (h, nh), (l, nl) = c["high"], c["low"]
            if l is not None:
                ax.plot([l, h], [j, j], color=GRID, lw=3, zorder=1)
                ax.plot(l, j, "o", color=SERIES[1], ms=8, zorder=2, label="low confidence" if j == 0 else None)
                ax.annotate(f"n={nl}", (l, j), xytext=(0, -13), textcoords="offset points", ha="center",
                            fontsize=7.5, color=INK2)
            ax.plot(h, j, "o", color=SERIES[0], ms=8, zorder=2, label="high confidence" if j == 0 else None)
            ax.annotate(f"n={nh}", (h, j), xytext=(0, 7), textcoords="offset points", ha="center",
                        fontsize=7.5, color=INK2)
        ax.set_yticks(range(len(RUNS)), [n for _, _, n in RUNS])
        ax.set_ylim(len(RUNS) - 0.4, -0.6)
        ax.set_xlim(-3, 100)
        ax.set_xlabel(f"accuracy (%) under {lab}")
        ax.set_title(f"Confidence split at tau={TAU}, {lab}")
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    return save(fig, "fig_b_confidence_split_by_label.png")


def fig_sweep(R):
    keys = [k for k in PAPER_SWEEP]
    fig, axes = plt.subplots(1, len(keys), figsize=(11, 3.3))
    for ax, k in zip(axes, keys):
        name = next(n for ds, be, n in RUNS if f"{ds}_{be}" == k)
        p = PAPER_SWEEP[k]
        taus = sorted(p)
        ax.plot([p[t][0] for t in taus], [p[t][1] for t in taus], "-", color=SERIES[0], lw=2, zorder=1,
                label="paper")
        for t in taus:
            ax.plot(*p[t], "o", ms=7, zorder=2, color=SERIES[0], mfc=SERIES[0] if t <= TAU else SURFACE, mew=1.6)
        rp = R[k]["sweep_replay"]["EM"]
        ax.plot([rp[t][0] for t in rp], [rp[t][1] for t in rp], "s--", color=SERIES[1], ms=6, lw=1.6,
                zorder=3, label="replayed from committed traces")
        at = {}
        for t in rp:                                   # one label per point, even when taus coincide
            at.setdefault(rp[t], []).append(f"{t:.2f}")
        for xy, ts in at.items():
            ax.annotate("tau " + ", ".join(ts), xy, xytext=(6, -11), textcoords="offset points",
                        fontsize=7.5, color=INK2)
        ax.margins(y=0.15)
        ax.set_title(name)
        ax.set_xlabel("avg. retrieval iterations")
    axes[0].set_ylabel("EM (%)")
    axes[0].legend(loc="upper left", fontsize=7.5)
    fig.text(0.01, -0.04, "Hollow paper points (tau 0.75, 0.90) cannot be replayed: the committed runs stopped "
             "at tau=0.6, so those rows came from runs that are not in the repository.", fontsize=8, color=INK2)
    fig.tight_layout()
    return save(fig, "fig_c_tau_sweep_paper_vs_replay.png")


def fig_auroc(R):
    sigs = ["S1", "S2", "S3", "conf", "brevity"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
    x = np.arange(len(RUNS))
    w = 0.16
    for ax, lab in zip(axes, ("EM", "contains")):
        for i, s in enumerate(sigs):
            vals = [R[f"{ds}_{be}"]["signal_auroc"][lab][s] for ds, be, _ in RUNS]
            ax.bar(x + (i - 2) * w, vals, w * 0.9, color=SERIES[i], label=s, zorder=2)
        if lab == "EM":
            for j, (ds, be, _) in enumerate(RUNS):
                for i, v in enumerate(PAPER_AUROC[f"{ds}_{be}"]):
                    ax.plot(x[j] + (i - 2) * w, v, "_", color=INK, ms=9, mew=2, zorder=3,
                            label="paper (S1-S3)" if i == 0 and j == 0 else None)
        ax.axhline(0.5, color=MUTED, lw=1, ls="--")
        ax.text(1.005, 0.5, "chance", fontsize=7.5, color=INK2, ha="left", va="center",
                transform=matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData))
        ax.set_xticks(x, [short(n) for _, _, n in RUNS])
        ax.set_ylim(0.25, 1.0)
        ax.set_title(f"First-round AUROC vs correctness, {lab}")
    axes[0].set_ylabel("AUROC")
    axes[0].legend(ncol=3, loc="upper left", fontsize=7.5)
    fig.tight_layout()
    return save(fig, "fig_d_signal_auroc_by_label.png")


def fig_n2(R):
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    for j, (ds, be, name) in enumerate(RUNS):
        d = R[f"{ds}_{be}"]["n2"]
        share = 100 * d["decision_changes"] / d["rounds"]
        ax.barh(j, share, 0.55, color=SERIES[0], zorder=2)
        w = d["weights_used"]
        ax.annotate(f"{d['decision_changes']}/{d['rounds']} rounds   weights used {w[0]:.2f} / {w[1]:.2f} / {w[2]:.2f}",
                    (share, j), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(RUNS)), [n for _, _, n in RUNS])
    ax.invert_yaxis()
    ax.set_xlim(0, 25)
    ax.set_xlabel("rounds whose stop decision flips when S2, S3 are held at their run mean (%)")
    ax.set_title("N2: do S2 and S3 change any stop decision? (weights each run used)")
    fig.tight_layout()
    return save(fig, "fig_e_n2_decision_influence.png")


def fig_n4(R):
    steps = ["static K=10", "round 1 (K=5)", "AB-RAG"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=False)
    for ax, lab in zip(axes, LABELS):
        for j, (ds, be, name) in enumerate(RUNS):
            d = R[f"{ds}_{be}"]["n4"][lab]
            ys = [d["static_K10"], d["first_round_K5"], d["abrag"]]
            xs = [i for i, y in enumerate(ys) if y is not None]
            ax.plot(xs, [ys[i] for i in xs], "o-", color=SERIES[j], lw=2, ms=6, label=name)
        ax.set_xticks(range(3), steps)
        ax.set_xlim(-0.3, 2.3)
        ax.set_title(f"N4: where the gain comes from, {lab}")
        ax.set_ylabel("accuracy (%)")
    fig.tight_layout()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=8)
    fig.text(0.01, -0.14, "Static K=10 is missing where the repository has no per-question static answers "
             "(HotpotQA under contains).", fontsize=8, color=INK2)
    return save(fig, "fig_f_n4_gain_decomposition.png")


def fig_recall(ret):
    if not any("pool_800" in v for v in ret.values()):
        return None
    ks = [5, 10, 20, 50]
    methods = ["bm25", "dense", "hybrid", "reranked"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, ds in zip(axes, ret):
        for i, m in enumerate(methods):
            ax.plot(ks, ret[ds]["paper"][m], "--", color=SERIES[i], lw=1.4, alpha=0.8)
            if "pool_800" in ret[ds]:
                ax.plot(ks, ret[ds]["pool_800"][m], "o-", color=SERIES[i], lw=2, ms=5, label=m)
        d = ret[ds].get("pool_800")
        pool = f"800-question pool, {d['corpus']:,} passages" if d else "800-question pool pending"
        ax.set_title(f"{ds} ({pool})", fontsize=10)
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.set_xlabel("k")
    axes[0].set_ylabel("recall@k (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], color=MUTED, lw=2, marker="o", ms=5),
                plt.Line2D([], [], color=MUTED, lw=1.4, ls="--")]
    labels += ["800-question pool (ours)", "paper"]
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    return save(fig, "fig_g_open_retrieval_recall.png")


def _box(ax, x, y, w, h, title, body, changed):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc="#e8f1fc" if changed else SURFACE, ec=SERIES[0] if changed else MUTED,
                                lw=1.6 if changed else 1.0))
    ax.text(x + 0.012, y + h - 0.03, title, fontsize=9, fontweight="bold", va="top", color=INK)
    ax.text(x + 0.012, y + h - 0.085, body, fontsize=7.4, va="top", color=INK2, linespacing=1.35)


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(12, 3.3))
    ax.set_axis_off()
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1)
    cols = [
        ("Data", "HotpotQA + TriviaQA\n+ TriviaQA aliases\n+ fixed splits: test 500\n  (has the paper's 200),\n  dev 300\n+ pooled corpus, 800 q", True),
        ("Retrieval", "(unchanged)\nBM25 k1 1.5, b 0.75\nbge-large dense\nRRF k=60 over top-20s\nMiniLM-L12 rerank\n+ rankings cached once", False),
        ("Generation", "Qwen / Llama / Claude\n+ Groq, OpenRouter\n  free tiers (quota-safe)\n+ multi-GPU / 4-bit\n+ Colab and Kaggle", True),
        ("Confidence + loop", "S1 token prob / self-cons.\nS2 evidence consistency\nS3 rerank variance\ntau 0.6, T_max 3, K 5-15\n+ weights logged per run", True),
        ("Evaluation (new)", "labels: EM, EM+alias,\n  contains\n+ McNemar, bootstrap\n+ full traces, K 5/10/15\n+ N2 decision influence\n+ N3 verbosity, brevity\n+ N4 decomposition,\n  random budget control", True)]
    w, gap = 0.18, 0.02
    for i, (t, b, ch) in enumerate(cols):
        x = i * (w + gap)
        _box(ax, x, 0.2, w, 0.7, t, b, ch)
        if i:
            _arrow(ax, x - gap + 0.002, 0.55, x - 0.002, 0.55)
    ax.add_patch(FancyBboxPatch((0.0, 0.03), 0.016, 0.07, boxstyle="round,pad=0.004", fc="#e8f1fc", ec=SERIES[0]))
    ax.text(0.024, 0.065, "changed or added in this project", fontsize=8, va="center", color=INK2)
    ax.add_patch(FancyBboxPatch((0.26, 0.03), 0.016, 0.07, boxstyle="round,pad=0.004", fc=SURFACE, ec=MUTED))
    ax.text(0.284, 0.065, "as in the paper", fontsize=8, va="center", color=INK2)
    ax.set_title("AB-RAG pipeline: what changed", fontsize=11)
    return save(fig, "fig_h_pipeline_changes.png")


def fig_protocol():
    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.set_axis_off()
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1)
    _box(ax, 0.0, 0.33, 0.16, 0.42, "Question", "ranked pool cached:\ntop-15 passages +\nreranker scores", False)
    for i, K in enumerate((5, 10, 15)):
        _box(ax, 0.22, 0.69 - i * 0.3, 0.2, 0.25, f"Round {i + 1}: K={K}", "answer + 5 samples\nS1, S2, S3, conf", True)
        _arrow(ax, 0.165, 0.55, 0.218, 0.815 - i * 0.3)
        _arrow(ax, 0.422, 0.815 - i * 0.3, 0.478, 0.55)
    _box(ax, 0.48, 0.33, 0.15, 0.42, "Full trace", "every question at\nevery round, saved\nonce (09)", True)
    _arrow(ax, 0.632, 0.55, 0.678, 0.55)
    _box(ax, 0.68, 0.06, 0.32, 0.88, "Offline replays (10, seconds)",
         "static K = 5 / 10 / 15\nAB-RAG at ANY tau (incl. 0.75, 0.90)\nAB-RAG + max-confidence answer\nrandom stopping, same budget (N4)\n"
         "oracle (any round right)\nall under EM / EM+alias / contains\nMcNemar + bootstrap vs static K=10\n"
         "gain decomposition (N4), S2+S3 influence (N2)", True)
    ax.set_title("Full-trace protocol: generate once, compare every policy on identical answers", fontsize=11)
    return save(fig, "fig_i_full_trace_protocol.png")


def fig_random_control(FT):
    grid = fulltrace_grid(FT)
    if not grid:
        return None
    variants = ["as generated", "+ I1"]
    fig, axes = plt.subplots(2, len(grid), figsize=(4.2 * len(grid), 5.6), squeeze=False)
    for j, (col, dd) in enumerate(grid.items()):
        for i, v in enumerate(variants):
            if v not in dd:
                axes[i][j].axis("off")
    for (j, (col, dd)), (i, v) in ((c, r) for c in enumerate(grid.items()) for r in enumerate(variants)):
        if v not in dd:
            continue
        ax, d = axes[i][j], dd[v]
        for i, lab in enumerate(LABELS):
            pol = d[lab]["policies"]
            r = pol["random (budget-matched)"]
            ax.errorbar(r["acc"], i, xerr=[[r["acc"] - r["acc_95"][0]], [r["acc_95"][1] - r["acc"]]],
                        fmt="o", color=MUTED, capsize=3, label="random, same budget" if i == 0 else None)
            ax.plot(pol["AB-RAG"]["acc"], i, "D", color=SERIES[0], ms=7, label="AB-RAG" if i == 0 else None)
            ax.plot(pol["static K=10"]["acc"], i, "s", color=SERIES[1], ms=6, label="static K=10" if i == 0 else None)
        ax.set_yticks(range(len(LABELS)), list(LABELS))
        ax.invert_yaxis()
        ax.set_xlabel("accuracy (%)")
        ax.set_title(f"{d['dataset']}, {variant_name(d['backend']).split(', ', 1)[1]} (n={d['n']})", fontsize=9.5)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03), fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, "fig_j_random_budget_control.png")


def fulltrace_grid(FT):
    """Test-split replays as {(dataset, retrieval): {"as generated" | "+ I1": policies dict}}."""
    grid = {}
    for d in FT.values():
        if d["split"] == "test":
            col = (d["dataset"], "R1 retrieval" if "-r1" in d["backend"] else "paper retrieval")
            grid.setdefault(col, {})["+ I1" if d["backend"].endswith("-i1") else "as generated"] = d
    return dict(sorted(grid.items(), key=lambda kv: (kv[0][0], kv[0][1] != "paper retrieval")))


def fig_fulltrace_sweep(FT):
    """EM vs passages read: AB-RAG at every tau, the random control's 95% band at the same budget,
    and static K. One column per (dataset, retrieval), one row without and one with I1."""
    grid = fulltrace_grid(FT)
    if not grid:
        return None
    variants = ["as generated", "+ I1"]
    fig, axes = plt.subplots(2, len(grid), figsize=(3.7 * len(grid), 6.0), squeeze=False, sharex=True)
    for j, ((ds, retr), dd) in enumerate(grid.items()):
        for i, v in enumerate(variants):
            ax = axes[i][j]
            if v not in dd:
                ax.axis("off")
                continue
            d = dd[v]
            sw = d["EM"]["tau_sweep"]
            x = [s["passages_read"] for s in sw]
            ax.fill_between(x, [s["random_95"][0] for s in sw], [s["random_95"][1] for s in sw], color=GRID,
                            lw=0, zorder=1, label="random stopping, same budget (95%)")
            ax.plot(x, [s["acc"] for s in sw], "-o", color=SERIES[0], ms=4, lw=2, zorder=3, label="AB-RAG, tau 0.30–0.95")
            at = next(s for s in sw if abs(s["tau"] - TAU) < 1e-9)
            ax.plot(at["passages_read"], at["acc"], "o", ms=8, mfc=SURFACE, mec=SERIES[0], mew=2, zorder=4)
            ax.annotate(f"tau={TAU}", (at["passages_read"], at["acc"]), xytext=(6, 6), textcoords="offset points",
                        fontsize=7.5, color=INK2)
            pol = d["EM"]["policies"]
            ks = d["Ks"]
            ax.plot(ks, [pol[f"static K={k}"]["acc"] for k in ks], "s", color=SERIES[1], ms=6, zorder=3,
                    label="static K = 5, 10, 15")
            ax.set_xscale("log")
            ax.minorticks_off()
            ax.set_xticks([5, 10, 15, 30], ["5", "10", "15", "30"])
            lo = min(min(s["random_95"][0] for s in sw), min(s["acc"] for s in sw))
            ax.set_ylim(lo - 1.5, lo + 6.5)
            ax.set_title(f"{ds}, {retr}{', + I1' if v == '+ I1' else ''}", fontsize=9.5)
            if j == 0:
                ax.set_ylabel("EM (%)")
            if i == 1:
                ax.set_xlabel("passages read per question")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03), fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, "fig_m_fulltrace_tau_sweep.png")


# ---------------------------------------------------------------- report

IMP_VARIANTS = ["paper AB-RAG", "AB-RAG + I1 short answer", "AB-RAG + I1 + I2"]
PAPER_RERANK = "rrf20 + ms-marco-MiniLM-L12-v2"


def fig_answer_improvements(IMP):
    if "answers" not in IMP:
        return None
    runs = IMP["answers"]["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    x = np.arange(len(RUNS))
    w = 0.26
    colors = [MUTED, SERIES[0], SERIES[2]]
    names = ["paper AB-RAG", "+ I1 short answer", "+ I1 + I2 best round"]
    for ax, metric in zip(axes, ("EM", "F1")):
        for i, v in enumerate(IMP_VARIANTS):
            vals = [runs[f"{ds}_{be}"]["scores"][v][metric] for ds, be, _ in RUNS]
            ax.bar(x + (i - 1) * w, vals, w * 0.92, color=colors[i], label=names[i], zorder=2)
            for xi, val in zip(x, vals):
                ax.text(xi + (i - 1) * w, val + 1, f"{val:.1f}", ha="center", fontsize=6.8, color=INK2)
        ax.set_xticks(x, [short(n) for _, _, n in RUNS])
        ax.set_ylim(0, 75)
        ax.set_title(f"{metric} on the paper's answers" if metric == "EM" else "Token F1 on the paper's answers")
    axes[0].set_ylabel("score (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return save(fig, "fig_k_answer_improvements.png")


def fig_retrieval_upgrade(IMP):
    if "retrieval" not in IMP:
        return None
    ks = [5, 10, 15, 20, 50]
    fig, axes = plt.subplots(1, len(IMP["retrieval"]), figsize=(10, 3.4), sharey=False, squeeze=False)
    for ax, (ds, d) in zip(axes[0], IMP["retrieval"].items()):
        s = d["paper"]["summary"]
        for m, color, style, label in ((PAPER_RERANK, MUTED, "--", "paper pipeline (reranked)"),
                                       ("dense", SERIES[1], ":", "dense only"),
                                       (d["pick"], SERIES[0], "-", f"R1: {d['pick']}")):
            ax.plot(ks, [s[m][f"recall@{k}"] for k in ks], style, marker="o", ms=4, lw=2, color=color, label=label)
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.set_xlabel("k (evidence passages)")
        ax.set_title(f"{ds}, paper setting ({d['paper']['n']} questions)")
        ax.legend(loc="lower right", fontsize=7.5)
    axes[0][0].set_ylabel("recall@k (%)")
    fig.tight_layout()
    return save(fig, "fig_l_retrieval_upgrade.png")


def variant_name(backend):
    base = backend.replace("-r1", "").replace("-i1", "")
    return f"{base}, {'R1' if '-r1' in backend else 'paper'} retrieval" + (" + I1" if backend.endswith("-i1") else "")


def fulltrace_variants(FT, figs):
    """Test-split replays of every traces variant: paper vs R1 retrieval, with and without I1."""
    test = sorted((d for d in FT.values() if d["split"] == "test"),
                  key=lambda d: (d["dataset"], "-r1" in d["backend"], d["backend"].endswith("-i1"), d["backend"]))
    if not test:
        return ["**On the new test split: pending.** No full traces with at least "
                f"{MIN_TRACES} test questions yet (Colab notebook).", ""]
    L = ["**On the new test split (full traces, frozen I1).** Each row is one traces file replayed by "
         "`10_policies.py`. “AB-RAG + I2” returns the most confident round; the random control shuffles AB-RAG's "
         "stopping rounds across questions (same budget, no confidence).", ""]
    for lab in ("EM", "contains"):
        rows = []
        for d in test:
            pol = d[lab]["policies"]
            r = pol["random (budget-matched)"]
            rows.append([d["dataset"], variant_name(d["backend"]), d["n"], pol["static K=10"]["acc"],
                         pol["AB-RAG"]["acc"], pol["AB-RAG + max-conf"]["acc"],
                         f"{r['acc']} [{r['acc_95'][0]}, {r['acc_95'][1]}]", r["p_random_ge_abrag"],
                         pol["oracle (any round)"]["acc"]])
        L += [f"{lab}:", "", md_table(["Dataset", "Variant", "n", "Static K=10", "AB-RAG", "AB-RAG + I2",
                                       "Random, same budget [95%]", "P(random ≥ AB-RAG)", "Oracle"], rows), ""]
    if figs.get("ft_sweep"):
        L += ["Every threshold, not only tau = 0.6: EM against passages read, with the random control's 95% band "
              "at each tau's budget.", "", f"![full-trace tau sweep]({figs['ft_sweep']})", ""]
    return L + fulltrace_n2(test) + fulltrace_paired(test)


def paper_qids(ds):
    """The paper's 200 questions (the first 200 of the test split). I1's instruction was revised on them."""
    return {json.loads(l)["qid"] for p in glob.glob(f"logs/adaptive_{ds}_*.jsonl") for l in open(p, encoding="utf-8")}


def fmt_pval(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def paired_cells(ref, new, seen):
    """ref, new: {qid: "0"/"1"}. Paired stats on all common questions, and Δ on those not in `seen`."""
    out = []
    for keep in (lambda q: True, lambda q: q not in seen):
        q = [k for k in new if k in ref and keep(k)]
        if not q:
            out.append(None)
            continue
        a, b = [ref[k] == "1" for k in q], [new[k] == "1" for k in q]
        plus, minus, p = mcnemar(a, b)
        out.append((len(q), pct(a), pct(b), plus, minus, p, paired_bootstrap(a, b)))
    (n, acc_a, acc_b, plus, minus, p, (lo, hi)), clean = out
    return [n, acc_a, acc_b, f"+{plus}/−{minus}", fmt_pval(p), f"[{lo:+.1f}, {hi:+.1f}]",
            f"{clean[2] - clean[1]:+.1f} (+{clean[3]}/−{clean[4]}, p{'' if clean[5] < 0.001 else '='}"
            f"{fmt_pval(clean[5])}, n={clean[0]})"
            if clean else "none yet"]


def fulltrace_paired(test):
    """Every improved pipeline vs the base traces' AB-RAG, paired by question, on all test questions
    and on questions 201-500 alone (outside the paper's runs, so never seen while designing I1).
    Then R1 for static retrieval: static K=10 with R1 vs with the paper's retrieval."""
    out = []
    head = ["n", "Base", "This pipeline", "+/− questions", "McNemar p", "95% CI (pts)", "Questions 201–500 only: Δ pts"]
    for lab in ("EM", "contains"):
        rows, static_rows = [], []
        for d in test:
            base = next((b for b in test if b["dataset"] == d["dataset"]
                         and b["backend"] == d["backend"].replace("-r1", "").replace("-i1", "")), None)
            if base is None or "qids" not in d or "qids" not in base:
                continue
            seen = paper_qids(d["dataset"])
            ref = dict(zip(base["qids"], base[lab]["per_question"]["AB-RAG"]))
            for pol in ("AB-RAG", "AB-RAG + max-conf"):
                if d is base and pol == "AB-RAG":
                    continue
                new = dict(zip(d["qids"], d[lab]["per_question"][pol]))
                rows.append([d["dataset"], variant_name(d["backend"]) + (" + I2" if pol != "AB-RAG" else "")]
                            + paired_cells(ref, new, seen))
            if "-r1" in d["backend"]:
                same = next((b for b in test if b["dataset"] == d["dataset"]
                             and b["backend"] == d["backend"].replace("-r1", "")), None)
                if same is not None:
                    st = lambda x: dict(zip(x["qids"], x[lab]["per_question"]["static K=10"]))
                    static_rows.append([d["dataset"], "static K=10" + (" + I1" if d["backend"].endswith("-i1") else "")]
                                       + paired_cells(st(same), st(d), seen))
        if rows:
            out += [f"{lab}, paired against the base pipeline's AB-RAG on the same questions:", "",
                    md_table(["Dataset", "Pipeline"] + head, rows), ""]
        if static_rows:
            out += [f"{lab}, R1 for static retrieval (Base = the paper's retrieval, same answer stage):", "",
                    md_table(["Dataset", "Policy"] + head, static_rows), ""]
    if out:
        out = ["Questions 201–500 are outside the paper's runs, so I1's instruction (revised once on the paper's "
               "200) has never seen them; R1 was chosen on dev questions only.", ""] + out
    return out


def fulltrace_n2(test):
    """N2 on the full traces: stop decisions that flip when S2 and S3 are held at their run mean."""
    rows = []
    for d in test:
        n2 = d.get("n2_held_at_mean")
        if n2 and not d["backend"].endswith("-i1"):      # I1 leaves every signal unchanged
            rows.append([d["dataset"], variant_name(d["backend"]).split(", ", 1)[1],
                         f"{n2['flips_executed']}/{n2['executed_rounds']}", f"{n2['flips_all']}/{n2['all_rounds']}",
                         f"{n2['sd_s2s3_contribution']} / {n2['sd_s1_contribution']}", n2["round1_stop_rate"]])
    if not rows:
        return []
    return ["**N2 on the full traces** (paper weights; S2 and S3 held at their run mean):", "",
            md_table(["Dataset", "Retrieval", "Flips, executed rounds", "Flips, all rounds",
                      "SD of conf. from S2+S3 / from S1", "Stops after round 1"], rows), ""]


def improvements_section(IMP, FT, figs):
    L = ["## 10. Improvements over the paper (I1, I2, R1)", ""]
    if "answers" not in IMP and "retrieval" not in IMP:
        return L + ["**Pending.** Run `13_answer_extract.py` and `14_retrieval_upgrade.py`.", ""] + fulltrace_variants(FT, figs)
    if "answers" in IMP:
        A = IMP["answers"]
        L += ["**I1, short-answer stage** and **I2, best-round answer** (`13_answer_extract.py`) are applied to the "
              "paper's own saved answers, so every comparison with the paper is paired, question by question. "
              f"I1 is {A['settings']['extractor']} (greedy) cutting a sentence answer down to the phrase that answers "
              "the question. It may only reuse words of the original answer and never sees the gold; answers of one "
              "or two words pass through. I2 answers with the highest-confidence round AB-RAG took instead of the last. "
              "I1's instruction was revised once after reading errors on these questions (revision 1 cut complete "
              "names, e.g. “Indianapolis Motor Speedway” to “Indianapolis”), so the clean test of the frozen stage is "
              "the 500-question test split of the new runs.", ""]
        rows = []
        for ds, be, name in RUNS:
            r = A["runs"][f"{ds}_{be}"]
            sc = r["scores"]
            labs = ["EM", "EM+alias", "F1", "contains"] if ds == "triviaqa" else ["EM", "F1", "contains"]
            for lab in labs:
                pv = r["paired_vs_paper"]["AB-RAG + I1 + I2"].get(lab)
                rows.append([name, lab] + [sc[v][lab] for v in IMP_VARIANTS] +
                            [f"{sc['AB-RAG + I1 + I2'][lab] - sc['paper AB-RAG'][lab]:+.1f}", fmt_p(pv) if pv else "–"])
        L += [md_table(["Run", "Metric", "Paper AB-RAG", "+ I1", "+ I1 + I2", "Change", "Paired vs paper (I1 + I2)"],
                       rows), ""]
        t = A["runs"]["triviaqa_claude"]
        L += ["Each “contains” drop under I1 is either an answer that said “unknown” but mentioned the gold in its "
              "explanation (credited by contains, rightly not by EM) or one of 3–4 answers per run where the stage cut "
              "the gold away. In no run did I1 turn an EM-correct answer wrong; the few losses in the last column "
              "come from I2.", "",
              f"Static K=10 gets the same stage on Claude / TriviaQA: EM {t['static']['static K=10']['EM']} → "
              f"{t['static']['static K=10 + I1']['EM']}. AB-RAG + I1 vs static + I1: "
              + "; ".join(f"{lab} {fmt_p(p)}" for lab, p in t["paired_vs_static_same_stage"].items()) +
              ". With the format fixed on both sides, AB-RAG's lead over static RAG is no longer significant.", "",
              "Confidence AUROC under EM, paper → + I1: " +
              "; ".join(f"{A['runs'][f'{ds}_{be}']['label']} {A['runs'][f'{ds}_{be}']['conf_auroc_EM']['paper AB-RAG']:.3f} → "
                        f"{A['runs'][f'{ds}_{be}']['conf_auroc_EM']['AB-RAG + I1 short answer']:.3f}" for ds, be, _ in RUNS) +
              ". It falls for Llama and both Claude runs: part of the paper's AUROC came from long answers being both "
              "low-confidence and EM-wrong (N3).", ""]
        if figs.get("answers"):
            L += [f"![answers]({figs['answers']})", ""]
    if "retrieval" in IMP:
        L += ["**R1, rerank pool and reranker** (`14_retrieval_upgrade.py`). The paper reranks at most 40 passages "
              "(RRF of BM25 top-20 and dense top-20, `rrf20`) with MiniLM-L12. R1 also tries the union of BM25 top-100 "
              "and dense top-100 (`union100`) and two BGE rerankers, picks the (pool, reranker) pair on dev questions "
              "(rows 500–800, none of which the paper evaluated) by mean recall@5/@10, and reports it on the paper's "
              "setting.", ""]
        for ds, d in IMP["retrieval"].items():
            sd, sp, pick = d["dev"]["summary"], d["paper"]["summary"], d["pick"]
            bge = max((m for m in sd if "bge-reranker" in m), key=lambda m: sd[m]["recall@5"] + sd[m]["recall@10"])
            line = (f"- {ds}: picked **{pick}**. Paper setting, recall@5 {sp[PAPER_RERANK]['recall@5']} → "
                    f"{sp[pick]['recall@5']} ({sp[pick]['recall@5'] - sp[PAPER_RERANK]['recall@5']:+.1f}), @10 "
                    f"{sp[PAPER_RERANK]['recall@10']} → {sp[pick]['recall@10']} "
                    f"({sp[pick]['recall@10'] - sp[PAPER_RERANK]['recall@10']:+.1f}).")
            if "bge-reranker" not in pick:
                line += (f" No BGE reranker beat MiniLM on dev (best, {bge}: recall@5 {sd[bge]['recall@5']} vs "
                         f"{sd[PAPER_RERANK]['recall@5']}), so R1 keeps the paper's reranker here and changes "
                         "little; the Colab notebook runs R1 traces only where it changes retrieval.")
            pool, model = CONFIG["r1_retrieval"][ds]
            if f"{pool} + {model.split('/')[-1]}" != pick:
                line += f" **[!] config r1_retrieval says {pool} + {model.split('/')[-1]}, not the dev pick.**"
            L.append(line)
        L.append("")
        rows = []
        for ds, d in IMP["retrieval"].items():
            s = d["paper"]["summary"]
            fmt = lambda m: " / ".join(str(s[m][f"recall@{k}"]) for k in (5, 10, 15, 20, 50))
            rows += [[ds, "paper, as published", " / ".join(str(v) for v in PAPER_RECALL[ds]["reranked"][:2]) +
                      " / – / " + " / ".join(str(v) for v in PAPER_RECALL[ds]["reranked"][2:])],
                     [ds, f"paper pipeline, our run ({PAPER_RERANK})", fmt(PAPER_RERANK)],
                     [ds, "dense only", fmt("dense")],
                     [ds, f"**R1 (picked on dev): {d['pick']}**", fmt(d["pick"])]]
        L += [md_table(["Dataset", "Retrieval", "Recall @5 / 10 / 15 / 20 / 50, paper setting"], rows), ""]
        if figs.get("retrieval_upgrade"):
            L += [f"![retrieval upgrade]({figs['retrieval_upgrade']})", ""]
    return L + fulltrace_variants(FT, figs)


def md_table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join("–" if c is None else str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fmt_p(d):
    p = d["mcnemar_p"]
    return (f"+{d['plus']}/−{d['minus']}, p{'' if p < 0.001 else '='}{fmt_pval(p)}, "
            f"CI [{d['ci95_pts'][0]:+}, {d['ci95_pts'][1]:+}]")


def write_report(R, ret, FT, IMP, figs):
    L = ["# AB-RAG follow-up: benchmarks for N2–N4", "",
         "Generated by `experiments/12_report.py`. Every number below is recomputed from saved outputs "
         "(the paper's committed runs in `logs/`, and our own `results/`); nothing here calls a model. "
         "N1 (value-of-retrieval stopping) is set aside and not reported.", "",
         "**Scope.** The paper's runs were re-scored, not re-generated: same answers, corrected evaluation. "
         "Results that need answers at every round for every question (the random budget-matched control, "
         "the tau sweep above 0.6) come from the new full traces (end of section 9, and section 10).", ""]

    L += ["## 1. Static RAG vs AB-RAG (paper Table `tab:static`)", "",
          "Under EM the paper's numbers match its saved answers in `logs/` exactly (re-scored, not re-generated: the models were not re-run). The other two labels use the same answers.", ""]
    rows = []
    for ds, be, name in RUNS:
        k = f"{ds}_{be}"
        P, S = PAPER_STATIC[k], R[k]["static"]
        sv = lambda lab: f"{S[lab].get('static_K10', '–')} / {S[lab]['abrag']}"
        rows.append([name, f"{P[0]} / {P[1]}", sv("EM"), sv("EM+alias"), sv("contains"),
                     f"{P[3]} / {S['EM']['abrag_f1']}", f"{P[4]:.2f} / {R[k]['avg_iters']:.2f}",
                     f"{P[5]:.3f} / {S['EM']['conf_auroc']:.3f}"])
    L += [md_table(["Backbone / dataset", "Paper EM static / AB-RAG", "Ours EM", "Ours EM+alias",
                    "Ours contains", "AB-RAG F1 paper / ours", "Avg iter paper / ours", "AUROC paper / ours"], rows), ""]
    t = R["triviaqa_claude"]["static"]
    L += ["Paired test, Claude / TriviaQA (the only run with per-question static answers in the repository): " +
          "; ".join(f"{lab} {fmt_p(t[lab]['vs_static'])}" for lab in LABELS) + ".", "",
          f"![labels]({figs['labels']})", ""]

    L += ["## 2. High- vs low-confidence accuracy (paper Table `tab:confsep`)", ""]
    rows = []
    for ds, be, name in RUNS:
        k = f"{ds}_{be}"
        (ph, pnh), (pl, pnl) = PAPER_CONFSEP[k]
        c = R[k]["confsep"]
        rows.append([name, f"{ph} ({pnh}) / {pl} ({pnl})"] +
                    [f"{c[lab]['high'][0]} ({c[lab]['high'][1]}) / {c[lab]['low'][0]} ({c[lab]['low'][1]})" for lab in LABELS])
    L += [md_table(["Backbone / dataset", "Paper EM high / low"] + [f"Ours {lab} high / low" for lab in LABELS], rows), "",
          f"![confsep]({figs['confsep']})", ""]

    L += ["## 3. Threshold sweep (paper Table `tab:sweep`)", "",
          "Replayed exactly from the committed traces for tau ≤ 0.6. Values are avg iterations / EM.", ""]
    rows = []
    for k, p in PAPER_SWEEP.items():
        name = next(n for ds, be, n in RUNS if f"{ds}_{be}" == k)
        rp = R[k]["sweep_replay"]["EM"]
        for tau in sorted(p):
            rows.append([name, f"{tau:.2f}", f"{p[tau][0]:.2f} / {p[tau][1]}",
                         f"{rp[tau][0]:.2f} / {rp[tau][1]}" if tau in rp else "not replayable from logs"])
    L += [md_table(["Run", "tau", "Paper", "Replayed"], rows), "",
          "At tau = 0.30 every question stops after round 1, so AB-RAG answers with K=5 (31.0 / 38.5 / 35.5 EM). "
          "The paper's tau = 0.30 row instead equals the static K=10 baseline in all three runs (32.5 / 35.5 / 34.5). "
          "The paper's other rows also differ, so its sweep came from separate runs that are not in the repository; "
          "with self-consistency sampling a rerun gives different answers. The full traces settle this for every tau.", "",
          f"![sweep]({figs['sweep']})", ""]

    L += ["## 4. Single-signal AUROC (paper Table `tab:auroc`)", "",
          "First-round signals as logged by the loop, against first-round correctness. S1 and S2 match the "
          "paper under EM.", ""]
    rows = []
    for ds, be, name in RUNS:
        k = f"{ds}_{be}"
        a = R[k]["signal_auroc"]
        rows.append([name, " / ".join(f"{v:.3f}" for v in PAPER_AUROC[k])] +
                    [" / ".join(f"{a[lab][s]:.3f}" for s in ("S1", "S2", "S3")) for lab in LABELS] +
                    [f"{a['EM']['brevity']:.3f} / {a['contains']['brevity']:.3f}"])
    L += [md_table(["Backbone / dataset", "Paper S1/S2/S3"] + [f"Ours {lab} S1/S2/S3" for lab in LABELS] +
                   ["Brevity EM / contains"], rows), "",
          "Two S3 entries in the paper are not the signal the loop used. Qwen's 0.429 is 1 − 0.571, i.e. variance "
          "scored as a penalty while the other rows score it as a reward. TriviaQA's 0.594 is the AUROC of raw "
          "rerank-score variance from `07_signal_diagnostics.py`; the min-max-normalized variance the loop "
          "actually used scores 0.456, below chance. Answer brevity alone beats every signal for Claude under EM.", "",
          f"![auroc]({figs['auroc']})", ""]

    L += ["## 5. Open-retrieval recall (paper Table `tab:recall`)", ""]
    rows = []
    for ds, d in ret.items():
        for m in ("bm25", "dense", "hybrid", "reranked"):
            row = [ds, m, " / ".join(str(v) for v in d["paper"][m])]
            for tag in ("rerun_paper_setting", "pool_800"):
                row.append(" / ".join(str(v) for v in d[tag][m]) if tag in d and m in d[tag] else "pending")
            rows.append(row)
    corp = {ds: (d.get("rerun_paper_setting", {}).get("corpus"), d.get("pool_800", {}).get("corpus")) for ds, d in ret.items()}
    L += [md_table(["Dataset", "Method", "Paper @5/10/20/50",
                    "Our rerun, paper setting", "800-question pool"], rows), "",
          "Corpus sizes (paper / rerun / 800 pool): " +
          "; ".join(f"{ds} {PAPER_RECALL[ds]['corpus']:,} / {f'{c[0]:,}' if c[0] else '–'} / {f'{c[1]:,}' if c[1] else '–'}" for ds, c in corp.items()) + ".", ""]
    if figs.get("recall"):
        L += [f"![recall]({figs['recall']})", ""]

    L += ["## 6. Worked examples (paper Table `tab:adaptive`)", ""]
    rows = []
    for e in R["triviaqa_claude"]["adaptive_examples"]:
        ex = "; ".join(f"{x['qid']} {x['confs']} EM+alias={int(x['EM'])} contains={int(x['contains'])}" for x in e["examples"])
        rows.append([e["claim"], e["first_conf"], e["second_conf"], e["matches"], ex or "none"])
    recs_t = load_run("triviaqa", "claude")[0]
    first = [r["trace"][0]["conf"] for r in recs_t]
    outside = [f"{c1:.2f}" for c1, _, _ in PAPER_ADAPTIVE if not round(min(first), 2) <= c1 <= round(max(first), 2)]
    L += [md_table(["Paper claim", "Round-1 conf", "Round-2 conf", "Matching traces", "Examples (first 3)"], rows), "",
          f"Round-1 confidence in the committed TriviaQA run ranges from {min(first):.2f} to {max(first):.2f}" +
          (f", so the {' and '.join(outside)} example(s) cannot come from this run." if outside else "."), ""]

    L += ["## 7. N2: do S2 and S3 influence decisions?", ""]
    rows = [[n, " / ".join(str(v) for v in R[f"{ds}_{be}"]["n2"]["weights_paper"]),
             " / ".join(str(v) for v in R[f"{ds}_{be}"]["n2"]["weights_used"]),
             f"{R[f'{ds}_{be}']['n2']['decision_changes']}/{R[f'{ds}_{be}']['n2']['rounds']}"] for ds, be, n in RUNS]
    L += [md_table(["Run", "Weights in paper", "Weights the run used (fit)",
                    "Decisions changed when S2, S3 are held at their mean (real weights)",
                    "Same, paper weights vs S1-only rule (as 08)", "SD of conf from S2+S3 / from S1"],
                   [r[:4] + [f"{R[f'{ds}_{be}']['n2']['decision_changes_paper_weights_vs_s1_only']}/{R[f'{ds}_{be}']['n2']['rounds']}",
                             f"{R[f'{ds}_{be}']['n2']['sd_s2s3_contribution']:.3f} / {R[f'{ds}_{be}']['n2']['sd_s1_contribution']:.3f}"]
                    for r, (ds, be, _) in zip(rows, RUNS)]), "",
          "Two of the four runs (Qwen and Claude on HotpotQA) were produced with weights 0.5 / 0.4 / −0.1, "
          "not the 0.7 / 0.05 / 0.25 the paper reports; the fit reproduces every stored confidence to 1e-4.", "",
          f"![n2]({figs['n2']})", ""]

    L += ["## 8. N3: format and verbosity confound", ""]
    rows = []
    for ds, be, n in RUNS:
        v = R[f"{ds}_{be}"]["n3"]
        rows.append([n, f"{v['em_wrong_but_contains_gold']}/{v['n']}", f"{v['mean_words_em_correct']} / {v['mean_words_em_wrong']}"] +
                    [f"{v['auroc_conf_final'][lab]:.3f} / {v['auroc_brevity_final'][lab]:.3f}" for lab in LABELS])
    L += [md_table(["Run", "EM-wrong answers that contain the gold", "Words: EM-right / EM-wrong"] +
                   [f"AUROC conf / brevity, {lab}" for lab in LABELS], rows), ""]
    ex = R["hotpotqa_claude"]["n3"]["paper_examples"] + R["triviaqa_claude"]["n3"]["paper_examples"]
    found = {(e["gold"], e["model"]) for e in ex if e["found_in_logs"]}
    L += ["Paper Table `tab:verbosity` examples found verbatim in the committed traces: " +
          ", ".join(f"“{m}” for “{g}”: {'yes' if (g, m) in found else 'no'}" for g, m in PAPER_VERBOSITY) + ".", ""]

    L += ["## 9. N4: gain decomposition and the random budget-matched control", ""]
    rows = []
    for ds, be, n in RUNS:
        for lab in LABELS:
            d = R[f"{ds}_{be}"]["n4"][lab]
            rows.append([n, lab, d["static_K10"], d["first_round_K5"], d["abrag"], d["start_K5_vs_K10"], d["iteration"]])
    L += [md_table(["Run", "Label", "Static K=10", "Round 1 (K=5)", "AB-RAG", "Start at K=5 (pts)",
                    "Iterating (pts)"], rows), "", f"![n4]({figs['n4']})", ""]
    if FT:
        L += ["### Full-trace replays (budget-matched random stopping)", ""]
        rows = []
        for k, d in FT.items():
            for lab in LABELS:
                pol = d[lab]["policies"]
                r = pol["random (budget-matched)"]
                rows.append([k, d["n"], lab, pol["static K=10"]["acc"], pol["AB-RAG"]["acc"],
                             f"{r['acc']} [{r['acc_95'][0]}, {r['acc_95'][1]}]", r["p_random_ge_abrag"],
                             pol["oracle (any round)"]["acc"]])
        L += [md_table(["Traces", "n", "Label", "Static K=10", "AB-RAG", "Random, same budget [95%]",
                        "P(random ≥ AB-RAG)", "Oracle"], rows), ""]
        if figs.get("random"):
            L += [f"![random]({figs['random']})", ""]
    else:
        L += [f"**Pending.** No full traces with at least {MIN_TRACES} questions in `results/` yet. The Colab "
              "(or Kaggle) notebook generates them, runs `10_policies.py` and this script; copy its `results/` here "
              "to rebuild locally. The random-control table, its figure and the full tau sweep then appear here.", ""]

    L += improvements_section(IMP, FT, figs)

    L += ["## 11. Cost", "", "The paper's Claude Haiku runs cost about $4.10 in API credit (its Table `tab:cost`). "
          "Everything in this report is re-scoring of saved outputs or local models; no API credit was spent. "
          "The full traces (three 500-question runs plus I1) took about 20 minutes on three H100 GPUs of a university "
          "cluster; the notebook in `notebooks/` reproduces them on a free Colab T4 in several hours.", ""]

    L += ["## 12. Figures", "", "**In the paper** (`paper/figures.zip`, extracted to `paper_figures/`):", ""]
    L += [md_table(["Figure", "Shows", "Status here"],
                   [[f"![{k}](paper_figures/{k}.png)" if os.path.exists(os.path.join(OUT, "paper_figures", f"{k}.png")) else k,
                     v, s] for (k, v), s in zip(PAPER_FIGS.items(),
                       ["unchanged", "updated as fig_h (what changed)", "unchanged; N2 adds fig_e",
                        "checked against logs in section 6", "re-measured in fig_g", "redone under 3 labels in fig_a",
                        "redone under 2 labels in fig_b", "checked in fig_c; full sweep with random band in fig_m",
                        "redone with brevity in fig_d"])]), ""]
    L += ["**New** (`figures/`): " + ", ".join(f"`{os.path.basename(p)}`" for p in figs.values() if p) + ".", ""]
    open(os.path.join(OUT, "REPORT.md"), "w", encoding="utf-8").write("\n".join(L))


def main():
    os.makedirs(FIG, exist_ok=True)
    R = {}
    for ds, be, name in RUNS:
        recs, static, exp = load_run(ds, be)
        srows = static_rows(recs, static, exp)
        R[f"{ds}_{be}"] = {
            "name": name, "n": len(recs), "avg_iters": round(float(np.mean([r["iterations_used"] for r in recs])), 2),
            "static": srows, "confsep": confsep(recs),
            "sweep_replay": {lab: replay_sweep(recs, fn) for lab, fn in LABELS.items()},
            "signal_auroc": signal_auroc(recs), "n2": n2_influence(recs), "n3": n3_verbosity(recs),
            "n4": n4_decomposition(srows)}
    R["triviaqa_claude"]["adaptive_examples"] = adaptive_examples(load_run("triviaqa", "claude")[0])
    ret, FT, IMP = retrieval(), full_trace_replays(), improvements()

    if os.path.exists("paper/figures.zip"):
        with zipfile.ZipFile("paper/figures.zip") as z:
            for m in z.namelist():
                if m.endswith(".png"):
                    os.makedirs(os.path.join(OUT, "paper_figures"), exist_ok=True)
                    open(os.path.join(OUT, "paper_figures", os.path.basename(m)), "wb").write(z.read(m))
    figs = {"labels": fig_labels(R), "confsep": fig_confsep(R), "sweep": fig_sweep(R), "auroc": fig_auroc(R),
            "n2": fig_n2(R), "n4": fig_n4(R), "recall": fig_recall(ret), "pipeline": fig_pipeline(),
            "protocol": fig_protocol(), "random": fig_random_control(FT),
            "answers": fig_answer_improvements(IMP), "retrieval_upgrade": fig_retrieval_upgrade(IMP), "ft_sweep": fig_fulltrace_sweep(FT)}
    json.dump({"runs": R, "retrieval": ret, "full_trace_replays": list(FT), "improvements": IMP},
              open(os.path.join(OUT, "report.json"), "w"), indent=1, default=str)
    write_report(R, ret, FT, IMP, figs)
    print(f"report -> {OUT}/REPORT.md  ({sum(1 for f in figs.values() if f)} figures in {FIG})")
    if not FT:
        print(f"[pending] no full traces with >= {MIN_TRACES} questions yet: random control not included")


if __name__ == "__main__":
    main()
