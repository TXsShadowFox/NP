"""
11_comparison_figures.py
------------------------
Figures comparing the paper's reported numbers with the same committed runs
re-scored by 08_rescore.py (run that first). No model calls.

Run:
    python experiments/08_rescore.py --runs_dir logs
    python experiments/11_comparison_figures.py
Output: results/figures/cmp*.png
"""
import json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from configs.config import CONFIG

RES = CONFIG["results_dir"]
OUT = f"{RES}/figures"
os.makedirs(OUT, exist_ok=True)
RUNS = [("hotpotqa_local", "Qwen-1.5B\nHotpotQA"), ("hotpotqa_ollama", "Llama-3.2-3B\nHotpotQA"),
        ("hotpotqa_claude", "Claude Haiku\nHotpotQA"), ("triviaqa_claude", "Claude Haiku\nTriviaQA")]
LABELS = ["EM", "EM+alias", "contains"]
COL = {"EM": "#9aa5b1", "EM+alias": "#4c78a8", "contains": "#f58518"}
R = {k: json.load(open(f"{RES}/rescore_{k}.json")) for k, _ in RUNS}
num = lambda s: float(str(s).split()[0])
names = [n for _, n in RUNS]
x = np.arange(len(RUNS))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}", dpi=200)
    plt.close(fig)


# 1. AB-RAG accuracy under the paper's EM vs fair scoring
fig, ax = plt.subplots(figsize=(8, 4))
for i, l in enumerate(LABELS):
    v = [R[k][l]["abrag"] for k, _ in RUNS]
    b = ax.bar(x + (i - 1) * 0.27, v, 0.27, label=l, color=COL[l])
    ax.bar_label(b, fmt="%.1f", fontsize=7)
ax.set_xticks(x, names); ax.set_ylabel("AB-RAG accuracy (%)")
ax.set_title("Same answers, three scoring rules: EM hides most correct answers")
ax.legend(); save(fig, "cmp1_accuracy_by_label.png")

# 2. high- vs low-confidence accuracy (paper's selective-prediction table)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, l in zip(axes, ["EM", "contains"]):
    hi = [num(R[k][l]["high_conf"]) for k, _ in RUNS]
    lo = [num(R[k][l]["low_conf"]) for k, _ in RUNS]
    ax.bar_label(ax.bar(x - 0.2, hi, 0.4, label="high conf (>= 0.6)", color="#54a24b"), fmt="%.1f", fontsize=7)
    ax.bar_label(ax.bar(x + 0.2, lo, 0.4, label="low conf (< 0.6)", color="#e45756"), fmt="%.1f", fontsize=7)
    ax.set_xticks(x, names, fontsize=7); ax.set_title(f"scored by {l}" + (" (paper)" if l == "EM" else " (fair)"))
axes[0].set_ylabel("accuracy (%)"); axes[0].legend(fontsize=8)
fig.suptitle("Confidence separation: '57.6% vs 0%' shrinks to '89% vs 61%' under fair scoring")
save(fig, "cmp2_confidence_separation.png")

# 3. static K=10 vs AB-RAG on Claude/TriviaQA (the only run with a static cache)
t = R["triviaqa_claude"]
fig, ax = plt.subplots(figsize=(6, 4))
st = [t[l]["static_K10"] for l in LABELS]; ab = [t[l]["abrag"] for l in LABELS]
xx = np.arange(3)
ax.bar_label(ax.bar(xx - 0.2, st, 0.4, label="static RAG (K=10)", color="#9aa5b1"), fmt="%.1f")
ax.bar_label(ax.bar(xx + 0.2, ab, 0.4, label="AB-RAG", color="#4c78a8"), fmt="%.1f")
for i, l in enumerate(LABELS):
    p = t[l]["vs_static"].split("McNemar ")[1].split(",")[0]
    ax.text(i, max(st[i], ab[i]) + 5, p, ha="center", fontsize=8)
ax.set_xticks(xx, LABELS); ax.set_ylim(0, 100); ax.set_ylabel("accuracy (%)")
ax.set_title("Claude/TriviaQA: the +4.5 EM gain vanishes under containment")
ax.legend(loc="upper left"); save(fig, "cmp3_static_vs_abrag_triviaqa.png")

# 4. confidence AUROC vs answer-brevity AUROC (N3)
fig, ax = plt.subplots(figsize=(8, 4))
c = [R[k]["EM"]["auroc_conf"] for k, _ in RUNS]; bv = [R[k]["EM"]["auroc_brevity"] for k, _ in RUNS]
ax.bar_label(ax.bar(x - 0.2, c, 0.4, label="AB-RAG confidence", color="#4c78a8"), fmt="%.3f", fontsize=8)
ax.bar_label(ax.bar(x + 0.2, bv, 0.4, label="answer brevity alone", color="#f58518"), fmt="%.3f", fontsize=8)
ax.axhline(0.5, ls="--", c="k", lw=0.8); ax.set_ylim(0.3, 1.0)
ax.set_xticks(x, names); ax.set_ylabel("AUROC vs EM correctness")
ax.set_title("N3: on Claude, answer length predicts EM better than the confidence score")
ax.legend(); save(fig, "cmp4_auroc_conf_vs_brevity.png")

# 5. gain decomposition (N4): first round (K=5) vs AB-RAG final, under fair scoring
fig, ax = plt.subplots(figsize=(8, 4))
f1 = [R[k]["contains"]["first_round_K5"] for k, _ in RUNS]; fa = [R[k]["contains"]["abrag"] for k, _ in RUNS]
ax.bar_label(ax.bar(x - 0.2, f1, 0.4, label="first round only (K=5, one pass)", color="#9aa5b1"), fmt="%.1f", fontsize=8)
ax.bar_label(ax.bar(x + 0.2, fa, 0.4, label="AB-RAG after extra rounds", color="#4c78a8"), fmt="%.1f", fontsize=8)
ax.set_xticks(x, names); ax.set_ylabel("accuracy, contains (%)")
ax.set_title("N4: what the extra retrieval rounds add on top of a single K=5 pass")
ax.legend(); save(fig, "cmp5_gain_decomposition.png")

# 6. reported vs actually-used confidence weights (reproducibility)
fig, ax = plt.subplots(figsize=(8, 3.6))
w = np.array([R[k]["weights_used"]["alpha_beta_gamma"] for k, _ in RUNS])
for i, (nm, col) in enumerate(zip(["alpha (S1)", "beta (S2)", "gamma (S3)"], ["#4c78a8", "#f58518", "#54a24b"])):
    ax.bar(x + (i - 1) * 0.27, w[:, i], 0.27, label=nm, color=col)
for i, v in enumerate([0.7, 0.05, 0.25]):
    ax.axhline(v, ls=":", c=["#4c78a8", "#f58518", "#54a24b"][i], lw=1)
ax.axhline(0, c="k", lw=0.6); ax.set_xticks(x, names)
ax.set_title("Weights that produced the stored confidences (dotted = paper's 0.7 / 0.05 / 0.25)")
ax.legend(fontsize=8); save(fig, "cmp6_weights_used.png")

# 7. paper's threshold sweep vs what the committed traces can produce
paper = {"hotpotqa_claude": [1.00, 1.14, 1.36, 1.98, 2.61], "triviaqa_claude": [1.00, 1.30, 1.81, 2.42, 2.88],
         "hotpotqa_local": [1.00, 1.02, 1.09, 1.18, 1.27]}
taus = [0.30, 0.45, 0.60, 0.75, 0.90]
fig, ax = plt.subplots(figsize=(7, 4))
for (k, col) in zip(paper, ["#4c78a8", "#f58518", "#54a24b"]):
    ax.plot(taus, paper[k], "o-", c=col, label=f"{k} (paper)")
    ax.axhline(R[k]["avg_iterations"], ls="--", c=col, lw=1)
ax.set_xlabel("tau"); ax.set_ylabel("avg. retrieval rounds")
ax.set_title("Threshold sweep: dashed = max rounds the committed traces contain")
ax.legend(fontsize=8); save(fig, "cmp7_sweep_reproducibility.png")

print(f"saved 7 figures -> {OUT}")
