"""
00d_make_splits.py
------------------
Freezes the dev/test question split for a dataset, so weights, thresholds and
new policies are tuned on dev only and reported on test.

  test = first n_test retrieval-evaluable questions (in file order; this keeps
         the paper's 200 questions inside test, so numbers stay comparable)
  dev  = the next n_dev questions

Run:
    python experiments/00d_make_splits.py --dataset hotpotqa
Output: data/splits_{dataset}.json  {"test": [qid, ...], "dev": [qid, ...]}
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config import CONFIG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hotpotqa")
    args = ap.parse_args()

    qids = []
    for line in open(f"data/{args.dataset}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["passages"] and r["gold_pids"]:
            qids.append(r["qid"])
    n_test, n_dev = CONFIG["n_test"], CONFIG["n_dev"]
    if len(qids) < n_test + n_dev:
        print(f"[!] only {len(qids)} evaluable questions; need {n_test + n_dev}. "
              f"Re-run the prep with a larger n.")
        return
    splits = {"test": qids[:n_test], "dev": qids[n_test:n_test + n_dev]}
    out = f"data/splits_{args.dataset}.json"
    json.dump(splits, open(out, "w"), indent=1)
    print(f"test={len(splits['test'])}  dev={len(splits['dev'])}  -> {out}")


if __name__ == "__main__":
    main()
