"""
00c_add_triviaqa_aliases.py
---------------------------
Backfills the accepted answer aliases into an existing data/triviaqa.jsonl
(older prep runs stored only the single answer value, so EM marked correct
alias answers wrong). Streams the small rc.nocontext config: no 17GB download.

Run:
    python experiments/00c_add_triviaqa_aliases.py
"""
import json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets import load_dataset

PATH = "data/triviaqa.jsonl"

if __name__ == "__main__":
    rows = [json.loads(l) for l in open(PATH, encoding="utf-8")]
    need = {r["qid"] for r in rows}
    found = {}
    for path in ["mandarjoshi/trivia_qa", "trivia_qa"]:
        try:
            ds = load_dataset(path, "rc.nocontext", split="validation", streaming=True,
                              trust_remote_code=True)
            for ex in ds:
                if ex["question_id"] in need:
                    found[ex["question_id"]] = ex["answer"].get("aliases", [])
                    if len(found) == len(need):
                        break
            break
        except Exception as e:
            print(f"[aliases] {path} failed: {type(e).__name__}: {str(e)[:100]}")
    for r in rows:
        r["aliases"] = found.get(r["qid"], r.get("aliases", []))
    with open(PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"aliases added for {len(found)}/{len(rows)} questions -> {PATH}")
