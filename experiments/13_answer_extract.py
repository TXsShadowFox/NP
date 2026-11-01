"""
13_answer_extract.py
--------------------
Improvements I1 and I2, applied to the paper's committed runs in logs/ (the
answers are re-used, not re-generated) and scored next to the paper's numbers.

  I1  short-answer stage. The generation prompt asks for "a few words", but the
      models often answer in a sentence ("Yes, both American.", "Wichita,
      Kansas"), which EM and F1 score as wrong. A small local model (the config's
      Qwen2.5-1.5B-Instruct, greedy) cuts each answer down to the shortest phrase
      that answers the question. Three fixed rules keep it a format fix:
        - answers of one or two words pass through unchanged;
        - the short answer may only use words of the original answer, else the
          original is kept (no new content can enter);
        - it sees the question and the answer, never the gold.
  I2  best-round answer. Answer with the round of highest confidence among the
      rounds AB-RAG took, instead of the last round.

Static K=10 answers (Claude / TriviaQA, the one run that stores them) go through
the same stage, so the paired test against static RAG stays like for like.

Run (GPU optional, a few minutes):
    python experiments/13_answer_extract.py
    python experiments/13_answer_extract.py --traces results/traces_hotpotqa_local.jsonl
        (full traces from 09_full_traces.py: writes a -i1 copy for 10_policies.py)
Output: results/extract_cache.jsonl (resumable), results/improved_answers.json,
        results/traces_*-i1.jsonl in --traces mode
"""
import argparse, hashlib, json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from configs.config import CONFIG
from experiments.metrics import (normalize, golds_of, exact_match, token_f1, contains,
                                 auroc, mcnemar, paired_bootstrap)

RUNS = [("hotpotqa", "local", "Qwen-1.5B / HotpotQA"),
        ("hotpotqa", "ollama", "Llama-3.2-3B / HotpotQA"),
        ("hotpotqa", "claude", "Claude Haiku / HotpotQA"),
        ("triviaqa", "claude", "Claude Haiku / TriviaQA")]

# Invented examples (fictional names), so no test question or answer leaks in.
# Revision 2: revision 1 asked for "the shortest phrase" and cut complete names
# ("Indianapolis Motor Speedway" -> "Indianapolis"); names are now kept whole.
INSTRUCTION = (
    "Extract the answer to the question from the response. Copy the exact words of the phrase "
    "that answers the question and keep names and titles complete. Remove only explanations, "
    "extra sentences and lead-ins. For a yes/no question write yes or no. If the response is "
    "already just the answer, copy it unchanged. If the response says the information is "
    "unknown or missing, write: unknown\n\n")
EXAMPLES = [
    ("Which stadium hosts the Riverton Owls' home games?",
     "The Riverton Owls play their home games at Carrow Hill Memorial Stadium, which opened in 1931.",
     "Carrow Hill Memorial Stadium"),
    ("Were the painters Ada Verne and Tomas Kell both born in Canada?",
     "Yes, both were born in Canada.", "yes"),
    ("Who wrote the novel the film The Glass Orchard was adapted from?",
     "The film was adapted from the 1952 novel by Helen Marsh Corbett.", "Helen Marsh Corbett"),
    ("In which city is the Northwind Rowing Club based?", "Lisbon, Portugal", "Lisbon, Portugal"),
    ("How many moons does the planet Arvo have?",
     "Unknown The evidence does not mention the number of moons.", "unknown"),
]
PASS_WORDS = 2          # answers this short are already minimal
# Fingerprint of the frozen instruction + examples; recorded with every I1 output.
PROMPT_SHA1 = hashlib.sha1((INSTRUCTION + repr(EXAMPLES)).encode("utf-8")).hexdigest()[:12]

LABELS = {"EM": lambda p, r: exact_match(p, r["gold_answer"]),
          "EM+alias": lambda p, r: exact_match(p, golds_of(r)),
          "contains": lambda p, r: contains(p, golds_of(r))}


def flat(answer):
    return " ".join(answer.split())


def build(question, answer):
    shots = "".join(f"Question: {q}\nResponse: {a}\nAnswer: {s}\n\n" for q, a, s in EXAMPLES)
    return INSTRUCTION + shots + f"Question: {question}\nResponse: {flat(answer)}\nAnswer:"


def key(question, answer):
    """Hash of the full prompt, so a changed instruction never reuses old outputs."""
    return hashlib.sha1(build(question, answer).encode("utf-8")).hexdigest()


def clean(text):
    t = text.strip().split("\n")[0].strip()
    for prefix in ("short answer:", "answer:"):
        if t.lower().startswith(prefix):
            t = t[len(prefix):].strip()
    return t.strip(" .\"'")


def allowed(short, answer):
    s = normalize(short).split()
    return bool(s) and set(s) <= set(normalize(answer).split())


class Extractor:
    def __init__(self, model_name):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        cuda = torch.cuda.is_available()
        print(f"loading {model_name} on {'cuda' if cuda else 'cpu'} ...")
        self.tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16 if cuda else torch.float32,
            device_map="cuda" if cuda else "cpu").eval()

    def __call__(self, pairs):
        texts = [self.tok.apply_chat_template([{"role": "user", "content": build(q, a)}],
                                              tokenize=False, add_generation_prompt=True)
                 for q, a in pairs]
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=16, do_sample=False, temperature=None,
                                      top_p=None, top_k=None, pad_token_id=self.tok.pad_token_id)
        n = enc["input_ids"].shape[1]
        return [clean(self.tok.decode(o[n:], skip_special_tokens=True)) for o in out]


def load_run(ds, be):
    recs = [json.loads(l) for l in open(f"logs/adaptive_{ds}_{be}.jsonl", encoding="utf-8")]
    aliases = {}
    for l in open(f"data/{ds}.jsonl", encoding="utf-8"):
        r = json.loads(l)
        aliases[r["qid"]] = r.get("aliases", [])
    for r in recs:
        r["aliases"] = aliases.get(r["qid"], [])
    p = f"logs/static_cache_{ds}_{be}.jsonl"
    static = {}
    if os.path.exists(p):
        static = {json.loads(l)["qid"]: json.loads(l)["pred"] for l in open(p, encoding="utf-8")}
    return recs, static


def best_round(rec):
    """Answer of the highest-confidence round taken (ties go to the later round)."""
    return max(rec["trace"], key=lambda t: (t["conf"], t["t"]))["answer"]


def pct(xs):
    return round(100 * float(np.mean(xs)), 1)


def paired(a, b):
    plus, minus, p = mcnemar(a, b)
    lo, hi = paired_bootstrap(a, b)
    return {"plus": plus, "minus": minus, "mcnemar_p": round(p, 4), "ci95_pts": [round(lo, 1), round(hi, 1)]}


def score(answers, recs):
    out = {lab: pct([fn(answers[r["qid"]], r) for r in recs]) for lab, fn in LABELS.items()}
    out["F1"] = pct([token_f1(answers[r["qid"]], r["gold_answer"]) for r in recs])
    out["F1+alias"] = pct([token_f1(answers[r["qid"]], golds_of(r)) for r in recs])
    return out


def correct(answers, recs, lab="EM"):
    return np.array([LABELS[lab](answers[r["qid"]], r) for r in recs])


def extract_all(pairs, cache_path, model, batch_size):
    """Short answers for every (question, answer) longer than PASS_WORDS, cached by prompt hash."""
    todo = {key(q, a): (q, a) for q, a in pairs if len(normalize(a).split()) > PASS_WORDS}
    cache = {}
    if os.path.exists(cache_path):
        cache = {d["key"]: d["short"] for d in map(json.loads, open(cache_path, encoding="utf-8"))}
    missing = [k for k in todo if k not in cache]
    print(f"{len(todo)} answers longer than {PASS_WORDS} words, {len(missing)} not cached yet")
    if missing:
        ext = Extractor(model)
        with open(cache_path, "a", encoding="utf-8") as f:
            for i in range(0, len(missing), batch_size):
                keys = missing[i:i + batch_size]
                for k, s in zip(keys, ext([todo[k] for k in keys])):
                    cache[k] = s
                    f.write(json.dumps({"key": k, "short": s}, ensure_ascii=False) + "\n")
                print(f"  {min(i + batch_size, len(missing))}/{len(missing)}", end="\r", flush=True)
        print()
    return cache


def make_shorten(cache, stats):
    def shorten(q, a):
        if len(normalize(a).split()) <= PASS_WORDS:
            return a
        s = cache[key(q, a)]
        if allowed(s, a):
            stats["changed"] += 1
            return s
        stats["kept_by_rule"] += 1
        return a
    return shorten


def shorten_traces(paths, cache_path, model, batch_size):
    """I1 on full traces (09_full_traces.py): writes traces_{dataset}_{label}-i1.jsonl with every
    round's answer shortened (the original kept as answer_raw; confidence signals untouched), so
    10_policies.py scores every policy with I1, and its "AB-RAG + max-conf" policy is I1 + I2."""
    files = [p for p in paths if not p.endswith("-i1.jsonl")]
    recs = {p: [json.loads(l) for l in open(p, encoding="utf-8")] for p in files}
    cache = extract_all([(r["question"], rd["answer"]) for rs in recs.values() for r in rs for rd in r["rounds"]],
                        cache_path, model, batch_size)
    stats = {"changed": 0, "kept_by_rule": 0}
    shorten = make_shorten(cache, stats)
    for p, rs in recs.items():
        out = p[:-len(".jsonl")] + "-i1.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for r in rs:
                rounds = [{**rd, "answer": shorten(r["question"], rd["answer"]), "answer_raw": rd["answer"]}
                          for rd in r["rounds"]]
                f.write(json.dumps({**r, "rounds": rounds, "i1": {"extractor": model, "prompt_sha1": PROMPT_SHA1}},
                                   ensure_ascii=False) + "\n")
        print(f"{len(rs)} questions -> {out}")
    print(f"round answers shortened: {stats['changed']}, kept by the copy rule: {stats['kept_by_rule']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=CONFIG["local_model"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--traces", nargs="+", default=None,
                    help="apply I1 to these full-trace files instead of the paper's logs")
    args = ap.parse_args()
    out_dir = CONFIG["results_dir"]
    os.makedirs(out_dir, exist_ok=True)
    cache_path = f"{out_dir}/extract_cache.jsonl"
    print(f"I1 prompt sha1 {PROMPT_SHA1}")
    if args.traces:
        shorten_traces(args.traces, cache_path, args.model, args.batch_size)
        return

    runs = {f"{ds}_{be}": (label, *load_run(ds, be)) for ds, be, label in RUNS}

    # every answer that could be shortened: final, every round, static
    pairs = []
    for _, recs, static in runs.values():
        for r in recs:
            answers = [r["pred_answer"]] + [t["answer"] for t in r["trace"]]
            if r["qid"] in static:
                answers.append(static[r["qid"]])
            pairs += [(r["question"], a) for a in answers]
    cache = extract_all(pairs, cache_path, args.model, args.batch_size)

    stats = {"changed": 0, "kept_by_rule": 0}
    shorten = make_shorten(cache, stats)

    report = {"settings": {"extractor": args.model, "prompt_sha1": PROMPT_SHA1, "pass_words": PASS_WORDS, "decoding": "greedy",
                           "examples": f"{len(EXAMPLES)} invented few-shot examples"}, "runs": {}}
    for name, (label, recs, static) in runs.items():
        stats.update(changed=0, kept_by_rule=0)
        paper = {r["qid"]: r["pred_answer"] for r in recs}
        short = {r["qid"]: shorten(r["question"], r["pred_answer"]) for r in recs}
        changed, kept = stats["changed"], stats["kept_by_rule"]
        best = {r["qid"]: best_round(r) for r in recs}
        best_short = {r["qid"]: shorten(r["question"], best[r["qid"]]) for r in recs}
        conf = [r["final_confidence"] for r in recs]
        best_conf = [max(t["conf"] for t in r["trace"]) for r in recs]
        tau = recs[0].get("tau", 0.6)
        high = np.array(conf) >= tau

        variants = {"paper AB-RAG": paper, "AB-RAG + I1 short answer": short,
                    "AB-RAG + I2 best round": best, "AB-RAG + I1 + I2": best_short}
        res = {"label": label, "n": len(recs), "changed_final": changed, "kept_by_rule_final": kept,
               "scores": {v: score(a, recs) for v, a in variants.items()},
               "paired_vs_paper": {v: {lab: paired(correct(paper, recs, lab), correct(a, recs, lab))
                                       for lab in ("EM", "EM+alias")}
                                   for v, a in variants.items() if v != "paper AB-RAG"},
               "conf_auroc_EM": {"paper AB-RAG": round(auroc(conf, correct(paper, recs)), 3),
                                 "AB-RAG + I1 short answer": round(auroc(conf, correct(short, recs)), 3),
                                 "AB-RAG + I1 + I2": round(auroc(best_conf, correct(best_short, recs)), 3)},
               "confsep_EM": {v: {"high": pct(correct(a, recs)[high]) if high.any() else None,
                                  "high_n": int(high.sum()),
                                  "low": pct(correct(a, recs)[~high]) if (~high).any() else None,
                                  "low_n": int((~high).sum())}
                              for v, a in (("paper AB-RAG", paper), ("AB-RAG + I1 short answer", short))},
               "examples": [{"qid": r["qid"], "question": r["question"], "gold": r["gold_answer"],
                             "before": flat(paper[r["qid"]]), "after": short[r["qid"]]}
                            for r in recs if short[r["qid"]] != paper[r["qid"]]][:6]}
        if static:
            st_short = {q: shorten(next(r["question"] for r in recs if r["qid"] == q), a)
                        for q, a in static.items()}
            res["static"] = {"static K=10": score(static, recs), "static K=10 + I1": score(st_short, recs)}
            res["paired_vs_static_same_stage"] = {
                lab: paired(correct(st_short, recs, lab), correct(short, recs, lab))
                for lab in ("EM", "EM+alias", "contains")}
        report["runs"][name] = res

    path = f"{out_dir}/improved_answers.json"
    json.dump(report, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"\n{'run':26s} {'variant':28s}  EM    EM+al  F1    cont")
    for name, res in report["runs"].items():
        for v, s in res["scores"].items():
            print(f"{res['label']:26s} {v:28s} {s['EM']:5.1f}  {s['EM+alias']:5.1f}  {s['F1']:5.1f}  {s['contains']:5.1f}")
        for v, s in res.get("static", {}).items():
            print(f"{res['label']:26s} {v:28s} {s['EM']:5.1f}  {s['EM+alias']:5.1f}  {s['F1']:5.1f}  {s['contains']:5.1f}")
        print(f"{'':26s} shortened {res['changed_final']}, kept by the copy rule {res['kept_by_rule_final']}")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
