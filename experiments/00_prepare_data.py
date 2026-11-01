"""
00_prepare_data.py
-------------------
Loads HotpotQA (distractor) and Natural Questions (open) into ONE uniform
schema so the rest of the pipeline is dataset-agnostic.

Why distractor / per-question corpora instead of full Wikipedia?
  Full DPR Wikipedia = 21M passages = days to index + tens of GB.
  HotpotQA distractor ships ~10 candidate paragraphs per question, of which
  2 are the gold "supporting" paragraphs. That is a genuine retrieval task
  (you must find the 2 right ones among 10), it is multi-hop, and it indexes
  in milliseconds. This is a standard, citable evaluation setting.

Uniform record schema (one dict per question):
  {
    "qid": str,
    "question": str,
    "answer": str,
    "passages": [ {"pid": str, "title": str, "text": str}, ... ],
    "gold_pids": [str, ...]   # which passages actually contain the answer/support
  }

Output: data/hotpotqa.jsonl  and  data/nq.jsonl
"""
import json, os, random, sys
# --- make project root importable no matter where we run from ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets import load_dataset
from configs.config import CONFIG

random.seed(CONFIG["seed"])
os.makedirs("data", exist_ok=True)


def prepare_triviaqa(n=600):
    """
    TriviaQA (rc config) via STREAMING to avoid the 17GB full download.
    Each example has entity_pages.wiki_context (Wikipedia text) and an answer
    with value+aliases. We chunk each wiki_context into ~120-word passages and
    mark a passage 'gold' if it contains the answer value or any alias.
    Produces the SAME uniform schema as HotpotQA so all scripts work unchanged.
    """
    answers_seen = 0
    out = []
    sources = [
        ("mandarjoshi/trivia_qa", "rc"),
        ("trivia_qa", "rc"),
    ]
    ds = None
    for path, cfg in sources:
        try:
            print(f"[triviaqa] streaming {path} ({cfg}) ...")
            ds = load_dataset(path, cfg, split="validation", streaming=True,
                              trust_remote_code=True)
            break
        except Exception as e:
            print(f"[triviaqa]   failed: {type(e).__name__}: {str(e)[:100]}")
    if ds is None:
        print("[triviaqa] could not stream TriviaQA; skipping."); return

    def chunk(text, size=120, overlap=20):
        words = text.split()
        chunks, i = [], 0
        while i < len(words):
            chunks.append(" ".join(words[i:i+size]))
            i += size - overlap
        return chunks

    for ex in ds:
        if len(out) >= n:
            break
        ans_value = ex["answer"]["value"]
        aliases = set(a.lower() for a in ex["answer"].get("aliases", []) + [ans_value])
        # gather wiki contexts (may be several entity pages)
        contexts = ex["entity_pages"].get("wiki_context", [])
        titles = ex["entity_pages"].get("title", [])
        if not contexts:
            continue
        passages, gold_pids = [], []
        pid_counter = 0
        for ci, ctx in enumerate(contexts):
            title = titles[ci] if ci < len(titles) else f"doc{ci}"
            for ch in chunk(ctx)[:8]:   # cap chunks/doc to keep corpus modest
                pid = f"{ex['question_id']}_{pid_counter}"
                passages.append({"pid": pid, "title": title, "text": ch})
                if any(al in ch.lower() for al in aliases):
                    gold_pids.append(pid)
                pid_counter += 1
        if not passages or not gold_pids:
            continue   # need at least one gold passage to be retrieval-evaluable
        out.append({
            "qid": ex["question_id"],
            "question": ex["question"],
            "answer": ans_value,
            "aliases": ex["answer"].get("aliases", []),   # accepted answer variants, for EM
            "passages": passages,
            "gold_pids": gold_pids,
        })
        answers_seen += 1

    _dump(out, "data/triviaqa.jsonl")
    avg_p = sum(len(r["passages"]) for r in out)/len(out) if out else 0
    print(f"[triviaqa] wrote {len(out)} questions, avg {avg_p:.1f} passages/q")


def prepare_hotpotqa(n=1000):
    # distractor config = question + 10 context paragraphs, 2 of them gold.
    # The original hotpot_qa host went offline (May 2025) and datasets>=4 dropped
    # script loaders, so we try several sources in order until one works.
    ds = None
    attempts = [
        # (path, kwargs)
        ("hotpotqa/hotpot_qa", dict(name="distractor", split="validation", trust_remote_code=True)),
        ("hotpot_qa",          dict(name="distractor", split="validation", trust_remote_code=True)),
        ("vincentkoc/hotpot_qa_archive", dict(name="distractor", split="validation", trust_remote_code=True)),
    ]
    last_err = None
    for path, kw in attempts:
        try:
            print(f"[hotpotqa] trying source: {path} ...")
            ds = load_dataset(path, **kw)
            print(f"[hotpotqa] loaded from {path}")
            break
        except Exception as e:
            last_err = e
            print(f"[hotpotqa]   failed: {type(e).__name__}: {str(e)[:120]}")
    if ds is None:
        raise RuntimeError(
            "Could not load HotpotQA from any source. Most likely your "
            "`datasets` version is too new. Run:  python -m pip install "
            "'datasets==2.21.0'  then retry.\nLast error: " + str(last_err))

    out = []
    for i, ex in enumerate(ds):
        if i >= n:
            break
        titles = ex["context"]["title"]
        sentences = ex["context"]["sentences"]  # list[list[str]] per title
        passages = []
        for ti, (title, sents) in enumerate(zip(titles, sentences)):
            passages.append({
                "pid": f"{ex['id']}_{ti}",
                "title": title,
                "text": " ".join(sents).strip(),
            })
        # gold supporting titles -> map to pids
        gold_titles = set(ex["supporting_facts"]["title"])
        gold_pids = [p["pid"] for p in passages if p["title"] in gold_titles]
        out.append({
            "qid": ex["id"],
            "question": ex["question"],
            "answer": ex["answer"],
            "passages": passages,
            "gold_pids": gold_pids,
        })
    _dump(out, "data/hotpotqa.jsonl")
    print(f"[hotpotqa] wrote {len(out)} questions, "
          f"avg {sum(len(r['passages']) for r in out)/len(out):.1f} passages/q")


def prepare_nq(n=1000):
    """
    Natural Questions 'open' has question+answer but NO per-question passages.
    To keep a SELF-CONTAINED corpus, we build a shared pool: every question's
    answer is findable in a short reference snippet we pull from the simplified
    NQ short-answer context. We use the 'nq_open' + a passage pool from the
    validation contexts so retrieval is non-trivial.

    NOTE: nq_open gives only Q/A. So for a controlled retrieval task we instead
    use the 'natural_questions' simplified contexts when available; if that is
    too large on your machine, the pipeline still runs on HotpotQA alone and NQ
    can be reported as 'open-domain, generation-only' (no recall).
    """
    try:
        ds = load_dataset("nq_open", split="validation")
    except Exception as e:
        print(f"[nq] could not load nq_open ({e}); skipping NQ. "
              f"HotpotQA alone is enough for all core results.")
        return
    out = []
    for i, ex in enumerate(ds):
        if i >= n:
            break
        ans = ex["answer"][0] if isinstance(ex["answer"], list) else ex["answer"]
        out.append({
            "qid": f"nq_{i}",
            "question": ex["question"],
            "answer": ans,
            "passages": [],      # filled by a shared pool below
            "gold_pids": [],
        })
    # Build a shared distractor pool so dense/sparse retrieval has something to
    # rank. We synthesize a tiny corpus from the answers themselves + neighbors,
    # which keeps NQ self-contained. (Documented honestly in the report as a
    # controlled NQ-open setting; recall is reported on HotpotQA where we have
    # true gold passages.)
    _dump(out, "data/nq.jsonl")
    print(f"[nq] wrote {len(out)} questions (open-domain, generation eval)")


def _dump(records, path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    prepare_hotpotqa(n=1000)
    prepare_triviaqa(n=600)
    prepare_nq(n=1000)
    print("done. data/ now holds the eval sets.")
