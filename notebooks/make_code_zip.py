"""
make_code_zip.py
----------------
Packs what a Kaggle or Colab GPU session needs into one zip: code, configs,
prepared data and splits, the notebooks, the cached retrieval rankings
(results/rankings_*.json + corpus_emb_*.npy) so the session skips re-ranking,
the results computed on this machine that 12_report.py reads (R1 retrieval,
open-retrieval recall, I1/I2 on the paper's answers and the I1 cache), and the
paper's committed runs (logs/) and figures that 12_report.py compares against. Leaves out .env, virtualenvs and every new trace or result file, so no
key and no half-finished run travels with it.

Run from the repo root:
    python notebooks/make_code_zip.py            # -> abrag_code.zip next to the repo folder
Then upload it as a PRIVATE Kaggle dataset (Datasets -> New Dataset) for
kaggle_phase1.ipynb, or to MyDrive/abrag/ for colab_phase1.ipynb.
"""
import argparse, glob, os, zipfile

INCLUDE = ["configs/*.py", "experiments/*.py", "notebooks/*.ipynb", "notebooks/*.py",
           "data/hotpotqa.jsonl", "data/triviaqa.jsonl", "data/splits_*.json",
           "results/rankings_*.json", "results/corpus_emb_*.npy",
           "results/retrieval_upgrade_*.json", "results/open_retrieval_*_n*.json",
           "results/improved_answers.json", "results/extract_cache.jsonl",
           "logs/*.json", "logs/*.jsonl", "paper/figures.zip",
           "requirements.txt", "README.md", "LICENSE", "CITATION.cff"]


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(root), "abrag_code.zip"))
    args = ap.parse_args()

    files = sorted({f for pat in INCLUDE for f in glob.glob(os.path.join(root, pat))})
    rel = [os.path.relpath(f, root).replace(os.sep, "/") for f in files]
    assert not any(os.path.basename(r).startswith(".env") for r in rel), "refusing to pack a .env file"
    for ds in ("hotpotqa", "triviaqa"):
        if f"results/rankings_{ds}.json" not in rel or f"results/corpus_emb_{ds}.npy" not in rel:
            print(f"[!] no cached rankings for {ds}: the GPU session will rebuild them (a few minutes)")
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for f, r in zip(files, rel):
            z.write(f, r)
    print(f"{len(rel)} files, {os.path.getsize(args.out) / 1e6:.1f} MB -> {args.out}")


if __name__ == "__main__":
    main()
