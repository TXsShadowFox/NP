"""
00b_prepare_triviaqa.py
-----------------------
Standalone TriviaQA prep (streaming, no 17GB download) so we don't re-fetch
HotpotQA. Produces data/triviaqa.jsonl in the uniform schema.

Run:
    python experiments/00b_prepare_triviaqa.py --n 800
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "prep", os.path.join(os.path.dirname(__file__), "00_prepare_data.py"))
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800,
                    help="questions to keep (the paper used 600; dev+test splits need 800)")
    prep.prepare_triviaqa(n=ap.parse_args().n)
    print("done. data/triviaqa.jsonl ready.")
