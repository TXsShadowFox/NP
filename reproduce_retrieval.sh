#!/usr/bin/env bash
# reproduce_retrieval.sh — one command from clean clone to the first real
# retrieval numbers (Recall@{5,10,20} for BM25 / dense / hybrid on HotpotQA).
set -e

echo "==> [1/3] installing dependencies"
pip install -r requirements.txt

echo "==> [2/3] downloading + preparing HotpotQA"
python experiments/00_prepare_data.py

echo "==> [3/3] hybrid retrieval + recall scoring"
python experiments/01_retrieval.py --dataset hotpotqa --n 500

echo ""
echo "Done. See results/retrieval_hotpotqa.json — compare against the"
echo "committed copy to verify reproduction. Continue with experiments/02..07 in"
echo "numeric order (see README)."
