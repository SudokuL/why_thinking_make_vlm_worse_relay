#!/bin/bash
# Block 0 master driver: E-01 (VisualSwap, MathVista_MINI, 200 pairs) + E-02
# (specificity controls: mmvp, hallu_vis, hallu_text) for all currently
# downloaded models. Resumable -- run scripts skip pids already in the
# output jsonl, so re-running this script after an interruption is safe.
#
# hallu_vis is capped at --limit 300 (951 available) to bound wall-clock
# time across 4 models; documented as a scope decision in WIKI.md (can be
# extended to the full set later if time allows -- result is resumable).
set -e
cd /home/liuruida/from84_relay/why_thinking_make_vlm_worse_relay/code
source ~/miniconda/etc/profile.d/conda.sh
conda activate wtmw

MODELS="2B-Instruct 2B-Thinking 8B-Instruct 8B-Thinking"

for M in $MODELS; do
  echo "=== [$(date)] E-01 $M ==="
  python run_E01_visualswap.py --model "$M" --subset MathVista_MINI --max_new_tokens 2048
done

for M in $MODELS; do
  echo "=== [$(date)] E-02 $M mmvp ==="
  python run_E02_specificity.py --model "$M" --task mmvp --max_new_tokens 1024
  echo "=== [$(date)] E-02 $M hallu_vis ==="
  python run_E02_specificity.py --model "$M" --task hallu_vis --limit 300 --max_new_tokens 1024
  echo "=== [$(date)] E-02 $M hallu_text ==="
  python run_E02_specificity.py --model "$M" --task hallu_text --max_new_tokens 1024
done

echo "=== [$(date)] Block 0 ALL DONE ==="
