#!/bin/bash
# 4B-Instruct/4B-Thinking finished downloading after run_block0_all.sh (2B/8B)
# was already launched, so they were left out of that script's MODELS list.
# This script waits for run_block0_all.sh to finish (single-GPU exclusivity)
# then runs E-01 + E-02 for the 4B pair, completing the "6 models" (2B/4B/8B
# x Instruct/Thinking) Block 0 matrix per EXPERIMENT_GUIDE.md.
set -e
cd /home/liuruida/from84_relay/why_thinking_make_vlm_worse_relay/code
source ~/miniconda/etc/profile.d/conda.sh
conda activate wtmw

while pgrep -f run_block0_all.sh > /dev/null; do
  sleep 60
done

echo "=== [$(date)] Block 0 (2B/8B) done -- starting Block 0 (4B) ==="

for M in 4B-Instruct 4B-Thinking; do
  echo "=== [$(date)] E-01 $M ==="
  python run_E01_visualswap.py --model "$M" --subset MathVista_MINI --max_new_tokens 2048
done

for M in 4B-Instruct 4B-Thinking; do
  echo "=== [$(date)] E-02 $M mmvp ==="
  python run_E02_specificity.py --model "$M" --task mmvp --max_new_tokens 1024
  echo "=== [$(date)] E-02 $M hallu_vis ==="
  python run_E02_specificity.py --model "$M" --task hallu_vis --limit 300 --max_new_tokens 1024
  echo "=== [$(date)] E-02 $M hallu_text ==="
  python run_E02_specificity.py --model "$M" --task hallu_text --max_new_tokens 1024
done

echo "=== [$(date)] Block 0 (4B) ALL DONE ==="
