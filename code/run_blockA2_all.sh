#!/bin/bash
# Queued after run_block0_all.sh AND run_blockA1_all.sh (single-GPU
# exclusivity). Runs E-A2 (pathway separation: image-token vs reasoning-token
# K/V patch) for all 4 currently-downloaded models.
set -e
cd /home/liuruida/from84_relay/why_thinking_make_vlm_worse_relay/code
source ~/miniconda/etc/profile.d/conda.sh
conda activate wtmw

while pgrep -f run_block0_all.sh > /dev/null || pgrep -f run_blockA1_all.sh > /dev/null; do
  sleep 60
done

echo "=== [$(date)] Block 0 + E-A1 done -- starting E-A2 ==="

for M in 8B-Instruct 8B-Thinking; do
  echo "=== [$(date)] E-A2 $M (full flippable) ==="
  python run_EA2_pathway.py --model "$M" --subset MathVista_MINI
done

for M in 2B-Instruct 2B-Thinking; do
  echo "=== [$(date)] E-A2 $M (limit 40) ==="
  python run_EA2_pathway.py --model "$M" --subset MathVista_MINI --limit 40
done

echo "=== [$(date)] E-A2 ALL DONE ==="
