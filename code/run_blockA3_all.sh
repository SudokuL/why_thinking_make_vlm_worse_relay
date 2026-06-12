#!/bin/bash
# Queued after run_block0_all.sh, run_blockA1_all.sh AND run_blockA2_all.sh
# (single-GPU exclusivity). Runs E-A3 (trigger-type causal comparison: T1
# self-reflection vs T2 new-turn-no-image vs T3 new-turn-with-fresh-image)
# for all 4 currently-downloaded models.
set -e
cd /home/liuruida/from84_relay/why_thinking_make_vlm_worse_relay/code
source ~/miniconda/etc/profile.d/conda.sh
conda activate wtmw

while pgrep -f run_block0_all.sh > /dev/null || pgrep -f run_blockA1_all.sh > /dev/null || pgrep -f run_blockA2_all.sh > /dev/null; do
  sleep 60
done

echo "=== [$(date)] Block 0 + E-A1 + E-A2 done -- starting E-A3 ==="

for M in 8B-Instruct 8B-Thinking; do
  echo "=== [$(date)] E-A3 $M (full flippable) ==="
  python run_EA3_trigger.py --model "$M" --subset MathVista_MINI
done

for M in 2B-Instruct 2B-Thinking; do
  echo "=== [$(date)] E-A3 $M (limit 40) ==="
  python run_EA3_trigger.py --model "$M" --subset MathVista_MINI --limit 40
done

echo "=== [$(date)] E-A3 ALL DONE ==="
