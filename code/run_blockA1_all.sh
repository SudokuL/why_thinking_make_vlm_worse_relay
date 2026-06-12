#!/bin/bash
# Queued to run after run_block0_all.sh finishes (waits for it to exit so we
# never run two GPU jobs at once on the single 4090). Runs E-A1 (CRI curve)
# for all 4 currently-downloaded models: 8B pair gets the full flippable set
# (subset of the 200 MathVista_MINI pairs where answer1 != answer2), 2B pair
# is sampled (--limit 40) per EXPERIMENT_GUIDE.md ("8B 对全量 + 2B/4B 对抽样复现").
set -e
cd /home/liuruida/from84_relay/why_thinking_make_vlm_worse_relay/code
source ~/miniconda/etc/profile.d/conda.sh
conda activate wtmw

while pgrep -f run_block0_all.sh > /dev/null; do
  sleep 60
done

echo "=== [$(date)] Block 0 done -- starting E-A1 ==="

for M in 8B-Instruct 8B-Thinking; do
  echo "=== [$(date)] E-A1 $M (full flippable) ==="
  python run_EA1_kv_patch.py --model "$M" --subset MathVista_MINI
done

for M in 2B-Instruct 2B-Thinking; do
  echo "=== [$(date)] E-A1 $M (limit 40) ==="
  python run_EA1_kv_patch.py --model "$M" --subset MathVista_MINI --limit 40
done

echo "=== [$(date)] E-A1 ALL DONE ==="
