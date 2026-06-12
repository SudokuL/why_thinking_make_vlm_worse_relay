#!/bin/bash
# Queued after run_block0_all.sh, run_blockA1_all.sh, run_blockA2_all.sh AND
# run_blockA3_all.sh (single-GPU exclusivity). Runs E-A4 (SynthCF region
# selectivity) for all 4 currently-downloaded models.
set -e
cd /data/liuruida/why_thinking_make_vlm_worse/code
source ~/miniconda/etc/profile.d/conda.sh
conda activate wtmw

while pgrep -f run_block0_all.sh > /dev/null || pgrep -f run_blockA1_all.sh > /dev/null \
   || pgrep -f run_blockA2_all.sh > /dev/null || pgrep -f run_blockA3_all.sh > /dev/null; do
  sleep 60
done

echo "=== [$(date)] Block 0 + E-A1 + E-A2 + E-A3 done -- starting E-A4 ==="

for M in 8B-Instruct 8B-Thinking 2B-Instruct 2B-Thinking; do
  echo "=== [$(date)] E-A4 $M (limit 40) ==="
  python run_EA4_synthcf.py --model "$M" --limit 40
done

echo "=== [$(date)] E-A4 ALL DONE ==="
