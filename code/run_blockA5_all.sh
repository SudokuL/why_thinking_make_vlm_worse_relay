#!/bin/bash
# Queued after run_block0_all.sh, run_blockA1_all.sh, run_blockA2_all.sh,
# run_blockA3_all.sh AND run_blockA4_all.sh (single-GPU exclusivity). Runs
# E-A5 (attention-flow descriptive stats -- NOT evidence, lowest priority)
# for all 4 currently-downloaded models.
set -e
cd /home/liuruida/from84_relay/why_thinking_make_vlm_worse_relay/code
source ~/miniconda/etc/profile.d/conda.sh
conda activate wtmw

while pgrep -f run_block0_all.sh > /dev/null || pgrep -f run_blockA1_all.sh > /dev/null \
   || pgrep -f run_blockA2_all.sh > /dev/null || pgrep -f run_blockA3_all.sh > /dev/null \
   || pgrep -f run_blockA4_all.sh > /dev/null; do
  sleep 60
done

echo "=== [$(date)] Block A (E-A1-A4) done -- starting E-A5 (descriptive only) ==="

for M in 2B-Instruct 2B-Thinking 8B-Instruct 8B-Thinking; do
  echo "=== [$(date)] E-A5 $M (limit 15) ==="
  python run_EA5_attention_flow.py --model "$M" --subset MathVista_MINI --limit 15
done

echo "=== [$(date)] Block A ALL DONE (E-A1...E-A5) ==="
