#!/bin/bash
# Completes the "8B 全量 + 2B/4B 抽样复现" matrix for Block A: runs E-A1...E-A5
# (--limit 40, or --limit 15 for E-A5) for the 4B-Instruct/4B-Thinking pair,
# which were missing from run_blockA{1,2,3,4,5}_all.sh (added after those
# were already queued). Waits for run_block0_4B_all.sh (needs 4B's E-01
# raw2/answer1/answer2) AND all of run_blockA{1..5}_all.sh (single-GPU
# exclusivity) to finish first.
set -e
cd /home/liuruida/from84_relay/why_thinking_make_vlm_worse_relay/code
source ~/miniconda/etc/profile.d/conda.sh
conda activate wtmw

while pgrep -f run_block0_4B_all.sh > /dev/null \
   || pgrep -f run_blockA1_all.sh > /dev/null || pgrep -f run_blockA2_all.sh > /dev/null \
   || pgrep -f run_blockA3_all.sh > /dev/null || pgrep -f run_blockA4_all.sh > /dev/null \
   || pgrep -f run_blockA5_all.sh > /dev/null; do
  sleep 60
done

echo "=== [$(date)] Block 0 (4B) + Block A (2B/8B) done -- starting Block A (4B) ==="

for M in 4B-Instruct 4B-Thinking; do
  echo "=== [$(date)] E-A1 $M (limit 40) ==="
  python run_EA1_kv_patch.py --model "$M" --subset MathVista_MINI --limit 40
  echo "=== [$(date)] E-A2 $M (limit 40) ==="
  python run_EA2_pathway.py --model "$M" --subset MathVista_MINI --limit 40
  echo "=== [$(date)] E-A3 $M (limit 40) ==="
  python run_EA3_trigger.py --model "$M" --subset MathVista_MINI --limit 40
  echo "=== [$(date)] E-A4 $M (limit 40) ==="
  python run_EA4_synthcf.py --model "$M" --limit 40
  echo "=== [$(date)] E-A5 $M (limit 15) ==="
  python run_EA5_attention_flow.py --model "$M" --subset MathVista_MINI --limit 15
done

echo "=== [$(date)] Block A (4B) ALL DONE -- full Block A matrix complete ==="
