#!/bin/bash
# E-B1: matched-sibling LoRA SFT (CoT-SFT vs direct-SFT, 3 seeds each = 6 runs).
# Queued behind the entire Block 0 + Block A pipeline (single-GPU
# exclusivity). Also waits for LLaVA-CoT-100k's 16 image.zip.* parts to
# finish downloading, then unzips them.
set -e
cd /data/liuruida/why_thinking_make_vlm_worse/code
source ~/miniconda/etc/profile.d/conda.sh
conda activate wtmw

while pgrep -f run_block0_all.sh > /dev/null || pgrep -f run_block0_4B_all.sh > /dev/null \
   || pgrep -f run_blockA1_all.sh > /dev/null || pgrep -f run_blockA2_all.sh > /dev/null \
   || pgrep -f run_blockA3_all.sh > /dev/null || pgrep -f run_blockA4_all.sh > /dev/null \
   || pgrep -f run_blockA5_all.sh > /dev/null || pgrep -f run_blockA_4B_all.sh > /dev/null; do
  sleep 60
done

echo "=== [$(date)] Block 0 + Block A (all 6 models) done -- preparing E-B1 ==="

DATA_DIR=/data/liuruida/why_thinking_make_vlm_worse/datasets/llava_cot_100k
cd "$DATA_DIR"

# Wait for all 16 image.zip.part-{aa..ap} to finish downloading. The original
# bulk download (download.log) permanently FAILED at 9/16 parts (cas-bridge
# CDN 403 on expired pre-signed URLs); a retry-loop redownload
# (redownload_llava_cot.sh, logs to download_llava_cot_retry.log) resumes the
# missing parts. hf download only materializes the final part-XX file once
# that part is fully downloaded (in-progress parts live under
# .cache/huggingface/download/*.incomplete), so just count the final files.
while [ "$(ls image.zip.part-* 2>/dev/null | wc -l)" -lt 16 ]; do
  echo "[$(date)] waiting for LLaVA-CoT-100k image.zip parts ($(ls image.zip.part-* 2>/dev/null | wc -l)/16)..."
  sleep 300
done

if [ ! -d images ] || [ -z "$(ls -A images 2>/dev/null)" ]; then
  echo "=== [$(date)] Unzipping LLaVA-CoT-100k images (this is large, ~160G) ==="
  cat image.zip.part-* > image.zip
  mkdir -p images
  unzip -q image.zip -d images
  # the zip may or may not contain a top-level "images/" dir -- normalize
  if [ ! -d images/coco ] && [ -d images/images ]; then
    mv images/images/* images/
    rmdir images/images
  fi
  rm -f image.zip
fi

cd /data/liuruida/why_thinking_make_vlm_worse/code
python build_blockb_sft_data.py

echo "=== [$(date)] Smoke test: train_lora_sft.py --max_steps 5 ==="
python train_lora_sft.py --model 2B-Instruct --data sft_cotsft.jsonl \
    --output_dir E-B1/smoke_test --seed 13 --max_steps 5

echo "=== [$(date)] Smoke test passed -- starting E-B1 6 runs (1 seed at a time) ==="

for SEED in 13 42 1234; do
  for COND in cotsft directsft; do
    echo "=== [$(date)] E-B1 $COND seed=$SEED ==="
    python train_lora_sft.py --model 2B-Instruct --data sft_${COND}.jsonl \
        --output_dir E-B1/${COND}_seed${SEED} --seed $SEED --max_steps 3000
  done
done

echo "=== [$(date)] E-B1 ALL DONE ==="
