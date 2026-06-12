"""E-B1: matched-sibling LoRA SFT for Block B causal attribution.

Trains Qwen3-VL-2B-Instruct with LoRA on either:
  - sft_cotsft.jsonl    (CoT-SFT: full <SUMMARY>/<CAPTION>/<REASONING>/<CONCLUSION> targets)
  - sft_directsft.jsonl (direct-SFT: CoT stripped to just the final answer)
both built by build_blockb_sft_data.py from the SAME LLaVA-CoT-100k
train.jsonl (same images, questions, ordering -> matched pair). Run 3x with
different --seed for each condition (6 runs total per E-B1).

Per-sample: build a multi-turn chat (image + alternating human/gpt turns),
tokenize the full conversation once via apply_chat_template, then mask
labels so loss is computed ONLY on assistant-turn tokens (re-rendering the
template incrementally to find each assistant span's [start,end) -- this is
O(n_turns) re-tokenizations per sample but n_turns is usually 1-3).

Single-sample batches (batch_size=1) + gradient accumulation, since Qwen-VL
multimodal inputs (pixel_values / image_grid_thw) have per-sample shapes
that don't trivially pad/stack.

NOT YET RUN ON GPU (queued behind Block 0 + Block A; image.zip.* must also
finish downloading + unzipping to datasets/llava_cot_100k/images/ first).
Validate on a tiny --max_steps before launching a full run.

Usage:
  python train_lora_sft.py --model 2B-Instruct --data sft_cotsft.jsonl \
      --output_dir ckpt/E-B1/cotsft_seed13 --seed 13 --max_steps 2000
  python train_lora_sft.py --model 2B-Instruct --data sft_directsft.jsonl \
      --output_dir ckpt/E-B1/directsft_seed13 --seed 13 --max_steps 2000
"""
import argparse
import json
import os
import random

import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from PIL import Image

from common.model_utils import load_model_and_processor

DATA_DIR = "/data/liuruida/why_thinking_make_vlm_worse/datasets/llava_cot_100k"
IMAGE_ROOT = os.path.join(DATA_DIR, "images")
CKPT_ROOT = "/data/liuruida/why_thinking_make_vlm_worse/ckpt"

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _build_messages(record):
    """Convert a {"image","conversations":[{"from":"human"/"gpt","value":...}]}
    record into a Qwen3-VL chat messages list. The image is attached to the
    FIRST human turn only (matches LLaVA-CoT-100k single-image convention).
    """
    img_path = os.path.join(IMAGE_ROOT, record["image"])
    messages = []
    for i, turn in enumerate(record["conversations"]):
        role = "user" if turn["from"] == "human" else "assistant"
        if role == "user" and i == 0:
            content = [{"type": "image", "image": img_path}, {"type": "text", "text": turn["value"]}]
        else:
            content = [{"type": "text", "text": turn["value"]}]
        messages.append({"role": role, "content": content})
    return messages


class LlavaCotSFTDataset(Dataset):
    def __init__(self, jsonl_path, processor, max_len=4096):
        self.records = []
        with open(jsonl_path) as f:
            for line in f:
                r = json.loads(line)
                if os.path.exists(os.path.join(IMAGE_ROOT, r["image"])):
                    self.records.append(r)
        print(f"[Data] {len(self.records)}/{sum(1 for _ in open(jsonl_path))} samples have images on disk")
        self.processor = processor
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        messages = _build_messages(record)

        full_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(text=[full_text], images=image_inputs, return_tensors="pt",
                                 truncation=True, max_length=self.max_len)
        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()
        labels[:] = -100

        # Re-render incrementally to find each assistant turn's token span.
        for i, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue
            prefix_msgs = messages[:i]
            prefix_text = self.processor.apply_chat_template(
                prefix_msgs, tokenize=False, add_generation_prompt=True)
            full_upto_text = self.processor.apply_chat_template(
                messages[:i + 1], tokenize=False, add_generation_prompt=False)

            # Tokenize via the full multimodal processor (not the bare text
            # tokenizer): the chat template inserts a single <|image_pad|>
            # placeholder, which processor() expands to image_grid_thw-many
            # tokens to build `input_ids` above. Tokenizing prefix/full_upto
            # with the bare tokenizer would leave that placeholder
            # un-expanded, making prefix_ids/full_upto_ids short by
            # (N_image_tokens - 1) and pointing labels at the wrong span.
            prefix_ids = self.processor(text=[prefix_text], images=image_inputs,
                                         return_tensors="pt")["input_ids"][0]
            full_upto_ids = self.processor(text=[full_upto_text], images=image_inputs,
                                            return_tensors="pt")["input_ids"][0]
            start, end = len(prefix_ids), len(full_upto_ids)
            end = min(end, labels.shape[0])
            if start < end:
                labels[start:end] = input_ids[start:end]

        item = {k: v[0] if v.dim() > 0 and v.shape[0] == 1 else v for k, v in inputs.items()}
        item["labels"] = labels
        return item


def collate_single(batch):
    assert len(batch) == 1, "use batch_size=1 (variable-shape multimodal inputs)"
    item = batch[0]
    return {k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in item.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="2B-Instruct")
    ap.add_argument("--data", required=True, help="sft_cotsft.jsonl or sft_directsft.jsonl (relative to llava_cot_100k/)")
    ap.add_argument("--output_dir", required=True, help="relative to ckpt/")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[Load] {args.model}")
    model, processor = load_model_and_processor(args.model)
    model.config.use_cache = False
    model.enable_input_require_grads()  # required for LoRA + gradient checkpointing

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=LORA_TARGET_MODULES, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    data_path = os.path.join(DATA_DIR, args.data)
    dataset = LlavaCotSFTDataset(data_path, processor)

    output_dir = os.path.join(CKPT_ROOT, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    targs = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=20,
        save_steps=max(1, args.max_steps // 4),
        save_total_limit=4,
        seed=args.seed,
        report_to=[],
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model, args=targs, train_dataset=dataset, data_collator=collate_single,
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f"[Done] saved to {output_dir}")


if __name__ == "__main__":
    main()
