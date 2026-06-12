"""E-A3: trigger-type causal comparison -- what reopens visual access after
the full reasoning trace R_a has been produced on I_a? (Block A, the
"standout" H3 experiment)

For each flippable sample (raw1's answer != raw2's answer, via
common.grading.canonical_keys), build cache_self over the
FULL prefix = prompt(I_a,Q) + R_a (the complete trace from E-01's raw2), then
patch its image-token K/V (all layers) with donor K/V from forward(prompt(I_b,Q))
-- same mechanism as E-A1's "late" cut but at cut=100%. This patched cache
represents "the clean (I_b) visual evidence is sitting in the K/V cache, but
has it been NOTICED yet?" Three trigger conditions are then appended and
continued:

  T1 self-reflection : same-turn continuation, append the E-01 reflection
                        phrase ("Wait, let me check the figure again...").
  T2 new user turn (no new image): close the assistant turn and open a new
                        user turn asking the model to double-check, WITHOUT
                        re-supplying any image -- only the patched K/V cache
                        provides (possibly) updated visual info.
  T3 new user turn + fresh I_b image (ceiling/positive control): a genuinely
                        fresh generation (no cache/patch) with I_b re-supplied
                        as a new image in a new user turn -- tests whether the
                        model CAN update at all when given literal new visual
                        tokens.

  CRI_T1/T2/T3 = P(continuation's answer == raw1's answer) over flippable
  samples (via common.grading.answers_match).

H3 (gating) predicts CRI_T2 and/or CRI_T3 >> CRI_T1 (a user-turn boundary --
with or without fresh image tokens -- reopens visual access; mere
self-reflection in the same turn does not, even though the patched K/V
already contains the clean evidence).
H1/H2 predict no T1 vs T2 gap from the patch alone (H1: T1 already works
because late patches are causal; H2: none of T1/T2 work because the info
was never "in" the image channel to begin with -- only T3, which provides
fresh tokens, can work).

Prerequisite: results/E-01/<model>_<subset>.jsonl must exist.

Outputs results/E-A3/<model>_<subset>.jsonl + _summary.json (resumable).

Usage:
  python run_EA3_trigger.py --model 8B-Instruct --subset MathVista_MINI
  python run_EA3_trigger.py --model 2B-Instruct --subset MathVista_MINI --limit 40
"""
import argparse
import json
import os
import time

import torch
from qwen_vl_utils import process_vision_info

from common.model_utils import load_model_and_processor, build_user_message, generate
from common.data_utils import load_vsbench_pairs
from common.metrics import aggregate
from common.grading import canonical_keys, answers_match, parse_choices
from common.patch_utils import (
    cache_prefix_for_continuation, continue_generate_from_cache,
    forward_with_cache, patch_image_kv, clone_cache,
)

E01_DIR = "/data/liuruida/why_thinking_make_vlm_worse/results/E-01"
RESULTS_DIR = "/data/liuruida/why_thinking_make_vlm_worse/results/E-A3"

REFLECT_PROMPT = "\n\nWait, let me check the figure again to make sure I haven't made a mistake."
NEWTURN_TEXT = "Please double check your answer by looking at the image again. Give your final answer."
NEWTURN_IMG_PREFIX = "Looking at the image again, please reconsider. "

T1T2_BUDGET = 1024
T3_BUDGET = 2048


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--subset", default="MathVista_MINI")
    ap.add_argument("--limit", type=int, default=None,
                     help="cap number of FLIPPABLE samples processed (for 2B/4B sampling)")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_short = args.model.replace("/", "_")

    e01_path = os.path.join(E01_DIR, f"{model_short}_{args.subset}.jsonl")
    if not os.path.exists(e01_path):
        raise FileNotFoundError(f"E-01 results not found: {e01_path} -- run run_E01_visualswap.py first")

    e01_records = {}
    with open(e01_path) as f:
        for line in f:
            r = json.loads(line)
            if "error" in r:
                continue
            e01_records[str(r["pid"])] = r
    print(f"[Load] {len(e01_records)} E-01 records from {e01_path}")

    out_path = os.path.join(RESULTS_DIR, f"{model_short}_{args.subset}.jsonl")
    summary_path = os.path.join(RESULTS_DIR, f"{model_short}_{args.subset}_summary.json")

    done_pids = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    done_pids.add(json.loads(line)["pid"])
        print(f"[Resume] {len(done_pids)} samples already done, skipping.")

    print(f"[Load] {args.model}")
    model, processor = load_model_and_processor(args.model)

    samples = load_vsbench_pairs(subset=args.subset)
    print(f"[Data] {len(samples)} pairs from {args.subset}")

    fout = open(out_path, "a")
    n_processed = 0
    for i, s in enumerate(samples):
        pid = str(s["pid"])
        if pid not in e01_records:
            continue
        e01 = e01_records[pid]
        raw1, raw2 = e01["raw1"], e01["raw2"]
        choices = parse_choices(s["query"])
        k1, k2 = canonical_keys(raw1, choices), canonical_keys(raw2, choices)
        flippable = (k1 is not None and k2 is not None and not (k1 & k2))
        if not flippable:
            continue
        if pid in done_pids:
            n_processed += 1
            continue
        if args.limit and n_processed >= args.limit:
            break

        t0 = time.time()
        try:
            msg_a = build_user_message(s["alt_image"], s["query"])
            msg_b = build_user_message(s["orig_image"], s["query"])

            full_ids_text = processor.apply_chat_template(msg_a, tokenize=False, add_generation_prompt=True)
            image_inputs_a, _ = process_vision_info(msg_a)
            base_a = processor(text=[full_ids_text], images=image_inputs_a, return_tensors="pt").to(model.device)
            prompt_len_a = base_a["input_ids"].shape[1]

            gen_ids = processor.tokenizer(raw2, return_tensors="pt", add_special_tokens=False)["input_ids"]
            n_gen = gen_ids.shape[1]
            if n_gen < 5:
                continue
            full_input_ids_a = torch.cat([base_a["input_ids"], gen_ids.to(model.device)], dim=1)
            L = full_input_ids_a.shape[1]

            # cache over the FULL trace (cut = end), then patch image K/V I_a -> I_b
            prefix_ids, attn, cache_self, span_self = cache_prefix_for_continuation(
                model, processor, msg_a, full_input_ids_a, cut_len=L)
            _, _, cache_donor, span_donor = forward_with_cache(model, processor, msg_b)
            patched_full = patch_image_kv(cache_self, cache_donor, span_self[0], span_donor[0], layers=None)

            # --- T1: self-reflection, same turn ---
            t1_ids = processor.tokenizer(REFLECT_PROMPT, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
            input_ids_t1 = torch.cat([prefix_ids, t1_ids], dim=1)
            attn_t1 = torch.ones_like(input_ids_t1)
            cont_t1 = continue_generate_from_cache(model, processor, input_ids_t1, attn_t1, clone_cache(patched_full), max_new_tokens=T1T2_BUDGET)

            # --- T2: new user turn, no fresh image ---
            t2_text = f"<|im_end|>\n<|im_start|>user\n{NEWTURN_TEXT}<|im_end|>\n<|im_start|>assistant\n"
            t2_ids = processor.tokenizer(t2_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
            input_ids_t2 = torch.cat([prefix_ids, t2_ids], dim=1)
            attn_t2 = torch.ones_like(input_ids_t2)
            cont_t2 = continue_generate_from_cache(model, processor, input_ids_t2, attn_t2, clone_cache(patched_full), max_new_tokens=T1T2_BUDGET)

            # --- T3: fresh generation, new user turn WITH I_b re-supplied (ceiling) ---
            msg_t3 = msg_a + [
                {"role": "assistant", "content": raw2},
                {"role": "user", "content": [
                    {"type": "image", "image": s["orig_image"]},
                    {"type": "text", "text": NEWTURN_IMG_PREFIX + s["query"]},
                ]},
            ]
            raw_t3, _ = generate(model, processor, msg_t3, max_new_tokens=T3_BUDGET)

            record = {
                "pid": pid,
                "flip_t1": answers_match(cont_t1, raw1, choices),
                "flip_t2": answers_match(cont_t2, raw1, choices),
                "flip_t3": answers_match(raw_t3, raw1, choices),
                "incomplete_t1": canonical_keys(cont_t1, choices) is None,
                "incomplete_t2": canonical_keys(cont_t2, choices) is None,
                "incomplete_t3": canonical_keys(raw_t3, choices) is None,
                "cont_t1": cont_t1, "cont_t2": cont_t2, "raw_t3": raw_t3,
            }
        except Exception as e:
            record = {"pid": pid, "error": str(e)}

        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()
        n_processed += 1
        dt = time.time() - t0
        if "error" in record:
            print(f"[{i+1}/{len(samples)}] pid={pid} dt={dt:.1f}s ERR {record['error']}")
        else:
            print(f"[{i+1}/{len(samples)}] pid={pid} dt={dt:.1f}s "
                  f"flip(t1,t2,t3)=({record['flip_t1']},{record['flip_t2']},{record['flip_t3']})")

    fout.close()

    results = []
    with open(out_path) as f:
        for line in f:
            r = json.loads(line)
            if "error" not in r:
                results.append(r)

    summary = {"model": args.model, "subset": args.subset, "n_flippable_done": len(results)}
    for key in ["flip_t1", "flip_t2", "flip_t3", "incomplete_t1", "incomplete_t2", "incomplete_t3"]:
        vals = [{"v": r[key]} for r in results]
        summary[key] = aggregate(vals, "v")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
