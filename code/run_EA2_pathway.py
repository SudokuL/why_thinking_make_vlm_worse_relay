"""E-A2: pathway separation -- where does the "clean" (I_b) visual evidence
live once the model has been reasoning on I_a for a while? (Block A)

Two patch conditions at the same early/mid/late cut points as E-A1, applied
to the SAME prefix = prompt(I_a,Q) + R_a[:cut]:

  IMG patch (= E-A1's patch): replace cache_self's IMAGE-token K/V (all
    layers) with the donor's image-token K/V from forward(prompt(I_b,Q)).
    -> tests whether the raw visual channel still carries recoverable
       "clean" evidence (H1-style).

  TXT patch: replace cache_self's REASONING-token K/V (positions
    [prompt_len_a, cut), i.e. R_a[:cut] itself -- NOT the image tokens) with
    the K/V those same tokens would have if the model had produced/attended
    to them while looking at I_b instead of I_a. Donor is built by force-
    feeding the identical token ids R_a[:cut] on top of prompt(I_b,Q) (a
    second forward pass) and reading off its K/V at the corresponding
    positions. -> tests whether the "visual belief" has already been written
    into the reasoning-token residual stream (H2-style "write-once").

  CRI_IMG(t) = P(ans_imgpatch == answer1)   (same quantity as E-A1's CRI(t))
  CRI_TXT(t) = P(ans_txtpatch == answer1)
  control(t) = P(ans_nopatch  == answer1)   (no-patch continuation)

H1 predicts CRI_IMG(late) high, CRI_TXT(*) low.
H2 predicts CRI_TXT(early/mid) high (information already "written"),
CRI_IMG(*) ~= 0.

Prerequisite: results/E-01/<model>_<subset>.jsonl must exist (reuses
raw1/raw2, must match the trace E-A1 operates on; "answer1"/"answer2" are
compared via common.grading.canonical_keys/answers_match against raw1/raw2).

Outputs results/E-A2/<model>_<subset>.jsonl + _summary.json (resumable).

Usage:
  python run_EA2_pathway.py --model 8B-Instruct --subset MathVista_MINI
  python run_EA2_pathway.py --model 2B-Instruct --subset MathVista_MINI --limit 40
"""
import argparse
import json
import os
import time

import torch
from qwen_vl_utils import process_vision_info

from common.model_utils import load_model_and_processor, build_user_message
from common.data_utils import load_vsbench_pairs
from common.metrics import aggregate
from common.grading import canonical_keys, answers_match, parse_choices
from common.patch_utils import (
    cache_prefix_for_continuation, continue_generate_from_cache,
    forward_with_cache, patch_image_kv, clone_cache,
)

E01_DIR = "/home/liuruida/from84_relay/why_thinking_make_vlm_worse_relay/results/E-01"
RESULTS_DIR = "/home/liuruida/from84_relay/why_thinking_make_vlm_worse_relay/results/E-A2"

CUT_FRACTIONS = {"early": 0.2, "mid": 0.5, "late": 0.8}
MAX_CONT_TOKENS = 800
MIN_CONT_TOKENS = 96
CONT_BUFFER = 96


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

            # IMG donor: forward over prompt(I_b, Q) alone
            _, _, cache_donor_img, span_donor_img = forward_with_cache(model, processor, msg_b)

            # base tokenization for I_b prompt (for building TXT donor below)
            text_b = processor.apply_chat_template(msg_b, tokenize=False, add_generation_prompt=True)
            image_inputs_b, _ = process_vision_info(msg_b)
            base_b = processor(text=[text_b], images=image_inputs_b, return_tensors="pt").to(model.device)
            prompt_len_b = base_b["input_ids"].shape[1]
            fwd_kwargs_b = {k: v for k, v in base_b.items() if k not in ("input_ids", "attention_mask")}

            cuts = {}
            for cut_name, frac in CUT_FRACTIONS.items():
                cut = prompt_len_a + max(1, min(n_gen - 1, int(n_gen * frac)))
                n_reason = cut - prompt_len_a  # number of R_a tokens included so far
                remaining = n_gen - n_reason
                budget = max(MIN_CONT_TOKENS, min(MAX_CONT_TOKENS, remaining + CONT_BUFFER))

                prefix_ids, attn, cache_self, span_self = cache_prefix_for_continuation(
                    model, processor, msg_a, full_input_ids_a, cut_len=cut)

                # --- IMG patch ---
                img_patched = patch_image_kv(cache_self, cache_donor_img, span_self[0], span_donor_img[0], layers=None)
                cont_imgpatch = continue_generate_from_cache(model, processor, prefix_ids, attn, img_patched, max_new_tokens=budget)

                # --- TXT patch: donor = forward(prompt(I_b,Q) + R_a[:n_reason]) ---
                reason_ids = gen_ids[:, :n_reason].to(model.device)
                txt_input_ids = torch.cat([base_b["input_ids"], reason_ids], dim=1)
                txt_attn = torch.ones_like(txt_input_ids)
                with torch.inference_mode():
                    out_txt = model(input_ids=txt_input_ids, attention_mask=txt_attn,
                                     use_cache=True, return_dict=True, **fwd_kwargs_b)
                cache_donor_txt = out_txt.past_key_values
                # target span = reasoning tokens [prompt_len_a, cut) in cache_self (len n_reason)
                # donor span   = corresponding tokens [prompt_len_b, prompt_len_b+n_reason) in cache_donor_txt
                txt_target_span = (prompt_len_a, cut)
                txt_donor_span = (prompt_len_b, prompt_len_b + n_reason)
                txt_patched = patch_image_kv(cache_self, cache_donor_txt, txt_target_span, txt_donor_span, layers=None)
                cont_txtpatch = continue_generate_from_cache(model, processor, prefix_ids, attn, txt_patched, max_new_tokens=budget)

                # --- no-patch control ---
                cont_nopatch = continue_generate_from_cache(model, processor, prefix_ids, attn, clone_cache(cache_self), max_new_tokens=budget)

                cuts[cut_name] = {
                    "cut_len": cut, "n_gen": n_gen, "n_reason": n_reason, "budget": budget,
                    "flip_img": answers_match(cont_imgpatch, raw1, choices),
                    "flip_txt": answers_match(cont_txtpatch, raw1, choices),
                    "flip_nopatch": answers_match(cont_nopatch, raw1, choices),
                    "incomplete_img": canonical_keys(cont_imgpatch, choices) is None,
                    "incomplete_txt": canonical_keys(cont_txtpatch, choices) is None,
                    "incomplete_nopatch": canonical_keys(cont_nopatch, choices) is None,
                    "cont_imgpatch": cont_imgpatch, "cont_txtpatch": cont_txtpatch, "cont_nopatch": cont_nopatch,
                }

            record = {"pid": pid, "cuts": cuts}
        except Exception as e:
            record = {"pid": pid, "error": str(e)}

        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()
        n_processed += 1
        dt = time.time() - t0
        if "error" in record:
            print(f"[{i+1}/{len(samples)}] pid={pid} dt={dt:.1f}s ERR {record['error']}")
        else:
            flips = {k: (v["flip_img"], v["flip_txt"], v["flip_nopatch"]) for k, v in record["cuts"].items()}
            print(f"[{i+1}/{len(samples)}] pid={pid} dt={dt:.1f}s flip(img,txt,nopatch)={flips}")

    fout.close()

    results = []
    with open(out_path) as f:
        for line in f:
            r = json.loads(line)
            if "error" not in r:
                results.append(r)

    summary = {"model": args.model, "subset": args.subset, "n_flippable_done": len(results)}
    for cut_name in CUT_FRACTIONS:
        for key, field in [("CRI_IMG", "flip_img"), ("CRI_TXT", "flip_txt"), ("control", "flip_nopatch")]:
            vals = [{"v": r["cuts"][cut_name][field]} for r in results if cut_name in r["cuts"]]
            summary[f"{key}_{cut_name}"] = aggregate(vals, "v")
        for key, field in [("incomplete_img", "incomplete_img"), ("incomplete_txt", "incomplete_txt")]:
            vals = [{"v": r["cuts"][cut_name][field]} for r in results if cut_name in r["cuts"]]
            summary[f"{key}_{cut_name}"] = aggregate(vals, "v")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
