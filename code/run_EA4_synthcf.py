"""E-A4: region selectivity on SynthCF (Block A).

Rules out the trivial "any K/V perturbation flips the answer" explanation
for E-A1's CRI(t) > 0 results. Uses the synthetic counterfactual pairs from
build_synthcf.py: a 392x392 canvas split into a TOP half and a BOTTOM half
(each = a CONTIGUOUS 72-token span of the 144-token image-token grid). One
half (the "target half") contains a single shape that is RED in I_b (clean,
ground-truth answer1="1") and some other color in I_a (counterfactual,
ground-truth answer2="0"); the other half ("distractor half") is pixel-
identical between I_a and I_b.

For each sample:
  1. raw1 = M(I_b,Q) -> answer1_model;  raw2 = M(I_a,Q) -> answer2_model
     (both fresh greedy generations -- SynthCF has no E-01 pre-pass).
  2. Only flippable samples (answer1_model != answer2_model) are scored.
  3. At cut fractions early/mid/late of raw2's token length:
       prefix      = prompt(I_a,Q) + raw2[:cut]
       cache_self  = forward cache over prefix (image = I_a)
       cache_donor = forward cache over prompt(I_b,Q) alone (image = I_b)
       REL  patch  = cache_self with image-token K/V at target_span (the
                      half containing the color-flipped shape) replaced by
                      cache_donor's K/V at the same span
       IRR  patch  = cache_self with image-token K/V at distractor_span (the
                      OTHER half, pixel-identical between I_a/I_b) replaced
                      by cache_donor's K/V at the same span
       continue greedy decoding from each patch (+ unpatched control)

  CRI_REL(t) = P(ans_rel == answer1_model)   -- should be high if H1-style
               re-consultation is REGION-SELECTIVE
  CRI_IRR(t) = P(ans_irr == answer1_model)   -- placebo; should stay near
               control(t) -- a perturbation of an irrelevant (but non-
               identical-K/V, since the vision tower has global receptive
               field) region should NOT recover answer1
  control(t) = P(ans_nopatch == answer1_model)

H1/H3 predict CRI_REL(late) >> CRI_IRR(late) ~= control(late) (selective
re-consultation of the relevant region). If CRI_IRR(t) ~= CRI_REL(t), CRI(t)
in E-A1 may be partly an artifact of perturbing the K/V cache per se, not of
recovering the specific visual evidence.

Prerequisite: datasets/synthcf/manifest.jsonl must exist (run
build_synthcf.py first).

Outputs results/E-A4/<model>_synthcf.jsonl + _summary.json (resumable).

Usage:
  python run_EA4_synthcf.py --model 8B-Instruct
  python run_EA4_synthcf.py --model 2B-Instruct --limit 40
"""
import argparse
import json
import os
import time

import torch

from common.model_utils import load_model_and_processor, build_user_message, generate
from common.data_utils import load_synthcf
from common.metrics import extract_answer, answers_equal, aggregate
from common.patch_utils import (
    cache_prefix_for_continuation, continue_generate_from_cache,
    forward_with_cache, patch_image_kv, clone_cache,
)

RESULTS_DIR = "/data/liuruida/why_thinking_make_vlm_worse/results/E-A4"

CUT_FRACTIONS = {"early": 0.2, "mid": 0.5, "late": 0.8}
MAX_CONT_TOKENS = 512
MIN_CONT_TOKENS = 64
CONT_BUFFER = 64
GEN_BUDGET = 768


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=None,
                     help="cap number of FLIPPABLE samples processed (for 2B/4B sampling)")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_short = args.model.replace("/", "_")

    out_path = os.path.join(RESULTS_DIR, f"{model_short}_synthcf.jsonl")
    summary_path = os.path.join(RESULTS_DIR, f"{model_short}_synthcf_summary.json")

    done_pids = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    done_pids.add(json.loads(line)["pid"])
        print(f"[Resume] {len(done_pids)} samples already done, skipping.")

    print(f"[Load] {args.model}")
    model, processor = load_model_and_processor(args.model)

    samples = load_synthcf()
    print(f"[Data] {len(samples)} SynthCF pairs")

    fout = open(out_path, "a")
    n_processed = 0
    for i, s in enumerate(samples):
        pid = str(s["pid"])
        if pid in done_pids:
            n_processed += 1
            continue
        if args.limit and n_processed >= args.limit:
            break

        t0 = time.time()
        try:
            msg_a = build_user_message(s["alt_image"], s["query"])
            msg_b = build_user_message(s["orig_image"], s["query"])

            raw1, _ = generate(model, processor, msg_b, max_new_tokens=GEN_BUDGET)
            raw2, _ = generate(model, processor, msg_a, max_new_tokens=GEN_BUDGET)
            answer1_model = extract_answer(raw1)
            answer2_model = extract_answer(raw2)
            flippable = (answer1_model is not None and answer2_model is not None
                          and not answers_equal(answer1_model, answer2_model))
            if not flippable:
                record = {
                    "pid": pid, "flippable": False,
                    "answer1_model": answer1_model, "answer2_model": answer2_model,
                    "gt_answer1": s["answer1"], "gt_answer2": s["answer2"],
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                n_processed += 1
                dt = time.time() - t0
                print(f"[{i+1}/{len(samples)}] pid={pid} dt={dt:.1f}s NOT FLIPPABLE "
                      f"({answer1_model} vs {answer2_model})")
                continue

            full_ids_text = processor.apply_chat_template(msg_a, tokenize=False, add_generation_prompt=True)
            from qwen_vl_utils import process_vision_info
            image_inputs_a, _ = process_vision_info(msg_a)
            base_a = processor(text=[full_ids_text], images=image_inputs_a, return_tensors="pt").to(model.device)
            prompt_len = base_a["input_ids"].shape[1]

            gen_ids = processor.tokenizer(raw2, return_tensors="pt", add_special_tokens=False)["input_ids"]
            n_gen = gen_ids.shape[1]
            if n_gen < 5:
                record = {
                    "pid": pid, "flippable": True, "skipped": "trace_too_short",
                    "answer1_model": answer1_model, "answer2_model": answer2_model,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                n_processed += 1
                continue
            full_input_ids_a = torch.cat([base_a["input_ids"], gen_ids.to(model.device)], dim=1)

            # donor cache: forward over prompt(I_b, Q) alone
            _, _, cache_donor, span_donor = forward_with_cache(model, processor, msg_b)
            d0 = span_donor[0][0]

            target_span = s["target_span"]      # (start,end) offsets within image span
            distractor_span = s["distractor_span"]

            cuts = {}
            for cut_name, frac in CUT_FRACTIONS.items():
                cut = prompt_len + max(1, min(n_gen - 1, int(n_gen * frac)))
                remaining = n_gen - (cut - prompt_len)
                budget = max(MIN_CONT_TOKENS, min(MAX_CONT_TOKENS, remaining + CONT_BUFFER))

                prefix_ids, attn, cache_self, span_self = cache_prefix_for_continuation(
                    model, processor, msg_a, full_input_ids_a, cut_len=cut)
                s0 = span_self[0][0]

                rel_target = (s0 + target_span[0], s0 + target_span[1])
                rel_donor = (d0 + target_span[0], d0 + target_span[1])
                irr_target = (s0 + distractor_span[0], s0 + distractor_span[1])
                irr_donor = (d0 + distractor_span[0], d0 + distractor_span[1])

                rel_patched = patch_image_kv(cache_self, cache_donor, rel_target, rel_donor, layers=None)
                cont_rel = continue_generate_from_cache(model, processor, prefix_ids, attn, rel_patched, max_new_tokens=budget)

                irr_patched = patch_image_kv(cache_self, cache_donor, irr_target, irr_donor, layers=None)
                cont_irr = continue_generate_from_cache(model, processor, prefix_ids, attn, irr_patched, max_new_tokens=budget)

                cont_nopatch = continue_generate_from_cache(model, processor, prefix_ids, attn, clone_cache(cache_self), max_new_tokens=budget)

                ans_rel = extract_answer(cont_rel)
                ans_irr = extract_answer(cont_irr)
                ans_nop = extract_answer(cont_nopatch)

                cuts[cut_name] = {
                    "cut_len": cut, "n_gen": n_gen, "budget": budget,
                    "ans_rel": ans_rel, "ans_irr": ans_irr, "ans_nopatch": ans_nop,
                    "flip_rel": bool(ans_rel is not None and answers_equal(ans_rel, answer1_model)),
                    "flip_irr": bool(ans_irr is not None and answers_equal(ans_irr, answer1_model)),
                    "flip_nopatch": bool(ans_nop is not None and answers_equal(ans_nop, answer1_model)),
                    "incomplete_rel": ans_rel is None,
                    "incomplete_irr": ans_irr is None,
                    "incomplete_nopatch": ans_nop is None,
                }

            record = {
                "pid": pid, "flippable": True,
                "answer1_model": answer1_model, "answer2_model": answer2_model,
                "target_half": s["target_half"], "cuts": cuts,
            }
        except Exception as e:
            record = {"pid": pid, "error": str(e)}

        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()
        n_processed += 1
        dt = time.time() - t0
        if "error" in record:
            print(f"[{i+1}/{len(samples)}] pid={pid} dt={dt:.1f}s ERR {record['error']}")
        elif not record.get("flippable", True) or "skipped" in record:
            print(f"[{i+1}/{len(samples)}] pid={pid} dt={dt:.1f}s SKIPPED")
        else:
            flips = {k: (v["flip_rel"], v["flip_irr"], v["flip_nopatch"]) for k, v in record["cuts"].items()}
            print(f"[{i+1}/{len(samples)}] pid={pid} dt={dt:.1f}s flip(rel,irr,nopatch)={flips}")

    fout.close()

    results = []
    with open(out_path) as f:
        for line in f:
            r = json.loads(line)
            if "error" not in r and r.get("flippable") and "skipped" not in r:
                results.append(r)

    summary = {"model": args.model, "n_flippable_done": len(results)}
    for cut_name in CUT_FRACTIONS:
        for key, field in [("CRI_REL", "flip_rel"), ("CRI_IRR", "flip_irr"), ("control", "flip_nopatch")]:
            vals = [{"v": r["cuts"][cut_name][field]} for r in results if cut_name in r["cuts"]]
            summary[f"{key}_{cut_name}"] = aggregate(vals, "v")
        for key, field in [("incomplete_rel", "incomplete_rel"), ("incomplete_irr", "incomplete_irr")]:
            vals = [{"v": r["cuts"][cut_name][field]} for r in results if cut_name in r["cuts"]]
            summary[f"{key}_{cut_name}"] = aggregate(vals, "v")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
