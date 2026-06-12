"""E-A1: position-segmented visual K/V patching -> CRI(t) curve (Block A).

Design (see EXPERIMENT_GUIDE.md sec 3, Block A):
  For each VisualSwap pair (I_a=alt/swap image with elicited reasoning trace
  R_a = raw2 from E-01, I_b=original image, Q, raw1=M(I_b,Q) -> "answer1",
  raw2=M(I_a,Q) -> "answer2", both compared via common.grading.canonical_keys
  / answers_match):
    - Only "flippable" samples where answer1 != answer2 are scored for CRI
      (otherwise there's no answer-gap to recover).
    - At cut fractions early/mid/late of R_a's token length:
        prefix      = prompt(I_a,Q) + R_a[:cut]
        cache_self  = forward cache over prefix (image = I_a, the "self"/
                      counterfactual image that produced this trace)
        cache_donor = forward cache over prompt(I_b,Q) alone (image = I_b,
                      the "clean"/original image)
        patched     = cache_self with image-token K/V (all layers) replaced
                      by cache_donor's image-token K/V
        continue greedy decoding from prefix with `patched`   -> ans_patch
        continue greedy decoding from prefix with `cache_self` (cloned,
          UNPATCHED) -> ans_nopatch  (control: does merely continuing the
          trace flip the answer anyway, with no visual evidence change?)

  CRI(t)        = P(ans_patch   == answer1) over flippable samples
  control(t)    = P(ans_nopatch == answer1) over flippable samples
    (control should be near 0; if it isn't, CRI(t) is confounded by
    spontaneous self-correction rather than the patch)

H1 predicts CRI(late) high; H2 predicts CRI(late) ~= 0 (with CRI(early)
possibly higher).

Prerequisite: results/E-01/<model>_<subset>.jsonl must already exist (run
run_E01_visualswap.py first) -- we reuse its raw1/raw2 instead of
re-generating, since R_a must be the SAME trace for the patch to operate on.

Outputs results/E-A1/<model>_<subset>.jsonl (one record per pid, all 3 cuts)
+ results/E-A1/<model>_<subset>_summary.json (resumable).

Usage:
  python run_EA1_kv_patch.py --model 8B-Instruct --subset MathVista_MINI
  python run_EA1_kv_patch.py --model 2B-Instruct --subset MathVista_MINI --limit 40
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

E01_DIR = "/data/liuruida/why_thinking_make_vlm_worse/results/E-01"
RESULTS_DIR = "/data/liuruida/why_thinking_make_vlm_worse/results/E-A1"

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
            image_inputs, _ = process_vision_info(msg_a)
            base = processor(text=[full_ids_text], images=image_inputs, return_tensors="pt").to(model.device)
            prompt_len = base["input_ids"].shape[1]

            gen_ids = processor.tokenizer(raw2, return_tensors="pt", add_special_tokens=False)["input_ids"]
            n_gen = gen_ids.shape[1]
            if n_gen < 5:
                continue
            full_input_ids = torch.cat([base["input_ids"], gen_ids.to(model.device)], dim=1)

            # donor cache: forward over prompt(I_b, Q) alone, computed once per sample
            _, _, cache_donor, span_donor = forward_with_cache(model, processor, msg_b)

            cuts = {}
            for cut_name, frac in CUT_FRACTIONS.items():
                cut = prompt_len + max(1, min(n_gen - 1, int(n_gen * frac)))
                remaining = n_gen - (cut - prompt_len)
                budget = max(MIN_CONT_TOKENS, min(MAX_CONT_TOKENS, remaining + CONT_BUFFER))

                prefix_ids, attn, cache_self, span_self = cache_prefix_for_continuation(
                    model, processor, msg_a, full_input_ids, cut_len=cut)

                patched = patch_image_kv(cache_self, cache_donor, span_self[0], span_donor[0], layers=None)
                cont_patch = continue_generate_from_cache(model, processor, prefix_ids, attn, patched, max_new_tokens=budget)

                cont_nopatch = continue_generate_from_cache(model, processor, prefix_ids, attn, clone_cache(cache_self), max_new_tokens=budget)

                cuts[cut_name] = {
                    "cut_len": cut, "n_gen": n_gen, "budget": budget,
                    "flip_patch": answers_match(cont_patch, raw1, choices),
                    "flip_nopatch": answers_match(cont_nopatch, raw1, choices),
                    "incomplete_patch": canonical_keys(cont_patch, choices) is None,
                    "incomplete_nopatch": canonical_keys(cont_nopatch, choices) is None,
                    "cont_patch": cont_patch, "cont_nopatch": cont_nopatch,
                }

            record = {
                "pid": pid, "img_span_self_lens": None, "cuts": cuts,
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
            flips = {k: (v["flip_patch"], v["flip_nopatch"]) for k, v in record["cuts"].items()}
            print(f"[{i+1}/{len(samples)}] pid={pid} dt={dt:.1f}s flip(patch,nopatch)={flips}")

    fout.close()

    # summary: CRI(t) and control(t) per cut, only over non-error records
    results = []
    with open(out_path) as f:
        for line in f:
            r = json.loads(line)
            if "error" not in r:
                results.append(r)

    summary = {"model": args.model, "subset": args.subset, "n_flippable_done": len(results)}
    for cut_name in CUT_FRACTIONS:
        flip_patch = [{"flip": r["cuts"][cut_name]["flip_patch"]} for r in results if cut_name in r["cuts"]]
        flip_nopatch = [{"flip": r["cuts"][cut_name]["flip_nopatch"]} for r in results if cut_name in r["cuts"]]
        incomplete_patch = [{"v": r["cuts"][cut_name]["incomplete_patch"]} for r in results if cut_name in r["cuts"]]
        incomplete_nopatch = [{"v": r["cuts"][cut_name]["incomplete_nopatch"]} for r in results if cut_name in r["cuts"]]
        summary[f"CRI_{cut_name}"] = aggregate(flip_patch, "flip")
        summary[f"control_{cut_name}"] = aggregate(flip_nopatch, "flip")
        summary[f"incomplete_patch_{cut_name}"] = aggregate(incomplete_patch, "v")
        summary[f"incomplete_nopatch_{cut_name}"] = aggregate(incomplete_nopatch, "v")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
