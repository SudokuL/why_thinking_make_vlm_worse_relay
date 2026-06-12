"""E-01: VisualSwap three-stage protocol reproduction (Block 0).

For each (I_b=original image, I_a=swap image, Q) triple:
  exp1: M(I_b, Q)                         -> answer1
  exp2: M(I_a, Q)                         -> R_a (reasoning trace), answer2
  exp3: prefill(R_a + reflection prompt) with image swapped BACK to I_b,
        continue decoding                 -> answer3

Outputs results/E-01/<model>_<subset>.jsonl (one record per sample, resumable)
and results/E-01/<model>_<subset>_summary.json (aggregate metrics).

Usage:
  python run_E01_visualswap.py --model 8B-Instruct --subset MathVista_MINI \
      --limit 20 --max_new_tokens 2048
"""
import argparse
import json
import os
import time

from common.model_utils import (
    load_model_and_processor, build_user_message, generate, continue_generate,
)
from common.data_utils import load_vsbench_pairs
from common.metrics import aggregate
from common.grading import grade, answers_match, canonical_keys, parse_choices

PREFILL_PROMPT = "Wait, let me check the figure again to make sure I haven't made a mistake."

RESULTS_DIR = "/data/liuruida/why_thinking_make_vlm_worse/results/E-01"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="registry key (e.g. 8B-Instruct) or path")
    ap.add_argument("--subset", default="MathVista_MINI")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_short = args.model.replace("/", "_")
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

    samples = load_vsbench_pairs(subset=args.subset, limit=args.limit)
    print(f"[Data] {len(samples)} pairs from {args.subset}")

    fout = open(out_path, "a")
    for i, s in enumerate(samples):
        if s["pid"] in done_pids:
            continue
        t0 = time.time()
        try:
            # exp1: M(I_b, Q)
            msg1 = build_user_message(s["orig_image"], s["query"])
            raw1, _ = generate(model, processor, msg1, max_new_tokens=args.max_new_tokens)

            # exp2: M(I_a, Q) -> R_a
            msg2 = build_user_message(s["alt_image"], s["query"])
            raw2, _ = generate(model, processor, msg2, max_new_tokens=args.max_new_tokens)

            # exp3: image swapped back to I_b, prefill R_a + reflection, continue
            msg3 = build_user_message(s["orig_image"], s["query"])
            msg3 = msg3 + [{"role": "assistant", "content": raw2 + "\n\n" + PREFILL_PROMPT}]
            cont3, _ = continue_generate(model, processor, msg3, max_new_tokens=args.max_new_tokens)
            raw3_combined = raw2 + "\n\n" + PREFILL_PROMPT + cont3

            # gt_orig = correct answer for I_b (orig image, stage1 + stage3
            # target); gt_alt = correct answer for I_a (alt image, stage2
            # target). These DIFFER by construction for VisualSwap pairs --
            # see data_utils.load_vsbench_pairs / WIKI 2026-06-12.
            choices = parse_choices(s["query"])
            gt_orig, gt_alt = s["answer_orig"], s["answer"]

            acc1 = grade(raw1, gt_orig, choices)
            acc2 = grade(raw2, gt_alt, choices)
            acc3 = grade(raw3_combined, gt_orig, choices)  # VUA: switched back to I_b's answer
            tir = grade(raw3_combined, gt_alt, choices)    # TIR: stuck on I_a's answer
            incomplete3 = canonical_keys(raw3_combined, choices) is None
            updated = (not incomplete3) and answers_match(raw3_combined, raw1, choices)
            inertia = ((not incomplete3) and answers_match(raw3_combined, raw2, choices)
                       and not answers_match(raw1, raw2, choices))

            record = {
                "pid": s["pid"], "gt_orig": gt_orig, "gt_alt": gt_alt,
                "raw1": raw1, "raw2": raw2, "raw3_continuation": cont3,
                "acc1": bool(acc1), "acc2": bool(acc2), "acc3": bool(acc3), "tir": bool(tir),
                "updated": updated, "inertia": inertia, "incomplete3": incomplete3,
            }
        except Exception as e:
            record = {"pid": s["pid"], "error": str(e)}

        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()
        dt = time.time() - t0
        print(f"[{i+1}/{len(samples)}] pid={s['pid']} dt={dt:.1f}s "
              f"{'ERR' if 'error' in record else record}")

    fout.close()

    # summary
    results = []
    with open(out_path) as f:
        for line in f:
            r = json.loads(line)
            if "error" not in r:
                results.append(r)
    summary = {}
    for key in ["acc1", "acc2", "acc3", "tir", "updated", "inertia", "incomplete3"]:
        summary[key] = aggregate(results, key)
    summary["delta_acc"] = summary["acc1"]["mean"] - summary["acc2"]["mean"]
    summary["model"] = args.model
    summary["subset"] = args.subset
    summary["n_total"] = len(results)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
