"""E-02: Specificity controls (Block 0).

Runs single-turn accuracy on tasks that do NOT involve mid-reasoning image
swaps, to check whether any Thinking-vs-Instruct gap is specific to the
"update visual belief mid-reasoning" setting (E-01) or just a general
visual/perception weakness of Thinking models.

Tasks:
  mmvp        -- 150 visually-contrastive perception pairs (single image)
  hallu_vis   -- HallusionBench 'image' subset (visual yes/no)
  hallu_text  -- HallusionBench 'non_image' subset (text-only control,
                 same question style, no image -> isolates visual contribution)

Outputs results/E-02/<model>_<task>.jsonl + _summary.json (resumable).

Usage:
  python run_E02_specificity.py --model 8B-Instruct --task mmvp --limit 30
"""
import argparse
import json
import os
import time

from common.model_utils import load_model_and_processor, build_user_message, generate
from common.data_utils import load_mmvp, load_hallusionbench
from common.metrics import extract_answer, is_correct, aggregate

RESULTS_DIR = "/data/liuruida/why_thinking_make_vlm_worse/results/E-02"

LOADERS = {
    "mmvp": lambda limit: load_mmvp(limit=limit),
    "hallu_vis": lambda limit: load_hallusionbench(visual=True, limit=limit),
    "hallu_text": lambda limit: load_hallusionbench(visual=False, limit=limit),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=list(LOADERS.keys()))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_short = args.model.replace("/", "_")
    out_path = os.path.join(RESULTS_DIR, f"{model_short}_{args.task}.jsonl")
    summary_path = os.path.join(RESULTS_DIR, f"{model_short}_{args.task}_summary.json")

    done_pids = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    done_pids.add(json.loads(line)["pid"])
        print(f"[Resume] {len(done_pids)} samples already done, skipping.")

    print(f"[Load] {args.model}")
    model, processor = load_model_and_processor(args.model)

    samples = LOADERS[args.task](args.limit)
    print(f"[Data] {len(samples)} samples for task={args.task}")

    fout = open(out_path, "a")
    for i, s in enumerate(samples):
        pid = str(s["pid"])
        if pid in done_pids:
            continue
        t0 = time.time()
        try:
            if "image" in s:
                msg = build_user_message(s["image"], s["query"])
            else:
                msg = [{"role": "user", "content": [{"type": "text", "text": s["query"]}]}]
            raw, _ = generate(model, processor, msg, max_new_tokens=args.max_new_tokens)
            ans = extract_answer(raw)
            record = {
                "pid": pid, "gt": s["answer"], "answer": ans, "raw": raw,
                "correct": is_correct(ans, s["answer"]),
            }
        except Exception as e:
            record = {"pid": pid, "error": str(e)}

        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()
        dt = time.time() - t0
        print(f"[{i+1}/{len(samples)}] pid={pid} dt={dt:.1f}s "
              f"{'ERR' if 'error' in record else record}")

    fout.close()

    results = []
    with open(out_path) as f:
        for line in f:
            r = json.loads(line)
            if "error" not in r:
                results.append(r)
    summary = aggregate(results, "correct")
    summary["model"] = args.model
    summary["task"] = args.task
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
