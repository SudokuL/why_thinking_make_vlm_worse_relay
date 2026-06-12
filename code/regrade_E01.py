"""Re-grade E-01 (VisualSwap) results using common/grading.py instead of
common/metrics.py's normalize/is_correct, which under-counts MC accuracy
(see grading.py docstring + WIKI 2026-06-12). Operates purely on the
already-saved raw1/raw2/raw3_continuation text -- no re-inference needed.

Writes <model>_<subset>_summary_v2.json next to the original summary, and
optionally a regraded per-sample jsonl with --write_jsonl.

Usage:
  python regrade_E01.py --model 2B-Instruct --subset MathVista_MINI
  python regrade_E01.py --all   # regrade every results/E-01/*.jsonl found
"""
import argparse
import glob
import json
import os

from common.data_utils import load_vsbench_pairs
from common.grading import grade, answers_match, canonical_keys, parse_choices
from common.metrics import aggregate

RESULTS_DIR = "/data/liuruida/why_thinking_make_vlm_worse/results/E-01"
PREFILL_PROMPT = "Wait, let me check the figure again to make sure I haven't made a mistake."


def regrade_one(model, subset, write_jsonl=False):
    in_path = os.path.join(RESULTS_DIR, f"{model}_{subset}.jsonl")
    if not os.path.exists(in_path):
        print(f"[skip] {in_path} not found")
        return None

    pairs = load_vsbench_pairs(subset)
    meta = {s["pid"]: s for s in pairs}

    regraded = []
    with open(in_path) as f:
        for line in f:
            r = json.loads(line)
            if "error" in r:
                continue
            pid = r["pid"]
            s = meta.get(pid)
            if s is None:
                continue
            choices = parse_choices(s["query"])
            gt_orig, gt_alt = s["answer_orig"], s["answer"]
            raw1, raw2, cont3 = r["raw1"], r["raw2"], r["raw3_continuation"]
            raw3_combined = raw2 + "\n\n" + PREFILL_PROMPT + cont3

            acc1 = grade(raw1, gt_orig, choices)
            acc2 = grade(raw2, gt_alt, choices)
            acc3 = grade(raw3_combined, gt_orig, choices)
            tir = grade(raw3_combined, gt_alt, choices)
            incomplete3 = canonical_keys(raw3_combined, choices) is None
            updated = (not incomplete3) and answers_match(raw3_combined, raw1, choices)
            inertia = ((not incomplete3) and answers_match(raw3_combined, raw2, choices)
                       and not answers_match(raw1, raw2, choices))

            regraded.append({
                "pid": pid,
                "acc1": bool(acc1), "acc2": bool(acc2), "acc3": bool(acc3), "tir": bool(tir),
                "updated": updated, "inertia": inertia, "incomplete3": incomplete3,
            })

    summary = {}
    for key in ["acc1", "acc2", "acc3", "tir", "updated", "inertia", "incomplete3"]:
        summary[key] = aggregate(regraded, key)
    summary["delta_acc"] = summary["acc1"]["mean"] - summary["acc2"]["mean"]
    summary["model"] = model
    summary["subset"] = subset
    summary["n_total"] = len(regraded)

    out_path = os.path.join(RESULTS_DIR, f"{model}_{subset}_summary_v2.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[{model}/{subset}] " + json.dumps(summary, ensure_ascii=False))

    if write_jsonl:
        out_jsonl = os.path.join(RESULTS_DIR, f"{model}_{subset}_regraded.jsonl")
        with open(out_jsonl, "w") as f:
            for r in regraded:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--subset", default="MathVista_MINI")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--write_jsonl", action="store_true")
    args = ap.parse_args()

    if args.all:
        for path in sorted(glob.glob(os.path.join(RESULTS_DIR, f"*_{args.subset}.jsonl"))):
            model = os.path.basename(path)[: -len(f"_{args.subset}.jsonl")]
            regrade_one(model, args.subset, args.write_jsonl)
    else:
        regrade_one(args.model, args.subset, args.write_jsonl)


if __name__ == "__main__":
    main()
