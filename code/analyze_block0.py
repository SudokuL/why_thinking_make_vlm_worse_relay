"""Cross-model aggregation for Block 0 (E-01 VisualSwap + E-02 specificity).

E-01 numbers come from <model>_<subset>_summary_v2.json, produced by
regrade_E01.py (see WIKI 2026-06-12: the live summary.json from
run_E01_visualswap.py BEFORE that fix used a single shared ground truth for
both stages and an MC-letter-blind extractor, making acc1 spuriously near 0
-- run `python regrade_E01.py --all` first for any model not yet on the
fixed run_E01_visualswap.py). Models run AFTER the fix write correct
acc1/acc2/acc3/tir directly into summary.json too, but summary_v2.json is
always the canonical source for the WIKI ledger.

results/E-02/<model>_<task>_summary.json for all models that have finished,
and prints a markdown table for WIKI.md Sec.2 (假设-实验台账).

Usage:
  python analyze_block0.py [--subset MathVista_MINI]
"""
import argparse
import json
import os

RESULTS_DIR = "/home/liuruida/from84_relay/why_thinking_make_vlm_worse_relay/results"

MODELS = ["2B-Instruct", "2B-Thinking", "4B-Instruct", "4B-Thinking", "8B-Instruct", "8B-Thinking"]
E02_TASKS = ["mmvp", "hallu_vis", "hallu_text"]


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _fmt(stat):
    if stat is None:
        return "—"
    return f"{stat['mean']:.3f} [{stat['ci_lo']:.3f},{stat['ci_hi']:.3f}] (n={stat['n']})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="MathVista_MINI")
    args = ap.parse_args()

    print(f"## E-01 VisualSwap ({args.subset})\n")
    print("| model | acc1 (I_b) | acc2 (I_a) | delta_acc | acc3/VUA | tir | self-updated | self-inertia | n |")
    print("|---|---|---|---|---|---|---|---|---|")
    for m in MODELS:
        s = _load(os.path.join(RESULTS_DIR, "E-01", f"{m}_{args.subset}_summary_v2.json"))
        if s is None:
            print(f"| {m} | — | — | — | — | — | — | — | — |")
            continue
        print(f"| {m} | {_fmt(s['acc1'])} | {_fmt(s['acc2'])} | {s['delta_acc']:.3f} | "
              f"{_fmt(s['acc3'])} | {_fmt(s['tir'])} | {_fmt(s['updated'])} | {_fmt(s['inertia'])} | {s['n_total']} |")

    print(f"\n## E-02 Specificity\n")
    print("| model | " + " | ".join(E02_TASKS) + " |")
    print("|---|" + "---|" * len(E02_TASKS))
    for m in MODELS:
        row = [m]
        for task in E02_TASKS:
            s = _load(os.path.join(RESULTS_DIR, "E-02", f"{m}_{task}_summary.json"))
            row.append(_fmt(s) if s else "—")
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
