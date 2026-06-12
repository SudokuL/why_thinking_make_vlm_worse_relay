"""Cross-model aggregation for Block A (E-A1 CRI(t) curve, E-A2 pathway
separation, E-A3 trigger comparison, E-A4 SynthCF region selectivity, E-A5
attention-flow descriptive stats).

Reads results/E-A{1,2,3,4,5}/<model>_<subset>_summary.json (E-A4 has no
<subset> in its filename) for all models that have finished, and prints
markdown tables for WIKI.md Sec.2.

Usage:
  python analyze_blockA.py [--subset MathVista_MINI]
"""
import argparse
import json
import os

RESULTS_DIR = "/data/liuruida/why_thinking_make_vlm_worse/results"

MODELS = ["2B-Instruct", "2B-Thinking", "4B-Instruct", "4B-Thinking", "8B-Instruct", "8B-Thinking"]
CUTS = ["early", "mid", "late"]


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _fmt(stat):
    if stat is None:
        return "—"
    if isinstance(stat, dict) and "mean" in stat:
        return f"{stat['mean']:.3f} (n={stat['n']})"
    return str(stat)


def _diff(a, b):
    if a is None or b is None:
        return "—"
    return f"{a['mean'] - b['mean']:+.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="MathVista_MINI")
    args = ap.parse_args()
    sub = args.subset

    # --- E-A1: CRI(t) curve ---
    print(f"## E-A1 CRI(t) curve ({sub})\n")
    print("| model | n | CRI_early | CRI_mid | CRI_late | control_early | control_mid | control_late |")
    print("|---|---|---|---|---|---|---|---|")
    for m in MODELS:
        s = _load(os.path.join(RESULTS_DIR, "E-A1", f"{m}_{sub}_summary.json"))
        if s is None:
            print(f"| {m} | — | — | — | — | — | — | — |")
            continue
        row = [m, str(s["n_flippable_done"])]
        for c in CUTS:
            row.append(_fmt(s.get(f"CRI_{c}")))
        for c in CUTS:
            row.append(_fmt(s.get(f"control_{c}")))
        print("| " + " | ".join(row) + " |")

    # --- E-A2: pathway separation ---
    print(f"\n## E-A2 Pathway separation ({sub})\n")
    print("| model | n | CRI_IMG_early | CRI_IMG_mid | CRI_IMG_late | CRI_TXT_early | CRI_TXT_mid | CRI_TXT_late | control_late |")
    print("|---|---|---|---|---|---|---|---|---|")
    for m in MODELS:
        s = _load(os.path.join(RESULTS_DIR, "E-A2", f"{m}_{sub}_summary.json"))
        if s is None:
            print(f"| {m} | " + " | ".join(["—"] * 8) + " |")
            continue
        row = [m, str(s["n_flippable_done"])]
        for c in CUTS:
            row.append(_fmt(s.get(f"CRI_IMG_{c}")))
        for c in CUTS:
            row.append(_fmt(s.get(f"CRI_TXT_{c}")))
        row.append(_fmt(s.get("control_late")))
        print("| " + " | ".join(row) + " |")

    # --- E-A3: trigger comparison ---
    print(f"\n## E-A3 Trigger comparison ({sub})\n")
    print("| model | n | CRI_T1 (self-reflect) | CRI_T2 (new turn, no img) | CRI_T3 (new turn + img) | gate(T2-T1) | gate(T3-T1) |")
    print("|---|---|---|---|---|---|---|")
    for m in MODELS:
        s = _load(os.path.join(RESULTS_DIR, "E-A3", f"{m}_{sub}_summary.json"))
        if s is None:
            print(f"| {m} | " + " | ".join(["—"] * 6) + " |")
            continue
        t1, t2, t3 = s.get("flip_t1"), s.get("flip_t2"), s.get("flip_t3")
        row = [m, str(s["n_flippable_done"]), _fmt(t1), _fmt(t2), _fmt(t3), _diff(t2, t1), _diff(t3, t1)]
        print("| " + " | ".join(row) + " |")

    # --- E-A4: SynthCF region selectivity ---
    print(f"\n## E-A4 SynthCF region selectivity\n")
    print("| model | n | SEL_early (REL-IRR) | SEL_mid | SEL_late | CRI_REL_late | CRI_IRR_late | control_late |")
    print("|---|---|---|---|---|---|---|---|")
    for m in MODELS:
        s = _load(os.path.join(RESULTS_DIR, "E-A4", f"{m}_summary.json"))
        if s is None:
            print(f"| {m} | " + " | ".join(["—"] * 7) + " |")
            continue
        row = [m, str(s["n_flippable_done"])]
        for c in CUTS:
            row.append(_diff(s.get(f"CRI_REL_{c}"), s.get(f"CRI_IRR_{c}")))
        row.append(_fmt(s.get("CRI_REL_late")))
        row.append(_fmt(s.get("CRI_IRR_late")))
        row.append(_fmt(s.get("control_late")))
        print("| " + " | ".join(row) + " |")

    # --- E-A5: attention-flow descriptive stats (not evidence) ---
    print(f"\n## E-A5 Attention-flow descriptive stats ({sub}, NOT evidence)\n")
    for m in MODELS:
        s = _load(os.path.join(RESULTS_DIR, "E-A5", f"{m}_{sub}_summary.json"))
        if s is None:
            continue
        print(f"- **{m}** (n={s['n_done']}):")
        for k, v in s.items():
            if k.startswith("img_attn_frac_"):
                vals = ", ".join(f"{x:.3f}" if x is not None else "—" for x in v)
                print(f"  - {k}: [{vals}]")


if __name__ == "__main__":
    main()
