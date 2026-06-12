"""Build SynthCF: a small synthetic counterfactual dataset for E-A4 (region
selectivity).

Each pair (I_a, I_b) is a 392x392 canvas split into a TOP half (y in
[0,196)) and a BOTTOM half (y in [196,392)). At this resolution Qwen3-VL's
processor produces a 12x12 = 144 image-token grid in row-major order, so the
TOP half is exactly image tokens [0,72) and the BOTTOM half is exactly
[72,144) of the image span -- two CONTIGUOUS spans, patchable directly with
patch_image_kv (verified empirically, see WIKI).

One half ("target half", randomly top or bottom) contains exactly ONE shape
whose color is the only difference between I_a and I_b:
  I_b (orig, "clean"): target shape is RED        -> answer1 = "1"
  I_a (alt, counterfactual): target shape is some other color -> answer2 = "0"
Question: "How many red shapes are in the image? Answer with a single
integer (0, 1, 2, ...)."

The other half ("distractor half") contains 1-3 shapes, none red, IDENTICAL
between I_a and I_b -- this is the "irrelevant region" for E-A4's selectivity
test (patch_image_kv on this span should be near-inert; patching the target
half's span should be the one that recovers answer1).

Outputs:
  datasets/synthcf/images/<pid>_alt.png   (I_a)
  datasets/synthcf/images/<pid>_orig.png  (I_b)
  datasets/synthcf/manifest.jsonl

Usage:
  python build_synthcf.py --n 60
"""
import argparse
import json
import os
import random

from PIL import Image, ImageDraw

OUT_DIR = "/data/liuruida/why_thinking_make_vlm_worse/datasets/synthcf"
CANVAS = 392
HALF = CANVAS // 2  # 196

# 12x12 merged-token grid for a 392x392 image (verified: grid_thw=[1,24,24] -> 12x12=144 tokens)
TOKEN_GRID = 12
TOKENS_PER_HALF = (TOKEN_GRID * TOKEN_GRID) // 2  # 72

NON_RED_COLORS = ["blue", "green", "yellow", "purple", "orange", "cyan", "gray", "brown"]
SHAPES = ["circle", "square", "triangle"]

# 2x2 cell layout within a half (each half is CANVAS x HALF px)
CELL_W = CANVAS // 2
CELL_H = HALF // 2
SHAPE_SIZE = 70


def _make_pair(rng, pid):
    target_half = rng.choice(["top", "bottom"])
    target_color_alt = rng.choice(NON_RED_COLORS)
    target_shape = rng.choice(SHAPES)
    target_cell = (rng.randint(0, 1), rng.randint(0, 1))

    n_distractors = rng.randint(1, 3)
    used_cells = set()
    distractors = []
    while len(distractors) < n_distractors:
        cell = (rng.randint(0, 1), rng.randint(0, 1))
        if cell in used_cells:
            continue
        used_cells.add(cell)
        distractors.append({
            "cell": cell,
            "shape": rng.choice(SHAPES),
            "color": rng.choice(NON_RED_COLORS),
        })

    def render(target_color):
        img = Image.new("RGB", (CANVAS, CANVAS), "white")
        draw = ImageDraw.Draw(img)
        # halves: top occupies y in [0,HALF), bottom y in [HALF,CANVAS)
        target_y_off = 0 if target_half == "top" else HALF
        distractor_y_off = HALF if target_half == "top" else 0

        cx, cy = target_cell
        _draw_shape_offset(draw, cx, cy, target_y_off, target_shape, target_color)
        for d in distractors:
            cx, cy = d["cell"]
            _draw_shape_offset(draw, cx, cy, distractor_y_off, d["shape"], d["color"])
        # faint dividing line between halves (visual aid, doesn't affect color count)
        draw.line([(0, HALF), (CANVAS, HALF)], fill="lightgray", width=1)
        return img

    img_b = render("red")          # I_b: clean, target=red -> answer1="1"
    img_a = render(target_color_alt)  # I_a: counterfactual, target!=red -> answer2="0"

    if target_half == "top":
        target_span = (0, TOKENS_PER_HALF)
        distractor_span = (TOKENS_PER_HALF, 2 * TOKENS_PER_HALF)
    else:
        target_span = (TOKENS_PER_HALF, 2 * TOKENS_PER_HALF)
        distractor_span = (0, TOKENS_PER_HALF)

    return img_a, img_b, target_half, target_span, distractor_span


def _draw_shape_offset(draw, cell_x, cell_y, y_off, shape, color):
    cx = cell_x * CELL_W + CELL_W // 2
    cy = y_off + cell_y * CELL_H + CELL_H // 2
    r = SHAPE_SIZE // 2
    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "triangle":
        draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=color)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    img_dir = os.path.join(OUT_DIR, "images")
    os.makedirs(img_dir, exist_ok=True)
    rng = random.Random(args.seed)

    query = ("How many red shapes are in the image? "
             "Think step by step, then give your final answer as a single integer (0, 1, 2, ...).")

    manifest_path = os.path.join(OUT_DIR, "manifest.jsonl")
    with open(manifest_path, "w") as f:
        for i in range(args.n):
            pid = f"synthcf_{i:04d}"
            img_a, img_b, target_half, target_span, distractor_span = _make_pair(rng, pid)
            alt_path = os.path.join(img_dir, f"{pid}_alt.png")
            orig_path = os.path.join(img_dir, f"{pid}_orig.png")
            img_a.save(alt_path)
            img_b.save(orig_path)
            rec = {
                "pid": pid,
                "query": query,
                "answer1": "1",
                "answer2": "0",
                "target_half": target_half,
                "target_span": list(target_span),
                "distractor_span": list(distractor_span),
                "alt_image": alt_path,
                "orig_image": orig_path,
            }
            f.write(json.dumps(rec) + "\n")

    print(f"[Done] wrote {args.n} pairs to {manifest_path}")


if __name__ == "__main__":
    main()
