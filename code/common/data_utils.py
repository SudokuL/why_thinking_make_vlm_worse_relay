"""Data loaders for Block 0 (E-01 VisualSwap reproduction, E-02 specificity controls).

All loaders return plain lists of dicts with PIL.Image objects already decoded,
so run scripts don't need to know about parquet/HF internals.
"""
import io
import os
import pandas as pd
from PIL import Image

DATA_ROOT = "/data/liuruida/why_thinking_make_vlm_worse/datasets"


def _bytes_to_image(d):
    """HF parquet image columns are dicts like {'bytes': b'...', 'path': ...}."""
    if isinstance(d, dict) and d.get("bytes") is not None:
        return Image.open(io.BytesIO(d["bytes"])).convert("RGB")
    if isinstance(d, str) and os.path.exists(d):
        return Image.open(d).convert("RGB")
    raise ValueError(f"Cannot decode image field: {type(d)}")


# ---------------------------------------------------------------------------
# E-01: VS-Bench (VisualSwap) pairs
# ---------------------------------------------------------------------------

# subset -> (alt-image parquet relative to vsbench/, original-dataset parquet
# relative to DATA_ROOT). Only MathVista_MINI's original set (MathVista) is
# downloaded so far (2026-06-11) -> Block 0 starts with this subset; other
# three VS-Bench subsets (MathVerse_MINI, MathVision, MMMU_Pro_10c_COT) need
# their own original datasets downloaded before they can be added (see WIKI).
_VSBENCH_SUBSETS = {
    "MathVista_MINI": {
        "alt": "vsbench/MathVista_MINI/test-00000-of-00001.parquet",
        "orig": "mathvista/data/testmini-00000-of-00001-725687bf7a18d64b.parquet",
    },
}


def load_vsbench_pairs(subset="MathVista_MINI", limit=None):
    """Return list of dicts:
        pid, question (str), query (str, full prompt incl. format hint),
        answer (str, ground truth FOR THE ALT/SWAPPED IMAGE I_a),
        answer_orig (str, ground truth FOR THE ORIGINAL IMAGE I_b),
        orig_image (PIL, I_b), alt_image (PIL, I_a)
    I_b = original image the question was written for.
    I_a = visually-similar-but-semantically-different swap image (VisualSwap).

    NOTE (2026-06-12, see WIKI): `answer` and `answer_orig` are DIFFERENT by
    construction for VisualSwap pairs (same question, different image ->
    different correct answer). acc1 = M(I_b,Q) must be graded against
    `answer_orig`, NOT `answer` -- using `answer` for both stages (the
    original Block 0 bug) makes acc1 spuriously near-zero.
    """
    if subset not in _VSBENCH_SUBSETS:
        raise ValueError(f"Unsupported VS-Bench subset (not downloaded yet): {subset}")
    cfg = _VSBENCH_SUBSETS[subset]
    mod = pd.read_parquet(os.path.join(DATA_ROOT, cfg["alt"]))
    orig = pd.read_parquet(os.path.join(DATA_ROOT, cfg["orig"]))
    orig_by_pid = {row["pid"]: row for _, row in orig.iterrows()}

    samples = []
    for _, row in mod.iterrows():
        pid = row["pid"]
        if pid not in orig_by_pid:
            continue
        orow = orig_by_pid[pid]
        samples.append({
            "pid": pid,
            "question": row["question"],
            "query": row["query"],
            "answer": row["answer"],
            "answer_orig": orow["answer"],
            "orig_image": _bytes_to_image(orow["decoded_image"]),
            "alt_image": _bytes_to_image(row["image"]),
        })
        if limit and len(samples) >= limit:
            break
    return samples


# ---------------------------------------------------------------------------
# E-02: specificity controls
# ---------------------------------------------------------------------------

def load_mmvp(limit=None):
    """MMVP: 150 visually-contrastive pairs (perception, no multi-step reasoning
    needed). Returns pid, question (incl. options), answer, image (PIL).
    """
    csv_path = os.path.join(DATA_ROOT, "mmvp", "Questions.csv")
    img_dir = os.path.join(DATA_ROOT, "mmvp", "MMVP Images")
    df = pd.read_csv(csv_path)
    samples = []
    for _, row in df.iterrows():
        idx = int(row["Index"])
        img_path = os.path.join(img_dir, f"{idx}.jpg")
        query = f"{row['Question']}\n{row['Options']}\nAnswer with the option's letter from the given choices directly."
        samples.append({
            "pid": idx,
            "question": row["Question"],
            "query": query,
            "answer": str(row["Correct Answer"]).strip(),
            "image": Image.open(img_path).convert("RGB"),
        })
        if limit and len(samples) >= limit:
            break
    return samples


# ---------------------------------------------------------------------------
# E-A4: SynthCF (synthetic counterfactual pairs, region selectivity)
# ---------------------------------------------------------------------------

def load_synthcf(limit=None):
    """SynthCF pairs built by build_synthcf.py. Returns list of dicts:
        pid, query, answer1, answer2 (str "1"/"0"),
        target_span, distractor_span ((start,end) image-token offsets within
          the image span -- target_span is the "relevant region", whose
          patch should recover answer1; distractor_span is the "irrelevant
          region", a placebo patch that should be near-inert),
        orig_image (PIL, I_b, target shape is red), alt_image (PIL, I_a,
          target shape is some other color).
    """
    import json
    manifest_path = os.path.join(DATA_ROOT, "synthcf", "manifest.jsonl")
    samples = []
    with open(manifest_path) as f:
        for line in f:
            r = json.loads(line)
            r["orig_image"] = Image.open(r["orig_image"]).convert("RGB")
            r["alt_image"] = Image.open(r["alt_image"]).convert("RGB")
            r["target_span"] = tuple(r["target_span"])
            r["distractor_span"] = tuple(r["distractor_span"])
            samples.append(r)
            if limit and len(samples) >= limit:
                break
    return samples


def load_hallusionbench(visual=True, limit=None):
    """HallusionBench.
    visual=True  -> 'image' subset (951 rows, has an image)
    visual=False -> 'non_image' subset (text-only control, no image)
    gt_answer is '0'/'1' -> map to No/Yes for the prompt.
    """
    fname = "image-00000-of-00001.parquet" if visual else "non_image-00000-of-00001.parquet"
    df = pd.read_parquet(os.path.join(DATA_ROOT, "hallusionbench", "data", fname))
    samples = []
    for _, row in df.iterrows():
        query = f"{row['question']}\nAnswer Yes or No."
        sample = {
            "pid": f"{row['category']}_{row['subcategory']}_{row['set_id']}_{row['figure_id']}_{row['question_id']}",
            "question": row["question"],
            "query": query,
            "answer": "Yes" if str(row["gt_answer"]).strip() == "1" else "No",
        }
        if visual:
            sample["image"] = _bytes_to_image(row["image"])
        samples.append(sample)
        if limit and len(samples) >= limit:
            break
    return samples
