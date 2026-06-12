"""Lightweight answer extraction + VisualSwap-style metrics for Block 0.

NOTE: this is a *heuristic* extractor (regex-based), not the LLM-judge used by
the official VS-Bench run_evaluation.py (which calls a 235B judge model -- too
heavy for our 4090). It is good enough for the core re-examination signal we
care about: whether the model's extracted answer changes across the three
VisualSwap stages (A1/A2/A3 below). Absolute accuracy numbers from this
extractor should be treated as approximate; if results are borderline,
upgrade to LLM-judge grading before drawing conclusions (see WIKI).
"""
import re
import random

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_FINAL_RE = re.compile(
    r"(?:final answer|the answer is|answer:)\s*[:\-]?\s*\**\(?([A-Za-z0-9.\-/ ]{1,30}?)\)?\**\s*[\.\n]",
    re.IGNORECASE,
)
_OPTION_RE = re.compile(r"^\(?([A-Ja-j])\)?$")
# leading option letter followed by punctuation/space + trailing description,
# e.g. "(a) Open" or "a) Open the door" -> "A"
_OPTION_PREFIX_RE = re.compile(r"^\(?([A-Ja-j])\)?[\.\:\)]?\s+")


def strip_thinking(text):
    """Drop the <think>...</think> block, keep only the final-answer portion.
    If </think> never appears (truncated thinking trace), return the full
    text but mark it as incomplete via a leading flag the caller can check.
    """
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip(), True
    if "<think>" in text:
        return text.strip(), False  # truncated -- no final answer emitted
    return text.strip(), True


def extract_answer(raw_text):
    """Best-effort extraction of the model's final answer string."""
    text, complete = strip_thinking(raw_text)
    if not complete:
        return None

    m = _BOXED_RE.findall(text)
    if m:
        return m[-1].strip()

    m = _FINAL_RE.findall(text)
    if m:
        return m[-1].strip()

    # fall back: last non-empty line, strip trailing punctuation
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return None
    last = lines[-1].strip(" .。*")
    return last


def normalize(ans):
    if ans is None:
        return None
    a = ans.strip().strip(".。*").strip()
    a = a.strip("()[]")
    om = _OPTION_RE.match(a)
    if om:
        return om.group(1).upper()
    pm = _OPTION_PREFIX_RE.match(a)
    if pm:
        return pm.group(1).upper()
    # try numeric normalization
    try:
        f = float(a.replace(",", ""))
        if f == int(f):
            return str(int(f))
        return f"{f:g}"
    except ValueError:
        pass
    return a.lower()


def answers_equal(a, b):
    na, nb = normalize(a), normalize(b)
    if na is None or nb is None:
        return False
    return na == nb


def is_correct(extracted, gt):
    return answers_equal(extracted, gt)


# ---------------------------------------------------------------------------
# VisualSwap-style aggregate metrics
# ---------------------------------------------------------------------------

def visualswap_outcome(answer1, answer2, answer3, gt):
    """Classify one sample's three-stage outcome.

    answer1 = M(I_b, Q)      -- on original image
    answer2 = M(I_a, Q)      -- on swapped image (produces reasoning R_a)
    answer3 = continuation after reflection prompt + swap BACK to I_b

    Returns a dict of booleans:
      acc1, acc2          : stage-1 / stage-2 correctness vs gt
      updated             : answer3 matches answer1 (re-examination succeeded,
                            model "noticed" the swap back to I_b)
      inertia             : answer3 matches answer2 and != answer1 (textual
                            inertia -- ignored the re-shown original image)
      incomplete3         : stage-3 generation never reached a final answer
                            (thinking trace truncated by max_new_tokens)
    """
    incomplete3 = answer3 is None
    updated = (not incomplete3) and answers_equal(answer3, answer1)
    inertia = (not incomplete3) and answers_equal(answer3, answer2) and not answers_equal(answer1, answer2)
    return {
        "acc1": is_correct(answer1, gt),
        "acc2": is_correct(answer2, gt),
        "updated": updated,
        "inertia": inertia,
        "incomplete3": incomplete3,
    }


def aggregate(results, key, n_boot=2000, seed=0):
    """Bootstrap mean + 95% CI for a boolean field across a list of per-sample
    result dicts (as produced by visualswap_outcome, merged with the sample).
    """
    vals = [1.0 if r[key] else 0.0 for r in results]
    if not vals:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0}
    mean = sum(vals) / len(vals)
    rng = random.Random(seed)
    n = len(vals)
    boot_means = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo = boot_means[int(0.025 * n_boot)]
    hi = boot_means[int(0.975 * n_boot) - 1]
    return {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": n}


def summarize_visualswap(results):
    """results: list of per-sample dicts each containing the visualswap_outcome
    keys (acc1, acc2, updated, inertia, incomplete3)."""
    out = {}
    for key in ["acc1", "acc2", "updated", "inertia", "incomplete3"]:
        out[key] = aggregate(results, key)
    # delta = stage1 acc - stage2 acc, the headline VisualSwap "vulnerability" number
    n = len(results)
    if n:
        out["delta_acc"] = out["acc1"]["mean"] - out["acc2"]["mean"]
    else:
        out["delta_acc"] = float("nan")
    return out
