"""Improved answer grading for MathVista-style multiple-choice + numeric
questions, used to fix a systematic acc1/acc2 under-counting bug in
common/metrics.py.

Bug (found 2026-06-12, see WIKI.md): MathVista_MINI's `query` embeds the
choices as "(A) <text>\\n(B) <text>...", and `answer` (gt) is the CHOICE
TEXT (e.g. "Yes", "Diffusion"), not the letter. Models frequently answer
with the LETTER ("A", "(B) No", "The correct answer is (A) MusicLDM
(mix-up)"). common.metrics.normalize() collapses "(B) no" to the letter
"B" (via _OPTION_PREFIX_RE) and compares it against gt "no" -> never
matches even when the model picked the choice matching gt. Result:
acc1 measured 0.02 for 2B-Instruct on MathVista_MINI (vs ~30-50% expected).

This module re-grades from the raw generation text (raw1/raw2/raw3_*
saved in results/E-01/*.jsonl) using the question's parsed choices, so no
re-inference is needed.
"""
import re

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_CHOICE_LINE_RE = re.compile(r"^\(([A-Za-z])\)\s*(.+?)\s*$", re.MULTILINE)
_LETTER_PAREN_RE = re.compile(r"\(([A-Za-z])\)")
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_FINAL_PHRASE_RE = re.compile(
    r"(?:final answer|correct answer|the answer|answer)\s*(?:is|:)\s*[:\-]?\s*(.+)",
    re.IGNORECASE,
)
_LEADING_LETTER_MARKER_RE = re.compile(r"^\(?([A-Ja-j])\)?[\.\:\)]\s*")
_BARE_LETTER_RE = re.compile(r"^([A-Ja-j])[\.\):]?$")


def parse_choices(query):
    """Parse '(A) Yes\\n(B) No' lines from a VS-Bench query string into
    {'A': 'Yes', 'B': 'No'}. Returns {} if the query has no MC choices
    (free-form numeric/text questions)."""
    choices = {}
    for m in _CHOICE_LINE_RE.finditer(query):
        letter = m.group(1).upper()
        if letter in "ABCDEFGHIJ":
            choices[letter] = m.group(2).strip()
    return choices


def _norm_text(s):
    s = s.lower().strip()
    s = re.sub(r"[^\w.\-]", "", s)
    return s


def strip_thinking(text):
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip(), True
    if "<think>" in text:
        return text.strip(), False
    return text.strip(), True


def _candidate_strings(text):
    cands = []
    m = _BOXED_RE.findall(text)
    if m:
        cands.append(m[-1])
    for m2 in _FINAL_PHRASE_RE.finditer(text):
        cands.append(m2.group(1))
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        cands.append(lines[-1])
    return [c.strip(" .。*\n") for c in cands if c.strip()]


def canonical_keys(raw_text, choices=None):
    """Set of normalized strings representing the model's final answer,
    or None if the generation was truncated mid-<think> (no final answer
    emitted). Letters are expanded to the corresponding choice text (and
    vice versa is implicit via the letter itself) so 'A' and the text of
    choice A are treated as the same answer."""
    text, complete = strip_thinking(raw_text)
    if not complete:
        return None
    keys = set()
    for c in _candidate_strings(text):
        for lm in _LETTER_PAREN_RE.finditer(c):
            letter = lm.group(1).upper()
            keys.add(letter.lower())
            if choices and letter in choices:
                keys.add(_norm_text(choices[letter]))
        bm = _BARE_LETTER_RE.match(c)
        if bm:
            letter = bm.group(1).upper()
            keys.add(letter.lower())
            if choices and letter in choices:
                keys.add(_norm_text(choices[letter]))
        c2 = _LEADING_LETTER_MARKER_RE.sub("", c)
        for cand in (c, c2):
            norm = _norm_text(cand)
            if norm:
                keys.add(norm)
            for n in _NUM_RE.findall(cand):
                keys.add(n.replace(",", ""))
    return keys


def answers_match(raw_a, raw_b, choices=None):
    ka, kb = canonical_keys(raw_a, choices), canonical_keys(raw_b, choices)
    if ka is None or kb is None:
        return False
    return bool(ka & kb)


def grade(raw_text, gt, choices=None):
    """True/False/None (None = incomplete generation, no final answer)."""
    keys = canonical_keys(raw_text, choices)
    if keys is None:
        return None
    acceptable = {_norm_text(gt)}
    for n in _NUM_RE.findall(gt):
        acceptable.add(n.replace(",", ""))
    if choices:
        for letter, txt in choices.items():
            if _norm_text(txt) == _norm_text(gt):
                acceptable.add(letter.lower())
                acceptable.add(_norm_text(txt))
    return bool(keys & acceptable)
