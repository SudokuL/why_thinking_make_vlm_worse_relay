"""Block B data prep: split LLaVA-CoT-100k's train.jsonl into a CoT-SFT
variant (used as-is) and a direct-SFT variant (CoT stripped to just the
final answer), per EXPERIMENT_GUIDE.md E-B1 ("LLaVA-CoT-100k 剥 CoT").

Each gpt turn in train.jsonl looks like:
    "<SUMMARY> ... </SUMMARY>\\n\\n<CAPTION> ... </CAPTION>\\n\\n"
    "<REASONING> ... </REASONING>\\n\\n<CONCLUSION> ... </CONCLUSION>"
(99.98% of the 254,927 gpt turns have a <CONCLUSION> tag; the remaining ~56
are already short direct answers and are passed through unchanged for both
variants.)

  CoT-SFT    : gpt turn = full text (all 4 tags), unchanged.
  direct-SFT : gpt turn = just the <CONCLUSION>...</CONCLUSION> content
               (tags stripped), i.e. the same Q -> short final-answer pairs
               with all CoT scaffolding removed. Same images/questions/order
               as CoT-SFT -> a natural matched pair for E-B1's "same data,
               steps, lr, batch" requirement.

Outputs:
  datasets/llava_cot_100k/sft_cotsft.jsonl
  datasets/llava_cot_100k/sft_directsft.jsonl
(same {"id","image","conversations"} schema as train.jsonl; image paths are
relative to datasets/llava_cot_100k/images/, unzipped from image.zip.* by
the caller.)

Usage:
  python build_blockb_sft_data.py
"""
import json
import os
import re

DATA_DIR = "/data/liuruida/why_thinking_make_vlm_worse/datasets/llava_cot_100k"
SRC = os.path.join(DATA_DIR, "train.jsonl")

_CONCLUSION_RE = re.compile(r"<CONCLUSION>(.*?)</CONCLUSION>", re.DOTALL)


def strip_to_conclusion(text):
    m = _CONCLUSION_RE.search(text)
    if m is None:
        return text.strip()
    return m.group(1).strip()


def main():
    cot_path = os.path.join(DATA_DIR, "sft_cotsft.jsonl")
    direct_path = os.path.join(DATA_DIR, "sft_directsft.jsonl")

    n = 0
    with open(SRC) as fin, open(cot_path, "w") as f_cot, open(direct_path, "w") as f_direct:
        for line in fin:
            r = json.loads(line)
            cot_conv = r["conversations"]
            direct_conv = []
            for c in cot_conv:
                if c["from"] == "gpt":
                    direct_conv.append({"from": "gpt", "value": strip_to_conclusion(c["value"])})
                else:
                    direct_conv.append(c)

            f_cot.write(json.dumps({"id": r["id"], "image": r["image"], "conversations": cot_conv},
                                    ensure_ascii=False) + "\n")
            f_direct.write(json.dumps({"id": r["id"], "image": r["image"], "conversations": direct_conv},
                                       ensure_ascii=False) + "\n")
            n += 1

    print(f"[Done] wrote {n} samples to {cot_path} and {direct_path}")


if __name__ == "__main__":
    main()
