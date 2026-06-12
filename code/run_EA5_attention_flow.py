"""E-A5: attention-flow descriptive statistics (Block A, lowest priority).

NOT evidence for H1/H2/H3 (per EXPERIMENT_GUIDE.md sec 3: "仅作描述性补充图，
不作为证据") -- purely descriptive: for each model, how much of each
generated reasoning token's attention mass (averaged over heads, a few
representative layers) lands on the image-token span, as a function of
normalized position in the reasoning trace R_a (from E-01)?

Method: single forward pass (no generation) over
prompt(I_a,Q) + R_a[:trunc], output_attentions=True, attn_implementation
"eager" (required for output_attentions). To bound memory we (a) truncate
R_a to TRUNC_TOKENS, (b) only keep attentions for LAYERS_OF_INTEREST
(first/middle/last), discarding the rest immediately, (c) free CUDA cache
between samples.

For each kept layer, for each reasoning-token query position q (0-indexed
within R_a[:trunc]), compute:
  img_attn_frac(q) = sum_k attn[q, k] over k in image_span / sum_k attn[q,k] over all k <= q
averaged over attention heads. Reasoning positions are then bucketed into
10 deciles of normalized trace position and averaged -> one curve per
(model, layer).

Outputs results/E-A5/<model>_<subset>.jsonl (per-sample decile curves, one
record per layer) + _summary.json (mean curve per layer across samples).

Usage:
  python run_EA5_attention_flow.py --model 8B-Instruct --subset MathVista_MINI --limit 15
"""
import argparse
import json
import os
import time

import torch
from qwen_vl_utils import process_vision_info

from common.model_utils import load_model_and_processor, build_user_message, find_image_token_spans, IMAGE_TOKEN_ID
from common.data_utils import load_vsbench_pairs

E01_DIR = "/data/liuruida/why_thinking_make_vlm_worse/results/E-01"
RESULTS_DIR = "/data/liuruida/why_thinking_make_vlm_worse/results/E-A5"

TRUNC_TOKENS = 400      # cap on R_a length used for the forward pass (memory)
N_DECILES = 10
LAYER_FRACS = {"early": 0.1, "mid": 0.5, "late": 0.9}  # which transformer layers to inspect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--subset", default="MathVista_MINI")
    ap.add_argument("--limit", type=int, default=15,
                     help="cap number of samples processed (descriptive only -- small N is fine)")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_short = args.model.replace("/", "_")

    e01_path = os.path.join(E01_DIR, f"{model_short}_{args.subset}.jsonl")
    if not os.path.exists(e01_path):
        raise FileNotFoundError(f"E-01 results not found: {e01_path} -- run run_E01_visualswap.py first")

    e01_records = []
    with open(e01_path) as f:
        for line in f:
            r = json.loads(line)
            if "error" not in r and r.get("raw2"):
                e01_records.append(r)
    print(f"[Load] {len(e01_records)} E-01 records from {e01_path}")

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
    model.config._attn_implementation = "eager"
    for m in model.modules():
        if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
            m.config._attn_implementation = "eager"

    n_layers = model.config.text_config.num_hidden_layers if hasattr(model.config, "text_config") else model.config.num_hidden_layers
    layer_idxs = {name: max(0, min(n_layers - 1, int(n_layers * frac))) for name, frac in LAYER_FRACS.items()}
    print(f"[Layers] n_layers={n_layers} -> {layer_idxs}")

    samples = load_vsbench_pairs(subset=args.subset)
    samples_by_pid = {str(s["pid"]): s for s in samples}

    fout = open(out_path, "a")
    n_processed = 0
    for rec in e01_records:
        pid = str(rec["pid"])
        if pid not in samples_by_pid:
            continue
        if pid in done_pids:
            n_processed += 1
            continue
        if args.limit and n_processed >= args.limit:
            break

        s = samples_by_pid[pid]
        t0 = time.time()
        try:
            msg_a = build_user_message(s["alt_image"], s["query"])
            text = processor.apply_chat_template(msg_a, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(msg_a)
            base = processor(text=[text], images=image_inputs, return_tensors="pt").to(model.device)
            prompt_len = base["input_ids"].shape[1]

            gen_ids = processor.tokenizer(rec["raw2"], return_tensors="pt", add_special_tokens=False)["input_ids"]
            n_gen_full = gen_ids.shape[1]
            if n_gen_full < 5:
                n_processed += 1
                continue
            gen_ids = gen_ids[:, :TRUNC_TOKENS]
            n_gen = gen_ids.shape[1]

            full_input_ids = torch.cat([base["input_ids"], gen_ids.to(model.device)], dim=1)
            attn_mask = torch.ones_like(full_input_ids)
            fwd_kwargs = {k: v for k, v in base.items() if k not in ("input_ids", "attention_mask")}

            img_spans = find_image_token_spans(full_input_ids[0], IMAGE_TOKEN_ID)
            img_start, img_end = img_spans[0]

            with torch.inference_mode():
                out = model(input_ids=full_input_ids, attention_mask=attn_mask,
                             output_attentions=True, use_cache=False, return_dict=True, **fwd_kwargs)

            curves = {}
            for layer_name, layer_idx in layer_idxs.items():
                attn = out.attentions[layer_idx][0]  # [n_heads, seq, seq]
                attn = attn.float().mean(dim=0)      # [seq, seq] averaged over heads
                deciles = [[] for _ in range(N_DECILES)]
                for q in range(prompt_len, prompt_len + n_gen):
                    row = attn[q, :q + 1]
                    img_mass = row[img_start:img_end].sum().item()
                    total_mass = row.sum().item()
                    frac = img_mass / total_mass if total_mass > 0 else 0.0
                    rel_pos = (q - prompt_len) / max(1, n_gen - 1)
                    decile = min(N_DECILES - 1, int(rel_pos * N_DECILES))
                    deciles[decile].append(frac)
                curves[layer_name] = [sum(d) / len(d) if d else None for d in deciles]
            del out
            torch.cuda.empty_cache()

            record = {"pid": pid, "n_gen_full": n_gen_full, "n_gen_used": n_gen,
                      "img_span_len": img_end - img_start, "curves": curves}
        except Exception as e:
            record = {"pid": pid, "error": str(e)}
            torch.cuda.empty_cache()

        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()
        n_processed += 1
        dt = time.time() - t0
        if "error" in record:
            print(f"[{n_processed}] pid={pid} dt={dt:.1f}s ERR {record['error']}")
        else:
            print(f"[{n_processed}] pid={pid} dt={dt:.1f}s n_gen={record['n_gen_used']}")

    fout.close()

    results = []
    with open(out_path) as f:
        for line in f:
            r = json.loads(line)
            if "error" not in r:
                results.append(r)

    summary = {"model": args.model, "subset": args.subset, "n_done": len(results)}
    for layer_name in layer_idxs:
        deciles_agg = [[] for _ in range(N_DECILES)]
        for r in results:
            for d, v in enumerate(r["curves"][layer_name]):
                if v is not None:
                    deciles_agg[d].append(v)
        summary[f"img_attn_frac_{layer_name}"] = [
            (sum(vals) / len(vals) if vals else None) for vals in deciles_agg
        ]

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
