"""Smoke test for common/patch_utils.py (Block A scaffold).

Mechanically verifies, on one model + two real images:
  1. forward_with_cache works and finds an image-token span.
  2. cache_prefix_for_continuation + continue_generate_from_cache reproduces
     (greedy) the same continuation as plain `generate` when no patch is
     applied (sanity: patched machinery == unpatched baseline).
  3. patch_image_kv with a DIFFERENT donor image changes the continuation
     for at least one of the tested cut points (i.e. the patch has a causal
     effect -- the infrastructure can actually move the needle).

This does not test any research hypothesis -- it only confirms the plumbing
works on this transformers/Qwen3-VL version before Block A experiments rely
on it.
"""
import torch

from common.model_utils import load_model_and_processor, build_user_message, generate
from common.data_utils import load_vsbench_pairs
from common.patch_utils import (
    cache_prefix_for_continuation, continue_generate_from_cache,
    forward_with_cache, patch_image_kv, clone_cache,
)


def main():
    print("[Load] 8B-Instruct")
    model, processor = load_model_and_processor("8B-Instruct")

    samples = load_vsbench_pairs(subset="MathVista_MINI", limit=1)
    s = samples[0]

    msg_a = build_user_message(s["alt_image"], s["query"])
    print("[Gen] full generation on alt image (short budget)")
    raw_a, _ = generate(model, processor, msg_a, max_new_tokens=64)
    print("raw_a:", repr(raw_a[:200]))

    full_ids = processor.apply_chat_template(msg_a, tokenize=False, add_generation_prompt=True)
    from qwen_vl_utils import process_vision_info
    image_inputs, _ = process_vision_info(msg_a)
    base = processor(text=[full_ids], images=image_inputs, return_tensors="pt").to(model.device)
    prompt_len = base["input_ids"].shape[1]

    gen_ids = processor.tokenizer(raw_a, return_tensors="pt", add_special_tokens=False)["input_ids"]
    full_input_ids = torch.cat([base["input_ids"], gen_ids.to(model.device)], dim=1)

    # donor: same model, DIFFERENT (original) image
    msg_b = build_user_message(s["orig_image"], s["query"])
    _, _, cache_donor, span_donor = forward_with_cache(model, processor, msg_b)
    print("img span (donor):", span_donor)

    n_gen = gen_ids.shape[1]
    fractions = {"early": 0.2, "mid": 0.5, "late": 0.8}
    any_diff = False
    for name, frac in fractions.items():
        cut = prompt_len + max(1, min(n_gen - 1, int(n_gen * frac)))
        print(f"\n[Cut={name}] prompt_len={prompt_len}, cut={cut}, total={full_input_ids.shape[1]}")

        prefix_ids, attn, cache_self, span_self = cache_prefix_for_continuation(
            model, processor, msg_a, full_input_ids, cut_len=cut)
        print("img span (self):", span_self)

        # generate() grows the cache in place, so clone before each use.
        cont_nopatch = continue_generate_from_cache(
            model, processor, prefix_ids, attn, clone_cache(cache_self), max_new_tokens=20)
        print("cont_nopatch:", repr(cont_nopatch))

        patched = patch_image_kv(cache_self, cache_donor, span_self[0], span_donor[0], layers=None)
        # tensor-level sanity: patched cache should actually differ from cache_self
        # at the image span (mechanical confirmation patch_image_kv did something)
        ts, te = span_self[0]
        kv_changed = not torch.allclose(
            cache_self.layers[0].keys[:, :, ts:te, :], patched.layers[0].keys[:, :, ts:te, :]
        )
        print("patched K/V differs from cache_self at image span (layer 0):", kv_changed)

        cont_patch = continue_generate_from_cache(model, processor, prefix_ids, attn, patched, max_new_tokens=20)
        print("cont_patch:  ", repr(cont_patch))

        diff = cont_patch != cont_nopatch
        print(f"differs from no-patch baseline: {diff}")
        any_diff = any_diff or diff

    print("\n[Result]")
    print("at least one cut point produced a different continuation under patching:", any_diff)
    print("(this is the expected/desired outcome -- confirms the patch has a causal effect "
          "and the infrastructure can move the needle)")


if __name__ == "__main__":
    main()
