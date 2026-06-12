"""Causal patching primitives for Block A (E-A1/E-A2/E-A3/E-A4).

Core idea (E-A1, position-segmented visual K/V patching):
  1. Generate the reasoning trace for image I_a (the "self" image).
  2. Pick a cut point t in that trace (early / mid / late fraction).
  3. Build prefix = prompt(I_a, Q) + trace[:t]; run a forward pass to get
     `cache_self` covering prefix[:-1].
  4. Build `cache_donor` from a forward pass over prompt(I_b, Q) alone (the
     counterfactual image).
  5. Patch: clone cache_self, and for the chosen transformer layers, overwrite
     the K/V at the image-token span with cache_donor's K/V at its own
     image-token span (sliced/truncated to matching length).
  6. Continue generation from `prefix` using the patched cache and compare the
     resulting answer to the unpatched continuation -> CRI(t).

Validated against transformers 4.57.3 / Qwen3-VL on ssh84 (2026-06-11) via
smoke_test_patch.py. DynamicCache layers expose `.keys` / `.values` tensors of
shape [batch, num_kv_heads, seq_len, head_dim] (see DynamicLayer.update).
"""
import copy
import torch

from .model_utils import find_image_token_spans, IMAGE_TOKEN_ID


@torch.inference_mode()
def forward_with_cache(model, processor, messages, extra_input_ids=None):
    """Run a single forward pass (no generation) over `messages` (+ optional
    extra continuation tokens appended to the prompt, e.g. a reasoning prefix
    rendered as plain token ids).

    Returns:
      input_ids: [1, L] full token ids actually fed to the model
      cache:     DynamicCache covering positions [0, L)
      img_spans: list of (start, end) image-token spans in input_ids
    """
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, return_tensors="pt", padding=True
    ).to(model.device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    if extra_input_ids is not None and extra_input_ids.numel() > 0:
        extra_input_ids = extra_input_ids.to(model.device)
        input_ids = torch.cat([input_ids, extra_input_ids], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(extra_input_ids)], dim=1
        )

    fwd_kwargs = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
        **fwd_kwargs,
    )
    img_spans = find_image_token_spans(input_ids[0], IMAGE_TOKEN_ID)
    return input_ids, attention_mask, out.past_key_values, img_spans


def clone_cache(cache):
    """Deep-copy a DynamicCache's per-layer key/value tensors."""
    new_cache = copy.deepcopy(cache)
    for layer_idx in range(len(new_cache.layers)):
        new_cache.layers[layer_idx].keys = cache.layers[layer_idx].keys.clone()
        new_cache.layers[layer_idx].values = cache.layers[layer_idx].values.clone()
    return new_cache


def patch_image_kv(target_cache, donor_cache, target_span, donor_span, layers=None):
    """Return a NEW cache (target_cache is not mutated) where, for the given
    transformer layers, the K/V at `target_span` positions are overwritten
    with `donor_cache`'s K/V at `donor_span` positions.

    If the two spans differ in length (different #image-tokens for the two
    images at the resolution the processor chose), the donor span is
    center-cropped/truncated to match the target span length so the patch is
    a clean drop-in replacement (logged by the caller -- length mismatches
    should be rare if both images are preprocessed at the same target size,
    but Qwen3-VL's dynamic resolution can still produce small differences).
    """
    patched = clone_cache(target_cache)
    n_layers = len(patched.layers)
    layer_idxs = range(n_layers) if layers is None else layers

    t_start, t_end = target_span
    d_start, d_end = donor_span
    t_len = t_end - t_start
    d_len = d_end - d_start
    n = min(t_len, d_len)
    # center-align if lengths differ
    t_off = (t_len - n) // 2
    d_off = (d_len - n) // 2
    ts, te = t_start + t_off, t_start + t_off + n
    ds, de = d_start + d_off, d_start + d_off + n

    for l in layer_idxs:
        patched.layers[l].keys[:, :, ts:te, :] = donor_cache.layers[l].keys[:, :, ds:de, :].to(
            patched.layers[l].keys.device
        )
        patched.layers[l].values[:, :, ts:te, :] = donor_cache.layers[l].values[:, :, ds:de, :].to(
            patched.layers[l].values.device
        )
    return patched


@torch.inference_mode()
def continue_generate_from_cache(model, processor, input_ids, attention_mask, cache,
                                  max_new_tokens=256, do_sample=False):
    """Continue generation given `input_ids` (full sequence so far, [1, L])
    and a `cache` covering positions [0, L-1) (i.e. one token short of L --
    generate() will process the final token of input_ids using the cache and
    then proceed autoregressively).
    """
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
    out_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=cache,
        use_cache=True,
        **gen_kwargs,
    )
    new_ids = out_ids[:, input_ids.shape[1]:]
    text = processor.batch_decode(
        new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return text.strip()


def cache_prefix_for_continuation(model, processor, messages, full_input_ids,
                                   cut_len):
    """Build (input_ids[:cut_len], cache covering [0, cut_len-1)) so that
    continue_generate_from_cache can resume exactly at position cut_len-1.

    `full_input_ids` is the [1, L] tensor of prompt+generated tokens (L >=
    cut_len) from a prior full-generation run; `messages` is only used to
    recover the multimodal kwargs (pixel_values etc.) via process_vision_info.
    """
    from qwen_vl_utils import process_vision_info

    image_inputs, _ = process_vision_info(messages)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    base_inputs = processor(text=[text], images=image_inputs, return_tensors="pt", padding=True)
    fwd_kwargs = {k: v.to(model.device) for k, v in base_inputs.items()
                   if k not in ("input_ids", "attention_mask")}

    # cache covers [0, cut_len-1); prefix_ids covers [0, cut_len) so that
    # generate() has exactly one new (uncached) token to process before
    # proceeding autoregressively.
    prefix_ids = full_input_ids[:, :cut_len].to(model.device)
    cache_input = full_input_ids[:, :cut_len - 1].to(model.device)
    cache_attn = torch.ones_like(cache_input)
    with torch.inference_mode():
        out = model(input_ids=cache_input, attention_mask=cache_attn,
                     use_cache=True, return_dict=True, **fwd_kwargs)
    cache = out.past_key_values
    attn = torch.ones_like(prefix_ids)
    img_spans = find_image_token_spans(prefix_ids[0], IMAGE_TOKEN_ID)
    return prefix_ids, attn, cache, img_spans
