"""Model loading and generation helpers for Qwen3-VL Instruct/Thinking variants.

Shared by Block 0 (E-01/E-02 behavioral runs) and Block A (causal patching).
"""
import os
import yaml
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

# Special token ids (Qwen3-VL, verified against Qwen3VL-8B config on 2026-06-11)
IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656
VISION_START_TOKEN_ID = 151652
VISION_END_TOKEN_ID = 151653

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REGISTRY_PATH = os.path.join(_THIS_DIR, "..", "configs", "models.yaml")


def load_registry():
    with open(_REGISTRY_PATH) as f:
        return yaml.safe_load(f)


def resolve_model_path(name_or_path):
    """Accept either a registry key (e.g. '8B-Thinking') or a raw filesystem path."""
    registry = load_registry()
    return registry.get(name_or_path, name_or_path)


def is_thinking_model(name_or_path):
    return "thinking" in name_or_path.lower()


def load_model_and_processor(name_or_path, device="cuda:0", dtype=torch.bfloat16,
                              attn_implementation="sdpa"):
    """Load a Qwen3-VL model + processor.

    sdpa (not flash_attention_2) is used by default because the patching
    infrastructure in Block A needs to read/write past_key_values directly;
    flash-attn's fused kernels make that harder to intercept cleanly.
    """
    path = resolve_model_path(name_or_path)
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        path,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return model, processor


def build_user_message(image, text):
    """Single-turn user message: one image + one text prompt."""
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text},
        ],
    }]


@torch.inference_mode()
def generate(model, processor, messages, max_new_tokens=2048, do_sample=False,
             temperature=0.1):
    """Standard single-turn generation. Returns (generated_text, num_input_tokens)."""
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, return_tensors="pt", padding=True
    ).to(model.device)

    gen_kwargs = dict(max_new_tokens=max_new_tokens)
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
    else:
        gen_kwargs.update(do_sample=False)

    out_ids = model.generate(**inputs, **gen_kwargs)
    input_len = inputs["input_ids"].shape[1]
    new_ids = out_ids[:, input_len:]
    out_text = processor.batch_decode(
        new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return out_text.strip(), input_len


@torch.inference_mode()
def continue_generate(model, processor, messages_with_partial_assistant,
                       max_new_tokens=2048, do_sample=False, temperature=0.1):
    """Stage-3 'prefill/probe' generation: the conversation already contains a
    partial assistant turn (reasoning R_a + reflection prompt) and decoding
    continues from there with a (possibly swapped) image.

    Uses HF's `continue_final_message=True`, which strips the closing
    <|im_end|> so generation picks up mid-turn (mirrors VS-Bench's manual
    string-surgery approach in run_inference.py but via the supported API).
    """
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(
        messages_with_partial_assistant, tokenize=False,
        add_generation_prompt=False, continue_final_message=True,
    )
    image_inputs, _ = process_vision_info(messages_with_partial_assistant)
    inputs = processor(
        text=[text], images=image_inputs, return_tensors="pt", padding=True
    ).to(model.device)

    gen_kwargs = dict(max_new_tokens=max_new_tokens)
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
    else:
        gen_kwargs.update(do_sample=False)

    out_ids = model.generate(**inputs, **gen_kwargs)
    input_len = inputs["input_ids"].shape[1]
    new_ids = out_ids[:, input_len:]
    out_text = processor.batch_decode(
        new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return out_text.strip(), input_len


def find_image_token_spans(input_ids, image_token_id=IMAGE_TOKEN_ID):
    """Return list of (start, end) exclusive index ranges of contiguous
    image-token runs in a 1D input_ids tensor. One span per image in the
    prompt (Qwen3-VL packs each image's patches as one contiguous run of
    <|image_pad|> tokens between <|vision_start|>/<|vision_end|>).
    """
    ids = input_ids.flatten()
    is_img = (ids == image_token_id)
    spans = []
    in_span = False
    start = None
    for i, v in enumerate(is_img.tolist()):
        if v and not in_span:
            start, in_span = i, True
        elif not v and in_span:
            spans.append((start, i))
            in_span = False
    if in_span:
        spans.append((start, len(ids)))
    return spans
