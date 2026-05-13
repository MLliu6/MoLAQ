#!/usr/bin/env python
"""
tests/test_vit_attention.py

Step 1（必须先运行）：检查 Qwen3-VL ViT 的 attention 接口，
决定 modal_stats.py 顶部的 SALIENCY_MODE。

运行方式：
    source ~/vllm/bin/activate
    cd ~/MoLAQ
    python tests/test_vit_attention.py
"""

import sys
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_PATH = "/home/lml/models/Qwen3-VL-2B-Instruct"
TEST_IMAGE  = "/mnt/e/BISHE_START/Datasets/MathVision/images/images/1.jpg"


def main():
    print("="*60)
    print("ViT Attention 可获取性检查")
    print("="*60)

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    img = Image.open(TEST_IMAGE).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text",  "text": "test"},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True,
        return_tensors="pt", add_generation_prompt=False,
    )
    pixel_values   = inputs["pixel_values"].to(torch.bfloat16)
    image_grid_thw = inputs.get("image_grid_thw")

    print(f"pixel_values shape : {pixel_values.shape}")
    print(f"image_grid_thw     : {image_grid_thw}")

    import inspect
    sig = inspect.signature(model.visual.forward)
    print(f"visual.forward params: {list(sig.parameters.keys())}")
    print()

    # 检查层名格式
    print("=== ViT Linear 层（前 5 个）===")
    vit_linears = [
        (n, m.weight.shape) for n, m in model.named_modules()
        if "visual" in n and isinstance(m, torch.nn.Linear)
    ]
    for n, s in vit_linears[:5]:
        print(f"  {n}: {s}")

    print("\n=== LLM Linear 层（前 5 个）===")
    llm_linears = [
        (n, m.weight.shape) for n, m in model.named_modules()
        if "model.layers" in n and isinstance(m, torch.nn.Linear)
    ]
    print(f"  LLM Linear 总数: {len(llm_linears)}")
    for n, s in llm_linears[:5]:
        print(f"  {n}: {s}")
    print()

    # image_token_id 检查
    try:
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        print(f"image_token_id (<|image_pad|>): {image_token_id}")
    except Exception as e:
        print(f"[WARN] 无法获取 image_token_id: {e}")

    # 尝试获取 ViT attention
    try:
        with torch.no_grad():
            vit_out = model.visual(
                pixel_values,
                grid_thw=image_grid_thw,
                output_attentions=True,
            )
        attns   = vit_out.attentions
        seq_len = attns[-1].shape[-1]
        thw     = image_grid_thw[0]            # Tensor[3]: (t, h, w)
        n_patches = int(thw[0] * thw[1] * thw[2])
        has_cls   = (seq_len == n_patches + 1)

        print(f"[OK] ViT attention 可获取")
        print(f"     num_layers  = {len(attns)}")
        print(f"     last layer  = {attns[-1].shape}")
        print(f"     n_patches   = {n_patches}")
        print(f"     seq_len     = {seq_len}")
        print(f"     has CLS     = {has_cls}")
        print()

        if has_cls:
            print(">>> 结论：SALIENCY_MODE = \"cls_attn\"")
        else:
            print(">>> 结论：SALIENCY_MODE = \"row_sum\"")
        print("请将上述结论写入 molaq/stats/modal_stats.py 顶部")

    except Exception as e:
        print(f"[FALLBACK] ViT attention 不可获取: {e}")
        print(">>> 结论：SALIENCY_MODE = \"act_norm\"")
        print("请将上述结论写入 molaq/stats/modal_stats.py 顶部")


if __name__ == "__main__":
    main()
