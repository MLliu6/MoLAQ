#!/usr/bin/env python
"""
scripts/run_molaq.py — MoLAQ 主调脚本

用法（完整 MoLAQ）：
  python scripts/run_molaq.py \\
      --model /home/lml/models/Qwen3-VL-2B-Instruct \\
      --calib_data /mnt/e/BISHE_START/Datasets/flickr30k/data \\
      --output /home/lml/models/Qwen3-VL-2B-MoLAQ \\
      --enable_A --enable_B --enable_C \\
      --budget_bits 4.5

用法（GPTQ-baseline，不启用任何创新点）：
  python scripts/run_molaq.py \\
      --model /home/lml/models/Qwen3-VL-2B-Instruct \\
      --calib_data /mnt/e/BISHE_START/Datasets/flickr30k/data \\
      --output /home/lml/models/Qwen3-VL-2B-GPTQ-baseline

消融配置开关：
  --enable_C  : 创新点 C（模态感知预平滑）
  --enable_A  : 创新点 A（加权 Hessian + GPTQ）
  --enable_B  : 创新点 B（显著 token 缩放搜索）

设备策略：
  - forward pass 收集激活：在 GPU 上运行（device_map=auto）。
  - 量化计算（Hessian / GPTQ / smooth / knapsack）：全部在 CPU 上进行。
  - LayerStats 中的张量全常居于 CPU（collect_modal_stats 内 .cpu() 存储）。
  - 写回：.to(orig_device, orig_dtype)，确保模型居于原设备。
"""

import argparse
import json
import os
from io import BytesIO
from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from PIL import Image

from molaq.stats.modal_stats      import collect_modal_stats, SALIENCY_MODE
from molaq.core.smooth            import compute_smooth_scale
from molaq.core.weighted_hessian  import (
    quantize_layer_A, compute_modal_hessian, sanity_check_layer
)
from molaq.core.saliency_scaling  import saliency_awq_quantize
from molaq.assign.knapsack        import greedy_bit_allocation, estimate_delta


# ─────────────────────────────────────────────────────────────
# 校准数据加载
# ─────────────────────────────────────────────────────────────

def resolve_image_from_field(image_field, data_dir: str, filename=None) -> Image.Image:
    if isinstance(image_field, Image.Image):
        return image_field.convert("RGB")
    if isinstance(image_field, dict) and image_field.get("path") is not None:
        raw_path = str(image_field["path"])
        candidates = (
            [raw_path] if os.path.isabs(raw_path)
            else [
                os.path.join(data_dir, raw_path),
                os.path.join(os.path.dirname(data_dir), raw_path),
            ]
        )
        for p in candidates:
            if os.path.exists(p):
                return Image.open(p).convert("RGB")
    if isinstance(image_field, dict) and image_field.get("bytes") is not None:
        return Image.open(BytesIO(image_field["bytes"])).convert("RGB")
    raise ValueError(
        f"Cannot resolve image from field of type {type(image_field)}, "
        f"keys={list(image_field.keys()) if isinstance(image_field, dict) else None}, "
        f"filename={filename}"
    )


class Flickr30kParquetDataset(Dataset):
    def __init__(self, data_dir, processor, n_samples=128, seed=42, max_seq_len=2048):
        self.processor   = processor
        self.data_dir    = data_dir
        self.max_seq_len = max_seq_len
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Flickr30k 数据目录不存在: {data_dir}")
        ds = load_dataset("parquet", data_dir=data_dir,
                          data_files={"train": "test-*.parquet"})["train"]
        ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))
        self.samples = ds
        print(f"[MoLAQ] Flickr30k 加载完毕，实际样本数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        example = self.samples[idx]
        image   = resolve_image_from_field(
            example["image"], data_dir=self.data_dir, filename=example.get("filename")
        )
        caption_field = example.get("caption")
        if caption_field is None:
            raise ValueError("Flickr30k 样本缺少 'caption' 字段")
        if isinstance(caption_field, (list, tuple)):
            caption_text = str(caption_field[0])
        else:
            try:
                caption_text = str(caption_field[0])
            except Exception:
                caption_text = str(caption_field)

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": caption_text},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt",
            padding=False, truncation=True, max_length=self.max_seq_len,
            add_special_tokens=False, add_generation_prompt=False,
        )
        inputs.pop("token_type_ids", None)
        if "pixel_values" not in inputs:
            raise ValueError(f"样本 {idx} processor 输出缺少 pixel_values")
        if "image_grid_thw" not in inputs:
            raise ValueError(f"样本 {idx} processor 输出缺少 image_grid_thw")
        return {
            "input_ids":      inputs["input_ids"].squeeze(0),
            "pixel_values":   inputs["pixel_values"].squeeze(0),
            "image_grid_thw": inputs["image_grid_thw"].squeeze(0),
        }


def collate_fn(batch):
    assert len(batch) == 1, "校准 DataLoader 必须 batch_size=1"
    return {
        "input_ids":      batch[0]["input_ids"].unsqueeze(0),
        "pixel_values":   batch[0]["pixel_values"].unsqueeze(0),
        "image_grid_thw": batch[0]["image_grid_thw"].unsqueeze(0),
    }


# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────

def get_linear_layers(model) -> List[Tuple[str, torch.nn.Module]]:
    result = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "model.language_model.layers" in name:
            result.append((name, module))
    return result


# ─────────────────────────────────────────────────────────────
# 参数解析
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MoLAQ 量化流水线")
    p.add_argument("--model",        required=True)
    p.add_argument("--calib_data",   required=True)
    p.add_argument("--output",       required=True)
    p.add_argument("--enable_A",     action="store_true")
    p.add_argument("--enable_B",     action="store_true")
    p.add_argument("--enable_C",     action="store_true")
    p.add_argument("--budget_bits",  type=float, default=4.5)
    p.add_argument("--bits",         type=int,   default=4)
    p.add_argument("--group_size",   type=int,   default=128)
    p.add_argument("--n_samples",    type=int,   default=128)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--max_seq_len",  type=int,   default=2048)
    p.add_argument("--device",       type=str,   default="cuda")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    print(f"[MoLAQ] 配置: enable_A={args.enable_A}, "
          f"enable_B={args.enable_B}, enable_C={args.enable_C}")
    print(f"[MoLAQ] SALIENCY_MODE={SALIENCY_MODE}")

    # ── 加载模型（GPU forward 用） ─────────────────────────────
    print("[MoLAQ] 加载模型...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    # ── 构建 DataLoader ───────────────────────────────────────
    dataset = Flickr30kParquetDataset(
        data_dir=args.calib_data, processor=processor,
        n_samples=args.n_samples, seed=args.seed, max_seq_len=args.max_seq_len,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # ── Stage 0+1：收集激活统计量（结果全部在 CPU）───────────────
    linear_layers = get_linear_layers(model)
    target_names  = [name for name, _ in linear_layers]
    print(f"[MoLAQ] 目标 Linear 层数: {len(target_names)}")
    print("[MoLAQ] 收集激活统计量（Stage 0+1）...")

    all_stats = collect_modal_stats(
        model, dataloader, target_names, processor,
        saliency_mode=SALIENCY_MODE,
        device=args.device,
    )

    # ── Stage 2/3/4：逐层量化（全部在 CPU 计算）─────────────────
    hessian_trace: Dict[str, float] = {}
    delta_4_dict:  Dict[str, float] = {}
    delta_8_dict:  Dict[str, float] = {}
    param_counts:  Dict[str, int]   = {}

    for layer_name, module in linear_layers:
        if layer_name not in all_stats:
            continue

        stats = all_stats[layer_name]  # LayerStats 全部 CPU tensor

        # 记录原设备/精度，用于最后写回
        orig_device = module.weight.device
        orig_dtype  = module.weight.dtype

        # 把权重拉到 CPU float32 做量化计算
        W = module.weight.data.detach().cpu().float()  # [d_out, d_in], CPU float32
        print(f"[MoLAQ] 量化层: {layer_name}  shape={W.shape}")

        # ── Stage 2 (C) + Stage 3 (A) ────────────────────────
        # 注意：x_bar / X_raw 已经是 CPU tensor，W 也在 CPU，没有设备冲突。
        if args.enable_C or args.enable_A:
            smooth_s = (
                compute_smooth_scale(stats.x_bar, W)         # 均在 CPU
                if args.enable_C
                else torch.ones(W.shape[1], dtype=torch.float32)  # CPU
            )
            W_hat_A, smooth_s = quantize_layer_A(
                stats, W, smooth_s,
                enable_C=args.enable_C,
                bits=args.bits,
                group_size=args.group_size,
            )
        else:
            W_hat_A = W

        # ── Stage 4 (B) ─────────────────────────────────────
        if args.enable_B:
            W_input_B = W_hat_A if args.enable_A else W
            W_final   = saliency_awq_quantize(
                W_input_B, stats.X_raw, stats.x_bar,         # 均在 CPU
                bits=args.bits, group_size=args.group_size,
            )
        else:
            W_final = W_hat_A

        # ── 写回权重：转回原设备 + 原精度 ───────────────────
        module.weight.data = W_final.to(device=orig_device, dtype=orig_dtype)

        # ── Sanity check 和 Hessian 收集（均在 CPU） ────────────────
        if args.enable_A:
            H = compute_modal_hessian(
                stats.X_raw.float(),
                stats.alpha_lang, stats.alpha_sal, stats.alpha_bg,
                stats.lang_mask, stats.sal_mask, stats.bg_mask,
            )
            sanity_check_layer(W_hat_A, W, H, stats.X_raw)
        else:
            X_f = stats.X_raw.float()
            H = (2.0 / X_f.shape[0]) * X_f.T @ X_f

        hessian_trace[layer_name] = H.trace().item()
        delta_4_dict[layer_name]  = estimate_delta(W, bits=4, group_size=args.group_size)
        delta_8_dict[layer_name]  = estimate_delta(W, bits=8, group_size=args.group_size)
        param_counts[layer_name]  = W.numel()

    # ── 混合精度分配 ───────────────────────────────────────
    if args.budget_bits < 8.0:
        layer_names = list(all_stats.keys())
        assignment  = greedy_bit_allocation(
            layer_names, param_counts,
            hessian_trace, delta_4_dict, delta_8_dict,
            budget_avg_bits=args.budget_bits,
        )
        os.makedirs(args.output, exist_ok=True)
        bit_config_path = os.path.join(args.output, "molaq_bits.json")
        with open(bit_config_path, "w") as f:
            json.dump(assignment, f, indent=2)
        print(f"[MoLAQ] bit 分配已保存到 {bit_config_path}")

    # ── 保存模型 ────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    processor.save_pretrained(args.output)
    print(f"[MoLAQ] 量化模型已保存到 {args.output}")


if __name__ == "__main__":
    main()
