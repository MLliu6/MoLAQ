#!/usr/bin/env python
"""
 scripts/run_molaq_chunked.py — MoLAQ 低显存分块量化脚本

 设计目标：在不改动主算法（A/B/C、Hessian 形式、GPTQ 内核）的前提下，
 通过“分块收集激活 + 分块量化”把显存占用从 O(#layers × #tokens)
 降到 O(chunk_size × #tokens)，避免在 16GB 显存上 OOM，支持 n_samples=128、
 甚至后续的 Qwen3-VL-4B。

 与 scripts/run_molaq.py 的差异：
   - 不再一次性对 196 个 Linear 层同时挂 hook 收集激活
   - 按 chunk_size（默认为 4）把层分块，每次只对一个子集收集激活并量化
   - 算法逻辑完全一致：A/B/C 三个模块 + Hessian/Knapsack 均保持不变

 使用建议：
   - 2B 正式实验（128 样本）：优先使用本脚本（--chunk_size 4 或 8）
   - 8 样本 smoke：可以用本脚本对比显存占用（应显著低于旧脚本的 ~9.9GB）
   - 旧脚本 scripts/run_molaq.py 保留，作为对照和备份。
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

from molaq.stats.modal_stats import (
    collect_modal_stats,
    SALIENCY_MODE,
)
from molaq.core.smooth import compute_smooth_scale
from molaq.core.weighted_hessian import (
    quantize_layer_A,
    compute_modal_hessian,
    sanity_check_layer,
)
from molaq.core.saliency_scaling import saliency_awq_quantize
from molaq.assign.knapsack import greedy_bit_allocation, estimate_delta


# ───────────────────────────────────────────────────────────────
# 校准数据加载（与 scripts/run_molaq.py 保持一致）
# ───────────────────────────────────────────────────────────────


def resolve_image_from_field(image_field, data_dir: str, filename=None) -> Image.Image:
    if isinstance(image_field, Image.Image):
        return image_field.convert("RGB")
    if isinstance(image_field, dict) and image_field.get("path") is not None:
        raw_path = str(image_field["path"])
        candidates = (
            [raw_path]
            if os.path.isabs(raw_path)
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
        self.processor = processor
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Flickr30k 数据目录不存在: {data_dir}")
        ds = load_dataset(
            "parquet", data_dir=data_dir, data_files={"train": "test-*.parquet"}
        )["train"]
        ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))
        self.samples = ds
        print(f"[MoLAQ] Flickr30k 加载完毕，实际样本数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        example = self.samples[idx]
        image = resolve_image_from_field(
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

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": caption_text},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_seq_len,
            add_special_tokens=False,
            add_generation_prompt=False,
        )
        inputs.pop("token_type_ids", None)
        if "pixel_values" not in inputs:
            raise ValueError(f"样本 {idx} processor 输出缺少 pixel_values")
        if "image_grid_thw" not in inputs:
            raise ValueError(f"样本 {idx} processor 输出缺少 image_grid_thw")
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "image_grid_thw": inputs["image_grid_thw"].squeeze(0),
        }


def collate_fn(batch):
    assert len(batch) == 1, "校准 DataLoader 必须 batch_size=1"
    return {
        "input_ids": batch[0]["input_ids"].unsqueeze(0),
        "pixel_values": batch[0]["pixel_values"].unsqueeze(0),
        "image_grid_thw": batch[0]["image_grid_thw"].unsqueeze(0),
    }


# ───────────────────────────────────────────────────────────────
# 工具函数
# ───────────────────────────────────────────────────────────────


def get_linear_layers(model) -> List[Tuple[str, torch.nn.Module]]:
    result = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "model.language_model.layers" in name:
            result.append((name, module))
    return result


def chunked(iterable: List[str], chunk_size: int) -> List[List[str]]:
    """把层名列表按 chunk_size 等分，最后一块可以更小。"""
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i : i + chunk_size]


# ───────────────────────────────────────────────────────────────
# 参数解析
# ───────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="MoLAQ 量化流水线（分块低显存版）")
    p.add_argument("--model", required=True)
    p.add_argument("--calib_data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--enable_A", action="store_true")
    p.add_argument("--enable_B", action="store_true")
    p.add_argument("--enable_C", action="store_true")
    p.add_argument("--budget_bits", type=float, default=4.5)
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--group_size", type=int, default=128)
    p.add_argument("--n_samples", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--chunk_size",
        type=int,
        default=4,
        help="每次同时收集并量化的 Linear 层数量，默认 4，显存与其成正比",
    )
    p.add_argument(
        "--sanity",
        action="store_true",
        help="是否启用每层 κ(H) 的 sanity check（默认关闭以节省显存与时间）",
    )
    return p.parse_args()


# ───────────────────────────────────────────────────────────────
# 主流程
# ───────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    dev = args.device  # "cuda"
    print(
        f"[MoLAQ] 配置: enable_A={args.enable_A}, "
        f"enable_B={args.enable_B}, enable_C={args.enable_C}"
    )
    print(f"[MoLAQ] SALIENCY_MODE={SALIENCY_MODE}")
    print(f"[MoLAQ] 量化计算设备: {dev}")
    print(f"[MoLAQ] 分块大小 chunk_size={args.chunk_size}")

    # ── 加载模型（GPU）──────────────────────────────────
    print("[MoLAQ] 加载模型...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    # ── 构建 DataLoader ──────────────────────────────────
    dataset = Flickr30kParquetDataset(
        data_dir=args.calib_data,
        processor=processor,
        n_samples=args.n_samples,
        seed=args.seed,
        max_seq_len=args.max_seq_len,
    )
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
    )

    # ── 目标 Linear 层 ──────────────────────────────────
    linear_layers = get_linear_layers(model)
    target_names = [name for name, _ in linear_layers]
    name_to_module = {name: module for name, module in linear_layers}
    print(f"[MoLAQ] 目标 Linear 层数: {len(target_names)}")

    # 记录 Hessian trace / Delta / 参数量，用于后续 knapsack
    hessian_trace: Dict[str, float] = {}
    delta_4_dict: Dict[str, float] = {}
    delta_8_dict: Dict[str, float] = {}
    param_counts: Dict[str, int] = {}

    # ── 按块收集激活 + 量化 ─────────────────────────────
    total_layers = len(target_names)
    for block_id, chunk_names in enumerate(chunked(target_names, args.chunk_size)):
        print(
            f"[MoLAQ] ==== 分块 {block_id+1}/{(total_layers + args.chunk_size - 1) // args.chunk_size} "
            f"（层数 {len(chunk_names)}）===="
        )
        print(
            f"[MoLAQ] 收集激活统计量（Stage 0+1，chunk_size={len(chunk_names)}）..."
        )

        # 只对 chunk 中的层挂 hook，显存占用 ~ O(chunk_size)
        all_stats_chunk = collect_modal_stats(
            model,
            dataloader,
            chunk_names,
            processor,
            saliency_mode=SALIENCY_MODE,
            device=dev,
        )

        # ── 对本块中的每一层执行 Stage 2/3/4 ─────────────
        for layer_name in chunk_names:
            if layer_name not in all_stats_chunk:
                continue

            module = name_to_module[layer_name]
            stats = all_stats_chunk[layer_name]

            orig_device = module.weight.device
            orig_dtype = module.weight.dtype

            # 权重保持在 cuda 上，转 float32 用于量化计算
            W = module.weight.data.detach().to(device=dev, dtype=torch.float32)
            print(f"[MoLAQ] 量化层: {layer_name}  shape={W.shape}")

            # ── Stage 2 (C) + Stage 3 (A) ─────────────────
            if args.enable_C or args.enable_A:
                smooth_s = (
                    compute_smooth_scale(stats.x_bar, W)
                    if args.enable_C
                    else torch.ones(W.shape[1], dtype=torch.float32, device=dev)
                )
                W_hat_A, smooth_s = quantize_layer_A(
                    stats,
                    W,
                    smooth_s,
                    enable_C=args.enable_C,
                    bits=args.bits,
                    group_size=args.group_size,
                )
            else:
                W_hat_A = W

            # ── Stage 4 (B) ─────────────────────────────
            if args.enable_B:
                W_input_B = W_hat_A if args.enable_A else W
                W_final = saliency_awq_quantize(
                    W_input_B,
                    stats.X_raw,
                    stats.x_bar,
                    bits=args.bits,
                    group_size=args.group_size,
                )
            else:
                W_final = W_hat_A

            # ── 写回权重：转回原设备 + 原精度 ─────────────
            module.weight.data = W_final.to(device=orig_device, dtype=orig_dtype)

            # ── 可选：sanity check + Hessian trace ────────
            if args.enable_A:
                # 注意：为了避免额外显存占用，sanity_check 可选开启
                H = compute_modal_hessian(
                    stats.X_raw,
                    stats.alpha_lang,
                    stats.alpha_sal,
                    stats.alpha_bg,
                    stats.lang_mask,
                    stats.sal_mask,
                    stats.bg_mask,
                )
                if args.sanity:
                    sanity_check_layer(W_hat_A, W, H, stats.X_raw)
            else:
                X_f = stats.X_raw.float()
                H = (2.0 / X_f.shape[0]) * X_f.T @ X_f

            hessian_trace[layer_name] = H.trace().item()
            delta_4_dict[layer_name] = estimate_delta(
                W, bits=4, group_size=args.group_size
            )
            delta_8_dict[layer_name] = estimate_delta(
                W, bits=8, group_size=args.group_size
            )
            param_counts[layer_name] = W.numel()

            # 释放本层 stats 显存
            del all_stats_chunk[layer_name]
            torch.cuda.empty_cache()

        # 整个 chunk 结束后再清一次缓存
        del all_stats_chunk
        torch.cuda.empty_cache()

    # ── 混合精度分配 ─────────────────────────────
    if args.budget_bits < 8.0:
        layer_names = list(hessian_trace.keys())
        assignment = greedy_bit_allocation(
            layer_names,
            param_counts,
            hessian_trace,
            delta_4_dict,
            delta_8_dict,
            budget_avg_bits=args.budget_bits,
        )
        os.makedirs(args.output, exist_ok=True)
        bit_config_path = os.path.join(args.output, "molaq_bits.json")
        with open(bit_config_path, "w") as f:
            json.dump(assignment, f, indent=2)
        print(f"[MoLAQ] bit 分配已保存到 {bit_config_path}")

    # ── 保存模型 ────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    processor.save_pretrained(args.output)
    print(f"[MoLAQ] 量化模型已保存到 {args.output}")


if __name__ == "__main__":
    main()
