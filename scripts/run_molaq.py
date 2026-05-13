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

校准数据：
  Flickr30k 的 test-*.parquet 文件，每个样本含真实 caption。
  图像字段可能是 PIL.Image、{'path':...} 或 {'bytes':...} 三种格式。

层名说明（Qwen3-VL-2B 实测）：
  LLM 侧 Linear 层前缀为 model.language_model.layers（不是 model.layers）
  ViT 侧 Linear 层前缀为 model.visual.blocks
  本脚本默认只量化 LLM 侧，若需量化 ViT 侧请修改 get_linear_layers。
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
# 校准数据加载（改自 AWQ 参考脚本，支持 parquet 格式）
# ─────────────────────────────────────────────────────────────

def resolve_image_from_field(image_field, data_dir: str, filename=None) -> Image.Image:
    """
    从 parquet 的 image 字段恢复 PIL.Image：
    1. 若已是 PIL.Image（datasets 自动解码），直接转成 RGB 返回。
    2. 若是 dict 且含 'path'，按路径读取磁盘图片。
    3. 若是 dict 且含 'bytes'，从 bytes 反序列化。
    """
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
    """
    从 flickr30k 的 test-*.parquet 文件加载校准数据。
    每个样本含真实图片（PIL.Image / bytes / path 三种格式均支持）
    和真实 caption（取第一条），构造 Qwen3-VL 模式的 messages。
    """
    def __init__(
        self,
        data_dir:  str,
        processor,
        n_samples: int = 128,
        seed:      int = 42,
        max_seq_len: int = 2048,
    ):
        self.processor   = processor
        self.data_dir    = data_dir
        self.max_seq_len = max_seq_len

        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Flickr30k 数据目录不存在: {data_dir}")

        ds = load_dataset(
            "parquet",
            data_dir=data_dir,
            data_files={"train": "test-*.parquet"},
        )["train"]
        ds = ds.shuffle(seed=seed)
        ds = ds.select(range(min(n_samples, len(ds))))
        self.samples = ds
        print(f"[MoLAQ] Flickr30k 加载完毕，实际样本数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        example = self.samples[idx]

        # ── 图像 ─────────────────────────────────────────
        image = resolve_image_from_field(
            example["image"],
            data_dir=self.data_dir,
            filename=example.get("filename"),
        )

        # ── Caption：取第一条 ──────────────────────────────
        caption_field = example.get("caption")
        if caption_field is None:
            raise ValueError(f"Flickr30k 样本缺少 'caption' 字段")
        if isinstance(caption_field, (list, tuple)):
            caption_text = str(caption_field[0])
        else:
            try:
                caption_text = str(caption_field[0])
            except Exception:
                caption_text = str(caption_field)

        # ── 构造 Qwen3-VL 多模态 messages ────────────────────
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": caption_text},
            ],
        }]

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
        inputs.pop("token_type_ids", None)  # Qwen3-VL 不需要

        if "pixel_values" not in inputs:
            raise ValueError(
                f"样本 {idx} 的 processor 输出缺少 pixel_values，请检查 messages 构造"
            )
        if "image_grid_thw" not in inputs:
            raise ValueError(
                f"样本 {idx} 的 processor 输出缺少 image_grid_thw"
            )

        return {
            "input_ids":      inputs["input_ids"].squeeze(0),       # [seq_len]
            "pixel_values":   inputs["pixel_values"].squeeze(0),    # [N_patches, C*p*p] 或类似
            "image_grid_thw": inputs["image_grid_thw"].squeeze(0),  # [3]
        }


def collate_fn(batch):
    """
    batch_size=1 的 collate。
    重新加回 batch 维度，确保 downstream 的 shape 一致。
    """
    assert len(batch) == 1, "校准 DataLoader 必须 batch_size=1"
    return {
        "input_ids":      batch[0]["input_ids"].unsqueeze(0),       # [1, seq_len]
        "pixel_values":   batch[0]["pixel_values"].unsqueeze(0),    # [1, ...]
        "image_grid_thw": batch[0]["image_grid_thw"].unsqueeze(0),  # [1, 3]
    }


# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────

def get_linear_layers(model) -> List[Tuple[str, torch.nn.Module]]:
    """
    返回 LLM 侧所有 nn.Linear 层的 (name, module) 列表。
    Qwen3-VL-2B 实测层名前缀为 model.language_model.layers，共 196 个。
    """
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
    p.add_argument("--model",        required=True,  help="模型路径")
    p.add_argument("--calib_data",   required=True,  help="Flickr30k parquet 目录")
    p.add_argument("--output",       required=True,  help="量化后模型保存路径")
    p.add_argument("--enable_A",     action="store_true", help="启用创新点 A")
    p.add_argument("--enable_B",     action="store_true", help="启用创新点 B")
    p.add_argument("--enable_C",     action="store_true", help="启用创新点 C")
    p.add_argument("--budget_bits",  type=float, default=4.5, help="混合精度平均 bit 预算")
    p.add_argument("--bits",         type=int,   default=4,   help="默认量化位数")
    p.add_argument("--group_size",   type=int,   default=128, help="量化分组大小")
    p.add_argument("--n_samples",    type=int,   default=128, help="校准样本数")
    p.add_argument("--seed",         type=int,   default=42,  help="随机种子")
    p.add_argument("--max_seq_len",  type=int,   default=2048, help="最大序列长度")
    p.add_argument("--device",       type=str,   default="cuda", help="运行设备")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"[MoLAQ] 配置: enable_A={args.enable_A}, "
          f"enable_B={args.enable_B}, enable_C={args.enable_C}")
    print(f"[MoLAQ] SALIENCY_MODE={SALIENCY_MODE}")

    # ── 加载模型 ────────────────────────────────────────────
    print("[MoLAQ] 加载模型...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(
        args.model, trust_remote_code=True
    )

    # ── 构建 DataLoader（parquet 格式的真实校准集）──────────────
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

    # ── Stage 0+1：一次 forward pass 收集所有统计量 ──────────
    linear_layers = get_linear_layers(model)
    target_names  = [name for name, _ in linear_layers]
    print(f"[MoLAQ] 目标 Linear 层数: {len(target_names)}")

    print("[MoLAQ] 收集激活统计量（Stage 0+1）...")
    all_stats = collect_modal_stats(
        model, dataloader, target_names, processor,
        saliency_mode=SALIENCY_MODE,
        device=args.device,
    )

    # ── Stage 2/3/4：逐层量化 ───────────────────────────────
    hessian_trace: Dict[str, float] = {}
    delta_4_dict:  Dict[str, float] = {}
    delta_8_dict:  Dict[str, float] = {}
    param_counts:  Dict[str, int]   = {}

    for layer_name, module in linear_layers:
        if layer_name not in all_stats:
            continue

        stats = all_stats[layer_name]
        W     = module.weight.data.float()
        print(f"[MoLAQ] 量化层: {layer_name}  shape={W.shape}")

        # ── Stage 2 (C) + Stage 3 (A) ──────────────────────
        if args.enable_C or args.enable_A:
            smooth_s = (
                compute_smooth_scale(stats.x_bar, W)
                if args.enable_C
                else torch.ones(W.shape[1], dtype=torch.float32, device=W.device)
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
                W_input_B, stats.X_raw, stats.x_bar,
                bits=args.bits,
                group_size=args.group_size,
            )
        else:
            W_final = W_hat_A

        # 写回权重
        module.weight.data = W_final.to(module.weight.dtype)

        # sanity check
        if args.enable_A:
            H = compute_modal_hessian(
                stats.X_raw.float(),
                stats.alpha_lang, stats.alpha_sal, stats.alpha_bg,
                stats.lang_mask, stats.sal_mask, stats.bg_mask,
            )
            sanity_check_layer(W_hat_A, W, H, stats.X_raw)
        else:
            H = (2.0 / stats.X_raw.shape[0]) * \
                stats.X_raw.float().T @ stats.X_raw.float()

        hessian_trace[layer_name] = H.trace().item()
        delta_4_dict[layer_name]  = estimate_delta(W, bits=4, group_size=args.group_size)
        delta_8_dict[layer_name]  = estimate_delta(W, bits=8, group_size=args.group_size)
        param_counts[layer_name]  = W.numel()

    # ── 混合精度分配（budget_bits < 8 时生效）──────────────
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

    # ── 保存量化后模型 ──────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    processor.save_pretrained(args.output)
    print(f"[MoLAQ] 量化模型已保存到 {args.output}")


if __name__ == "__main__":
    main()
