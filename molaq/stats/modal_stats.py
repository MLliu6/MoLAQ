"""
molaq/stats/modal_stats.py
§3.2–3.3: Token 分组、激活收集、共享统计量

实现前必须先运行 tests/test_vit_attention.py 确认 SALIENCY_MODE。
根据检查结果在下方设置（三选一）：
"""

# ============================================================
# 根据 tests/test_vit_attention.py 的输出在此处设置（三选一）
# SALIENCY_MODE = "cls_attn"   # 优先方案：有 CLS token
# SALIENCY_MODE = "row_sum"  # 备选方案 1：无 CLS，用被注意度
SALIENCY_MODE = "act_norm" # 备选方案 2：sdpa 不支持 output_attentions，用激活 L2
# ============================================================

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader


@dataclass
class LayerStats:
    """
    单层共享统计量，由 collect_modal_stats 一次性收集。
    后续 smooth / weighted_hessian / saliency_scaling 均从此读取，
    不重复 forward pass。
    """
    X_raw:      Tensor   # [N, d_in]  原始激活（float32，所有校准样本拼接）
    e_lang:     float    # 语言组激活能量
    e_sal:      float    # 显著视觉组激活能量
    e_bg:       float    # 背景视觉组激活能量
    alpha_lang: float    # A 模块 Hessian 权重，归一化到 sum=3
    alpha_sal:  float
    alpha_bg:   float
    a_lang:     float    # B/C 模块幅度权重，归一化到 max=1
    a_sal:      float
    a_bg:       float
    x_bar:      Tensor   # [d_in]  显著性加权激活幅度
    lang_mask:  Tensor   # [N] bool
    sal_mask:   Tensor   # [N] bool
    bg_mask:    Tensor   # [N] bool


def compute_saliency(
    model,
    pixel_values: Tensor,
    image_grid_thw: Tensor,
    mode: str = "act_norm",
) -> Tensor:
    """
    计算视觉 patch 的显著性分数 p_i，满足 sum(p) == 1。

    Args:
        model           : Qwen3VLForConditionalGeneration
        pixel_values    : 已经过 processor 预处理的像素张量
        image_grid_thw  : [num_images, 3]，每行 (t, h, w)
        mode            : "cls_attn" | "row_sum" | "act_norm"
    Returns:
        p : Tensor [N_patches]

    注意（act_norm 模式）：
        Qwen3-VL 的 model.visual() 直接返回 Tensor（不是 dataclass），
        shape 为 [N_patches, hidden_dim]。
        sdpa attention 不支持 output_attentions=True，因此只能用 act_norm。
    """
    if mode == "cls_attn":
        with torch.no_grad():
            out = model.visual(
                pixel_values, grid_thw=image_grid_thw, output_attentions=True
            )
        last_attn = out.attentions[-1]       # [1, N_heads, seq, seq]
        cls_attn  = last_attn[0, :, 0, 1:]  # [N_heads, N_patches]
        p = cls_attn.mean(dim=0)

    elif mode == "row_sum":
        with torch.no_grad():
            out = model.visual(
                pixel_values, grid_thw=image_grid_thw, output_attentions=True
            )
        last_attn = out.attentions[-1]           # [1, N_heads, N_p, N_p]
        p = last_attn[0].mean(dim=0).mean(dim=0) # [N_p]

    elif mode == "act_norm":
        # Qwen3-VL visual 返回 Tensor，shape [N_patches, hidden_dim]
        with torch.no_grad():
            vit_out = model.visual(pixel_values, grid_thw=image_grid_thw)
        # vit_out 是 Tensor，直接取 L2 范数
        if isinstance(vit_out, Tensor):
            p = vit_out.float().norm(dim=-1)   # [N_patches]
        else:
            # 兜底：若未来版本返回 dataclass
            p = vit_out.last_hidden_state[0].float().norm(dim=-1)

    else:
        raise ValueError(f"Unknown saliency mode: {mode}")

    p = p.float()
    p = p / (p.sum() + 1e-8)
    return p


def get_token_masks(
    input_ids: Tensor,
    image_token_id: int,
) -> Tuple[Tensor, Tensor]:
    """
    Args:
        input_ids      : [seq_len] 或 [1, seq_len]
        image_token_id : processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    Returns:
        vis_mask  : [seq_len] bool
        lang_mask : [seq_len] bool
    """
    ids = input_ids.squeeze(0)  # -> [seq_len]
    vis_mask  = (ids == image_token_id)
    lang_mask = ~vis_mask
    return vis_mask, lang_mask


def compute_stats_for_layer(
    X:         Tensor,  # [N, d_in]，float32
    lang_mask: Tensor,  # [N] bool
    sal_mask:  Tensor,  # [N] bool
    bg_mask:   Tensor,  # [N] bool
    p_sal:     Tensor,  # [N_sal]，长度必须严格等于 sal_mask.sum()
) -> LayerStats:
    """
    计算单层所有共享统计量。
    注意：p_sal 长度必须等于 sal_mask.sum()，传入前须 assert 验证。
    """
    assert p_sal.shape[0] == sal_mask.sum().item(), (
        f"p_sal 长度 {p_sal.shape[0]} != sal_mask.sum() {sal_mask.sum().item()}"
    )

    def energy(mask: Tensor) -> float:
        """计算子组激活能量，空组返回 1e-8 防零除。"""
        if mask.sum() == 0:
            return 1e-8
        return X[mask].pow(2).mean().item()

    e = {
        "lang": energy(lang_mask),
        "sal":  energy(sal_mask),
        "bg":   energy(bg_mask),
    }
    e_sum = sum(e.values())

    # A 模块权重：归一化使 sum = 3（保持与 GPTQ Hessian 同量级）
    alpha = {m: 3.0 * e[m] / e_sum for m in e}

    # B/C 模块权重：归一化使 max = 1
    e_max = max(e.values())
    a = {m: e[m] / e_max for m in e}

    # 显著性加权激活幅度 x̄_j [d_in]
    N = X.shape[0]  # 取行数标量
    w = torch.zeros(N, dtype=torch.float32)
    w[lang_mask] = a["lang"]
    w[bg_mask]   = a["bg"]
    sal_indices  = sal_mask.nonzero(as_tuple=True)[0]
    w[sal_indices] = a["sal"] * p_sal.to(w.device)

    w_sum = w.sum() + 1e-8
    x_bar = (w.unsqueeze(1) * X.abs()).sum(dim=0) / w_sum  # [d_in]

    return LayerStats(
        X_raw=X,
        e_lang=e["lang"], e_sal=e["sal"], e_bg=e["bg"],
        alpha_lang=alpha["lang"], alpha_sal=alpha["sal"], alpha_bg=alpha["bg"],
        a_lang=a["lang"], a_sal=a["sal"], a_bg=a["bg"],
        x_bar=x_bar,
        lang_mask=lang_mask, sal_mask=sal_mask, bg_mask=bg_mask,
    )


def collect_modal_stats(
    model,
    dataloader: DataLoader,
    target_layer_names: List[str],
    processor,
    top_k_ratio: float = 0.2,
    saliency_mode: str = SALIENCY_MODE,
    device: str = "cuda",
) -> Dict[str, LayerStats]:
    """
    一次 forward pass 收集所有层的共享统计量。

    Args:
        model               : Qwen3VLForConditionalGeneration（已 .to(device)）
        dataloader          : batch_size=1，每 batch 含 pixel_values + input_ids
                              + image_grid_thw
        target_layer_names  : 需要统计的 nn.Linear 层名列表
        processor           : AutoProcessor（用于获取 image_token_id）
        top_k_ratio         : 显著 token 比例（默认 top-20%）
        saliency_mode       : 由 test_vit_attention.py 决定
        device              : "cuda" 或 "cpu"
    Returns:
        Dict[layer_name -> LayerStats]
    """
    model.eval()
    model.to(device)

    # ── 1. 注册 forward hook 收集激活 ──────────────────────────
    storage: Dict[str, List[Tensor]] = {}

    def make_hook(name: str):
        def hook(module, inp, out):
            x = inp[0].detach().float()         # 取第一个输入；统一 float32
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])  # [B*seq, d_in]
            storage.setdefault(name, []).append(x.cpu())
        return hook

    handles = []
    for name, module in model.named_modules():
        if name in target_layer_names:
            handles.append(module.register_forward_hook(make_hook(name)))

    # ── 2. 获取 image_token_id ────────────────────────────────
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    # ── 3. forward pass 收集激活 + 显著性 ──────────────────────
    all_lang_masks: List[Tensor] = []
    all_sal_masks:  List[Tensor] = []
    all_bg_masks:   List[Tensor] = []
    all_p_sal:      List[Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            pixel_values   = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            input_ids      = batch["input_ids"].to(device)
            image_grid_thw = batch.get("image_grid_thw")
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(device)

            # 显著性分数
            p = compute_saliency(model, pixel_values, image_grid_thw, mode=saliency_mode)
            # p: [N_patches]

            # token 分组
            vis_mask, lang_mask = get_token_masks(input_ids, image_token_id)
            seq_len = vis_mask.shape[0]
            n_vis   = vis_mask.sum().item()

            # top-K 显著 token
            K = max(1, int(top_k_ratio * p.shape[0]))
            # p 的长度是 N_patches，vis_mask 中 True 的数量应等于 N_patches
            assert n_vis == p.shape[0], (
                f"vis token 数 {n_vis} != p 长度 {p.shape[0]}，"
                "请检查 image_grid_thw 和 image_token_id"
            )
            topk_vals, _ = torch.topk(p, K)
            threshold    = topk_vals[-1]
            sal_vis_mask = (p >= threshold)           # [N_patches] bool
            bg_vis_mask  = ~sal_vis_mask              # [N_patches] bool

            # 将 vis patch 级别的掩码展开到 seq 级别
            sal_mask_full = torch.zeros(seq_len, dtype=torch.bool)
            bg_mask_full  = torch.zeros(seq_len, dtype=torch.bool)
            vis_indices   = vis_mask.nonzero(as_tuple=True)[0]
            sal_mask_full[vis_indices[sal_vis_mask]] = True
            bg_mask_full[vis_indices[bg_vis_mask]]   = True

            p_sal = p[sal_vis_mask]  # [K]
            p_sal = p_sal / (p_sal.sum() + 1e-8)  # 重归一化

            all_lang_masks.append(lang_mask.cpu())
            all_sal_masks.append(sal_mask_full.cpu())
            all_bg_masks.append(bg_mask_full.cpu())
            all_p_sal.append(p_sal.cpu())

            # 触发 forward hook（需要完整 forward）
            model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

    # ── 4. 移除 hook ─────────────────────────────────────────
    for h in handles:
        h.remove()

    # ── 5. 拼接激活，计算每层统计量 ───────────────────────────
    lang_mask_cat = torch.cat(all_lang_masks, dim=0)  # [N_total]
    sal_mask_cat  = torch.cat(all_sal_masks,  dim=0)
    bg_mask_cat   = torch.cat(all_bg_masks,   dim=0)
    p_sal_cat     = torch.cat(all_p_sal,      dim=0)  # [sum_K]

    results: Dict[str, LayerStats] = {}
    for name in target_layer_names:
        if name not in storage:
            continue
        X = torch.cat(storage[name], dim=0).float()   # [N_total, d_in]

        # 截断到与 mask 长度一致（部分层 N 可能因 batching 存在微小差异）
        N = min(X.shape[0], lang_mask_cat.shape[0])
        X_trunc    = X[:N]
        lang_trunc = lang_mask_cat[:N]
        sal_trunc  = sal_mask_cat[:N]
        bg_trunc   = bg_mask_cat[:N]
        p_sal_trunc = p_sal_cat[:sal_trunc.sum().item()]

        results[name] = compute_stats_for_layer(
            X_trunc, lang_trunc, sal_trunc, bg_trunc, p_sal_trunc
        )
        del storage[name]  # 释放显存

    return results
