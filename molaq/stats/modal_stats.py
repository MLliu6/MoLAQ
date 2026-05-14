"""
molaq/stats/modal_stats.py
§3.2–3.3: Token 分组、激活收集、共享统计量

设备策略：所有激活、掩码、统计量全部在 device（cuda）上存储和计算。
不允许任何 .cpu() 调用出现在热点路径中。
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
    单层共享统计量。所有 Tensor 均在 device（cuda）上。
    """
    X_raw:      Tensor   # [N, d_in]  float32，cuda
    e_lang:     float
    e_sal:      float
    e_bg:       float
    alpha_lang: float
    alpha_sal:  float
    alpha_bg:   float
    a_lang:     float
    a_sal:      float
    a_bg:       float
    x_bar:      Tensor   # [d_in]  float32，cuda
    lang_mask:  Tensor   # [N] bool，cuda
    sal_mask:   Tensor   # [N] bool，cuda
    bg_mask:    Tensor   # [N] bool，cuda


def compute_saliency(
    model,
    pixel_values: Tensor,
    image_grid_thw: Tensor,
    mode: str = "act_norm",
) -> Tensor:
    """
    计算视觉 patch 的显著性分数 p_i，sum(p)==1。
    返回张量与 pixel_values 在同一设备。
    """
    if mode == "cls_attn":
        with torch.no_grad():
            out = model.visual(
                pixel_values, grid_thw=image_grid_thw, output_attentions=True
            )
        last_attn = out.attentions[-1]
        cls_attn  = last_attn[0, :, 0, 1:]
        p = cls_attn.mean(dim=0)

    elif mode == "row_sum":
        with torch.no_grad():
            out = model.visual(
                pixel_values, grid_thw=image_grid_thw, output_attentions=True
            )
        last_attn = out.attentions[-1]
        p = last_attn[0].mean(dim=0).mean(dim=0)

    elif mode == "act_norm":
        with torch.no_grad():
            vit_out = model.visual(pixel_values, grid_thw=image_grid_thw)
        if isinstance(vit_out, Tensor):
            feat = vit_out
        elif isinstance(vit_out, tuple):
            feat = vit_out[0]
        elif hasattr(vit_out, "last_hidden_state"):
            feat = vit_out.last_hidden_state[0]
        else:
            raise TypeError(f"model.visual() 返回未知类型 {type(vit_out)}")
        p = feat.float().norm(dim=-1)  # [N_patches]，在 cuda 上

    else:
        raise ValueError(f"Unknown saliency mode: {mode}")

    p = p.float()
    p = p / (p.sum() + 1e-8)
    return p  # cuda Tensor


def get_token_masks(
    input_ids: Tensor,
    image_token_id: int,
) -> Tuple[Tensor, Tensor]:
    """input_ids 和返回掩码均保持在原设备上。"""
    ids = input_ids.squeeze(0)
    vis_mask  = (ids == image_token_id)
    lang_mask = ~vis_mask
    return vis_mask, lang_mask


def compute_stats_for_layer(
    X:         Tensor,  # [N, d_in]，float32，cuda
    lang_mask: Tensor,  # [N] bool，cuda
    sal_mask:  Tensor,  # [N] bool，cuda
    bg_mask:   Tensor,  # [N] bool，cuda
    p_sal:     Tensor,  # [N_sal]，cuda
) -> LayerStats:
    assert p_sal.shape[0] == sal_mask.sum().item(), (
        f"p_sal 长度 {p_sal.shape[0]} != sal_mask.sum() {sal_mask.sum().item()}"
    )

    def energy(mask: Tensor) -> float:
        if mask.sum() == 0:
            return 1e-8
        return X[mask].pow(2).mean().item()

    e = {"lang": energy(lang_mask), "sal": energy(sal_mask), "bg": energy(bg_mask)}
    e_sum = sum(e.values())
    alpha = {m: 3.0 * e[m] / e_sum for m in e}
    e_max = max(e.values())
    a = {m: e[m] / e_max for m in e}

    dev = X.device
    N   = X.shape[0]
    w   = torch.zeros(N, dtype=torch.float32, device=dev)
    w[lang_mask] = a["lang"]
    w[bg_mask]   = a["bg"]
    sal_indices  = sal_mask.nonzero(as_tuple=True)[0]
    w[sal_indices] = a["sal"] * p_sal.to(dev)

    w_sum = w.sum() + 1e-8
    x_bar = (w.unsqueeze(1) * X.abs()).sum(dim=0) / w_sum  # [d_in]，cuda

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
) -> Dict[str, "LayerStats"]:
    """
    一次 forward pass 收集所有层的共享统计量。
    所有张量均在 device（cuda）上，不做任何 .cpu() 操作。
    """
    model.eval()

    # hook 存储：activations 留在 cuda
    storage: Dict[str, List[Tensor]] = {}

    def make_hook(name: str):
        def hook(module, inp, out):
            x = inp[0].detach().float()
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])  # [B*seq, d_in]。不 .cpu()
            storage.setdefault(name, []).append(x)
        return hook

    handles = []
    for name, module in model.named_modules():
        if name in target_layer_names:
            handles.append(module.register_forward_hook(make_hook(name)))

    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

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

            p = compute_saliency(model, pixel_values, image_grid_thw, mode=saliency_mode)

            vis_mask, lang_mask = get_token_masks(input_ids, image_token_id)
            seq_len = vis_mask.shape[0]
            n_vis   = vis_mask.sum().item()

            K = max(1, int(top_k_ratio * p.shape[0]))
            assert n_vis == p.shape[0], (
                f"vis token 数 {n_vis} != p 长度 {p.shape[0]}"
            )
            topk_vals, _ = torch.topk(p, K)
            threshold    = topk_vals[-1]
            sal_vis_mask = (p >= threshold)
            bg_vis_mask  = ~sal_vis_mask

            # 在 cuda 上构造 seq_len 长度掩码
            dev = pixel_values.device
            sal_mask_full = torch.zeros(seq_len, dtype=torch.bool, device=dev)
            bg_mask_full  = torch.zeros(seq_len, dtype=torch.bool, device=dev)
            vis_indices   = vis_mask.nonzero(as_tuple=True)[0]
            sal_mask_full[vis_indices[sal_vis_mask]] = True
            bg_mask_full[vis_indices[bg_vis_mask]]   = True

            p_sal = p[sal_vis_mask]
            p_sal = p_sal / (p_sal.sum() + 1e-8)

            # 全部保持在 cuda
            all_lang_masks.append(lang_mask)
            all_sal_masks.append(sal_mask_full)
            all_bg_masks.append(bg_mask_full)
            all_p_sal.append(p_sal)

            model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

    for h in handles:
        h.remove()

    lang_mask_cat = torch.cat(all_lang_masks, dim=0)
    sal_mask_cat  = torch.cat(all_sal_masks,  dim=0)
    bg_mask_cat   = torch.cat(all_bg_masks,   dim=0)
    p_sal_cat     = torch.cat(all_p_sal,      dim=0)

    results: Dict[str, LayerStats] = {}
    for name in target_layer_names:
        if name not in storage:
            continue
        X = torch.cat(storage[name], dim=0).float()  # cuda float32

        N = min(X.shape[0], lang_mask_cat.shape[0])
        X_trunc     = X[:N]
        lang_trunc  = lang_mask_cat[:N]
        sal_trunc   = sal_mask_cat[:N]
        bg_trunc    = bg_mask_cat[:N]
        p_sal_trunc = p_sal_cat[:sal_trunc.sum().item()]

        results[name] = compute_stats_for_layer(
            X_trunc, lang_trunc, sal_trunc, bg_trunc, p_sal_trunc
        )
        del storage[name]
        torch.cuda.empty_cache()

    return results
