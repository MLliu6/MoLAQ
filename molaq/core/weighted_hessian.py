"""
molaq/core/weighted_hessian.py
§3.5: 创新点 A — 模态感知加权 Hessian + GPTQ 逐列量化

核心改变：用模态加权 Hessian H^MoLAQ 替换 GPTQ 的等权 H，
Cholesky 分解与逐列更新循环完全复用 GPTQ。
所有计算均在传入张量所在设备上执行。
"""

import torch
from torch import Tensor
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from molaq.stats.modal_stats import LayerStats

from molaq.core.smooth import apply_smooth


def compute_modal_hessian(
    X_tilde:    Tensor,
    alpha_lang: float,
    alpha_sal:  float,
    alpha_bg:   float,
    lang_mask:  Tensor,
    sal_mask:   Tensor,
    bg_mask:    Tensor,
    damp:       float = 1e-3,
) -> Tensor:
    """
    计算 MoLAQ 模态加权 Hessian。所有计算在 X_tilde.device 上进行。

    Args:
        X_tilde    : [N, d_in]  平滑后激活
        alpha_lang/sal/bg : 各组 Hessian 权重系数（sum=3）
        lang/sal/bg_mask  : [N] bool，token 分组掩码
        damp       : 正则化强度 λ，默认 1e-3
    Returns:
        H : [d_in, d_in]，正定 Hessian。与 X_tilde 在同一设备
    """
    dev  = X_tilde.device
    d_in = X_tilde.shape[1]

    # 掩码统一到 X_tilde 设备
    lang_mask = lang_mask.to(dev)
    sal_mask  = sal_mask.to(dev)
    bg_mask   = bg_mask.to(dev)

    def gram(mask: Tensor) -> Tensor:
        if mask.sum() == 0:
            return torch.zeros(d_in, d_in, dtype=torch.float32, device=dev)
        x_m = X_tilde[mask].float()
        N_m = x_m.shape[0]
        return x_m.T @ x_m / N_m

    G_lang = gram(lang_mask)
    G_sal  = gram(sal_mask)
    G_bg   = gram(bg_mask)

    H = (2.0 / 3.0) * (
        alpha_lang * G_lang +
        alpha_sal  * G_sal  +
        alpha_bg   * G_bg
    )

    trace_H = H.trace().item()
    H = H + damp * trace_H * torch.eye(d_in, dtype=H.dtype, device=dev)
    return H


def gptq_quantize(
    W_tilde:    Tensor,
    H:          Tensor,
    bits:       int  = 4,
    group_size: int  = 128,
    sym:        bool = True,
) -> Tensor:
    """
    GPTQ Cholesky 逐列量化。所有计算在 W_tilde.device 上进行。
    """
    dev = W_tilde.device
    d_out, d_in = W_tilde.shape
    W_q = W_tilde.clone().float()
    H   = H.to(device=dev, dtype=torch.float32)

    try:
        L = torch.linalg.cholesky(H)
    except torch.linalg.LinAlgError:
        d = H.shape[0]
        H = H + 1e-2 * H.trace() / d * torch.eye(d, dtype=H.dtype, device=dev)
        L = torch.linalg.cholesky(H)

    H_inv = torch.cholesky_inverse(L)  # [d_in, d_in]

    for i in range(d_in):
        w_col   = W_q[:, i]
        g_start = (i // group_size) * group_size
        g_end   = min(g_start + group_size, d_in)
        w_group = W_q[:, g_start:g_end]

        scale = (
            w_group.abs().max(dim=1, keepdim=True).values
            / (2 ** (bits - 1) - 1)
        ).clamp(min=1e-8)

        if sym:
            w_col_q = (
                (w_col / scale.squeeze(1)).round()
                .clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
                * scale.squeeze(1)
            )
        else:
            raise NotImplementedError("非对称量化暂未实现")

        quant_err = w_col_q - w_col
        h_inv_row = H_inv[i, i + 1:]
        h_inv_ii  = H_inv[i, i].clamp(min=1e-8)

        W_q[:, i]      = w_col_q
        W_q[:, i + 1:] -= torch.outer(quant_err, h_inv_row / h_inv_ii)

    return W_q


def quantize_layer_A(
    stats,
    W:          Tensor,
    s:          Tensor,
    enable_C:   bool = True,
    bits:       int  = 4,
    group_size: int  = 128,
) -> Tuple[Tensor, Tensor]:
    """
    Stage 3 完整量化接口。所有 stats 张量统一到 W.device。
    """
    dev = W.device

    # 将 stats 中所有 Tensor 统一到 W.device
    X_raw      = stats.X_raw.to(device=dev, dtype=torch.float32)
    lang_mask  = stats.lang_mask.to(dev)
    sal_mask   = stats.sal_mask.to(dev)
    bg_mask    = stats.bg_mask.to(dev)

    if enable_C:
        s_dev = s.to(device=dev, dtype=torch.float32)
        X_tilde, W_tilde = apply_smooth(X_raw, W, s_dev)
    else:
        X_tilde = X_raw
        W_tilde = W.float()
        s_dev   = torch.ones(W.shape[1], dtype=torch.float32, device=dev)

    H = compute_modal_hessian(
        X_tilde,
        stats.alpha_lang, stats.alpha_sal, stats.alpha_bg,
        lang_mask, sal_mask, bg_mask,
    )

    W_tilde_q = gptq_quantize(W_tilde, H, bits=bits, group_size=group_size)

    # 恢复到原始空间
    W_hat_A = W_tilde_q / s_dev.unsqueeze(0)

    return W_hat_A, s_dev


def sanity_check_layer(W_hat_A: Tensor, W: Tensor, H: Tensor, X_raw: Tensor):
    """
    量化后 sanity check。打印量化误差和条件数改善情况。
    """
    dev     = W_hat_A.device
    W_f     = W.to(device=dev, dtype=torch.float32)
    X_raw_f = X_raw.to(device=dev, dtype=torch.float32)

    err      = (W_hat_A - W_f).abs()
    mean_err = err.mean().item()
    max_err  = err.max().item()
    print(f"  [sanity] quant error: mean={mean_err:.4f}, max={max_err:.4f}", end="")
    if mean_err > 0.02:
        print("  ⚠️  mean > 0.02，检查 damp 正则化强度", end="")
    if max_err > 0.5:
        print("  ⚠️  max > 0.5，检查该层权重分布", end="")
    print()

    N       = X_raw_f.shape[0]
    H_gptq  = (2.0 / N) * X_raw_f.T @ X_raw_f
    kappa_gptq  = torch.linalg.cond(H_gptq).item()
    kappa_molaq = torch.linalg.cond(H.to(device=dev, dtype=torch.float32)).item()
    print(f"  [sanity] κ(H_GPTQ)={kappa_gptq:.1f}, κ(H_MoLAQ)={kappa_molaq:.1f}", end="")
    if kappa_molaq > kappa_gptq:
        print("  ⚠️  κ(H_MoLAQ) > κ(H_GPTQ)，检查 smooth_scale 是否正确施加", end="")
    print()
