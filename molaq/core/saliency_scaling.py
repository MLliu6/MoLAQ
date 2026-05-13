"""
molaq/core/saliency_scaling.py
§3.6: 创新点 B — 显著 Token 驱动的缩放搜索

AWQ 风格的 grid search，用显著性加权统计量 x̄_j 替代全 token 均值。
B 的输入权重在不同消融配置下不同：
  - +B only  : W_input = 原始浮点 W_l
  - MoLAQ   : W_input = Ŵ_l^(A)（A 模块量化结果）
"""

import torch
from torch import Tensor


def rtn_quantize(W: Tensor, bits: int, group_size: int) -> Tensor:
    """
    RTN（Round-To-Nearest）量化。
    用于 AWQ grid search 的快速误差估计，不需要 Hessian。

    Args:
        W          : [d_out, d_in]  浮点权重
        bits       : 量化位数
        group_size : 量化分组大小
    Returns:
        W_q : [d_out, d_in]  量化后权重
    """
    d_out, d_in = W.shape
    W_q = W.clone()
    for g in range(0, d_in, group_size):
        wg    = W[:, g:g + group_size]
        scale = (
            wg.abs().max(dim=1, keepdim=True).values
            / (2 ** (bits - 1) - 1)
        )
        scale = scale.clamp(min=1e-8)
        W_q[:, g:g + group_size] = (
            (wg / scale).round()
            .clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
            * scale
        )
    return W_q


def saliency_awq_quantize(
    W_input:    Tensor,
    X_orig:     Tensor,
    x_bar:      Tensor,
    bits:       int = 4,
    group_size: int = 128,
    n_grid:     int = 20,
) -> Tensor:
    """
    显著 Token 驱动的 AWQ 缩放搜索量化。

    Args:
        W_input    : [d_out, d_in]
                     +B only 配置 → 原始浮点 W_l
                     MoLAQ 配置  → Ŵ_l^(A)
        X_orig     : [N, d_in]  原始激活（始终用原始 X，不用 X̃）
        x_bar      : [d_in]     显著性加权激活幅度（来自 modal_stats）
        bits       : 量化位数
        group_size : 量化分组大小
        n_grid     : grid search 步数（默认 20）
    Returns:
        best_W_q : [d_out, d_in]  最优量化权重

    公式（§3.6）：
        s_j = x̄_j^t，t ∈ [0,1] grid search
        误差参考点为 W_input（相对于 B 的输入，不是原始 W_l）
    """
    W_fp     = W_input.float()
    X_fp     = X_orig.float()
    x_bar_fp = x_bar.float().clamp(min=1e-8)

    # 子采样降显存压力（必须用 shape[0]，不能比较整个 shape）
    if X_fp.shape[0] > 4096:
        X_fp = X_fp[::4]

    best_err = float("inf")
    best_W_q = None

    for t in torch.linspace(0, 1, n_grid):
        s = x_bar_fp.pow(t.item()).clamp(min=1e-8)  # [d_in]

        W_scaled   = W_fp * s.unsqueeze(0)           # W · diag(s)
        W_scaled_q = rtn_quantize(W_scaled, bits=bits, group_size=group_size)
        W_q        = W_scaled_q / s.unsqueeze(0)     # 逆变换

        # 误差参考点为 W_input（B 的参考基准）
        err = (
            (W_q @ X_fp.T) - (W_fp @ X_fp.T)
        ).pow(2).mean().item()

        if err < best_err:
            best_err = err
            best_W_q = W_q

    return best_W_q
