"""
molaq/core/saliency_scaling.py
§3.6: 创新点 B — 显著 token 缩放搜索
所有计算均在 W.device 上进行。
"""

import torch
from torch import Tensor


def rtn_quantize(W: Tensor, bits: int, group_size: int) -> Tensor:
    """
    Round-to-Nearest 对称量化（用于 B 模块搜索随机种子 t*）。
    W 和返回张量均在 W.device 上。
    """
    d_out, d_in = W.shape
    W_q = W.clone().float()
    for g_start in range(0, d_in, group_size):
        g_end   = min(g_start + group_size, d_in)
        w_group = W_q[:, g_start:g_end]
        scale   = (
            w_group.abs().max(dim=1, keepdim=True).values
            / (2 ** (bits - 1) - 1)
        ).clamp(min=1e-8)
        W_q[:, g_start:g_end] = (
            (w_group / scale).round()
            .clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
            * scale
        )
    return W_q


def saliency_awq_quantize(
    W:          Tensor,
    X_raw:      Tensor,
    x_bar:      Tensor,
    bits:       int   = 4,
    group_size: int   = 128,
    grid_size:  int   = 20,
    alpha_min:  float = 0.0,
    alpha_max:  float = 1.0,
) -> Tensor:
    """
    显著 token 缩放搜索（B 模块）。

    公式：搜索最优 t* = argmin_{t in grid} ||W_q(t) - W|| * ||X_raw||
    缩放因子：s_j(t) = x_bar_j^t

    所有张量统一到 W.device 上进行。
    """
    dev  = W.device
    W_f  = W.to(device=dev, dtype=torch.float32)
    X_f  = X_raw.to(device=dev, dtype=torch.float32)
    xb   = x_bar.to(device=dev, dtype=torch.float32).clamp(min=1e-8)

    best_err = float("inf")
    best_W_q = W_f.clone()

    for i in range(grid_size + 1):
        t = alpha_min + (alpha_max - alpha_min) * i / grid_size
        s = xb ** t                              # [d_in]
        W_scaled = W_f * s.unsqueeze(0)          # [d_out, d_in]
        X_scaled = X_f / s.unsqueeze(0)          # [N, d_in]
        W_q_s    = rtn_quantize(W_scaled, bits, group_size)
        W_q      = W_q_s / s.unsqueeze(0)        # 恢复原始空间

        x_norm   = X_f.norm(dim=0)               # [d_in]
        err      = ((W_q - W_f) * x_norm.unsqueeze(0)).norm().item()
        if err < best_err:
            best_err = err
            best_W_q = W_q

    return best_W_q
