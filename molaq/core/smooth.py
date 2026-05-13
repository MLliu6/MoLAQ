"""
molaq/core/smooth.py
§3.4: 创新点 C — 模态感知激活预平滑

等价变换：W̃ · X̃ᵀ = W · Xᵀ（数值误差 < 1e-3，见 tests/test_smooth.py）
"""

import torch
from torch import Tensor
from typing import Tuple


def compute_smooth_scale(x_bar: Tensor, W: Tensor) -> Tensor:
    """
    计算逐通道缩放因子 s_j。

    Args:
        x_bar : [d_in]  显著性加权激活幅度（来自 modal_stats.x_bar）
        W     : [d_out, d_in]  原始浮点权重
    Returns:
        s     : [d_in]  缩放因子，值域 (0, 1e4]

    公式（§3.4）：
        ρ_j = x̄_j / (x̄_j + max_k|W_{k,j}|)
        s_j = x̄_j^{ρ_j} · max_k|W_{k,j}|^{-(1-ρ_j)}
    """
    assert x_bar.dim() == 1, f"x_bar 应为 1D，实际 shape={x_bar.shape}"
    assert W.dim() == 2,     f"W 应为 2D，实际 shape={W.shape}"
    assert x_bar.shape[0] == W.shape[1], (
        f"d_in 不一致：x_bar={x_bar.shape[0]}, W={W.shape[1]}"
    )

    w_max = W.abs().max(dim=0).values   # [d_in]
    x_bar = x_bar.clamp(min=1e-8)
    w_max = w_max.clamp(min=1e-8)

    rho   = x_bar / (x_bar + w_max)    # [d_in], ∈ (0, 1)

    # log-exp 计算，避免极端值溢出
    log_s = rho * x_bar.log() - (1.0 - rho) * w_max.log()
    s     = log_s.exp().clamp(max=1e4)  # [d_in]
    return s


def apply_smooth(X: Tensor, W: Tensor, s: Tensor) -> Tuple[Tensor, Tensor]:
    """
    施加等价缩放变换。

    Args:
        X : [N, d_in]   原始激活
        W : [d_out, d_in]  原始权重
        s : [d_in]      缩放因子（来自 compute_smooth_scale）
    Returns:
        X_tilde : [N, d_in]     平滑后激活，= X / diag(s)
        W_tilde : [d_out, d_in] 平滑后权重，= W * diag(s)

    等价性：W_tilde @ X_tilde.T == W @ X.T
    """
    assert X.dim() == 2 and W.dim() == 2 and s.dim() == 1, (
        "shape 错误：X/W 应为 2D，s 应为 1D"
    )
    assert X.shape[1] == W.shape[1] == s.shape[0], (
        f"d_in 不一致：X={X.shape[1]}, W={W.shape[1]}, s={s.shape[0]}"
    )

    X_tilde = X / s.unsqueeze(0)    # [N, d_in]
    W_tilde = W * s.unsqueeze(0)    # [d_out, d_in]
    return X_tilde, W_tilde
