"""
molaq/core/smooth.py
§3.4: 创新点 C — 模态感知激活预平滑

等价变换：W̃ · X̃ᵀ = W · Xᵀ（数値误差 < 1e-3，见 tests/test_smooth.py）
所有计算均在 x_bar/W 所在设备上执行。
"""

import torch
from torch import Tensor
from typing import Tuple


def compute_smooth_scale(x_bar: Tensor, W: Tensor) -> Tensor:
    """
    计算逐通道缩放因子 s_j。所有计算在 W.device 上进行。

    Args:
        x_bar : [d_in]  显著性加权激活幅度（来自 modal_stats.x_bar）
        W     : [d_out, d_in]  原始浮点权重
    Returns:
        s     : [d_in]  缩放因子，值域 (0, 1e4]，与 W 在同一设备

    公式（§3.4）：
        ρ_j = x̄_j / (x̄_j + max_k|W_{k,j}|)
        s_j = x̄_j^{ρ_j} · max_k|W_{k,j}|^{-(1-ρ_j)}
    """
    assert x_bar.dim() == 1, f"x_bar 应为 1D，实际 shape={x_bar.shape}"
    assert W.dim() == 2,     f"W 应为 2D，实际 shape={W.shape}"
    assert x_bar.shape[0] == W.shape[1], (
        f"d_in 不一致：x_bar={x_bar.shape[0]}, W={W.shape[1]}"
    )

    # 统一到 W.device，float32
    dev   = W.device
    x_bar = x_bar.to(device=dev, dtype=torch.float32)
    W_f   = W.float()

    w_max = W_f.abs().max(dim=0).values    # [d_in]
    x_bar = x_bar.clamp(min=1e-8)
    w_max = w_max.clamp(min=1e-8)

    rho   = x_bar / (x_bar + w_max)       # [d_in], ∈ (0, 1)

    # log-exp 计算，避免极端値溢出
    log_s = rho * x_bar.log() - (1.0 - rho) * w_max.log()
    s     = log_s.exp().clamp(max=1e4)    # [d_in]
    return s


def apply_smooth(X: Tensor, W: Tensor, s: Tensor) -> Tuple[Tensor, Tensor]:
    """
    施加等价缩放变换。所有张量统一到 W.device。

    Args:
        X : [N, d_in]      原始激活
        W : [d_out, d_in]  原始权重
        s : [d_in]         缩放因子（来自 compute_smooth_scale）
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

    dev   = W.device
    X_f   = X.to(device=dev, dtype=torch.float32)
    W_f   = W.float()
    s_f   = s.to(device=dev, dtype=torch.float32)

    X_tilde = X_f / s_f.unsqueeze(0)    # [N, d_in]
    W_tilde = W_f * s_f.unsqueeze(0)    # [d_out, d_in]
    return X_tilde, W_tilde
