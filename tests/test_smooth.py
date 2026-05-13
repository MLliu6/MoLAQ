#!/usr/bin/env python
"""
tests/test_smooth.py

Sanity check for smooth.py (创新点 C)。
通过标准：max_err < 1e-3。

运行方式：
    source ~/vllm/bin/activate
    cd ~/MoLAQ
    pytest -q tests/test_smooth.py
    # 或直接运行：
    python tests/test_smooth.py
"""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from molaq.core.smooth import compute_smooth_scale, apply_smooth


def test_smooth_equivalence():
    """验证等价变换：W̃ @ X̃.T ≈ W @ X.T，误差 < 1e-3"""
    torch.manual_seed(42)
    N, d_in, d_out = 64, 128, 256
    X = torch.randn(N, d_in)
    W = torch.randn(d_out, d_in)
    x_bar = X.abs().mean(dim=0)

    s = compute_smooth_scale(x_bar, W)
    X_t, W_t = apply_smooth(X, W, s)

    out_orig   = X   @ W.T
    out_smooth = X_t @ W_t.T
    max_err = (out_orig - out_smooth).abs().max().item()

    print(f"test_smooth_equivalence: max_err={max_err:.2e}")
    assert max_err < 1e-3, f"FAILED: max_err={max_err} >= 1e-3"
    print("PASSED")


def test_smooth_scale_range():
    """验证 s_j 在合理范围 (0, 1e4] 内"""
    torch.manual_seed(0)
    X = torch.randn(32, 64)
    W = torch.randn(128, 64)
    x_bar = X.abs().mean(dim=0)
    s = compute_smooth_scale(x_bar, W)

    assert (s > 0).all(),    f"s 中存在非正值: min={s.min().item()}"
    assert (s <= 1e4).all(), f"s 中存在超出上限的值: max={s.max().item()}"
    print(f"test_smooth_scale_range: s in ({s.min().item():.4f}, {s.max().item():.4f})  PASSED")


def test_apply_smooth_shapes():
    """验证 apply_smooth 输出 shape 正确"""
    N, d_in, d_out = 16, 64, 32
    X = torch.randn(N, d_in)
    W = torch.randn(d_out, d_in)
    s = torch.ones(d_in)
    X_t, W_t = apply_smooth(X, W, s)
    assert X_t.shape == (N, d_in),    f"X_tilde shape 错误: {X_t.shape}"
    assert W_t.shape == (d_out, d_in), f"W_tilde shape 错误: {W_t.shape}"
    print("test_apply_smooth_shapes: PASSED")


if __name__ == "__main__":
    test_smooth_equivalence()
    test_smooth_scale_range()
    test_apply_smooth_shapes()
    print("\n所有 smooth 测试通过 ✓")
