"""
molaq/assign/knapsack.py
§3.7: 混合精度贪心分配

基于边际代价 δ_l 排序，在平均 bit 预算约束下决定每层 INT4/INT8。
"""

from typing import Dict, List

import torch
from torch import Tensor

from molaq.core.saliency_scaling import rtn_quantize


def estimate_delta(W: Tensor, bits: int, group_size: int = 128) -> float:
    """
    用 RTN 误差估计 Δ_l(b)（不需要校准集 forward pass）。

    Args:
        W          : [d_out, d_in]  原始浮点权重
        bits       : 目标位数（4 或 8）
        group_size : 量化分组大小
    Returns:
        delta : float  MSE 量化误差
    """
    W_q = rtn_quantize(W.float(), bits=bits, group_size=group_size)
    return (W_q - W.float()).pow(2).mean().item()


def greedy_bit_allocation(
    layer_names:     List[str],
    param_counts:    Dict[str, int],
    hessian_trace:   Dict[str, float],
    delta_4:         Dict[str, float],
    delta_8:         Dict[str, float],
    budget_avg_bits: float = 4.5,
) -> Dict[str, int]:
    """
    贪心背包求解混合精度分配。

    Args:
        layer_names     : 需要分配的层名列表
        param_counts    : {layer_name: 参数量（元素个数）}
        hessian_trace   : {layer_name: tr(H^MoLAQ)}，即层级敏感度 σ_l
        delta_4         : {layer_name: Δ_l(4)，RTN 误差}
        delta_8         : {layer_name: Δ_l(8)，RTN 误差}
        budget_avg_bits : 平均 bit 预算（默认 4.5）
    Returns:
        assignment : {layer_name: bit_width}，bit_width ∈ {4, 8}

    算法（§3.7）：
        δ_l = σ_l · (Δ_l(8) - Δ_l(4))
        δ_l 升序排列，优先将 δ_l 最小的层分配为 INT4（精度损失最小）
        δ_l 最大的层对精度最敏感，保留 INT8
        直到再翻转一层会超出 bit 预算为止
    """
    total_params = sum(param_counts[l] for l in layer_names)
    budget_total = budget_avg_bits * total_params  # 总 bit 上限

    delta_margin = {
        l: hessian_trace[l] * (delta_8[l] - delta_4[l])
        for l in layer_names
    }

    assignment   = {l: 8 for l in layer_names}   # 初始全 INT8
    current_bits = float(total_params * 8)

    # δ_l 升序：优先翻转损失最小的层为 INT4
    for l in sorted(layer_names, key=lambda l: delta_margin[l]):
        bits_saved = param_counts[l] * 4        # INT8 → INT4 节省的 bit 数
        if current_bits - bits_saved < budget_total:
            # 翻转后低于预算下界，停止
            break
        assignment[l]  = 4
        current_bits  -= bits_saved

    achieved = current_bits / total_params
    n_int4   = sum(1 for b in assignment.values() if b == 4)
    print(
        f"[knapsack] budget={budget_avg_bits:.2f}b | "
        f"achieved={achieved:.3f}b | "
        f"INT4={n_int4}/{len(layer_names)}"
    )
    return assignment
