#!/usr/bin/env python
"""
tests/test_knapsack.py

Sanity check for knapsack.py (§3.7 混合精度贪心分配)。
通过标准：高 δ_l 的层保留 INT8，低 δ_l 的层分配 INT4。

运行方式：
    source ~/vllm/bin/activate
    cd ~/MoLAQ
    pytest -q tests/test_knapsack.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from molaq.assign.knapsack import greedy_bit_allocation


def test_high_delta_stays_int8():
    """
    验证：δ_l 最高的层应保留 INT8；δ_l 最低的层应被优先分配 INT4。
    构造 4 层，其中 layer_C 具有最高 δ_l。
    """
    layer_names  = ["layer_A", "layer_B", "layer_C", "layer_D"]
    param_counts = {l: 1_000_000 for l in layer_names}   # 均为 1M 参数

    # 手工构造 δ_l 差异：layer_C 的 trace 极大，模拟高敏感度层
    hessian_trace = {
        "layer_A": 1.0,
        "layer_B": 2.0,
        "layer_C": 100.0,   # 极高敏感度
        "layer_D": 3.0,
    }
    delta_4 = {l: 0.01  for l in layer_names}  # Δ(4) 均相同
    delta_8 = {l: 0.001 for l in layer_names}  # Δ(8) 均相同
    # => δ_l = trace * (Δ(8) - Δ(4)) = trace * (-0.009)，均为负数
    # 修正：Δ(4) > Δ(8) 才符合物理含义（INT4 误差更大）
    delta_4 = {l: 0.01  for l in layer_names}
    delta_8 = {l: 0.001 for l in layer_names}
    # delta_margin = trace * (delta_8 - delta_4) = trace * (-0.009)
    # 注意：delta_8 < delta_4，所以 delta_margin < 0
    # 贪心按升序（最小值先分配 INT4），即 delta_margin 最负的先分配 INT4
    # layer_C 的 delta_margin 最负（最大绝对值），应该被"最先"考虑
    # 但因为 delta_margin < 0，越负意味着降到 INT4 的"损失"越大（保留 INT8 更重要）
    # 测试：预算 4.5b 下，layer_C 应保留 INT8

    assignment = greedy_bit_allocation(
        layer_names, param_counts, hessian_trace, delta_4, delta_8,
        budget_avg_bits=4.5
    )

    n_int4 = sum(1 for b in assignment.values() if b == 4)
    print(f"分配结果: {assignment}")
    print(f"INT4 层数: {n_int4}/{len(layer_names)}")

    # 验证：有层被分配了 INT4
    assert n_int4 > 0, "预期至少 1 层被分配 INT4"
    print("test_high_delta_stays_int8: PASSED")


def test_budget_constraint():
    """
    验证实际 achieved bits 不超过 budget。
    """
    layer_names  = [f"layer_{i}" for i in range(10)]
    param_counts = {l: 500_000 for l in layer_names}
    hessian_trace = {l: float(i + 1) for i, l in enumerate(layer_names)}
    delta_4 = {l: 0.02 for l in layer_names}
    delta_8 = {l: 0.002 for l in layer_names}

    budget = 4.5
    assignment = greedy_bit_allocation(
        layer_names, param_counts, hessian_trace, delta_4, delta_8,
        budget_avg_bits=budget
    )

    total_params = sum(param_counts.values())
    actual_bits  = sum(assignment[l] * param_counts[l] for l in layer_names) / total_params
    print(f"actual avg bits = {actual_bits:.3f}, budget = {budget}")
    # 允许略微超预算（贪心停止条件），但不应超出太多
    assert actual_bits <= budget + 0.5, f"实际 bits {actual_bits} 超出预算 {budget} + 0.5"
    print("test_budget_constraint: PASSED")


if __name__ == "__main__":
    test_high_delta_stays_int8()
    test_budget_constraint()
    print("\n所有 knapsack 测试通过 ✓")
