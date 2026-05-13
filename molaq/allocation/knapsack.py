"""
0-1 Knapsack solver for mixed-precision bit allocation.

Optimization problem:
    min  Σ_l  σ_l · Δ_l(b_l)
    s.t. Σ_l  |W_l| · b_l ≤ B_total
    b_l ∈ {4, 8}

Greedy heuristic: assign INT4 to layers with best
(sensitivity-weighted error reduction) / (bit savings) ratio.
"""
import json
from typing import Dict, List, Tuple


def greedy_bit_allocation(
    layer_names: List[str],
    param_counts: Dict[str, int],
    sigma: Dict[str, float],
    delta: Dict[str, Dict[int, float]],
    budget_bits: float,          # average bits per param budget, e.g. 4.5
    candidate_bits: List[int] = [4, 8]
) -> Dict[str, int]:
    """
    Greedy knapsack: start all layers at INT8, greedily flip to INT4
    layers with the best gain/cost ratio until budget is met.

    gain = σ_l · (Δ_l(8) - Δ_l(4))   [lower is better for error, so gain = error saved]
    cost = |W_l| · 4                   [bits saved by going 8->4]
    ratio = gain / cost                [we want HIGH gain with LOW cost = flip safely]

    Args:
        layer_names: list of layer identifiers
        param_counts: {layer_name: number of params}
        sigma: {layer_name: sensitivity score}
        delta: {layer_name: {4: error4, 8: error8}}
        budget_bits: target average bits per param
        candidate_bits: [4, 8]

    Returns:
        bit_assignment: {layer_name: bit_width}
    """
    total_params = sum(param_counts.values())
    B_total = budget_bits * total_params

    # Start all at INT8
    assignment = {l: 8 for l in layer_names}
    current_bits = total_params * 8.0

    # Compute gain/cost ratio for flipping each layer from 8 -> 4
    scores = []
    for l in layer_names:
        error_reduction = sigma[l] * (delta[l][8] - delta[l][4])
        bit_saving = param_counts[l] * 4  # bits saved
        if bit_saving <= 0:
            continue
        ratio = error_reduction / bit_saving  # want to minimize this when flipping to INT4
        scores.append((ratio, l))

    # Sort ascending: flip layers with smallest sensitivity-weighted error loss first
    scores.sort(key=lambda x: x[0])

    for ratio, l in scores:
        if current_bits - param_counts[l] * 4 >= B_total:
            break
        assignment[l] = 4
        current_bits -= param_counts[l] * 4

    achieved_avg_bits = current_bits / total_params
    print(f"[MoLAQ] Budget: {budget_bits:.2f} bits/param | "
          f"Achieved: {achieved_avg_bits:.3f} bits/param")
    int4_count = sum(1 for b in assignment.values() if b == 4)
    print(f"[MoLAQ] INT4 layers: {int4_count} / {len(layer_names)}")
    return assignment


def save_bit_assignment(assignment: Dict[str, int], path: str):
    with open(path, "w") as f:
        json.dump(assignment, f, indent=2)
    print(f"[MoLAQ] Bit assignment saved to {path}")


def load_bit_assignment(path: str) -> Dict[str, int]:
    with open(path) as f:
        return json.load(f)
