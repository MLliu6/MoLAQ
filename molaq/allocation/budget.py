"""
Memory and bit budget utilities for MoLAQ.
"""


def estimate_vram_gb(param_counts: dict, assignment: dict) -> float:
    """Estimates total model weight VRAM given a bit assignment."""
    total_bits = sum(param_counts[l] * assignment.get(l, 16) for l in param_counts)
    return total_bits / 8 / 1e9


def compute_average_bits(param_counts: dict, assignment: dict) -> float:
    total_params = sum(param_counts.values())
    total_bits = sum(param_counts[l] * assignment.get(l, 8) for l in param_counts)
    return total_bits / total_params if total_params > 0 else 0.0
