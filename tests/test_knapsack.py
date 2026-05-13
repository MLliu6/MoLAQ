"""Sanity check for the greedy knapsack allocator."""
from molaq.allocation.knapsack import greedy_bit_allocation


def test_greedy_basic():
    layers = ["layer_a", "layer_b", "layer_c"]
    param_counts = {"layer_a": 1000, "layer_b": 2000, "layer_c": 500}
    sigma = {"layer_a": 0.1, "layer_b": 5.0, "layer_c": 0.05}
    delta = {
        "layer_a": {4: 0.01, 8: 0.001},
        "layer_b": {4: 0.10, 8: 0.005},
        "layer_c": {4: 0.005, 8: 0.001},
    }
    # Budget = 6.0 bits/param, should flip some layers to INT4
    assignment = greedy_bit_allocation(layers, param_counts, sigma, delta, budget_bits=6.0)

    # High sensitivity layer_b should stay INT8
    assert assignment["layer_b"] == 8, "High sensitivity layer should stay INT8"
    # Low sensitivity layers should be INT4 first
    assert assignment["layer_a"] == 4 or assignment["layer_c"] == 4

    print("test_greedy_basic PASSED")


if __name__ == "__main__":
    test_greedy_basic()
