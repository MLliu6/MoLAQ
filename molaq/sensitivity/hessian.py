"""
Hessian approximation for per-layer sensitivity in MoLAQ.
H_l = (2/N) * X_l^T X_l  (diagonal approximation)
Reuses activation collection hooks, compatible with llm-compressor style.
"""
import torch
from typing import Dict


def compute_hessian_diag(activations: torch.Tensor) -> torch.Tensor:
    """
    Args:
        activations: [N, d_in] collected input activations for layer l
    Returns:
        hessian_diag: [d_in] diagonal of Hessian approximation
    """
    N = activations.shape[0]
    hessian_diag = (2.0 / N) * (activations ** 2).sum(dim=0)
    return hessian_diag


def collect_activations_hook(storage: Dict, layer_name: str):
    """
    Returns a forward hook that stores input activations for a named layer.
    Usage:
        hook = collect_activations_hook(storage, "model.layers.0.self_attn.q_proj")
        handle = module.register_forward_hook(hook)
    """
    def hook(module, input, output):
        x = input[0].detach().float()
        if x.dim() == 3:
            # [batch, seq, d] -> [batch*seq, d]
            x = x.view(-1, x.shape[-1])
        if layer_name not in storage:
            storage[layer_name] = x
        else:
            storage[layer_name] = torch.cat([storage[layer_name], x], dim=0)
    return hook
