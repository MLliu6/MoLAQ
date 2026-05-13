"""
Modal gradient collection for MoLAQ sensitivity: σ_l = tr(H_l) * (||g_lang||_1 + α||g_vis_sal||_1)
"""
import torch
from typing import Dict, List, Optional


def compute_modal_sensitivity(
    hessian_diag: torch.Tensor,
    grad_lang: torch.Tensor,
    grad_vis_sal: torch.Tensor,
    alpha: float = 10.0
) -> float:
    """
    Computes σ_l for layer l.

    σ_l = tr(H_l) · E[||g_lang||_1 + α·||g_vis_sal||_1]

    Args:
        hessian_diag: [d_in] diagonal Hessian approximation
        grad_lang: [n_lang_samples, d] gradients for language tokens
        grad_vis_sal: [n_vis_samples, d] gradients for salient visual tokens
        alpha: modality balance coefficient (default 10.0 from MBQ empirical finding)
    Returns:
        sigma_l: scalar sensitivity score
    """
    trace_H = hessian_diag.sum().item()

    g_lang_norm = grad_lang.abs().mean().item() if grad_lang.numel() > 0 else 0.0
    g_vis_norm = grad_vis_sal.abs().mean().item() if grad_vis_sal.numel() > 0 else 0.0

    sigma_l = trace_H * (g_lang_norm + alpha * g_vis_norm)
    return sigma_l


def estimate_alpha_from_calibration(
    grad_lang_list: List[float],
    grad_vis_list: List[float]
) -> float:
    """
    Estimates alpha as inverse of (mean_vis_grad / mean_lang_grad) ratio,
    calibrated to make salient visual tokens as important as language tokens.
    Based on MBQ finding: lang grad ~10x vis grad on average.
    """
    mean_lang = sum(grad_lang_list) / len(grad_lang_list)
    mean_vis = sum(grad_vis_list) / len(grad_vis_list)
    if mean_vis < 1e-8:
        return 10.0
    alpha = mean_lang / mean_vis
    return alpha
