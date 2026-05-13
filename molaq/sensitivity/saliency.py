"""
Visual token saliency estimation via ViT attention map top-K selection.
Core innovation of MoLAQ: distinguishes salient visual patches from background.
"""
import torch
from typing import Tuple


def get_salient_visual_indices(
    attn_weights: torch.Tensor,
    top_k_ratio: float = 0.2
) -> torch.Tensor:
    """
    Extracts top-K salient visual patch token indices from ViT attention map.

    Args:
        attn_weights: [num_heads, seq_len, seq_len] attention weights from last ViT layer
        top_k_ratio: fraction of visual tokens to select as salient (default 20%)
    Returns:
        salient_indices: [K] indices of salient visual tokens
    """
    # Average across heads; use CLS token's attention to patches as saliency proxy
    avg_attn = attn_weights.mean(dim=0)  # [seq_len, seq_len]
    cls_attn = avg_attn[0, 1:]           # CLS -> all patch tokens, shape [num_patches]

    K = max(1, int(top_k_ratio * cls_attn.shape[0]))
    salient_indices = cls_attn.topk(K).indices
    return salient_indices


def separate_modal_tokens(
    token_ids: torch.Tensor,
    visual_token_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Splits token sequence into language indices and visual indices.

    Args:
        token_ids: [seq_len] full token id sequence
        visual_token_mask: [seq_len] bool mask, True for visual tokens
    Returns:
        lang_indices, visual_indices
    """
    lang_indices = (~visual_token_mask).nonzero(as_tuple=True)[0]
    visual_indices = visual_token_mask.nonzero(as_tuple=True)[0]
    return lang_indices, visual_indices
