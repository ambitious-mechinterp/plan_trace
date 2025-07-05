"""
SAE hook functions and masking utilities for circuit interventions.

Shape Suffix Definition: 
- B: batch size 
- L: Num of Input Tokens 
- S: Number of SAE neurons in a layer
"""

import os
import torch
from torch import nn
from functools import partial
from typing import List, Optional, Dict, Any, Callable
from .utils import clear_memory


class SAEMasks(nn.Module):
    """
    Binary masks for selectively ablating SAE latents across different hook points.
    
    Attributes:
        hook_points: List of hook point names (e.g., ["blocks.0.hook_mlp_out"])
        masks: List of binary tensors, one per hook point
    """
    
    def __init__(self, hook_points: List[str], masks: List[torch.Tensor]):
        super().__init__()
        self.hook_points = hook_points  
        self.masks = masks

    def forward(
        self, 
        x: torch.Tensor, 
        sae_hook_point: str, 
        mean_ablation: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply or ablate activations according to the binary masks for each hook point.

        Args:
            x: The original activations [L, S] or [B, L, S]
            sae_hook_point: Identifier for which mask to apply
            mean_ablation: If given, use it as the "off" activation instead of zero

        Returns:
            Tensor: Masked (or mean-ablated) activations with same shape as input
        """
        index = self.hook_points.index(sae_hook_point)
        mask = self.masks[index]
        censored_activations = torch.ones_like(x)
        if mean_ablation is not None:
            censored_activations = censored_activations * mean_ablation
        else:
            censored_activations = censored_activations * 0

        diff_to_x = x - censored_activations
        return censored_activations + diff_to_x * mask

    def print_mask_statistics(self) -> None:
        """Log shape, total latents, on-latent count, and avg on-latents/token for each mask."""
        for i, mask in enumerate(self.masks):
            shape = list(mask.shape)
            total_latents = mask.numel()
            total_on = mask.sum().item()  # number of 1's in the mask

            # Average on-latents per token depends on dimensions
            if len(shape) == 1:
                # e.g., shape == [S]
                avg_on_per_token = total_on  # only one token
            elif len(shape) == 2:
                # e.g., shape == [L, S]
                seq_len = shape[0]
                avg_on_per_token = total_on / seq_len if seq_len > 0 else 0
            else:
                # If there's more than 2 dims, adapt as needed;
                # we'll just define "token" as the first dimension.
                seq_len = shape[0]
                avg_on_per_token = total_on / seq_len if seq_len > 0 else 0

            print(f"Statistics for mask '{self.hook_points[i]}':")
            print(f"  - Shape: {shape}")
            print(f"  - Total latents: {total_latents}")
            print(f"  - Latents ON (mask=1): {int(total_on)}")
            print(f"  - Average ON per token: {avg_on_per_token:.4f}\n")

    def save(self, save_dir: str, file_name: str = "sae_masks.pt") -> None:
        """
        Serialize hook_points and their masks to a file for later reuse.

        Args:
            save_dir: Directory to write into (created if missing)
            file_name: Name of the checkpoint file
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, file_name)
        checkpoint = {"hook_points": self.hook_points, "masks": self.masks}
        torch.save(checkpoint, save_path)
        print(f"SAEMasks saved to {save_path}")

    @classmethod
    def load(cls, load_dir: str, file_name: str = "sae_masks.pt") -> "SAEMasks":
        """
        Load previously saved masks from disk.

        Args:
            load_dir: Directory holding the checkpoint
            file_name: Which mask set to load

        Returns:
            SAEMasks instance reconstructed from file
        """
        load_path = os.path.join(load_dir, file_name)
        if not os.path.isfile(load_path):
            raise FileNotFoundError(f"No saved SAEMasks found at {load_path}")

        checkpoint = torch.load(load_path)
        hook_points = checkpoint["hook_points"]
        masks = checkpoint["masks"]

        instance = cls(hook_points=hook_points, masks=masks)
        print(f"SAEMasks loaded from {load_path}")
        return instance

    def get_num_latents(self) -> int:
        """Count how many latents across all masks are "on" (mask > 0)."""
        num_latents = 0
        for mask in self.masks:
            num_latents += (mask > 0).sum().item()
        return num_latents


def build_sae_hook_fn(
    sae,
    sequence: torch.Tensor,
    bos_token_id: int,
    circuit_mask: Optional[SAEMasks] = None,
    mean_mask: bool = False,
    cache_masked_activations: bool = False,
    cache_sae_activations: bool = False,
    mean_ablate: bool = False,  
    fake_activations = False,  
    calc_error: bool = False,
    use_error: bool = False,
    use_mean_error: bool = False,
) -> Callable:
    """
    Construct a forward hook for an SAE: encode → optional mask/ablation → decode → merge.

    Args:
        sae: The SAE module whose encode/decode to use
        sequence: Token indices [L] that define padding/BOS for masking
        bos_token_id: ID marking start-of-sequence (excluded from mask)
        circuit_mask: If provided, zero or mean-ablate selected latents
        mean_mask: If True, use `sae.mean_ablation` as "off" value in mask
        cache_masked_activations: Store masked acts in `sae.feature_acts`
        cache_sae_activations: Store raw SAE activations in `sae.feature_acts`
        mean_ablate: Always ablate to `sae.mean_ablation`
        fake_activations: If (layer, acts), replaces acts at that layer
        calc_error: Compute `sae.error_term = original − updated`
        use_error: Add error term back into the returned activations
        use_mean_error: Add stored `sae.mean_error` back into the returned activations

    Returns:
        Hook function matching the model's fwd_hooks API
    """
    # make the mask for the sequence
    mask = torch.ones_like(sequence, dtype=torch.bool)
    mask[sequence == bos_token_id] = False  # where mask is false, keep original

    def sae_hook(value: torch.Tensor, hook) -> torch.Tensor:
        """
        SAE hook function that processes activations.
        
        Args:
            value: Input activations [B, L, D_model] where B=batch, L=seq_len, D_model=model_dim
            hook: TransformerLens hook object
            
        Returns:
            Modified activations [B, L, D_model]
        """
        feature_acts = sae.encode(value)  # [B, L, S] where S=sae_neurons
        feature_acts = feature_acts * mask.unsqueeze(-1)

        if fake_activations and sae.cfg.hook_layer == fake_activations[0]:
            feature_acts = fake_activations[1]

        if cache_sae_activations:
            sae.feature_acts = feature_acts.detach().clone()

        if circuit_mask is not None:
            hook_point = sae.cfg.hook_name
            if mean_mask == True:
                feature_acts = circuit_mask(
                    feature_acts, hook_point, mean_ablation=sae.mean_ablation
                )
            else:
                feature_acts = circuit_mask(feature_acts, hook_point)

        if cache_masked_activations:
            sae.feature_acts = feature_acts.detach().clone()

        if mean_ablate:
            feature_acts = sae.mean_ablation

        out = sae.decode(feature_acts)  # [B, L, D_model]
        mask_expanded = mask.unsqueeze(-1).expand_as(value)
        updated_value = torch.where(mask_expanded, out, value)

        if calc_error:
            sae.error_term = value - updated_value
            if use_error:
                return updated_value + sae.error_term

        if use_mean_error:
            return updated_value + sae.mean_error
        
        return updated_value

    return sae_hook


def register_sae_hooks(model, saes: List, tokens: torch.Tensor, **hook_kwargs) -> List:
    """
    Create a list of (hook_name, hook_fn) tuples ready for model.run_with_hooks().
    
    Args:
        model: The language model
        saes: List of SAE objects to register hooks for
        tokens: Input token sequence [B, L] for masking BOS tokens
        **hook_kwargs: Additional arguments passed to build_sae_hook_fn
        
    Returns:
        List of (hook_name, hook_function) tuples
    """
    bos_id = model.tokenizer.bos_token_id
    return [
        (
            sae.cfg.hook_name,
            build_sae_hook_fn(
                sae,
                tokens,
                bos_id,
                **hook_kwargs,
            ),
        )
        for sae in saes
    ]


def run_with_saes(model, saes: List, tokens: torch.Tensor, **hook_kwargs) -> tuple:
    """
    Convenience wrapper around model.run_with_hooks that uses register_sae_hooks().
    
    Args:
        model: The language model
        saes: List of SAE objects
        tokens: Input tokens [B, L]
        **hook_kwargs: Arguments for SAE hook construction
        
    Returns:
        Tuple of (logits, saes) where logits are [B, L, V] model outputs
    """
    hooks = register_sae_hooks(model, saes, tokens, **hook_kwargs)
    return model.run_with_hooks(tokens, return_type="logits", fwd_hooks=hooks), saes 