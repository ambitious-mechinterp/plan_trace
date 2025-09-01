"""
Steering interventions for testing causal effects of SAE latents on generation.

Shape Suffix Definition: 
- B: batch size 
- L: Num of Input Tokens 
- S: Number of SAE neurons in a layer
- D: model dimension
"""

import torch
from functools import partial
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Sequence, Any, Union
from .utils import cleanup_cuda


def _steer_dtype(activations: torch.Tensor) -> torch.dtype:
    """Return dtype used to apply steering (fp32 for precision)."""
    return torch.float32


def steering_hook(
    activations: torch.Tensor, 
    hook, 
    sae, 
    latent_idx: int,
    coeff: float, 
    steering_token_index: Optional[int] = None
) -> torch.Tensor:
    """
    Apply coeff · W_dec[latent_idx] either to the whole sequence or
    a single position (steering_token_index), using high‑precision arithmetic.
    
    Args:
        activations: Input activations [B, L, D_model]
        hook: TransformerLens hook object
        sae: SAE object containing W_dec
        latent_idx: Index of SAE latent to steer with
        coeff: Steering coefficient (positive or negative)
        steering_token_index: If specified, only steer this token position
        
    Returns:
        Modified activations [B, L, D_model]
    """
    tgt_dtype = _steer_dtype(activations)
    steer_vec = (
        sae.W_dec[latent_idx]
        .to(device=activations.device, dtype=tgt_dtype) * float(coeff)
    )

    # Cast activations to target dtype only for the in‑place add
    act_view = activations.to(dtype=tgt_dtype)

    if steering_token_index is None:
        act_view += steer_vec
    else:
        act_view[:, steering_token_index] += steer_vec

    # Cast back to original dtype
    return act_view.to(dtype=activations.dtype)


def steering_effect_on_next_token(
    model, 
    inter_toks_BL: torch.Tensor, 
    saes: List, 
    interventions: List[Tuple[int, int, int]], 
    coeff: float, 
    stop_tok: int
) -> Tuple[bool, int, int]:
    """
    Test if steering interventions change the next predicted token.
    
    Args:
        model: The language model
        inter_toks_BL: Input tokens [B, L]
        saes: List of SAE objects
        interventions: List of (layer_idx, tok_pos, latent_idx) tuples
        coeff: Steering coefficient
        stop_tok: Stop token ID (unused but kept for API compatibility)
        
    Returns:
        Tuple of (changed: bool, new_id: int, baseline_id: int)
    """
    # Get baseline prediction
    model.reset_hooks(including_permanent=True)
    with torch.no_grad():
        logits = model(inter_toks_BL)[:, -1]
    baseline_id = logits.argmax(-1).item()

    # Apply steering interventions
    for layer_idx, tok_pos, latent_idx in interventions:
        sae = saes[layer_idx]
        model.add_hook(
            sae.cfg.hook_name,
            partial(
                steering_hook, sae=sae,
                latent_idx=latent_idx, coeff=coeff,
                steering_token_index=tok_pos,
            ),
        )

    # Get steered prediction
    with torch.no_grad():
        logits = model(inter_toks_BL)[:, -1]        
    new_id = logits.argmax(-1).item()
    
    model.reset_hooks(including_permanent=True)
    return new_id != baseline_id, new_id, baseline_id


def generate_once(
    model, 
    inter_toks_BL: torch.Tensor, 
    stop_tok: int, 
    hooks: Optional[List] = None, 
    *, 
    max_tokens: int = 256, 
    device: str = "cuda",
    return_tokens: bool = False
) -> str:
    """
    Generate text continuation from input tokens with optional hooks.
    
    Args:
        model: The language model
        inter_toks_BL: Input tokens [B, L]
        stop_tok: Token ID to stop generation at
        hooks: Optional list of (hook_name, hook_fn) tuples
        max_tokens: Maximum tokens to generate
        device: Device for computation
        
    Returns:
        Generated text string (suffix only, not including input)
    """
    toks = inter_toks_BL.to(device)
    out = toks.clone()

    if hooks:
        for hook_name, fn in hooks:
            model.add_hook(hook_name, fn)

    while out.shape[-1] - toks.shape[-1] < max_tokens:
        with torch.no_grad():
            logits = model(out)[:, -1]
        next_id = logits.argmax(-1).item()
        if next_id == stop_tok:
            break
        if model.to_string(next_id) == "<end_of_turn>" or model.to_string(next_id) == "<eos>":
            break
        out = torch.cat([out, torch.tensor([[next_id]], device=device)], dim=1)

    if hooks:
        model.reset_hooks(including_permanent=True)

    if return_tokens:
        return out[0, toks.shape[-1]:]
    else:
        return model.to_string(out[0, toks.shape[-1]:])


def sweep_coefficients_multi(
    model,
    saes: Dict[int, Any],
    interventions: List[Tuple[int, int, int]],   # (layer, token_pos, latent)
    coefficients: List[float],
    inter_toks_BL: torch.Tensor,
    stop_tok: int = 1917,
    device: str = "cuda",
    max_tokens: int = 100,
    return_tokens: bool = False
) -> Dict[float, str]:
    """
    Sweep over multiple steering coefficients and generate text for each.
    
    Pre-aggregates steering vectors for efficiency when multiple latents
    are steered at the same position.
    
    Args:
        model: The language model
        saes: Dict mapping layer_idx -> SAE object
        interventions: List of (layer, token_pos, latent) intervention tuples
        coefficients: List of steering coefficients to test
        inter_toks_BL: Input tokens [B, L]
        stop_tok: Token ID to stop generation at
        device: Device for computation
        max_tokens: Maximum tokens to generate
        
    Returns:
        Dict mapping coefficient -> generated_text
    """
    d_model = saes[0].W_dec.shape[1]
    act_dtype = next(model.parameters()).dtype

    # Pre-aggregate: (coeff → layer → {pos → steer_vec})
    steer_by_coeff: Dict[float, Dict[int, Dict[int, torch.Tensor]]] = {}
    for c in coefficients:
        per_coeff = defaultdict(                       # layer →
            lambda: defaultdict(                       #   pos →
                lambda: torch.zeros(                   #     steer_vec
                    d_model, device=device,
                    dtype=_steer_dtype(torch.empty((), dtype=act_dtype))
                )
            )
        )
        for layer_idx, pos, latent_idx in interventions:
            vec = (
                saes[layer_idx].W_dec[latent_idx]
                .to(device=device, dtype=_steer_dtype(torch.empty((), dtype=act_dtype)))
                * float(c)
            )
            per_coeff[layer_idx][pos] += vec
        steer_by_coeff[c] = per_coeff

    # Generate for each coefficient
    outputs: Dict[float, Union[str, List[int]]] = {}
    for c in coefficients:
        # Check if steering actually changes the next token
        changed, new_id, bl_id = steering_effect_on_next_token(
            model, inter_toks_BL, saes, interventions, c, stop_tok
        )
        if not changed:
            continue 
            
        per_coeff = steer_by_coeff[c]

        # Register one hook per layer that adds the summed vector
        for layer_idx, pos_dict in per_coeff.items():
            hook_name = saes[layer_idx].cfg.hook_name

            def make_layer_hook(pos_dict_local):
                def layer_hook(activations, hook):
                    for pos, vec in pos_dict_local.items():
                        if pos < activations.size(1):
                            act_view = activations.to(_steer_dtype(activations))
                            act_view[:, pos] += vec
                            # Copy back with dtype cast
                            activations.copy_(act_view.to(dtype=activations.dtype))
                    return activations
                return layer_hook

            model.add_hook(hook_name, make_layer_hook(pos_dict))

        # Generate with hooks active
        gen_suffix = generate_once(
            model, inter_toks_BL=inter_toks_BL, stop_tok=stop_tok,
            hooks=None, max_tokens=max_tokens, device=device,
            return_tokens=return_tokens
        )
        outputs[c] = gen_suffix
        model.reset_hooks(including_permanent=True)

    return outputs


def run_steering_sweep(
    model,
    saes: List,
    inter_toks_BL: torch.Tensor,
    saved_pair_dict: Dict[str, List[Tuple[int, int, List[int]]]],
    baseline_text: str,
    *,
    coeff_grid: Sequence[int],
    stop_tok: int,
    max_tokens: int = 100,
    return_tokens: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Run steering sweeps for all clusters in saved_pair_dict.
    
    For each predicted token cluster, steer all associated latents
    at their original positions and measure generation changes.
    
    Args:
        model: The language model
        saes: List of SAE objects
        inter_toks_BL: Input tokens [B, L]
        saved_pair_dict: Dict mapping predicted_token -> [(layer, latent, [positions])]
        baseline_text: Baseline generation text for comparison
        coeff_grid: Steering coefficients to test
        stop_tok: Token ID to stop generation at
        max_tokens: Maximum tokens to generate
        
    Returns:
        Dict mapping label -> {"base_text": str, "steered": [{"coeff": float, "steered_text": str}]}
    """
    results: Dict[str, Dict[str, Any]] = {}

    for label, latent_infos in saved_pair_dict.items():
        # Build (layer, tok_pos, latent) triples
        interventions = [
            (layer_i, tok_pos, latent_i)
            for layer_i, latent_i, tok_positions in latent_infos
            for tok_pos in tok_positions
        ]
        
        # Convert saes list to dict for sweep_coefficients_multi
        saes_dict = {i: sae for i, sae in enumerate(saes)}
        
        sweep_out = sweep_coefficients_multi(
            model=model,
            saes=saes_dict,
            interventions=interventions,
            coefficients=list(coeff_grid),
            inter_toks_BL=inter_toks_BL,
            stop_tok=stop_tok,
            max_tokens=max_tokens,
            return_tokens=return_tokens
        )

        label_metrics: List[Dict[str, Any]] = []
        for c in coeff_grid:
            txt = sweep_out.get(c, "")
            label_metrics.append(dict(coeff=c, steered_text=txt))

        results[label] = {
            "base_text": baseline_text,
            "steered": label_metrics
        }

    return results 