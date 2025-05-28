"""
Shape Suffix Definition: 

- B: batch size 
- L: Num of Input Tokens 
- O: Num of Output Tokens
- V: vocabulary size
- F: feed-forward subnetwork hidden size
- D: Depth or number of layers
- H: number of attention heads in a layer
- S: Number of SAE neurons in a layer
- A: Number of SAEs attached
- tx: variables dealing with the prediction of the x'th output token
"""

import argparse
from tqdm import tqdm
import sys
import os
import torch
import itertools
from IPython.display import display, HTML, IFrame
import numpy as np
import torch.nn.functional as F
import re
from tqdm import tqdm
from typing import  Tuple,  Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from helpers.utils import (
    load_model, 
    get_device,
    cleanup_cuda,
    clear_memory,
)
from helpers.hook_manager import (
    SAEMasks,
    build_sae_hook_fn,
    run_sae_hook_fn, 
)

### METRIC ### 

def compute_metric(model, base_saes, sae_index, tokens, circuit_mask, label, use_mean_error=True):
    hooks = []
    bos_token_id = model.tokenizer.bos_token_id
    for sae in base_saes:
        hooks.append(
            (
                sae.cfg.hook_name,
                build_sae_hook_fn(
                    sae,
                    tokens,
                    bos_token_id,
                    circuit_mask=circuit_mask if sae.cfg.hook_layer == sae_index else None,
                    use_mean_error=use_mean_error,
                ),
            )
        )
    circuit_logits = model.run_with_hooks(tokens, return_type="logits", fwd_hooks=hooks)
    answer_logits = circuit_logits[..., -1, :]  # Get the logits of the last tokens
    answer_logprobs = F.softmax(answer_logits, dim=-1)
    clean_logprobs = answer_logprobs[..., torch.arange(answer_logits.shape[-2]), label]
    return torch.sum(clean_logprobs).item()

### ATTRIBUTION ### 

def run_integrated_gradients(
    model,
    base_saes,
    token_list,
    clean_sae_cache,
    clean_error_cache,
    corr_sae_cache,
    corr_error_cache,
    labels,
    save_dir,
    ig_steps=10,
    save_and_use=True,   # <--- new,
    logstats=False,  # <--- new
):
    """
    Runs a simple integrated-gradients-like calculation on the SAE activations
    and associated error signals.

    For each SAE in `base_saes`, it interpolates from clean activations
    to a counterfactual baseline (which is currently set to 0),
    measuring how the log-prob of the correct token changes.

    Args:
        model (HookedSAETransformer): The main model.
        base_saes (List[SAE]): List of SAE objects, one per layer/hook.
        token_list (torch.Tensor): The tokens for which we want to evaluate.
        clean_sae_cache (Dict[str, torch.Tensor]): Dictionary of "clean" SAE activations.
        clean_error_cache (Dict[str, torch.Tensor]): Dictionary of the "clean" error term.
        corr_sae_cache (Dict[str, torch.Tensor]): Dictionary of "counterfactual" (or zero) activations.
        corr_error_cache (Dict[str, torch.Tensor]): Dictionary of "counterfactual" error terms.
        labels (torch.Tensor): The correct token IDs for measuring log-prob.
        save_dir (str): Where to save the resulting effect Tensors.
        ig_steps (int): Number of integration steps.

    Returns:
        None. (Effects are saved to disk.)
    """
    if not os.path.exists(save_dir) and save_and_use:
        os.makedirs(save_dir, exist_ok=True)
    
    results_sae = {}
    results_err = {}

    # We'll need a placeholder to store the updated SAEs while running
    for sae_index, sae in tqdm(enumerate(base_saes)):
        for inner_sae_index in range(len(base_saes)):
            base_saes[inner_sae_index].mean_error = clean_error_cache[base_saes[inner_sae_index].cfg.hook_name]
        if logstats:
            print(f"[IG] Processing SAE {sae_index} at hook {sae.cfg.hook_name} ...")

        # We will accumulate partial effect from each interpolation step
        effects_sae = []
        effects_err = []

        # For clarity, define the shape references
        clean_acts = clean_sae_cache[sae.cfg.hook_name]
        clean_err = clean_error_cache[sae.cfg.hook_name]

        # We'll integrate from "clean" to "corr" (which is zero in your example),
        # ratio=0 => 100% clean, ratio=1 => 100% corr
        for step in range(ig_steps):
            ratio = step / float(ig_steps)
            # Interpolate
            interpolation_acts = (clean_acts * (1 - ratio) + corr_sae_cache[sae.cfg.hook_name] * ratio).requires_grad_(True)
            interpolation_acts.retain_grad()

            interpolation_err = (clean_err * (1 - ratio) + corr_error_cache[sae.cfg.hook_name] * ratio).requires_grad_(True)
            interpolation_err.retain_grad()

            # Replace the mean error for the current SAE with the interpolated error
            base_saes[sae_index].mean_error = interpolation_err

            # We run the model with these fake_activations. 
            interpolated_out, _ = run_sae_hook_fn(
                model,
                base_saes,
                token_list,
                calc_error=False,
                use_error=False,
                fake_activations=(sae.cfg.hook_layer, interpolation_acts),
                use_mean_error=True
            )

            # Evaluate log-prob for the correct label
            # Your code snippet suggests shape: [batch, seq, vocab]
            answer_logits = interpolated_out[..., -1, :]
            answer_logprobs = F.softmax(answer_logits, dim=-1)
            # We sum or average across the batch dimension (assumed batch=1 in example).
            clean_logprobs = answer_logprobs[..., labels[-1]]
            metric = torch.sum(clean_logprobs)
            if logstats:
                print(f"  [Step={step}/{ig_steps}] ratio={ratio}, metric={metric.item():.4f}")

            # Backprop
            metric.backward()

            # zero attribution formula
            counterfactual_delta_sae = -clean_acts
            counterfactual_delta_err = -clean_err

            effect_sae = (interpolation_acts.grad * counterfactual_delta_sae).mean(dim=0).detach()
            effect_err = (interpolation_err.grad * counterfactual_delta_err).mean(dim=0).detach()

            effects_sae.append(effect_sae.cpu())
            effects_err.append(effect_err.cpu())

            # Clear out grads from model and SAEs to avoid accumulation
            clear_memory(base_saes, model)

        # Average over steps
        effects_sae = torch.stack(effects_sae)
        effects_err = torch.stack(effects_err)
        final_effect_sae = effects_sae.mean(dim=0)
        final_effect_err = effects_err.mean(dim=0)

        if save_and_use:
            sae_effect_path = os.path.join(save_dir, f"sae_effect_{sae_index}.pt")
            err_effect_path = os.path.join(save_dir, f"err_effect_{sae_index}.pt")
            torch.save(final_effect_sae, sae_effect_path)
            torch.save(final_effect_err, err_effect_path)
            print(f"  => Saved SAE effect to {sae_effect_path}")
            print(f"  => Saved Error effect to {err_effect_path}")
        else:
            results_sae[sae.cfg.hook_name] = final_effect_sae
            results_err[sae.cfg.hook_name] = final_effect_err

    if not save_and_use:
        return results_sae, results_err
    else:
        return None, None

def get_saes_cache(saes):
    """
    Get the cache of SAEs
    """
    clean_sae_cache = {}
    clean_error_cache = {}
    for sae in saes:
        clean_sae_cache[sae.cfg.hook_name] = sae.feature_acts
        clean_error_cache[sae.cfg.hook_name] = sae.error_term

    corr_sae_cache = {}
    corr_error_cache = {}
    for sae in saes:
        corr_sae_cache[sae.cfg.hook_name] = torch.zeros_like(clean_sae_cache[sae.cfg.hook_name])
        corr_error_cache[sae.cfg.hook_name] = torch.zeros_like(clean_error_cache[sae.cfg.hook_name])

    return clean_sae_cache, clean_error_cache, corr_sae_cache, corr_error_cache


### K FUNCTIONS ### 

def compute_k_metrics(model, saes, res_sae_effects, device, hook_names, tokens, label, k_max=7001, k_step=500):
    """
    Compute metrics for different K values but don't find a threshold.
    
    Args:
        model: The model to evaluate
        saes: List of sparse autoencoders
        res_sae_effects: Dictionary mapping hook names to effect tensors
        device: Device to run computations on
        hook_names: List of hook names
        tokens: Input tokens
        label: Target label
        k_max: Maximum K value to try (default: 7001)
        k_step: Step size for K values (default: 500)
        
    Returns:
        tuple: (K_vals, metrics_negative, metrics_absolute)
    """
    model.reset_hooks(including_permanent=True)
    
    # Store metrics
    K_vals = list(range(1, k_max, k_step))
    metrics_negative = []
    metrics_absolute = []
    
    for k in K_vals:
        model.reset_hooks(including_permanent=True)
        print(f"Processing K={k}")

        # Gather all effects into a single list with their layer indices
        all_effects = []
        for layer_idx, sae in enumerate(saes):
            final_effect_sae = res_sae_effects[sae.cfg.hook_name]
            L, S = final_effect_sae.shape  # L: sequence length, S: number of latents

            # Create indices for all positions
            token_indices = torch.arange(L).repeat_interleave(S)
            latent_indices = torch.arange(S).repeat(L)
            layer_indices = torch.full_like(token_indices, layer_idx)

            # Stack with effects
            effects = final_effect_sae.view(-1)
            all_effects.append(torch.stack([
                layer_indices,
                token_indices,
                latent_indices,
                effects
            ], dim=0))

        # Concatenate all effects and their indices
        all_effects = torch.cat(all_effects, dim=1)  # Shape: [4, total_positions]

        # === NEGATIVE EFFECTS ===
        neg_effects = -all_effects[3]  # Negate effects for negative selection
        topk_values_neg, topk_indices_neg = torch.topk(neg_effects, k)

        # Get layer, token, latent indices for top K negative values
        selected_layers_neg = all_effects[0][topk_indices_neg].long()
        selected_tokens_neg = all_effects[1][topk_indices_neg].long()
        selected_latents_neg = all_effects[2][topk_indices_neg].long()

        # Create masks for negative effects
        circ_masks_list = []
        for layer_idx in range(len(saes)):
            final_effect_sae = res_sae_effects[saes[layer_idx].cfg.hook_name]
            mask_neg = torch.zeros_like(final_effect_sae, dtype=torch.float)

            # Find indices for this layer
            layer_mask = (selected_layers_neg == layer_idx)
            if layer_mask.any():
                layer_tokens = selected_tokens_neg[layer_mask]
                layer_latents = selected_latents_neg[layer_mask]
                mask_neg[layer_tokens, layer_latents] = 1.0

            circ_masks_list.append(mask_neg.to(device))

        circuit_mask_neg = SAEMasks(
            hook_points=hook_names,
            masks=circ_masks_list
        ).to(device)

        # Evaluate negative masks
        hooks = []
        bos_token_id = model.tokenizer.bos_token_id
        for sae in saes:
            hooks.append(
                (
                    sae.cfg.hook_name,
                    build_sae_hook_fn(
                        sae,
                        tokens,
                        bos_token_id,
                        circuit_mask=circuit_mask_neg,
                        use_mean_error=True,
                    ),
                )
            )
        circuit_logits = model.run_with_hooks(tokens, return_type="logits", fwd_hooks=hooks)
        answer_logits = circuit_logits[..., -1, :]
        answer_logprobs = F.softmax(answer_logits, dim=-1)
        clean_logprobs = answer_logprobs[..., torch.arange(answer_logits.shape[-2]), label]
        metric_negative = torch.sum(clean_logprobs).item()
        metrics_negative.append(metric_negative)
        
        # === ABSOLUTE EFFECTS ===
        model.reset_hooks(including_permanent=True)
        abs_effects = torch.abs(all_effects[3])
        topk_values_abs, topk_indices_abs = torch.topk(abs_effects, k)

        selected_layers_abs = all_effects[0][topk_indices_abs].long()
        selected_tokens_abs = all_effects[1][topk_indices_abs].long()
        selected_latents_abs = all_effects[2][topk_indices_abs].long()

        circ_masks_list = []
        for layer_idx in range(len(saes)):
            final_effect_sae = res_sae_effects[saes[layer_idx].cfg.hook_name]
            mask_abs = torch.zeros_like(final_effect_sae, dtype=torch.float)

            layer_mask = (selected_layers_abs == layer_idx)
            if layer_mask.any():
                layer_tokens = selected_tokens_abs[layer_mask]
                layer_latents = selected_latents_abs[layer_mask]
                mask_abs[layer_tokens, layer_latents] = 1.0

            circ_masks_list.append(mask_abs.to(device))

        circuit_mask_abs = SAEMasks(
            hook_points=hook_names,
            masks=circ_masks_list
        ).to(device)

        # Evaluate absolute masks
        hooks = []
        for sae in saes:
            hooks.append(
                (
                    sae.cfg.hook_name,
                    build_sae_hook_fn(
                        sae,
                        tokens,
                        bos_token_id,
                        circuit_mask=circuit_mask_abs,
                        use_mean_error=True,
                    ),
                )
            )
        circuit_logits = model.run_with_hooks(tokens, return_type="logits", fwd_hooks=hooks)
        answer_logits = circuit_logits[..., -1, :]
        answer_logprobs = F.softmax(answer_logits, dim=-1)
        clean_logprobs = answer_logprobs[..., torch.arange(answer_logits.shape[-2]), label]
        metric_absolute = torch.sum(clean_logprobs).item()
        metrics_absolute.append(metric_absolute)
            
    return K_vals, metrics_negative, metrics_absolute


def plot_k_metrics(K_vals, metrics_negative, metrics_absolute, clean_probs_baseline, min_k_negative=None, min_k_absolute=None, threshold=None):
    """
    Plot the results of K analysis with optional threshold and minimum K values.
    
    Args:
        K_vals: List of K values
        metrics_negative: List of metrics for negative selection
        metrics_absolute: List of metrics for absolute selection
        clean_probs_baseline: Clean baseline performance
        min_k_negative: Optional minimum K for negative effects
        min_k_absolute: Optional minimum K for absolute effects
        threshold: Optional threshold as fraction of baseline
    """
    plt.figure(figsize=(10, 6))
    plt.plot(K_vals[:len(metrics_negative)], metrics_negative, label="Top K Negative Values", marker="o")
    plt.plot(K_vals[:len(metrics_absolute)], metrics_absolute, label="Top K Absolute Values", marker="s")
    plt.hlines(y=clean_probs_baseline.item(), xmin=min(K_vals), xmax=max(K_vals[:max(len(metrics_negative), len(metrics_absolute))]), 
              color='r', linestyle='--', label="Baseline Probs")
    
    if threshold is not None:
        target_value = clean_probs_baseline.item() * threshold
        plt.hlines(y=target_value, xmin=min(K_vals), xmax=max(K_vals[:max(len(metrics_negative), len(metrics_absolute))]), 
                  color='g', linestyle='--', label=f"{threshold*100}% of Baseline")
    
    if min_k_negative is not None:
        plt.axvline(x=min_k_negative, color='b', linestyle=':', label=f"Min K Negative: {min_k_negative}")
    if min_k_absolute is not None:
        plt.axvline(x=min_k_absolute, color='purple', linestyle=':', label=f"Min K Absolute: {min_k_absolute}")
    
    plt.xlabel("K (Total Number of Latent-Token Pairs across all layers)")
    plt.ylabel("P(correct)")
    plt.title("P(correct) vs K latent-token pairs across all RES SAEs")
    plt.grid(True)
    plt.legend()
    plt.show()


def find_min_k_for_threshold(K_vals, metrics_negative, metrics_absolute, clean_probs_baseline, threshold=0.8):
    """
    Find minimum K value where performance exceeds threshold*baseline
    
    Args:
        K_vals: List of K values
        metrics_negative: List of metrics for negative selection
        metrics_absolute: List of metrics for absolute selection
        clean_probs_baseline: Clean baseline performance
        threshold: Target threshold as fraction of baseline (default: 0.8)
        
    Returns:
        tuple: (min_k_negative, min_k_absolute)
    """
    target_value = clean_probs_baseline.item() * threshold
    min_k_negative = None
    min_k_absolute = None
    
    # Find minimum K for negative effects
    for i, (k, metric) in enumerate(zip(K_vals, metrics_negative)):
        if metric >= target_value:
            min_k_negative = k
            print(f"Found minimum K for negative effects: {k} (metric: {metric:.4f}, target: {target_value:.4f})")
            break
    
    # Find minimum K for absolute effects
    for i, (k, metric) in enumerate(zip(K_vals, metrics_absolute)):
        if metric >= target_value:
            min_k_absolute = k
            print(f"Found minimum K for absolute effects: {k} (metric: {metric:.4f}, target: {target_value:.4f})")
            break
    
    return min_k_negative, min_k_absolute


def iter_topk_negative_effects(
    res_sae_effects: Dict[str, torch.Tensor],
    saes:        List[Any],
    hook_names:  List[str],
    K:           int
):
    """
    Yields (layer_idx, token_idx, latent_idx, neg_effect_value)
    for the top‑K *negative* effects across all layers.
    """
    all_parts = []
    # 1) collect layer, token, latent indices + effects into one big [4, N] tensor
    for layer_idx, sae in enumerate(saes):
        effects = res_sae_effects[sae.cfg.hook_name]  # shape [L, S]
        L, S    = effects.shape
        idx_layer  = torch.full((L*S,), layer_idx, dtype=torch.long)
        idx_token  = torch.arange(L, device=effects.device).repeat_interleave(S)
        idx_latent = torch.arange(S, device=effects.device).repeat(L)
        vals       = effects.view(-1)
        all_parts.append(torch.stack([idx_layer, idx_token, idx_latent, vals], dim=0))
    all_effects = torch.cat(all_parts, dim=1)  # [4, total_positions]

    # 2) turn into negative, grab top‑K
    neg_vals = -all_effects[3]
    topk_vals, topk_idxs = torch.topk(neg_vals, K)

    # 3) yield the tuples
    for neg_val, idx in zip(topk_vals, topk_idxs):
        layer_i  = int(all_effects[0, idx])
        token_i  = int(all_effects[1, idx])
        latent_i = int(all_effects[2, idx])
        yield (layer_i, token_i, latent_i, float(neg_val))

def iter_topk_absolute_effects(
    res_sae_effects: Dict[str, torch.Tensor],
    saes:        List[Any],
    hook_names:  List[str],
    K:           int
):
    """
    Yields (layer_idx, token_idx, latent_idx, abs_effect_value)
    for the top‑K *absolute* effects across all layers.
    """
    all_parts = []
    # 1) collect layer, token, latent indices + effects into one big [4, N] tensor
    for layer_idx, sae in enumerate(saes):
        effects = res_sae_effects[sae.cfg.hook_name]  # shape [L, S]
        L, S    = effects.shape
        idx_layer  = torch.full((L*S,), layer_idx, dtype=torch.long)
        idx_token  = torch.arange(L, device=effects.device).repeat_interleave(S)
        idx_latent = torch.arange(S, device=effects.device).repeat(L)
        vals       = effects.view(-1)
        all_parts.append(torch.stack([idx_layer, idx_token, idx_latent, vals], dim=0))
    all_effects = torch.cat(all_parts, dim=1)  # [4, total_positions]

    # 2) take absolute value of effects and get top‑K
    abs_vals = torch.abs(all_effects[3])
    topk_vals, topk_idxs = torch.topk(abs_vals, K)

    # 3) yield the tuples
    for abs_val, idx in zip(topk_vals, topk_idxs):
        layer_i  = int(all_effects[0, idx])
        token_i  = int(all_effects[1, idx])
        latent_i = int(all_effects[2, idx])
        yield (layer_i, token_i, latent_i, float(abs_val))

def discover_circuit(model, 
                     saes, 
                     changable_toks, 
                     label,
                     device,
                     ig_steps=10,
                     k_max=7001,
                     k_step=500,
                     k_thres=0.6,):

    with torch.no_grad():
        logits_BLV, saes = run_sae_hook_fn(model, saes, changable_toks, calc_error=True, use_error=True, cache_sae_activations=True)
    cleanup_cuda()
    clean_logits = logits_BLV[0, -1, label]
    probs_BV = F.softmax(logits_BLV[:, -1, :], dim=-1)
    clean_probs_baseline = probs_BV[0, label]

    for ind, sae in enumerate(saes):
        sae.mean_error = sae.error_term.detach()
    
    # Get Cache
    clean_sae_cache, clean_error_cache, corr_sae_cache, corr_error_cache = get_saes_cache(saes)

    # Run Intergrated Gradients & Save Results
    res_sae_effects, res_err_effects = run_integrated_gradients(
        model=model,
        base_saes=saes,
        token_list=changable_toks,
        clean_sae_cache=clean_sae_cache,
        clean_error_cache=clean_error_cache,
        corr_sae_cache=corr_sae_cache,
        corr_error_cache=corr_error_cache,
        labels=torch.tensor([label]).to(device),
        ig_steps=ig_steps,
        save_dir="", 
        save_and_use=False,
        logstats=False,)  

    hook_names = [sae.cfg.hook_name for sae in saes]

    ### K METRICS
    K_vals, metrics_negative, metrics_absolute = compute_k_metrics(
        model, saes, res_sae_effects, device, hook_names, changable_toks, label, k_max=k_max, k_step=k_step
    )
    min_k_neg, min_k_abs = find_min_k_for_threshold(
        K_vals, metrics_negative, metrics_absolute, clean_probs_baseline, threshold=k_thres
    )
    if min_k_neg is not None:
        K = min_k_neg
        topk_iter = iter_topk_negative_effects(
            res_sae_effects=res_sae_effects,
            saes=saes,
            hook_names=hook_names,
            K=K
        )
    elif min_k_abs is not None:
        K = min_k_abs
        topk_iter = iter_topk_absolute_effects(
            res_sae_effects=res_sae_effects,
            saes=saes,
            hook_names=hook_names,
            K=K
        )
    else:
        # print(f"Skipping as no K found")
        return None

    # we only need to store entries
    entries = list(itertools.islice(topk_iter, K)) # preserves order

    return entries

