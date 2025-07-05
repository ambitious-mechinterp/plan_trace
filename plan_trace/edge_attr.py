# %%
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

import json
import torch
import os
import sys
import argparse
import itertools
import numpy as np
import torch.nn.functional as F
import re
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import copy
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from IPython.display import display, HTML, IFrame

# Add parent directory to path so plan_trace package can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plan_trace.utils import load_model, load_pretrained_saes, cleanup_cuda, clear_memory
from plan_trace.circuit_discovery import discover_circuit, run_integrated_gradients, get_saes_cache
from plan_trace.logit_lens import find_logit_lens_clusters  
from plan_trace.steering import run_steering_sweep
from plan_trace.analysis import CircuitAnalyzer, Config, analyze_batch
from plan_trace.hooks import run_with_saes, register_sae_hooks, SAEMasks, build_sae_hook_fn

# %%

# EDGE ATTRIBUTION

# ─────────────────────────────────────────────────────────────────────────────
# JVP-based edge attribution helpers (B,L,S aware)
# ─────────────────────────────────────────────────────────────────────────────
def _jvp_edge_attr(
    model,
    base_saes,
    token_list,
    res_sae_effects,
    clean_sae_cache,
    clean_error_cache,
    labels,
    device,
    *,
    max_features_per_layer: int = 100,
    use_mean_error: bool = True,
    logstats: bool = False,
    edge_includes_loss_grad: bool = True,
    feature_selection: str = "max",
):
    """
    Compute layer-to-layer edge attribution with a *forward-mode*
    Jacobian-vector product (JVP).

    Shapes
    -------
    Activations are [B, L, S] throughout.

    Returns
    -------
    Dict[str, Dict[str, torch.Tensor]]
        Sparse COO edge tensors keyed upstream_hook → downstream_hook
        with shape  [S_down , S_up].
    """
    # ── 1. feature pre-selection ──────────────────────────────────────────
    if feature_selection not in {"max", "sum", "negative"}:
        raise ValueError("feature_selection must be 'max', 'sum', or 'negative'.")

    important_feats: Dict[str, List[int]] = {}
    for sae in base_saes:
        hname   = sae.cfg.hook_name
        effects = res_sae_effects[hname]                       # [B, L, S]
        effects_flat = effects.reshape(-1, effects.shape[-1])  # [B·L, S]

        if feature_selection == "max":
            scores = effects_flat.abs().max(0).values
        elif feature_selection == "sum":
            scores = effects_flat.abs().sum(0)
        elif feature_selection == "negative":
            scores = -effects_flat.min(0).values  # Most negative becomes most positive
        else:
            raise ValueError(f"Unknown feature_selection: {feature_selection}")
        k       = min(max_features_per_layer, scores.numel())
        topk    = torch.topk(scores, k).indices if k else torch.tensor([], dtype=torch.long)
        important_feats[hname] = topk.tolist()
        if logstats:
            print(f"[edge-attr-jvp] {hname}: kept {len(topk)}/{scores.numel()} features")

    # ── 2. sequential edge loop ───────────────────────────────────────────
    edges: Dict[str, Dict[str, torch.Tensor]] = {}
    for i in range(len(base_saes) - 1):
        up_sae,   down_sae   = base_saes[i], base_saes[i + 1]
        up_hook,  down_hook  = up_sae.cfg.hook_name, down_sae.cfg.hook_name
        up_feats, down_feats = important_feats.get(up_hook, []), important_feats.get(down_hook, [])

        if not up_feats or not down_feats:
            if logstats:
                print(f"[edge-attr-jvp] skip {up_hook}->{down_hook} (no feats)")
            continue

        if logstats:
            print(f"[edge-attr-jvp] computing {up_hook}->{down_hook}")

        edge = _compute_jvp_edge(
            model,
            base_saes,
            up_sae,
            down_sae,
            token_list,
            up_feats,
            down_feats,
            clean_sae_cache,
            clean_error_cache,
            res_sae_effects,
            labels,
            device,
            use_mean_error=use_mean_error,
            edge_includes_loss_grad=edge_includes_loss_grad,
            logstats=logstats,
        )
        if edge is not None:
            edges.setdefault(up_hook, {})[down_hook] = edge

    if logstats:
        n_edges = sum(len(v) for v in edges.values())
        print(f"[edge-attr-jvp] finished – {n_edges} non-zero edge tensors")

    return edges


# ─────────────────────────────────────────────────────────────────────────────
def _compute_jvp_edge(
    model,
    base_saes,
    upstream_sae,
    downstream_sae,
    token_list,
    upstream_features: List[int],
    downstream_features: List[int],
    clean_sae_cache,
    clean_error_cache,
    res_sae_effects,
    labels,
    device,
    *,
    use_mean_error: bool = True,
    edge_includes_loss_grad: bool = True,
    logstats: bool = False,
):
    """
    Single-pair JVP edge attribution.

    For each (up_idx, down_idx) pair computes

        Σ_{b,t}  ∂ down_latent[b,t,down_idx] / ∂ up_latent[b,t,up_idx]
      or, with `edge_includes_loss_grad`,
        Σ_{b,t}  grad_loss[b,t,down_idx] *
                  ∂ down_latent[b,t,down_idx] / ∂ up_latent[b,t,up_idx]

    using `torch.autograd.functional.jvp`.
    """
    if logstats:
        print("[edge-attr-jvp] running JVP edge attribution")

    up_hook, down_hook = upstream_sae.cfg.hook_name, downstream_sae.cfg.hook_name

    # Baseline activations [B, L, S]
    up_base   = clean_sae_cache[up_hook].to(device)      # [B, L, S_up]
    down_base = clean_sae_cache[down_hook].to(device)    # [B, L, S_down]

    # Optional downstream loss gradient
    down_grad = res_sae_effects[down_hook].to(device) if edge_includes_loss_grad else None

    # Container: (down_idx, up_idx) -> list[Tensor]
    bucket: Dict[Tuple[int, int], List[torch.Tensor]] = {}

    # Reset SAE state once (no grads required for forward-mode)
    for sae in base_saes:
        sae.mean_error   = clean_error_cache[sae.cfg.hook_name].detach()
        sae.feature_acts = clean_sae_cache[sae.cfg.hook_name].detach().to(device)

    def _forward_fn(up_act: torch.Tensor) -> torch.Tensor:
        """
        up_act  : [B, L, S_up]
        returns : [B, L, S_down]
        """
        _, saes_out = run_with_saes( # run_sae_hook_fn(
            model,
            base_saes,
            token_list,
            calc_error=False,
            use_error=False,
            fake_activations=(upstream_sae.cfg.hook_layer, up_act),
            use_mean_error=use_mean_error,
            cache_sae_activations=True,
        )
        return saes_out[downstream_sae.cfg.hook_layer].feature_acts

    # ── loop over upstream features ────────────────────────────────────────
    for up_idx in upstream_features:
        # 1-hot direction in latent dim, broadcast over B and L
        direction = torch.zeros_like(up_base)
        direction[..., up_idx] = 1.0

        # Forward-mode Jacobian-vector product
        _, jvp = torch.autograd.functional.jvp(
            _forward_fn, (up_base,), (direction,), create_graph=False, strict=False
        )  # jvp: [B, L, S_down]

        # ── accumulate contributions ───────────────────────────────────────
        for down_idx in downstream_features:
            contrib = jvp[..., down_idx]                      # [B, L]
            if down_grad is not None:
                contrib = contrib * down_grad[..., down_idx]  # weighting
            val = contrib.sum()                               # scalar
            if val.abs() < 1e-6:
                continue
            bucket.setdefault((down_idx, up_idx), []).append(val.detach().cpu())

        # hygiene
        clear_memory(base_saes, model)

    if not bucket:
        return None

    # ── assemble sparse COO tensor ─────────────────────────────────────────
    idxs, vals = zip(
        *[((d, u), torch.stack(v).mean()) for (d, u), v in bucket.items()]
    )
    idx_mat = torch.tensor(list(zip(*idxs)), dtype=torch.long)  # [2, N]
    val_mat = torch.stack(list(vals))                           # [N]

    edge_tensor = torch.sparse_coo_tensor(
        idx_mat,
        val_mat,
        size=(len(downstream_features), len(upstream_features)),
    ).coalesce()

    return edge_tensor



def _finite_differences_edge_attr(
    model,
    base_saes,
    token_list,
    res_sae_effects,
    clean_sae_cache,
    clean_error_cache,
    labels,
    device,
    *,
    max_features_per_layer: int = 100,
    fd_steps: int = 10,
    use_mean_error: bool = True,
    logstats: bool = False,
    edge_includes_loss_grad: bool = True,
    feature_selection: str = "max",
):
    """Compute edge attribution
    """

    if feature_selection not in {"max", "sum", "negative"}:
        raise ValueError("feature_selection must be 'max', 'sum', or 'negative'.")

    important_features: Dict[str, List[int]] = {}
    for sae in base_saes:
        hname = sae.cfg.hook_name
        effects = res_sae_effects[hname]  # [L, S]
        if feature_selection == "max":
            scores = effects.abs().max(dim=0).values  # per‑latent
        elif feature_selection == "sum":
            scores = effects.abs().sum(dim=0)
        elif feature_selection == "negative":
            scores = -effects.min(dim=0).values  # Most negative becomes most positive
        else:
            raise ValueError(f"Unknown feature_selection: {feature_selection}")
        k = min(max_features_per_layer, scores.numel())
        top_idx = torch.topk(scores, k).indices if k > 0 else torch.tensor([], dtype=torch.long)
        important_features[hname] = top_idx.tolist()
        if logstats:
            print(f"[edge‑attr-fd] {hname}: kept {len(top_idx)}/{scores.numel()} features")

    edges: Dict[str, Dict[str, torch.Tensor]] = {}
    for i in range(len(base_saes) - 1):
        up_sae, down_sae = base_saes[i], base_saes[i + 1]
        up_hook, down_hook = up_sae.cfg.hook_name, down_sae.cfg.hook_name

        up_feats = important_features.get(up_hook, [])
        down_feats = important_features.get(down_hook, [])
        if not up_feats or not down_feats:
            if logstats:
                print(f"[edge‑attr-fd] skip {up_hook}->{down_hook} (no feats)")
            continue

        if logstats:
            print(f"[edge‑attr-fd] computing {up_hook}->{down_hook}")

        edge = _compute_finite_differences_edge(
            model,
            base_saes,
            up_sae,
            down_sae,
            token_list,
            up_feats,
            down_feats,
            clean_sae_cache,
            clean_error_cache,
            res_sae_effects,
            labels,
            device,
            steps=fd_steps,
            use_mean_error=use_mean_error,
            edge_includes_loss_grad=edge_includes_loss_grad,
            logstats=logstats,
        )
        if edge is None:
            continue
        edges.setdefault(up_hook, {})[down_hook] = edge

    if logstats:
        n_edges = sum(len(v) for v in edges.values())
        print(f"[edge‑attr-fd] finished – {n_edges} non‑zero edge tensors")

    return edges

def _compute_finite_differences_edge(
    model,
    base_saes,
    upstream_sae,
    downstream_sae,
    token_list,
    upstream_features: List[int],
    downstream_features: List[int],
    clean_sae_cache,
    clean_error_cache,
    res_sae_effects,
    labels,
    device,
    *,
    steps: int = 5,
    use_mean_error: bool = True,
    edge_includes_loss_grad: bool = True,
    logstats: bool = False,
):
    """Finite‑step perturbation implementation.

    In practice this still adds a small α and measures Δ – i.e. it is the
    perturb‑and‑measure approximation.
    """
    print(f"[edge‑attr-fd] running finite differences edge attribution")

    up_hook, down_hook = upstream_sae.cfg.hook_name, downstream_sae.cfg.hook_name
    up_clean = clean_sae_cache[up_hook].detach().to(device)  # [L,S]
    down_clean = clean_sae_cache[down_hook].detach().to(device)
    down_grad = res_sae_effects[down_hook].to(device) if edge_includes_loss_grad else None

    # Collect contributions across `steps` different α magnitudes, then average.
    bucket: Dict[Tuple[int, int], List[torch.Tensor]] = {}

    for s in range(steps):
        α = 0.01 * (s + 1)
        # reset SAE caches each outer loop
        for sae in base_saes:
            sae.mean_error = clean_error_cache[sae.cfg.hook_name].detach()
            sae.feature_acts = clean_sae_cache[sae.cfg.hook_name].detach().to(device)

        for up_idx in upstream_features:
            pert = up_clean.clone()
            pert[..., up_idx] += α

            _, updated_saes = run_with_saes( # run_sae_hook_fn(
                model,
                base_saes,
                token_list,
                calc_error=False,
                use_error=False,
                fake_activations=(upstream_sae.cfg.hook_layer, pert),
                use_mean_error=use_mean_error,
                cache_sae_activations=True,
            )
            down_pert = updated_saes[downstream_sae.cfg.hook_layer].feature_acts
            delta = (down_pert - down_clean) / α  # [L,S]

            for down_idx in downstream_features:
                val = torch.sum(
                    (down_grad[..., down_idx] if down_grad is not None else 1.0)
                    * delta[..., down_idx]
                )
                if val.abs() < 1e-6:
                    continue
                bucket.setdefault((down_idx, up_idx), []).append(val.detach().cpu())

            clear_memory(base_saes, model)

    if not bucket:
        return None

    idxs, vals = zip(*[((d, u), torch.stack(v).mean()) for (d, u), v in bucket.items()])
    idx_mat = torch.tensor(list(zip(*idxs)), dtype=torch.long)  # [2, N]
    val_mat = torch.stack(list(vals))
    edge_tensor = torch.sparse_coo_tensor(
        idx_mat, val_mat, size=(len(downstream_features), len(upstream_features))
    ).coalesce()
    return edge_tensor

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

def discover_circuit_edge_attr(model, 
                     saes, 
                     changable_toks, 
                    #  label,
                     device,
                     ig_steps=10,
                     k_max=7001,
                     k_step=500,
                     k_thres=0.6,
                     compute_edges=True,  
                     edge_method="jvp",    
                     max_edge_features=100, 
                     edge_includes_loss_grad=True,  
                     edge_feature_selection="max"   
                     ):
    model.reset_hooks(including_permanent=True)
    with torch.no_grad():
        logits_BLV, saes =run_with_saes(model, saes, changable_toks, calc_error=True, use_error=True, cache_sae_activations=True)

    label = logits_BLV[0, -1, :].argmax(-1).item()
    cleanup_cuda()
    # clean_logits = logits_BLV[0, -1, label]
    probs_BV = F.softmax(logits_BLV[:, -1, :], dim=-1)
    clean_probs_baseline = probs_BV[0, label]

    for ind, sae in enumerate(saes):
        sae.mean_error = sae.error_term.detach()

    del logits_BLV
    cleanup_cuda()

    # Get Cache
    clean_sae_cache, clean_error_cache, corr_sae_cache, corr_error_cache = get_saes_cache(saes)

    # Run Intergrated Gradients & Save Results
    res_sae_effects, _ = run_integrated_gradients(
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

    # NEW: Compute edges if requested
    edges = None
    if compute_edges and res_sae_effects:
        print("Computing edge attributions...")
        if edge_method == "finite_differences":
            edges = _finite_differences_edge_attr(
                model=model,
                base_saes=saes,
                token_list=changable_toks,
                res_sae_effects=res_sae_effects,
                clean_sae_cache=clean_sae_cache,
                clean_error_cache=clean_error_cache,
                labels=torch.tensor([label]).to(device),
                device=device,
                max_features_per_layer=max_edge_features,
                edge_includes_loss_grad=edge_includes_loss_grad,
                feature_selection=edge_feature_selection,
                logstats=True, 
                fd_steps=5
            )
        elif edge_method == "jvp":
                edges = _jvp_edge_attr(
                model=model,
                base_saes=saes,
                token_list=changable_toks,
                res_sae_effects=res_sae_effects,
                clean_sae_cache=clean_sae_cache,
                clean_error_cache=clean_error_cache,
                labels=torch.tensor([label]).to(device),
                device=device,
                max_features_per_layer=max_edge_features,
                use_mean_error=True,
                logstats=True,
                edge_includes_loss_grad=edge_includes_loss_grad,
                feature_selection=edge_feature_selection,
            )
        
        
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
        if compute_edges:
            return None, edges
        else:
            return None

    # we only need to store entries
    entries = list(itertools.islice(topk_iter, K)) # preserves order

    # Return both entries and edges if computed
    if compute_edges:
        return entries, edges
    else:
        return entries

# %%

model_name = "gemma-2-2b-it"
device = "cuda"
model = load_model(model_name, device=device, use_custom_cache=False, dtype=torch.bfloat16)

layers = list(range(model.cfg.n_layers))
saes = load_pretrained_saes(
    layers=layers, 
    release="gemma-scope-2b-pt-mlp-canonical", 
    width="16k", 
    device=device, 
    canon=True
)

#%%

data_path = "../data/first_100_passing_examples.json"
prompt_idx = 15
inter_token_id = 297
stop_token_id = 1917
ig_steps = 10
k_max = 90001
k_step = 10000
k_thres = 0.6

with open(data_path, 'r') as f:
        data = json.load(f)

# %%
entry = data[prompt_idx]
prompt = (
    "You are an expert Python programmer, and here is your task: "
    f"{entry['prompt']} Your code should pass these tests:\n\n"
    + "\n".join(entry["test_list"]) + "\nWrite your code below starting with \"```python\" and ending with \"```\".\n```python\n"
)

toks_BL = model.to_tokens(prompt).to(device)
out_BL = toks_BL.clone()

while out_BL.shape[-1] - toks_BL.shape[-1] < 150:
    with torch.no_grad():
        logits_V = model(out_BL)[0, -1]
    next_id = logits_V.argmax(-1).item()
    del logits_V
    cleanup_cuda()
    if next_id == stop_token_id:
        break
    out_BL = torch.cat([out_BL, torch.tensor([[next_id]], device=device)], dim=1)

# Extract the specific prediction position
inter_toks_BL = out_BL[:, :inter_token_id]
baseline_suffix = model.to_string(out_BL[0, inter_token_id:])
print(baseline_suffix)
print(model.to_string(inter_toks_BL[0]))

# %%

entries = discover_circuit_edge_attr(
        model=model,
        saes=saes,
        changable_toks=inter_toks_BL,
        # inter_toks_BL=inter_toks_BL,
        device=device,
        ig_steps=ig_steps,
        k_max=k_max,
        k_step=k_step,
        k_thres=k_thres,
        compute_edges=True,
        edge_method="finite_differences",
        max_edge_features=50,
        edge_includes_loss_grad=True,
        edge_feature_selection="max"
    )


# %% DEBUGGING RUNS FROM HERE

model.reset_hooks(including_permanent=True)
with torch.no_grad():
    logits_BLV, saes =run_with_saes(model, saes, inter_toks_BL, calc_error=True, use_error=True, cache_sae_activations=True)

label = logits_BLV[0, -1, :].argmax(-1).item()
cleanup_cuda()
probs_BV = F.softmax(logits_BLV[:, -1, :], dim=-1)
clean_probs_baseline = probs_BV[0, label]

for ind, sae in enumerate(saes):
    sae.mean_error = sae.error_term.detach()

del logits_BLV
cleanup_cuda()

# Get Cache
clean_sae_cache, clean_error_cache, corr_sae_cache, corr_error_cache = get_saes_cache(saes)

# Run Intergrated Gradients & Save Results
res_sae_effects, _ = run_integrated_gradients(
    model=model,
    base_saes=saes,
    token_list=inter_toks_BL,
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

# %%

max_features_per_layer = 100
feature_selection = "negative"  # NEW: negative selection strategy
logstats = True
fd_steps = 5
use_mean_error = True
edge_includes_loss_grad = True

important_features: Dict[str, List[int]] = {}
for sae in saes:
    hname = sae.cfg.hook_name
    effects = res_sae_effects[hname]  # [L, S]
    if feature_selection == "max":
        scores = effects.abs().max(dim=0).values  # per‑latent
    elif feature_selection == "sum":
        scores = effects.abs().sum(dim=0)
    elif feature_selection == "negative":
        scores = -effects.min(dim=0).values  # Most negative becomes most positive
    else:
        raise ValueError(f"Unknown feature_selection: {feature_selection}")
    k = min(max_features_per_layer, scores.numel())
    top_idx = torch.topk(scores, k).indices if k > 0 else torch.tensor([], dtype=torch.long)
    important_features[hname] = top_idx.tolist()
    if logstats:
        print(f"[edge‑attr-fd] {hname}: kept {len(top_idx)}/{scores.numel()} features")

# %%
edges: Dict[str, Dict[str, torch.Tensor]] = {}
for i in range(len(saes) - 1):
    up_sae, down_sae = saes[i], saes[i + 1]
    up_hook, down_hook = up_sae.cfg.hook_name, down_sae.cfg.hook_name

    up_feats = important_features.get(up_hook, [])
    down_feats = important_features.get(down_hook, [])
    if not up_feats or not down_feats:
        if logstats:
            print(f"[edge‑attr-fd] skip {up_hook}->{down_hook} (no feats)")
        continue

    if logstats:
        print(f"[edge‑attr-fd] computing {up_hook}->{down_hook}")

    edge = _compute_finite_differences_edge(
        model,
        saes,
        up_sae,
        down_sae,
        inter_toks_BL, # token_list,
        up_feats,
        down_feats,
        clean_sae_cache,
        clean_error_cache,
        res_sae_effects,
        torch.tensor([label]).to(device),
        device,
        steps=fd_steps,
        use_mean_error=use_mean_error,
        edge_includes_loss_grad=edge_includes_loss_grad,
        logstats=logstats,
    )
    if edge is not None:
        edges.setdefault(up_hook, {})[down_hook] = edge
    break

print(edges)

# if logstats:
#     n_edges = sum(len(v) for v in edges.values())
#     print(f"[edge‑attr-fd] finished – {n_edges} non‑zero edge tensors")

# %%




# %% finite differences step by step run, step 1

max_features_per_layer = 100
feature_selection = "negative"  # NEW: negative selection strategy
logstats = True
fd_steps = 5
use_mean_error = True
edge_includes_loss_grad = True

print("=== FEATURE SELECTION DEBUG ===")
important_features: Dict[str, List[int]] = {}
for sae in saes:
    hname = sae.cfg.hook_name
    effects = res_sae_effects[hname]  # [L, S]
    print(f"\n--- Processing {hname} ---")
    print(f"Effects shape: {effects.shape}")
    print(f"Effects range: [{effects.min().item():.4f}, {effects.max().item():.4f}]")
    
    if feature_selection == "max":
        scores = effects.abs().max(dim=0).values  # per‑latent
    elif feature_selection == "sum":
        scores = effects.abs().sum(dim=0)
    elif feature_selection == "negative":
        # Take the most negative effects (reverse sort)
        scores = -effects.min(dim=0).values  # per-latent, most negative becomes most positive
    else:
        raise ValueError(f"Unknown feature_selection: {feature_selection}")
    
    print(f"Scores shape: {scores.shape}")
    print(f"Scores range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
    
    k = min(max_features_per_layer, scores.numel())
    top_idx = torch.topk(scores, k).indices if k > 0 else torch.tensor([], dtype=torch.long)
    important_features[hname] = top_idx.tolist()
    
    print(f"Selected {len(top_idx)}/{scores.numel()} features")
    print(f"Top 10 feature indices: {top_idx[:10].tolist()}")
    print(f"Top 10 scores: {scores[top_idx[:10]].tolist()}")
    
    if logstats:
        print(f"[edge‑attr-fd] {hname}: kept {len(top_idx)}/{scores.numel()} features")

print(f"\nTotal layers with features: {len(important_features)}")

# %%

"""
implement a single upstream layer and multiple downstream layers but only calculating delta, not the down_grad. 
This specific cell does zero ablation instead of adding an alpha. 
"""

print("=== UPSTREAM TO MULTIPLE DOWNSTREAM DELTA ANALYSIS ===")

# Set parameters
upstream_layer_idx = 0
downstream_layer_range = (1, 16)  # layers 1-15
alpha_values = [0.01, 0.1, 0.5, 1.0]
threshold = 1e-10
max_features_per_layer = 100

# Get upstream SAE
if upstream_layer_idx >= len(saes):
    print(f"ERROR: upstream_layer_idx {upstream_layer_idx} >= num_saes {len(saes)}")
else:
    up_sae = saes[upstream_layer_idx]
    up_hook = up_sae.cfg.hook_name
    up_feats = important_features.get(up_hook, [])
    
    if not up_feats:
        print(f"ERROR: No features found for upstream layer {upstream_layer_idx}")
    else:
        # Limit upstream features
        up_feats = up_feats[:5]  # Just look at first feature for initial testing
        
        print(f"Upstream layer: {upstream_layer_idx} ({up_hook})")
        print(f"Upstream features: {len(up_feats)} features")
        print(f"Downstream layers: {downstream_layer_range[0]} to {downstream_layer_range[1]-1}")
        
        # Get upstream baseline
        up_clean = clean_sae_cache[up_hook].detach().to(device)
        print(f"Upstream baseline shape: {up_clean.shape}")
        
        # Get downstream SAEs and baselines
        downstream_saes = []
        downstream_baselines = {}
        downstream_features = {}
        
        for down_layer_idx in range(downstream_layer_range[0], min(downstream_layer_range[1], len(saes))):
            down_sae = saes[down_layer_idx]
            down_hook = down_sae.cfg.hook_name
            down_feats = important_features.get(down_hook, [])
            
            if down_feats:
                downstream_saes.append((down_layer_idx, down_sae))
                downstream_baselines[down_layer_idx] = clean_sae_cache[down_hook].detach().to(device)
                downstream_features[down_layer_idx] = down_feats[:max_features_per_layer]
                
                print(f"  Downstream layer {down_layer_idx}: {len(down_feats)} features -> {len(downstream_features[down_layer_idx])} used")
        
        print(f"Total downstream layers with features: {len(downstream_saes)}")
        
        if downstream_saes:
            print(f"\nStarting ablation analysis...")
            print(f"Testing {len(up_feats)} upstream features")
            print(f"Zeroing out activations instead of perturbation")
            
            # Storage for significant deltas
            significant_deltas = []
            total_deltas = 0
            
            # Track maximum absolute delta
            max_abs_delta = 0.0
            max_delta_info = None
            
            # Reset SAE states once
            for sae in saes:
                if hasattr(sae, 'mean_error'):
                    sae.mean_error = clean_error_cache[sae.cfg.hook_name].detach()
                if hasattr(sae, 'feature_acts'):
                    sae.feature_acts = clean_sae_cache[sae.cfg.hook_name].detach().to(device)
            
            # Loop through upstream features
            for up_feat_idx in up_feats:
                print(f"  Zeroing out upstream feature {up_feat_idx}")
                
                # Create ablation - set feature to 0
                pert = up_clean.clone()
                pert[..., up_feat_idx] = 0.0
                
                try:
                    # Run forward pass with ablation
                    _, updated_saes = run_with_saes(
                        model,
                        saes,
                        inter_toks_BL,
                        calc_error=False,
                        use_error=False,
                        fake_activations=(up_sae.cfg.hook_layer, pert),
                        use_mean_error=use_mean_error,
                        cache_sae_activations=True,
                    )
                    
                    # Check all downstream layers
                    for down_layer_idx, down_sae in downstream_saes:
                        down_pert = updated_saes[down_sae.cfg.hook_layer].feature_acts
                        down_clean = downstream_baselines[down_layer_idx]
                        
                        # Calculate delta (no division by alpha needed)
                        delta = down_pert - down_clean
                        
                        # Check downstream features
                        down_feats = downstream_features[down_layer_idx]
                        for down_feat_idx in down_feats:
                            delta_val = delta[..., down_feat_idx]
                            
                            # Sum across all positions (tokens)
                            delta_sum = torch.sum(delta_val).item()
                            total_deltas += 1
                            
                            # Track maximum absolute delta
                            abs_delta = abs(delta_sum)
                            if abs_delta > max_abs_delta:
                                max_abs_delta = abs_delta
                                max_delta_info = {
                                    'upstream_feature': up_feat_idx,
                                    'downstream_layer': down_layer_idx,
                                    'downstream_feature': down_feat_idx,
                                    'ablation': 'zeroed',
                                    'delta_sum': delta_sum,
                                    'delta_abs': abs_delta
                                }
                            
                            # Check if significant and print
                            if abs_delta > threshold:
                                significant_deltas.append({
                                    'upstream_feature': up_feat_idx,
                                    'downstream_layer': down_layer_idx,
                                    'downstream_feature': down_feat_idx,
                                    'ablation': 'zeroed',
                                    'delta_sum': delta_sum,
                                    'delta_abs': abs_delta
                                })
                                print(f"    SIGNIFICANT: up_feat={up_feat_idx}, down_layer={down_layer_idx}, down_feat={down_feat_idx}, delta={delta_sum:.8f}")
                    
                    # Clean up memory
                    clear_memory(saes, model)
                    
                except Exception as e:
                    print(f"    ERROR in forward pass for up_feat {up_feat_idx}: {e}")
                    continue
            
            # Print final summary
            print(f"\n=== ANALYSIS COMPLETE ===")
            print(f"Total deltas computed: {total_deltas}")
            print(f"Significant deltas (abs > {threshold}): {len(significant_deltas)}")
            
            # Print maximum absolute delta information
            if max_delta_info:
                print(f"\nMAXIMUM ABSOLUTE DELTA:")
                print(f"  Max |delta| = {max_abs_delta:.2e}")
                print(f"  Upstream feature: {max_delta_info['upstream_feature']}")
                print(f"  Downstream layer: {max_delta_info['downstream_layer']}")
                print(f"  Downstream feature: {max_delta_info['downstream_feature']}")
                print(f"  Ablation: {max_delta_info['ablation']}")
                print(f"  Delta value: {max_delta_info['delta_sum']:.12e}")
            else:
                print(f"\nNo deltas computed (this shouldn't happen!)")
            
            # Sort significant deltas by magnitude
            significant_deltas.sort(key=lambda x: x['delta_abs'], reverse=True)
            
            # Print top significant deltas
            print(f"\nTop 20 significant deltas:")
            for i, delta_info in enumerate(significant_deltas[:20]):
                print(f"  {i+1:2d}. up_feat={delta_info['upstream_feature']:4d}, down_layer={delta_info['downstream_layer']:2d}, down_feat={delta_info['downstream_feature']:4d}, ablation={delta_info['ablation']}, δ={delta_info['delta_sum']:10.6f}")
            
            # Summary by downstream layer
            layer_summary = {}
            for delta_info in significant_deltas:
                layer = delta_info['downstream_layer']
                if layer not in layer_summary:
                    layer_summary[layer] = {'count': 0, 'max_delta': 0}
                layer_summary[layer]['count'] += 1
                layer_summary[layer]['max_delta'] = max(layer_summary[layer]['max_delta'], delta_info['delta_abs'])
            
            print(f"\nSummary by downstream layer:")
            for layer in sorted(layer_summary.keys()):
                info = layer_summary[layer]
                print(f"  Layer {layer:2d}: {info['count']:4d} significant deltas, max_delta={info['max_delta']:8.6f}")
        
        else:
            print("ERROR: No downstream layers have features!")
            
print("=== END OF UPSTREAM TO MULTIPLE DOWNSTREAM ABLATION ANALYSIS ===")



# %%
# STEP 2: Toy Scenario - Just Layers 0 and 1 Debug
print("=== TOY SCENARIO: LAYERS 0 AND 1 DEBUG ===")

# Get first two SAEs only
if len(saes) >= 2:
    up_sae, down_sae = saes[0], saes[1]
    up_hook, down_hook = up_sae.cfg.hook_name, down_sae.cfg.hook_name
    
    print(f"Upstream SAE: {up_hook}")
    print(f"Downstream SAE: {down_hook}")
    
    # Get features for both layers
    up_feats = important_features.get(up_hook, [])
    down_feats = important_features.get(down_hook, [])
    
    print(f"Upstream features: {len(up_feats)} features")
    print(f"Downstream features: {len(down_feats)} features")
    
    if not up_feats or not down_feats:
        print("ERROR: No features found for one or both layers!")
        if not up_feats:
            print(f"  - No upstream features for {up_hook}")
        if not down_feats:
            print(f"  - No downstream features for {down_hook}")
    else:
        print(f"SUCCESS: Both layers have features")
        print(f"  - Upstream: {up_feats[:5]}... ({len(up_feats)} total)")
        print(f"  - Downstream: {down_feats[:5]}... ({len(down_feats)} total)")
        
        # Show the actual effects for these features
        up_effects = res_sae_effects[up_hook]
        down_effects = res_sae_effects[down_hook]
        
        print(f"\nEffect tensors:")
        print(f"  - Upstream effects shape: {up_effects.shape}")
        print(f"  - Downstream effects shape: {down_effects.shape}")
        
        # Sample some effects
        print(f"\nSample upstream effects for first 5 features:")
        for i, feat_idx in enumerate(up_feats[:5]):
            feat_effects = up_effects[:, feat_idx]
            print(f"  - Feature {feat_idx}: {feat_effects.tolist()}")
            
        print(f"\nSample downstream effects for first 5 features:")
        for i, feat_idx in enumerate(down_feats[:5]):
            feat_effects = down_effects[:, feat_idx]
            print(f"  - Feature {feat_idx}: {feat_effects.tolist()}")
else:
    print("ERROR: Need at least 2 SAEs for edge computation!")

# %%

# STEP 3: Detailed Finite Differences Computation
print("=== DETAILED FINITE DIFFERENCES COMPUTATION ===")

# Assuming we have up_sae and down_sae from the previous cell
if len(saes) >= 2 and len(important_features.get(saes[0].cfg.hook_name, [])) > 0 and len(important_features.get(saes[1].cfg.hook_name, [])) > 0:
    up_sae, down_sae = saes[0], saes[1]
    up_hook, down_hook = up_sae.cfg.hook_name, down_sae.cfg.hook_name
    up_feats = important_features[up_hook]
    down_feats = important_features[down_hook]
    
    print(f"Starting computation for {up_hook} -> {down_hook}")
    print(f"Steps: {fd_steps}")
    print(f"Use mean error: {use_mean_error}")
    print(f"Edge includes loss grad: {edge_includes_loss_grad}")
    
    # Get baseline activations
    up_clean = clean_sae_cache[up_hook].detach().to(device)
    down_clean = clean_sae_cache[down_hook].detach().to(device)
    
    print(f"\nBaseline activations:")
    print(f"  - Upstream clean shape: {up_clean.shape}")
    print(f"  - Downstream clean shape: {down_clean.shape}")
    print(f"  - Upstream clean range: [{up_clean.min().item():.4f}, {up_clean.max().item():.4f}]")
    print(f"  - Downstream clean range: [{down_clean.min().item():.4f}, {down_clean.max().item():.4f}]")
    
    # Get downstream gradient if needed
    down_grad = res_sae_effects[down_hook].to(device) if edge_includes_loss_grad else None
    if down_grad is not None:
        print(f"\nDownstream gradient:")
        print(f"  - Shape: {down_grad.shape}")
        print(f"  - Range: [{down_grad.min().item():.4f}, {down_grad.max().item():.4f}]")
    else:
        print(f"\nNo downstream gradient (edge_includes_loss_grad=False)")
    
    # Initialize bucket for collecting contributions
    bucket: Dict[Tuple[int, int], List[torch.Tensor]] = {}
    
    print(f"\nStarting perturbation loop...")
    print(f"Will test {len(up_feats)} upstream features x {len(down_feats)} downstream features")
    
    # Limit to first few features for debugging
    up_feats_debug = up_feats[:3]  # Just first 3 for debugging
    down_feats_debug = down_feats[:3]  # Just first 3 for debugging
    
    print(f"DEBUG: Limited to {len(up_feats_debug)} x {len(down_feats_debug)} for debugging")
    
    # Store original SAE states
    original_states = {}
    for sae in saes:
        original_states[sae.cfg.hook_name] = {
            'mean_error': sae.mean_error.clone() if hasattr(sae, 'mean_error') else None,
            'feature_acts': sae.feature_acts.clone() if hasattr(sae, 'feature_acts') else None
        }
    
    print(f"Stored original SAE states for {len(original_states)} SAEs")
    
else:
    print("ERROR: Cannot proceed - missing SAEs or features!")

# %%

# STEP 4: Perturbation Loop with Detailed Logging
print("=== PERTURBATION LOOP ===")

if len(saes) >= 2 and len(important_features.get(saes[0].cfg.hook_name, [])) > 0 and len(important_features.get(saes[1].cfg.hook_name, [])) > 0:
    
    # Get our variables from previous cell
    up_sae, down_sae = saes[0], saes[1]
    up_hook, down_hook = up_sae.cfg.hook_name, down_sae.cfg.hook_name
    up_feats = important_features[up_hook]
    down_feats = important_features[down_hook]
    up_clean = clean_sae_cache[up_hook].detach().to(device)
    down_clean = clean_sae_cache[down_hook].detach().to(device)
    down_grad = res_sae_effects[down_hook].to(device) if edge_includes_loss_grad else None
    
    # Debug with very limited features
    up_feats_debug = up_feats[:100]  # Just 2 features
    down_feats_debug = down_feats[:100]  # Just 2 features
    
    bucket: Dict[Tuple[int, int], List[torch.Tensor]] = {}
    
    print(f"Testing {len(up_feats_debug)} upstream x {len(down_feats_debug)} downstream features")
    print(f"Upstream features: {up_feats_debug}")
    print(f"Downstream features: {down_feats_debug}")
    
    # Outer loop: different alpha values
    for s in range(fd_steps):
        α = 0.1 * (s + 1)
        print(f"\n--- Step {s+1}/{fd_steps}, α = {α} ---")
        
        # Reset SAE caches for each alpha
        for sae in saes:
            if hasattr(sae, 'mean_error'):
                sae.mean_error = clean_error_cache[sae.cfg.hook_name].detach()
            if hasattr(sae, 'feature_acts'):
                sae.feature_acts = clean_sae_cache[sae.cfg.hook_name].detach().to(device)
        
        # Inner loop: upstream features
        for up_idx in up_feats_debug:
            print(f"\n  Perturbing upstream feature {up_idx}")
            
            # Create perturbation
            pert = up_clean.clone()
            pert[..., up_idx] += α
            
            # print(f"    Original value at feature {up_idx}: {up_clean[..., up_idx].tolist()}")
            # print(f"    Perturbed value at feature {up_idx}: {pert[..., up_idx].tolist()}")
            # print(f"    Perturbation magnitude: {α}")
            
            # Run forward pass with perturbation
            try:
                print(f"    Running forward pass with perturbation...")
                _, updated_saes = run_with_saes(
                    model,
                    saes,
                    inter_toks_BL,
                    calc_error=False,
                    use_error=False,
                    fake_activations=(up_sae.cfg.hook_layer, pert),
                    use_mean_error=use_mean_error,
                )
                
                down_pert = updated_saes[down_sae.cfg.hook_layer].feature_acts
                # print(f"    Got downstream perturbed activations: {down_pert.shape}")
                
                # Compute delta
                delta = (down_pert - down_clean) / α
                # print(f"    Delta range: [{delta.min().item():.6f}, {delta.max().item():.6f}]")
                
                # Check specific downstream features
                for down_idx in down_feats_debug:
                    delta_val = delta[..., down_idx]
                    # print(f"    Delta for downstream feature {down_idx}: {delta_val.tolist()}")
                    
                    # Apply gradient weighting if needed
                    if down_grad is not None:
                        grad_weight = down_grad[..., down_idx]
                        weighted_delta = grad_weight * delta_val
                        # print(f"    Gradient weight: {grad_weight.tolist()}")
                        # print(f"    Weighted delta: {weighted_delta.tolist()}")
                        val = torch.sum(weighted_delta)
                    else:
                        val = torch.sum(delta_val)
                    
                    print(f"    Final contribution for ({down_idx}, {up_idx}): {val.item():.6f}")
                    
                    # Store if significant
                    if val.abs() >= 1e-10:
                        bucket.setdefault((down_idx, up_idx), []).append(val.detach().cpu())
                        print(f"    ✓ Stored contribution for ({down_idx}, {up_idx})")
                    else:
                        print(f"    ✗ Contribution too small, skipping")
                
                # Clean up
                clear_memory(saes, model)
                # print(f"    Cleared memory")
                
            except Exception as e:
                print(f"    ERROR in forward pass: {e}")
                
        print(f"  Bucket status: {len(bucket)} entries")
        for key, values in bucket.items():
            print(f"    {key}: {len(values)} contributions")
    
    print(f"\nFinal bucket: {len(bucket)} unique (downstream, upstream) pairs")
    for (down_idx, up_idx), values in bucket.items():
        mean_val = torch.stack(values).mean().item()
        print(f"  ({down_idx}, {up_idx}): {len(values)} values, mean = {mean_val:.6f}")
    
else:
    print("ERROR: Cannot proceed - missing SAEs or features!")

# %%

# STEP 5: Create Sparse Tensor from Bucket
print("=== CREATING SPARSE TENSOR ===")

# This assumes we have the bucket from the previous cell
if 'bucket' in locals() and len(bucket) > 0:
    print(f"Creating sparse tensor from {len(bucket)} entries...")
    
    # Show what we have
    print(f"Bucket contents:")
    for (down_idx, up_idx), values in bucket.items():
        values_tensor = torch.stack(values)
        mean_val = values_tensor.mean().item()
        std_val = values_tensor.std().item()
        print(f"  ({down_idx}, {up_idx}): {len(values)} values, mean={mean_val:.6f}, std={std_val:.6f}")
    
    # Create the sparse tensor
    idxs, vals = zip(*[((d, u), torch.stack(v).mean()) for (d, u), v in bucket.items()])
    
    print(f"\nTensor creation:")
    print(f"  Indices: {idxs}")
    print(f"  Values: {[v.item() for v in vals]}")
    
    # Convert to matrix format
    idx_mat = torch.tensor(list(zip(*idxs)), dtype=torch.long)  # [2, N]
    val_mat = torch.stack(list(vals))  # [N]
    
    print(f"  Index matrix shape: {idx_mat.shape}")
    print(f"  Value matrix shape: {val_mat.shape}")
    print(f"  Index matrix: {idx_mat}")
    print(f"  Value matrix: {val_mat}")
    
    # Determine tensor size
    # Note: this should be based on the actual number of features we tested
    max_down_idx = max(down_idx for down_idx, up_idx in bucket.keys())
    max_up_idx = max(up_idx for down_idx, up_idx in bucket.keys())
    
    # But we need to map back to positions in our feature lists
    down_feats_debug = down_feats[:2]  # Same as used in previous cell
    up_feats_debug = up_feats[:2]      # Same as used in previous cell
    
    tensor_size = (len(down_feats_debug), len(up_feats_debug))
    print(f"  Tensor size: {tensor_size}")
    
    # Create sparse tensor
    edge_tensor = torch.sparse_coo_tensor(
        idx_mat,
        val_mat,
        size=tensor_size,
    ).coalesce()
    
    print(f"  Created sparse tensor:")
    print(f"    Shape: {edge_tensor.shape}")
    print(f"    NNZ: {edge_tensor._nnz()}")
    print(f"    Dense representation:")
    print(f"    {edge_tensor.to_dense()}")
    
    # Show the mapping
    print(f"\nFeature mapping:")
    print(f"  Downstream features: {down_feats_debug}")
    print(f"  Upstream features: {up_feats_debug}")
    
    # Convert to dense to see the full matrix
    dense_edge = edge_tensor.to_dense()
    print(f"\nEdge matrix (downstream x upstream):")
    for i, down_feat in enumerate(down_feats_debug):
        row_str = f"  Down {down_feat}: ["
        for j, up_feat in enumerate(up_feats_debug):
            row_str += f"{dense_edge[i, j].item():8.4f} "
        row_str += "]"
        print(row_str)
    
    SUCCESS = True
    
elif 'bucket' in locals():
    print("Bucket exists but is empty!")
    print("This means no significant contributions were found.")
    SUCCESS = False
else:
    print("No bucket found - previous step may have failed")
    SUCCESS = False

print(f"\nSUCCESS: {SUCCESS}")

# %%
# SUMMARY: How to Run the Complete Edge Attribution Debug Pipeline

print("=== EDGE ATTRIBUTION DEBUG PIPELINE ===")
print("""
This file now contains a complete step-by-step debugging pipeline for edge attribution.
Run the cells in order to debug your edge attribution issues:

STEP 1: Feature Selection Debug
- Tests the new "negative" feature selection strategy
- Shows you exactly which features are selected and why
- Prints shapes, ranges, and top features for each layer

STEP 2: Toy Scenario Setup  
- Focuses on just layers 0 and 1 for simplicity
- Validates that both layers have selected features
- Shows sample effects for the selected features

STEP 3: Detailed Computation Setup
- Prepares all variables for finite differences
- Shows baseline activations and gradients
- Stores original SAE states for proper cleanup

STEP 4: Perturbation Loop
- Runs the finite differences computation step by step
- Shows exactly what happens for each perturbation
- Prints deltas, gradients, and final contributions
- Limited to 2x2 features for detailed debugging

STEP 5: Sparse Tensor Creation
- Creates the final edge tensor from collected contributions
- Shows the complete feature mapping
- Displays the final edge matrix

DEBUGGING TIPS:
1. If Step 1 shows no features selected, try different feature_selection strategies
2. If Step 4 shows no contributions, check that perturbations are large enough
3. If Step 4 shows error in forward pass, check your SAE setup
4. If contributions are too small (< 1e-6), try larger perturbations or different features

NEXT STEPS:
- Once this works, increase the number of features in Step 4
- Try different alpha values in the perturbation loop
- Test with more layer pairs
- Compare with JVP method for validation
""")

# %%

# DEBUGGING: Test Larger Perturbations and Check Baseline Values
print("=== DEBUGGING: PERTURBATION MAGNITUDE TEST ===")

if len(saes) >= 2 and len(important_features.get(saes[0].cfg.hook_name, [])) > 0:
    up_sae, down_sae = saes[0], saes[1]
    up_hook, down_hook = up_sae.cfg.hook_name, down_sae.cfg.hook_name
    up_feats = important_features[up_hook]
    down_feats = important_features[down_hook]
    up_clean = clean_sae_cache[up_hook].detach().to(device)
    down_clean = clean_sae_cache[down_hook].detach().to(device)
    
    # Check baseline values of selected features
    print(f"Checking baseline values for first 10 upstream features:")
    for i, feat_idx in enumerate(up_feats[:10]):
        baseline_val = up_clean[..., feat_idx]
        print(f"  Feature {feat_idx}: {baseline_val.tolist()} (min: {baseline_val.min().item():.6f}, max: {baseline_val.max().item():.6f})")
    
    print(f"\nChecking baseline values for first 10 downstream features:")
    for i, feat_idx in enumerate(down_feats[:10]):
        baseline_val = down_clean[..., feat_idx]
        print(f"  Feature {feat_idx}: {baseline_val.tolist()} (min: {baseline_val.min().item():.6f}, max: {baseline_val.max().item():.6f})")
    
    # Test different perturbation magnitudes
    test_alphas = [5, 10, 20, 50] #[0.01, 0.1, 0.5, 1.0, 2.0]
    print(f"\nTesting different perturbation magnitudes:")
    
    # Reset SAE states
    for sae in saes:
        if hasattr(sae, 'mean_error'):
            sae.mean_error = clean_error_cache[sae.cfg.hook_name].detach()
        if hasattr(sae, 'feature_acts'):
            sae.feature_acts = clean_sae_cache[sae.cfg.hook_name].detach().to(device)
    
    # Test just one feature pair
    test_up_feat = up_feats[0]
    test_down_feat = down_feats[0]
    
    print(f"Testing upstream feature {test_up_feat} -> downstream feature {test_down_feat}")
    
    for α in test_alphas:
        print(f"\n  α = {α}")
        
        # Create perturbation
        pert = up_clean.clone()
        pert[..., test_up_feat] += α
        
        print(f"    Original: {up_clean[..., test_up_feat].tolist()}")
        print(f"    Perturbed: {pert[..., test_up_feat].tolist()}")
        
        try:
            # Run forward pass
            _, updated_saes = run_with_saes(
                model, saes, inter_toks_BL,
                calc_error=False, use_error=False,
                fake_activations=(up_sae.cfg.hook_layer, pert),
                use_mean_error=use_mean_error,
            )
            
            down_pert = updated_saes[down_sae.cfg.hook_layer].feature_acts
            delta = (down_pert - down_clean) / α
            delta_val = delta[..., test_down_feat]
            
            print(f"    Delta: {delta_val.tolist()}")
            print(f"    Delta sum: {torch.sum(delta_val).item():.8f}")
            
            clear_memory(saes, model)
            
        except Exception as e:
            print(f"    ERROR: {e}")

# %%

