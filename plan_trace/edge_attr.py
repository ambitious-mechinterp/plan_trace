#%%

"""
Complete pipeline demonstration for planning detection in language models.

This script shows how to use all the modular components together to:
1. Discover circuits using integrated gradients
2. Cluster latents by logit lens  
3. Test steering effects
4. Analyze planning positions

Shape Suffix Definition: 
- B: batch size 
- L: Num of Input Tokens 
- O: Num of Output Tokens
- V: vocabulary size
- S: Number of SAE neurons in a layer
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

# """
# Shape Suffix Definition: 

# - B: batch size 
# - L: Num of Input Tokens 
# - O: Num of Output Tokens
# - V: vocabulary size
# - F: feed-forward subnetwork hidden size
# - D: Depth or number of layers
# - H: number of attention heads in a layer
# - S: Number of SAE neurons in a layer
# - A: Number of SAEs attached
# - tx: variables dealing with the prediction of the x'th output token
# """

# import argparse
# from tqdm import tqdm
# import sys
# import os
# import torch
# import itertools
# from IPython.display import display, HTML, IFrame
# import numpy as np
# import torch.nn.functional as F
# import re
# from tqdm import tqdm
# from typing import  Tuple,  Dict, Any, List
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import copy

# from helpers.utils import (
#     load_model, 
#     cleanup_cuda,
#     clear_memory,
# )
# from helpers.hook_manager import (
#     SAEMasks,
#     build_sae_hook_fn,
#     run_sae_hook_fn, 
# )
# from tracing_plans import load_pretrained_saes

### METRIC ### 

# def compute_metric(model, base_saes, sae_index, tokens, circuit_mask, label, use_mean_error=True):
#     hooks = []
#     bos_token_id = model.tokenizer.bos_token_id
#     for sae in base_saes:
#         hooks.append(
#             (
#                 sae.cfg.hook_name,
#                 build_sae_hook_fn(
#                     sae,
#                     tokens,
#                     bos_token_id,
#                     circuit_mask=circuit_mask if sae.cfg.hook_layer == sae_index else None,
#                     use_mean_error=use_mean_error,
#                 ),
#             )
#         )
#     circuit_logits = model.run_with_hooks(tokens, return_type="logits", fwd_hooks=hooks)
#     answer_logits = circuit_logits[..., -1, :]  # Get the logits of the last tokens
#     answer_logprobs = F.softmax(answer_logits, dim=-1)
#     clean_logprobs = answer_logprobs[..., torch.arange(answer_logits.shape[-2]), label]
#     return torch.sum(clean_logprobs).item()

# ### ATTRIBUTION ### 

# def run_integrated_gradients(
#     model,
#     base_saes,
#     token_list,
#     clean_sae_cache,
#     clean_error_cache,
#     corr_sae_cache,
#     corr_error_cache,
#     labels,
#     save_dir,
#     ig_steps=10,
#     save_and_use=True,   # <--- new,
#     logstats=False,  # <--- new
# ):
#     """
#     Runs a simple integrated-gradients-like calculation on the SAE activations
#     and associated error signals.

#     For each SAE in `base_saes`, it interpolates from clean activations
#     to a counterfactual baseline (which is currently set to 0),
#     measuring how the log-prob of the correct token changes.

#     Args:
#         model (HookedSAETransformer): The main model.
#         base_saes (List[SAE]): List of SAE objects, one per layer/hook.
#         token_list (torch.Tensor): The tokens for which we want to evaluate.
#         clean_sae_cache (Dict[str, torch.Tensor]): Dictionary of "clean" SAE activations.
#         clean_error_cache (Dict[str, torch.Tensor]): Dictionary of the "clean" error term.
#         corr_sae_cache (Dict[str, torch.Tensor]): Dictionary of "counterfactual" (or zero) activations.
#         corr_error_cache (Dict[str, torch.Tensor]): Dictionary of "counterfactual" error terms.
#         labels (torch.Tensor): The correct token IDs for measuring log-prob.
#         save_dir (str): Where to save the resulting effect Tensors.
#         ig_steps (int): Number of integration steps.

#     Returns:
#         None. (Effects are saved to disk.)
#     """
#     if not os.path.exists(save_dir) and save_and_use:
#         os.makedirs(save_dir, exist_ok=True)
    
#     results_sae = {}
#     results_err = {}

#     # We'll need a placeholder to store the updated SAEs while running
#     for sae_index, sae in tqdm(enumerate(base_saes)):
#         for inner_sae_index in range(len(base_saes)):
#             base_saes[inner_sae_index].mean_error = clean_error_cache[base_saes[inner_sae_index].cfg.hook_name]
#         if logstats:
#             print(f"[IG] Processing SAE {sae_index} at hook {sae.cfg.hook_name} ...")

#         # We will accumulate partial effect from each interpolation step
#         effects_sae = []
#         effects_err = []

#         # For clarity, define the shape references
#         clean_acts = clean_sae_cache[sae.cfg.hook_name]
#         clean_err = clean_error_cache[sae.cfg.hook_name]

#         # We'll integrate from "clean" to "corr" (which is zero in your example),
#         # ratio=0 => 100% clean, ratio=1 => 100% corr
#         for step in range(ig_steps):
#             ratio = step / float(ig_steps)
#             # Interpolate
#             interpolation_acts = (clean_acts * (1 - ratio) + corr_sae_cache[sae.cfg.hook_name] * ratio).requires_grad_(True)
#             interpolation_acts.retain_grad()

#             interpolation_err = (clean_err * (1 - ratio) + corr_error_cache[sae.cfg.hook_name] * ratio).requires_grad_(True)
#             interpolation_err.retain_grad()

#             # Replace the mean error for the current SAE with the interpolated error
#             base_saes[sae_index].mean_error = interpolation_err

#             # We run the model with these fake_activations. 
#             interpolated_out, _ = run_sae_hook_fn(
#                 model,
#                 base_saes,
#                 token_list,
#                 calc_error=False,
#                 use_error=False,
#                 fake_activations=(sae.cfg.hook_layer, interpolation_acts),
#                 use_mean_error=True
#             )

#             # Evaluate log-prob for the correct label
#             # Your code snippet suggests shape: [batch, seq, vocab]
#             answer_logits = interpolated_out[..., -1, :]
#             answer_logprobs = F.softmax(answer_logits, dim=-1)
#             # We sum or average across the batch dimension (assumed batch=1 in example).
#             clean_logprobs = answer_logprobs[..., labels[-1]]
#             metric = torch.sum(clean_logprobs)
#             if logstats:
#                 print(f"  [Step={step}/{ig_steps}] ratio={ratio}, metric={metric.item():.4f}")

#             # Backprop
#             metric.backward()

#             # zero attribution formula
#             counterfactual_delta_sae = -clean_acts
#             counterfactual_delta_err = -clean_err

#             effect_sae = (interpolation_acts.grad * counterfactual_delta_sae).mean(dim=0).detach()
#             effect_err = (interpolation_err.grad * counterfactual_delta_err).mean(dim=0).detach()

#             effects_sae.append(effect_sae.cpu())
#             effects_err.append(effect_err.cpu())

#             # Clear out grads from model and SAEs to avoid accumulation
#             clear_memory(base_saes, model)

#         # Average over steps
#         effects_sae = torch.stack(effects_sae)
#         effects_err = torch.stack(effects_err)
#         final_effect_sae = effects_sae.mean(dim=0)
#         final_effect_err = effects_err.mean(dim=0)

#         if save_and_use:
#             sae_effect_path = os.path.join(save_dir, f"sae_effect_{sae_index}.pt")
#             err_effect_path = os.path.join(save_dir, f"err_effect_{sae_index}.pt")
#             torch.save(final_effect_sae, sae_effect_path)
#             torch.save(final_effect_err, err_effect_path)
#             print(f"  => Saved SAE effect to {sae_effect_path}")
#             print(f"  => Saved Error effect to {err_effect_path}")
#         else:
#             results_sae[sae.cfg.hook_name] = final_effect_sae
#             results_err[sae.cfg.hook_name] = final_effect_err

#     if not save_and_use:
#         return results_sae, results_err
#     else:
#         return None, None

# def get_saes_cache(saes):
#     """
#     Get the cache of SAEs
#     """
#     clean_sae_cache = {}
#     clean_error_cache = {}
#     for sae in saes:
#         clean_sae_cache[sae.cfg.hook_name] = sae.feature_acts
#         clean_error_cache[sae.cfg.hook_name] = sae.error_term

#     corr_sae_cache = {}
#     corr_error_cache = {}
#     for sae in saes:
#         corr_sae_cache[sae.cfg.hook_name] = torch.zeros_like(clean_sae_cache[sae.cfg.hook_name])
#         corr_error_cache[sae.cfg.hook_name] = torch.zeros_like(clean_error_cache[sae.cfg.hook_name])

#     return clean_sae_cache, clean_error_cache, corr_sae_cache, corr_error_cache

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
    if feature_selection not in {"max", "sum"}:
        raise ValueError("feature_selection must be 'max' or 'sum'.")

    important_feats: Dict[str, List[int]] = {}
    for sae in base_saes:
        hname   = sae.cfg.hook_name
        effects = res_sae_effects[hname]                       # [B, L, S]
        effects_flat = effects.reshape(-1, effects.shape[-1])  # [B·L, S]

        scores = (
            effects_flat.abs().max(0).values
            if feature_selection == "max"
            else effects_flat.abs().sum(0)
        )
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

    if feature_selection not in {"max", "sum"}:
        raise ValueError("feature_selection must be 'max' or 'sum'.")

    important_features: Dict[str, List[int]] = {}
    for sae in base_saes:
        hname = sae.cfg.hook_name
        effects = res_sae_effects[hname]  # [L, S]
        if feature_selection == "max":
            scores = effects.abs().max(dim=0).values  # per‑latent
        else:
            scores = effects.abs().sum(dim=0)
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

# %%



model.reset_hooks(including_permanent=True)
with torch.no_grad():
    logits_BLV, saes =run_with_saes(model, saes, inter_toks_BL, calc_error=True, use_error=True, cache_sae_activations=True)

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

# %%

hook_names = [sae.cfg.hook_name for sae in saes]

# NEW: Compute edges if requested
edges = None
compute_edges = True
edge_method = "finite_differences"
max_edge_features = 50
edge_includes_loss_grad = True
edge_feature_selection = "max"

if compute_edges and res_sae_effects:
    print("Computing edge attributions...")
    if edge_method == "finite_differences":
        edges = _finite_differences_edge_attr(
            model=model,
            base_saes=saes,
            token_list=inter_toks_BL,
            res_sae_effects=res_sae_effects,
            clean_sae_cache=clean_sae_cache,
            clean_error_cache=clean_error_cache,
            labels=torch.tensor([label]).to(device),
            device=device,
            max_features_per_layer=max_edge_features,
            edge_includes_loss_grad=edge_includes_loss_grad,
            feature_selection=edge_feature_selection,
            logstats=True
        )
    elif edge_method == "jvp":
            edges = _jvp_edge_attr(
            model=model,
            base_saes=saes,
            token_list=inter_toks_BL,
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







# %%
# if __name__ == "__main__":
#     device = "cuda"
#     model_name = "gemma-2-2b"
#     print(model_name)
#     model = load_model(model_name, device=device, use_custom_cache=False, dtype=torch.bfloat16)

#     layers = list(range(model.cfg.n_layers))
#     saes = load_pretrained_saes(layers=layers, release="gemma-scope-2b-pt-mlp-canonical", width="16k", device=device, canon=True)

#     changable_toks = torch.tensor([1917]).to(device)
#     label = 1917
#     discover_circuit(model, saes, changable_toks, label, device)

# %%


