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

from functools import partial
from typing import List, Optional, Dict, Any, Callable


# %%



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
    no_detach: bool = False,
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
            if no_detach:
                sae.feature_acts = feature_acts.clone()
            else:
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
            if no_detach:
                sae.feature_acts = feature_acts.clone()
            else:
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

# EDGE ATTRIBUTION

# ─────────────────────────────────────────────────────────────────────────────
# JVP-based edge attribution helpers (B,L,S aware)
# ─────────────────────────────────────────────────────────────────────────────
def _jvp_edge_attr(
    model,
    base_saes,
    token_list,
    circuit_entries,
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
):
    """
    Compute layer-to-layer edge attribution with a *forward-mode*
    Jacobian-vector product (JVP).

    Args:
        circuit_entries: List of (layer_idx, token_idx, latent_idx, effect_val) tuples

    Shapes
    -------
    Activations are [B, L, S] throughout.

    Returns
    -------
    Dict[str, Dict[str, torch.Tensor]]
        Sparse COO edge tensors keyed upstream_hook → downstream_hook
        with shape  [S_down , S_up].
    """
    # ── 1. Extract important features from circuit entries ──────────────────
    important_feats: Dict[str, List[int]] = {}
    for sae in base_saes:
        important_feats[sae.cfg.hook_name] = []
    
    # Group circuit entries by layer and collect unique latent indices
    layer_to_latents = {}
    for layer_idx, token_idx, latent_idx, effect_val in circuit_entries:
        if layer_idx not in layer_to_latents:
            layer_to_latents[layer_idx] = set()
        layer_to_latents[layer_idx].add(latent_idx)
    
    # Convert to lists and limit by max_features_per_layer
    for layer_idx, latent_set in layer_to_latents.items():
        if layer_idx < len(base_saes):
            hname = base_saes[layer_idx].cfg.hook_name
            latent_list = list(latent_set) #[:max_features_per_layer]
            important_feats[hname] = latent_list
            if logstats:
                print(f"[edge-attr-jvp] {hname}: using {len(latent_list)} features from circuit entries")

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

        edge = _compute_jvp_edge_v2(
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

# def _compute_jvp_edge_v2(
# ─────────────────────────────────────────────────────────────────────────────
# Reverse-mode (VJP) edge attribution  ─ replaces old forward-mode JVP helper
# ─────────────────────────────────────────────────────────────────────────────
def _compute_jvp_edge_v2(        # ← name kept so callers don’t change
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
    Reverse-mode **VJP** implementation of layer-to-layer edge attribution.

    For every pair (down_idx, up_idx) we accumulate

        Σ_{b,t}  ∂ y[b,t,down_idx] / ∂ x[b,t,up_idx]
        (or the gradient-weighted version if `edge_includes_loss_grad=True`).

    Compared with the old forward-mode JVP version this is:

      • robust (all ops in PyTorch have a VJP rule)  
      • slightly heavier on memory, but still much faster than
        finite-difference or zero-ablation baselines.
    """
    if logstats:
        print("[edge-attr-vjp] running reverse-mode edge attribution")

    up_hook,   down_hook  = upstream_sae.cfg.hook_name, downstream_sae.cfg.hook_name
    up_feats,  down_feats = upstream_features, downstream_features
    if not up_feats or not down_feats:
        if logstats:
            print(f"[edge-attr-vjp] skip {up_hook}->{down_hook} (no feats)")
        return None

    # ----------------------------------------------------------------------
    # 1. Baseline upstream activation that we will differentiate *through*
    # ----------------------------------------------------------------------
    up_base = (
        clean_sae_cache[up_hook]
        .detach()
        .clone()
        .to(device)
        .requires_grad_()                 # crucial for reverse-mode
    )

    # ----------------------------------------------------------------------
    # 2. Helper: forward pass that returns downstream SAE latents *attached*
    # ----------------------------------------------------------------------
    def _forward_fn() -> torch.Tensor:    # returns [B, L, S_down]
        _, saes_out = run_with_saes(
            model,
            base_saes,
            token_list,
            calc_error=False,
            use_error=False,
            fake_activations=(upstream_sae.cfg.hook_layer, up_base),
            use_mean_error=use_mean_error,
            cache_sae_activations=True,   # we need the graph intact
            no_detach=True,
        )
        feats = saes_out[downstream_sae.cfg.hook_layer].feature_acts
        if not feats.requires_grad:
            raise RuntimeError(
                "[edge-attr-vjp] downstream activations are detached; "
                "remove `.detach()` inside your SAE hook or clone with "
                "`.requires_grad_()` earlier in the graph."
            )
        return feats

    # ----------------------------------------------------------------------
    # 3. Single forward pass (re-used for every downstream feature)
    # ----------------------------------------------------------------------
    down_base = _forward_fn()             # [B,L,S_down]
    down_grad = (
        res_sae_effects[down_hook].to(device)
        if edge_includes_loss_grad else None
    )

    # Container: (down_idx, up_idx) → list[val]
    bucket: Dict[Tuple[int, int], List[torch.Tensor]] = {}

    # ----------------------------------------------------------------------
    # 4. Loop over downstream features (rows of the Jacobian)
    # ----------------------------------------------------------------------
    for d_idx in down_feats:
        # Select the scalar we will back-prop; optionally weight by loss grad
        scalar_field = down_base[..., d_idx]
        if down_grad is not None:
            scalar_field = scalar_field * down_grad[..., d_idx]
        scalar = scalar_field.sum()

        # Jᵀ ▽  – gradient w.r.t. *entire* upstream latent tensor
        grad_tensor = torch.autograd.grad(
            scalar,
            up_base,
            retain_graph=True,   # keep graph for next d_idx
            create_graph=False,  # we only need first-order grads
        )[0]                     # shape [B,L,S_up]

        # Accumulate entries we care about
        for u_idx in up_feats:
            val = grad_tensor[..., u_idx].sum()  # Σ_{b,t}
            if val.abs() < 1e-6:                 # keep/raise threshold as needed
                continue
            bucket.setdefault((d_idx, u_idx), []).append(val.detach().cpu())

        if logstats and (d_idx == down_feats[0] or d_idx % 10 == 0):
            print(f"[edge-attr-vjp] processed downstream idx {d_idx}")

    # ----------------------------------------------------------------------
    # 5. Assemble sparse COO tensor
    # ----------------------------------------------------------------------
    if not bucket:
        return None

    idxs, vals = zip(
        *[((d, u), torch.stack(v).mean()) for (d, u), v in bucket.items()]
    )
    idx_mat = torch.tensor(list(zip(*idxs)), dtype=torch.long)  # [2, N]
    val_mat = torch.stack(list(vals))                           # [N]

    edge_tensor = torch.sparse_coo_tensor(
        idx_mat,
        val_mat,
        size=(len(down_feats), len(up_feats)),
    ).coalesce()

    if logstats:
        nnz = edge_tensor._nnz()
        print(f"[edge-attr-vjp] finished – {nnz} non-zero entries")

    return edge_tensor


def _finite_differences_edge_attr(
    model,
    base_saes,
    token_list,
    circuit_entries,
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
    zero_ablation: bool = False,
):
    """Compute edge attribution using finite differences or zero ablation.
    
    Args:
        circuit_entries: List of (layer_idx, token_idx, latent_idx, effect_val) tuples
        zero_ablation: If True, sets activations to 0 instead of adding perturbations.
                      When True, fd_steps parameter is ignored.
    """

    # Extract important features from circuit entries
    important_features: Dict[str, List[int]] = {}
    for sae in base_saes:
        important_features[sae.cfg.hook_name] = []
    
    # Group circuit entries by layer and collect unique latent indices
    layer_to_latents = {}
    for layer_idx, token_idx, latent_idx, effect_val in circuit_entries:
        if layer_idx not in layer_to_latents:
            layer_to_latents[layer_idx] = set()
        layer_to_latents[layer_idx].add(latent_idx)
    
    # Convert to lists and limit by max_features_per_layer
    for layer_idx, latent_set in layer_to_latents.items():
        if layer_idx < len(base_saes):
            hname = base_saes[layer_idx].cfg.hook_name
            latent_list = list(latent_set)[:max_features_per_layer]
            important_features[hname] = latent_list
            if logstats:
                print(f"[edge‑attr-fd] {hname}: using {len(latent_list)} features from circuit entries")

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
            zero_ablation=zero_ablation,
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
    zero_ablation: bool = False,
):
    """Finite‑step perturbation implementation.

    Args:
        zero_ablation: If True, sets activations to 0 instead of adding α.
                      When True, steps parameter is ignored and only one pass is made.
    """
    method = "zero ablation" if zero_ablation else "finite differences"
    if logstats:
        print(f"[edge‑attr-fd] running {method} edge attribution")

    up_hook, down_hook = upstream_sae.cfg.hook_name, downstream_sae.cfg.hook_name
    up_clean = clean_sae_cache[up_hook].detach().to(device)  # [L,S]
    down_clean = clean_sae_cache[down_hook].detach().to(device)
    down_grad = res_sae_effects[down_hook].to(device) if edge_includes_loss_grad else None

    # Collect contributions across `steps` different α magnitudes, then average.
    bucket: Dict[Tuple[int, int], List[torch.Tensor]] = {}

    if zero_ablation:
        # Zero ablation: single pass, set features to 0
        for sae in base_saes:
            sae.mean_error = clean_error_cache[sae.cfg.hook_name].detach()
            sae.feature_acts = clean_sae_cache[sae.cfg.hook_name].detach().to(device)

        for up_idx in upstream_features:
            pert = up_clean.clone()
            pert[..., up_idx] = 0.0

            _, updated_saes = run_with_saes(
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
            delta = down_pert - down_clean  # [L,S] - no division by α

            for down_idx in downstream_features:
                val = torch.sum(
                    (down_grad[..., down_idx] if down_grad is not None else 1.0)
                    * delta[..., down_idx]
                )
                if val.abs() < 1e-6:
                    continue
                bucket.setdefault((down_idx, up_idx), []).append(val.detach().cpu())

            clear_memory(base_saes, model)
    else:
        # Regular finite differences with multiple steps
        for s in range(steps):
            α = 0.01 * (s + 1)
            # reset SAE caches each outer loop
            for sae in base_saes:
                sae.mean_error = clean_error_cache[sae.cfg.hook_name].detach()
                sae.feature_acts = clean_sae_cache[sae.cfg.hook_name].detach().to(device)

            for up_idx in upstream_features:
                pert = up_clean.clone()
                pert[..., up_idx] += α

                _, updated_saes = run_with_saes(
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
    """
    Discover circuit with edge attribution support.
    
    Args:
        edge_method: Method for computing edge attribution
            - "jvp": Jacobian-vector product method
            - "finite_differences": Finite differences with perturbations
            - "zero_ablation": Zero ablation (sets activations to 0)
        edge_feature_selection: Feature selection strategy
            - "max": Maximum absolute effect across positions
            - "sum": Sum of absolute effects across positions  
            - "negative": Most negative effects (for inhibitory features)
        edge_includes_loss_grad: Whether to weight edges by downstream gradients
    """
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
                circuit_entries=entries,
                res_sae_effects=res_sae_effects,
                clean_sae_cache=clean_sae_cache,
                clean_error_cache=clean_error_cache,
                labels=torch.tensor([label]).to(device),
                device=device,
                max_features_per_layer=max_edge_features,
                edge_includes_loss_grad=edge_includes_loss_grad,
                logstats=True, 
                fd_steps=5,
                zero_ablation=False
            )
        elif edge_method == "zero_ablation":
            edges = _finite_differences_edge_attr(
                model=model,
                base_saes=saes,
                token_list=changable_toks,
                circuit_entries=entries,
                res_sae_effects=res_sae_effects,
                clean_sae_cache=clean_sae_cache,
                clean_error_cache=clean_error_cache,
                labels=torch.tensor([label]).to(device),
                device=device,
                max_features_per_layer=max_edge_features,
                edge_includes_loss_grad=edge_includes_loss_grad,
                logstats=True, 
                fd_steps=1,  # ignored for zero ablation
                zero_ablation=True
            )
        elif edge_method == "jvp":
                edges = _jvp_edge_attr(
                model=model,
                base_saes=saes,
                token_list=changable_toks,
                circuit_entries=entries,
                res_sae_effects=res_sae_effects,
                clean_sae_cache=clean_sae_cache,
                clean_error_cache=clean_error_cache,
                labels=torch.tensor([label]).to(device),
                device=device,
                max_features_per_layer=max_edge_features,
                use_mean_error=True,
                logstats=True,
                edge_includes_loss_grad=edge_includes_loss_grad,
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

# Example usage and testing

# Load model and SAEs
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

# %%

# Load data and setup
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

# Generate prompt and tokens
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
print(f"Generated text: {baseline_suffix}")
print(f"Input tokens: {model.to_string(inter_toks_BL[0])}")

# %%

# CELL 1: Run integrated gradients and get circuit rankings
print("Running integrated gradients and computing circuit rankings...")

# Reset hooks and run SAEs to get baseline
model.reset_hooks(including_permanent=True)
with torch.no_grad():
    logits_BLV, saes = run_with_saes(model, saes, inter_toks_BL, calc_error=True, use_error=True, cache_sae_activations=True)

# Get label and clean baseline
label = logits_BLV[0, -1, :].argmax(-1).item()
cleanup_cuda()
probs_BV = F.softmax(logits_BLV[:, -1, :], dim=-1)
clean_probs_baseline = probs_BV[0, label]

print(f"Label: {label}")
print(f"Clean baseline probability: {clean_probs_baseline.item():.4f}")

# Set mean error for SAEs
for ind, sae in enumerate(saes):
    sae.mean_error = sae.error_term.detach()

del logits_BLV
cleanup_cuda()

# Get SAE caches
clean_sae_cache, clean_error_cache, corr_sae_cache, corr_error_cache = get_saes_cache(saes)

# Run Integrated Gradients
print("Running integrated gradients...")
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
    logstats=False,
)

hook_names = [sae.cfg.hook_name for sae in saes]

# Compute K metrics and find circuit entries
print("Computing K metrics...")
K_vals, metrics_negative, metrics_absolute = compute_k_metrics(
    model, saes, res_sae_effects, device, hook_names, inter_toks_BL, label, k_max=k_max, k_step=k_step
)

# Find threshold and get entries
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
    selection_method = "negative"
    print(f"Using negative selection with K={K}")
elif min_k_abs is not None:
    K = min_k_abs
    topk_iter = iter_topk_absolute_effects(
        res_sae_effects=res_sae_effects,
        saes=saes,
        hook_names=hook_names,
        K=K
    )
    selection_method = "absolute"
    print(f"Using absolute selection with K={K}")
else:
    print("No suitable K found - skipping circuit discovery")
    circuit_entries = None
    K = None
    selection_method = None

if K is not None:
    # Get circuit entries
    circuit_entries = list(itertools.islice(topk_iter, K))
    print(f"Found {len(circuit_entries)} circuit entries using {selection_method} selection")
    
    # Show some example entries
    print("Top 10 circuit entries:")
    for i, (layer_idx, token_idx, latent_idx, effect_val) in enumerate(circuit_entries[:10]):
        print(f"  {i+1}. Layer {layer_idx}, Token {token_idx}, Latent {latent_idx}, Effect: {effect_val:.6f}")

# %%

for stuff in circuit_entries:
    if stuff[0] == 0 or stuff[0] == 1:
        print(stuff)

# %%

edges = _jvp_edge_attr(
        model=model,
        base_saes=saes,
        token_list=inter_toks_BL,
        circuit_entries=circuit_entries,
        res_sae_effects=res_sae_effects,
        clean_sae_cache=clean_sae_cache,
        clean_error_cache=clean_error_cache,
        labels=torch.tensor([label]).to(device),
        device=device,
        max_features_per_layer=50,
        use_mean_error=True,
        logstats=True,
        edge_includes_loss_grad=True,
    )


# %%
# Save the full edges variable
import pickle

with open('edges_v2.pkl', 'wb') as f:
    pickle.dump(edges, f)

print("Edges saved to edges_v2.pkl")

# %% Finite difference 

edge_method = "zero_ablation"


edges = _finite_differences_edge_attr(
                model=model,
                base_saes=saes,
                token_list=inter_toks_BL,
                circuit_entries=circuit_entries,
                res_sae_effects=res_sae_effects,
                clean_sae_cache=clean_sae_cache,
                clean_error_cache=clean_error_cache,
                labels=torch.tensor([label]).to(device),
                device=device,
                max_features_per_layer=50,
                edge_includes_loss_grad=True,
                logstats=True, 
                fd_steps=1,  # ignored for zero ablation
                zero_ablation=True
            )








# %%
# CELL 2: Compute edge attributions
print("Computing edge attributions...")

if K is not None and circuit_entries is not None:
    # Test different edge attribution methods
    edge_methods = ["jvp"] #, "finite_differences", "zero_ablation"]
    edge_results = {}
    
    for method in edge_methods:
        print(f"\nComputing {method} edges...")
        
        if method == "finite_differences":
            edges = _finite_differences_edge_attr(
                model=model,
                base_saes=saes,
                token_list=inter_toks_BL,
                circuit_entries=circuit_entries,
                res_sae_effects=res_sae_effects,
                clean_sae_cache=clean_sae_cache,
                clean_error_cache=clean_error_cache,
                labels=torch.tensor([label]).to(device),
                device=device,
                max_features_per_layer=50,
                edge_includes_loss_grad=True,
                logstats=True, 
                fd_steps=5,
                zero_ablation=False
            )
        elif method == "zero_ablation":
            edges = _finite_differences_edge_attr(
                model=model,
                base_saes=saes,
                token_list=inter_toks_BL,
                circuit_entries=circuit_entries,
                res_sae_effects=res_sae_effects,
                clean_sae_cache=clean_sae_cache,
                clean_error_cache=clean_error_cache,
                labels=torch.tensor([label]).to(device),
                device=device,
                max_features_per_layer=50,
                edge_includes_loss_grad=True,
                logstats=True, 
                fd_steps=1,  # ignored for zero ablation
                zero_ablation=True
            )
        elif method == "jvp":
            edges = _jvp_edge_attr(
                model=model,
                base_saes=saes,
                token_list=inter_toks_BL,
                circuit_entries=circuit_entries,
                res_sae_effects=res_sae_effects,
                clean_sae_cache=clean_sae_cache,
                clean_error_cache=clean_error_cache,
                labels=torch.tensor([label]).to(device),
                device=device,
                max_features_per_layer=50,
                use_mean_error=True,
                logstats=True,
                edge_includes_loss_grad=True,
            )
        
        edge_results[method] = edges
        
        # Show edge statistics
        if edges:
            total_edges = sum(len(v) for v in edges.values())
            print(f"  {method}: {total_edges} edge tensors")
            
            for upstream_hook, downstream_edges in edges.items():
                for downstream_hook, edge_tensor in downstream_edges.items():
                    print(f"    {upstream_hook} -> {downstream_hook}: {edge_tensor.shape} ({edge_tensor._nnz()} non-zero)")
                    # Show some statistics
                    if edge_tensor._nnz() > 0:
                        dense_edges = edge_tensor.to_dense()
                        print(f"      Range: [{dense_edges.min().item():.4f}, {dense_edges.max().item():.4f}]")
                        print(f"      Mean magnitude: {dense_edges.abs().mean().item():.4f}")
        else:
            print(f"  {method}: No edges computed")
    
    print(f"\nEdge attribution complete! Computed edges using {len(edge_results)} methods.")
    
    # Store results in variables for easy access
    edges_jvp = edge_results.get("jvp", None)
    edges_fd = edge_results.get("finite_differences", None) 
    edges_zero = edge_results.get("zero_ablation", None)
    
else:
    print("Cannot compute edges - no circuit entries found")
    edges_jvp = None
    edges_fd = None
    edges_zero = None

# %%

# Test JVP edge attribution
print("Testing JVP edge attribution...")
entries_jvp, edges_jvp = discover_circuit_edge_attr(
    model=model,
    saes=saes,
    changable_toks=inter_toks_BL,
    device=device,
    ig_steps=ig_steps,
    k_max=k_max,
    k_step=k_step,
    k_thres=k_thres,
    compute_edges=True,
    edge_method="jvp",
    max_edge_features=50,
    edge_includes_loss_grad=True,
    edge_feature_selection="max"
)

print(f"JVP Results: {len(entries_jvp) if entries_jvp else 0} entries")
if edges_jvp:
    print(f"JVP Edges: {sum(len(v) for v in edges_jvp.values())} edge tensors")

# %%

# Test finite differences edge attribution  
print("Testing finite differences edge attribution...")
entries_fd, edges_fd = discover_circuit_edge_attr(
    model=model,
    saes=saes,
    changable_toks=inter_toks_BL,
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

print(f"Finite Differences Results: {len(entries_fd) if entries_fd else 0} entries")
if edges_fd:
    print(f"Finite Differences Edges: {sum(len(v) for v in edges_fd.values())} edge tensors")

# %%

# Test zero ablation edge attribution
print("Testing zero ablation edge attribution...")
entries_zero, edges_zero = discover_circuit_edge_attr(
    model=model,
    saes=saes,
    changable_toks=inter_toks_BL,
    device=device,
    ig_steps=ig_steps,
    k_max=k_max,
    k_step=k_step,
    k_thres=k_thres,
    compute_edges=True,
    edge_method="zero_ablation",
    max_edge_features=50,
    edge_includes_loss_grad=True,
    edge_feature_selection="max"
)

print(f"Zero Ablation Results: {len(entries_zero) if entries_zero else 0} entries")
if edges_zero:
    print(f"Zero Ablation Edges: {sum(len(v) for v in edges_zero.values())} edge tensors")

# %%

# Test negative feature selection
print("Testing negative feature selection...")
entries_negative, edges_negative = discover_circuit_edge_attr(
    model=model,
    saes=saes,
    changable_toks=inter_toks_BL,
    device=device,
    ig_steps=ig_steps,
    k_max=k_max,
    k_step=k_step,
    k_thres=k_thres,
    compute_edges=True,
    edge_method="jvp",
    max_edge_features=50,
    edge_includes_loss_grad=True,
    edge_feature_selection="negative"
)

print(f"Negative Selection Results: {len(entries_negative) if entries_negative else 0} entries")
if edges_negative:
    print(f"Negative Selection Edges: {sum(len(v) for v in edges_negative.values())} edge tensors")

# %%

# Compare edge attribution methods
print("Comparing edge attribution methods...")

methods = [
    ("JVP", edges_jvp),
    ("Finite Differences", edges_fd), 
    ("Zero Ablation", edges_zero),
    ("Negative Selection", edges_negative)
]

for method_name, edges in methods:
    if edges:
        print(f"\n{method_name}:")
        for upstream_hook, downstream_edges in edges.items():
            for downstream_hook, edge_tensor in downstream_edges.items():
                print(f"  {upstream_hook} -> {downstream_hook}: {edge_tensor.shape} ({edge_tensor._nnz()} non-zero)")
                # Show some statistics
                dense_edges = edge_tensor.to_dense()
                print(f"    Range: [{dense_edges.min().item():.4f}, {dense_edges.max().item():.4f}]")
                print(f"    Mean magnitude: {dense_edges.abs().mean().item():.4f}")
    else:
        print(f"\n{method_name}: No edges computed")

# %%

print("Edge attribution analysis complete!")

