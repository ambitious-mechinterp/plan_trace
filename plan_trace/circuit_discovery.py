"""
Circuit discovery using integrated gradients attribution on SAE latents.

Shape Suffix Definition: 
- B: batch size 
- L: Num of Input Tokens 
- V: vocabulary size
- S: Number of SAE neurons in a layer
"""

import os
import torch
import itertools
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any, Iterable
from .utils import cleanup_cuda, clear_memory
from .hooks import run_with_saes, register_sae_hooks, SAEMasks


def run_integrated_gradients(
    model,
    base_saes: List,
    token_list: torch.Tensor,
    clean_sae_cache: Dict[str, torch.Tensor],
    clean_error_cache: Dict[str, torch.Tensor],
    corr_sae_cache: Dict[str, torch.Tensor],
    corr_error_cache: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    save_dir: str,
    *,
    ig_steps: int = 10,
    save_and_use: bool = True,
    logstats: bool = False,
) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[Dict[str, torch.Tensor]]]:
    """
    Perform integrated-gradients attribution on SAE latents & errors.

    Steps:
    1. For each SAE, interpolate from "clean" to "corrupted" activations.
    2. Hook in interpolated acts, backprop a metric, gather grads.
    3. Compute per-latent effect via zero-attribution rule.
    4. Average across `ig_steps` and either save or return results.

    Args:
        model: The language model
        base_saes: List of SAE objects
        token_list: Input tokens [B, L] 
        clean_sae_cache: Dict mapping hook_name -> clean SAE activations [L, S]
        clean_error_cache: Dict mapping hook_name -> clean error terms [L, D_model]
        corr_sae_cache: Dict mapping hook_name -> corrupted SAE activations [L, S] (typically zeros)
        corr_error_cache: Dict mapping hook_name -> corrupted error terms [L, D_model] (typically zeros)
        labels: Target token indices [B] for metric computation
        save_dir: Where to write `.pt` files if `save_and_use`
        ig_steps: Number of integration points
        save_and_use: If True, write effects to disk; else return them
        logstats: If True, print per-step diagnostics

    Returns:
        Tuple of (sae_effects, err_effects) dicts if save_and_use=False, else (None, None)
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
            interpolated_out, _ = run_with_saes(
                model,
                base_saes,
                token_list,
                calc_error=False,
                use_error=False,
                fake_activations=(sae.cfg.hook_layer, interpolation_acts),
                use_mean_error=True
            )

            # Evaluate log-prob for the correct label
            # Shape: [B, L, V]
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


def get_saes_cache(saes: List) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Extract cached activations and error terms from SAEs, plus create zero-filled corrupted versions.
    
    Args:
        saes: List of SAE objects with cached feature_acts and error_term
        
    Returns:
        Tuple of (clean_sae_cache, clean_error_cache, corr_sae_cache, corr_error_cache)
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


def iter_topk_effects(
    res_sae_effects: Dict[str, torch.Tensor],
    saes: List[Any],
    K: int,
    *,
    mode: str = "neg"
) -> Iterable[Tuple[int, int, int, float]]:
    """
    Yield (layer, token, latent, value) for the top‑K chosen effects.
    
    Args:
        res_sae_effects: Dict mapping hook_name -> effect tensor [L, S]
        saes: List of SAE objects
        K: Number of top effects to yield
        mode: "neg" for most negative effects, "abs" for largest absolute effects
        
    Yields:
        Tuples of (layer_idx, token_idx, latent_idx, effect_value)
    """
    assert mode in {"neg", "abs"}
    parts = []
    for l, sae in enumerate(saes):
        eff = res_sae_effects[sae.cfg.hook_name]              # [L, S]
        L, S = eff.shape
        parts.append(
            torch.stack(
                [
                    torch.full((L * S,), l, dtype=torch.long),           # layer idx
                    torch.arange(L).repeat_interleave(S),                # token idx
                    torch.arange(S).repeat(L),                           # latent idx
                    eff.view(-1),                                        # effect
                ],
                dim=0,
            )
        )
    big = torch.cat(parts, dim=1)                                     # [4, N]
    vals = -big[3] if mode == "neg" else torch.abs(big[3])
    top_v, top_i = torch.topk(vals, K)

    for v, idx in zip(top_v, top_i):
        yield (
            int(big[0, idx]),
            int(big[1, idx]),
            int(big[2, idx]),
            float(v),
        )


def _metric_for_topk(
    *,
    model,
    saes: List,
    res_sae_effects: Dict[str, torch.Tensor],
    all_effects: torch.Tensor,
    tokens: torch.Tensor,
    label: int,
    K: int,
    mode: str,
    device: str,
) -> float:
    """
    Return summed prob for the target label after masking top‑K effects.
    
    Args:
        model: The language model
        saes: List of SAE objects
        res_sae_effects: Dict mapping hook_name -> effect tensor [L, S]
        all_effects: Pre-computed tensor [4, N] of (layer, token, latent, effect)
        tokens: Input tokens [B, L]
        label: Target token ID
        K: Number of top effects to mask
        mode: "neg" or "abs" for selection criteria
        device: Device for computation
        
    Returns:
        Float probability for target label after masking
    """
    vals = -all_effects[3] if mode == "neg" else torch.abs(all_effects[3])
    _, idx = torch.topk(vals, K)

    sel_layer = all_effects[0, idx].long()
    sel_tok = all_effects[1, idx].long()
    sel_lat = all_effects[2, idx].long()

    # build one binary mask tensor per SAE, shape [L, S]
    masks = []
    for l, sae in enumerate(saes):
        base = torch.zeros_like(res_sae_effects[sae.cfg.hook_name], dtype=torch.float)
        pick = sel_layer == l
        if pick.any():
            base[sel_tok[pick], sel_lat[pick]] = 1.0
        masks.append(base.to(device))

    circuit_mask = SAEMasks(
        hook_points=[s.cfg.hook_name for s in saes],
        masks=masks,
    ).to(device)

    hooks = register_sae_hooks(
        model,
        saes,
        tokens,
        circuit_mask=circuit_mask,
        use_mean_error=True,
    )
    logits = model.run_with_hooks(tokens, return_type="logits", fwd_hooks=hooks)
    probs = F.softmax(logits[..., -1, :], dim=-1)
    return probs[..., torch.arange(probs.shape[-2]), label].sum().item()


def compute_k_metrics(
    model,
    saes: List,
    res_sae_effects: Dict[str, torch.Tensor],
    device: str,
    tokens: torch.Tensor,
    label: int,
    *,
    k_max: int = 7001,
    k_step: int = 500,
) -> Tuple[List[int], List[float], List[float]]:
    """
    Sweep K=1…k_max by k_step, measuring model performance when masking
    the top-K negative or absolute effects. Prints each result.

    Args:
        model: The language model
        saes: List of SAE objects
        res_sae_effects: Dict mapping hook_name -> effect tensor [L, S]
        device: Device for computation
        tokens: Input tokens [B, L]
        label: Target token ID
        k_max: Maximum K to test
        k_step: Step size between successive K values

    Returns:
        Tuple of (K_vals, neg_metrics, abs_metrics)
    """
    # pre‑assemble big [4, N] tensor ONCE
    parts = []
    for l, sae in enumerate(saes):
        eff = res_sae_effects[sae.cfg.hook_name]      # [L, S]
        L, S = eff.shape
        parts.append(
            torch.stack(
                [
                    torch.full((L * S,), l, dtype=torch.long),
                    torch.arange(L).repeat_interleave(S),
                    torch.arange(S).repeat(L),
                    eff.view(-1),
                ],
                dim=0,
            )
        )
    big = torch.cat(parts, dim=1).to(device)          # [4, N]

    K_vals, negs, abss = [], [], []
    for K in range(1, k_max, k_step):
        metric_neg = _metric_for_topk(
            model=model,
            saes=saes,
            res_sae_effects=res_sae_effects,
            all_effects=big,
            tokens=tokens,
            label=label,
            K=K,
            mode="neg",
            device=device,
        )
        metric_abs = _metric_for_topk(
            model=model,
            saes=saes,
            res_sae_effects=res_sae_effects,
            all_effects=big,
            tokens=tokens,
            label=label,
            K=K,
            mode="abs",
            device=device,
        )
        K_vals.append(K)
        negs.append(metric_neg)
        abss.append(metric_abs)
        print(f"K={K:<5} | neg={metric_neg:.4f} | abs={metric_abs:.4f}")
        model.reset_hooks(including_permanent=True)

    return K_vals, negs, abss


def find_min_k_for_threshold(
    K_vals: List[int], 
    metrics_negative: List[float], 
    metrics_absolute: List[float], 
    clean_probs_baseline: torch.Tensor, 
    threshold: float = 0.8
) -> Tuple[Optional[int], Optional[int]]:
    """
    Scan the two metric lists to find the smallest K that exceeds
    `threshold × clean_probs_baseline` for negative and absolute modes.

    Args:
        K_vals: List of K values tested
        metrics_negative: List of performance metrics for negative mode
        metrics_absolute: List of performance metrics for absolute mode
        clean_probs_baseline: Baseline performance value
        threshold: Fraction of baseline to reach

    Returns:
        Tuple of (min_k_negative, min_k_absolute), either int or None
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


def discover_circuit(
    model, 
    saes: List, 
    inter_toks_BL: torch.Tensor, 
    device: str, 
    ig_steps: int = 10, 
    k_max: int = 7001, 
    k_step: int = 500, 
    k_thres: float = 0.6
) -> Optional[List[Tuple[int, int, int, float]]]:
    """
    Orchestrate a full circuit discovery pipeline:
    1. Run model with SAEs to get clean activations + error.
    2. Integrated gradients to get per-latent effects.
    3. Compute k-sweep metrics and find minimum K threshold.
    4. Return the top-K hits via `iter_topk_effects`.

    Args:
        model: The language model
        saes: List of SAE objects
        inter_toks_BL: Input tokens [B, L]
        device: Device for computation
        ig_steps: Number of integration steps for IG
        k_max: Maximum K to test in sweep
        k_step: Step size for K sweep
        k_thres: Fraction of baseline performance to aim for

    Returns:
        List of (layer, token, latent, value) tuples, or None if none exceed threshold
    """
    model.reset_hooks(including_permanent=True)
    with torch.no_grad():
        inter_logits_BLV, saes = run_with_saes(model, saes, inter_toks_BL, calc_error=True, use_error=True, cache_sae_activations=True)
    
    inter_label = inter_logits_BLV[0, -1, :].argmax(-1).item()
    clean_probs_baseline = F.softmax(inter_logits_BLV[0, -1, :], dim=-1)[inter_label]

    for sae in saes:
        sae.mean_error = sae.error_term.detach()

    del inter_logits_BLV
    cleanup_cuda()

    # Get Cache
    clean_sae_cache, clean_error_cache, corr_sae_cache, corr_error_cache = get_saes_cache(saes)

    # Run Integrated Gradients & Save Results
    res_sae_effects, _ = run_integrated_gradients(
        model=model,
        base_saes=saes,
        token_list=inter_toks_BL,
        clean_sae_cache=clean_sae_cache,
        clean_error_cache=clean_error_cache,
        corr_sae_cache=corr_sae_cache,
        corr_error_cache=corr_error_cache,
        labels=torch.tensor([inter_label]).to(device),
        ig_steps=ig_steps,
        save_dir="", 
        save_and_use=False,
        logstats=False,
    ) 
    
    K_vals, metrics_negative, metrics_absolute = compute_k_metrics(
        model, saes, res_sae_effects, device,
        inter_toks_BL, inter_label,
        k_max=k_max, k_step=k_step
    )

    min_k_neg, min_k_abs = find_min_k_for_threshold(
        K_vals, metrics_negative, metrics_absolute, clean_probs_baseline, threshold=k_thres
    )

    if min_k_neg is not None or min_k_abs is not None:
        # Choose the smallest non-None value
        if min_k_neg is not None and min_k_abs is not None:
            K = min(min_k_neg, min_k_abs)
            mode = "neg" if K == min_k_neg else "abs"
        elif min_k_neg is not None:
            K = min_k_neg
            mode = "neg"
        else:
            K = min_k_abs
            mode = "abs"
        topk_iter = iter_topk_effects(
            res_sae_effects=res_sae_effects,
            saes=saes,
            K=K,
            mode=mode,
        )
    else:
        return None
    
    # we only need to store entries
    entries = list(itertools.islice(topk_iter, K))  # preserves order

    return entries 