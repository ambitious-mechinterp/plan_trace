# %% 
"""
This is a self contained script for tracing plans. 
GOALS: 
for a given prompt and token index,
- STEP1: discover circuit with >60% perf recovery
- STEP2: cluster based on decoding directions logit lens 
- STEP3: test steering effect of clusters
- STEP4: test effect of each latent / position in cluster

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
"""
# Imports 
import os
import gc
import re 
import json
import torch
import itertools
import pandas as pd
from torch import nn
from tqdm import tqdm
from functools import partial
from dotenv import load_dotenv
import torch.nn.functional as F
from huggingface_hub import login
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from sae_lens import SAE, HookedSAETransformer
from typing import List, Optional, Dict, Any, Sequence, Tuple, Iterable, Union, Set
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory

# %% ALL HELPER FUNCTIONS

# Basic utils 
def load_model(model_name: str, device: str, use_custom_cache: bool = False, dtype: torch.dtype = torch.bfloat16, weights_dir: str = None) -> HookedSAETransformer:
    """
    Load a pretrained HookedSAETransformer, optionally redirecting HuggingFace/Torch cache.

    Args:
        model_name (str): HF identifier of the model to load.
        device (str): Compute device, e.g. "cuda" or "cpu".
        use_custom_cache (bool): If True, force both TORCH_HOME and HF_HOME to a local directory.
        dtype (torch.dtype): Numeric precision for model weights.
        weights_dir (str | None): Alternate cache directory (if exists).

    Returns:
        HookedSAETransformer: The model with gradients disabled.
    """
    if use_custom_cache:
        WEIGHTS_DIR = "/project/pi_jensen_umass_edu/jnainani_umass_edu/mechinterp_cache"
        os.environ["TORCH_HOME"] = WEIGHTS_DIR
        os.environ["HF_HOME"] = WEIGHTS_DIR
    elif weights_dir and os.path.exists(weights_dir):
        WEIGHTS_DIR = weights_dir
        os.environ["TORCH_HOME"] = WEIGHTS_DIR
        os.environ["HF_HOME"] = WEIGHTS_DIR
    else:
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        login(token=hf_token)
        WEIGHTS_DIR = None  # Use default cache directory
    
    model = HookedSAETransformer.from_pretrained(
        model_name,
        device=device,
        cache_dir=WEIGHTS_DIR if use_custom_cache else None,
        dtype=dtype
    )

    # Disable gradient updates
    for param in model.parameters():
        param.requires_grad_(False)

    return model


def cleanup_cuda():
    """Run Python GC and empty the CUDA cache to free GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()

def clear_memory(saes, model, mask_bool=False):
    """
    Zeroes out all gradients on SAEs (and their masks if requested) and on the main model.

    Args:
        saes (Iterable[SAE]): List of SAE modules whose gradients to clear.
        model (nn.Module): The language model to clear gradients on.
        mask_bool (bool): If True, also clear grads on any mask modules attached to SAEs.
    """
    for sae in saes:
        for param in sae.parameters():
            if param.grad is not None:
                param.grad = None
        if mask_bool and hasattr(sae, 'mask'):
            for param in sae.mask.parameters():
                if param.grad is not None:
                    param.grad = None
    for param in model.parameters():
        if param.grad is not None:
            param.grad = None

def get_pretrained_saes_ids() -> pd.DataFrame:
    """
    Query the SAE directory and return a cleaned DataFrame of available IDs and metadata.

    Returns:
        DataFrame: One row per SAE ID, with columns filtered for clarity.
    """
    df = pd.DataFrame.from_records(
        {k: v.__dict__ for k, v in get_pretrained_saes_directory().items()}
    ).T
    
    # Drop unnecessary columns for cleaner output
    df.drop(
        columns=[
            "expected_var_explained",
            "expected_l0",
            "config_overrides",
            "conversion_func",
        ],
        inplace=True,
    )
    
    return df

def load_pretrained_saes(layers: List[int], 
                        release: str,
                        width: str = "16k",
                        device: str = "cuda", 
                        canon: bool = True) -> List:
    """
    For each layer in `layers`, pick the median-l0 SAE variant and load it.

    Args:
        layers (List[int]): Transformer layers to attach SAEs to.
        release (str): The scope/release name for SAEs (e.g. "gemma-scope-2b-pt-mlp-canonical").
        width (str): Width identifier in the SAE directory (default "16k").
        device (str): Target device for the loaded SAEs.
        canon (bool): If True, use the canonical naming convention when matching.

    Returns:
        List[SAE]: One SAE instance per requested layer.
    """
    pretrained_saes_ids = get_pretrained_saes_ids()
    sae_dict = pretrained_saes_ids.saes_map[release]
    base_saes = []
    
    for layer in layers:
        layer_str = f"layer_{layer}"
        width_str = f"width_{width}"
        pattern = r'average_l0_(\d+)'
        
        matching_saes = []
        for key in sae_dict.keys():
            parts = key.split('/')
            val_parts = sae_dict[key].split('/')
            if len(parts) >= 3 and parts[0] == layer_str and parts[1] == width_str:
                match = re.search(pattern, parts[2]) if not canon else re.search(pattern, val_parts[2])
                if match:
                    l0_value = int(match.group(1))
                    matching_saes.append((l0_value, key))
        
        matching_saes.sort(key=lambda x: x[0])
        middle_idx = len(matching_saes) // 2
        _, middle_key = matching_saes[middle_idx]
        sae = SAE.from_pretrained(release=release, sae_id=middle_key, device=device)[0]
        base_saes.append(sae)
    return base_saes

class SAEMasks(nn.Module):
    def __init__(self, hook_points, masks):
        super().__init__()
        self.hook_points = hook_points  # list of strings
        self.masks = masks

    def forward(self, x, sae_hook_point, mean_ablation=None):
        """
        Apply or ablate activations according to the binary masks for each hook point.

        Args:
            x (Tensor): The original activations [seq, latent_dim] or [batch, seq, latent_dim].
            sae_hook_point (str): Identifier for which mask to apply.
            mean_ablation (Tensor | None): If given, use it as the “off” activation instead of zero.

        Returns:
            Tensor: Masked (or mean-ablated) activations.
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

    def print_mask_statistics(self):
        """Log shape, total latents, on-latent count, and avg on-latents/token for each mask."""
        for i, mask in enumerate(self.masks):
            shape = list(mask.shape)
            total_latents = mask.numel()
            total_on = mask.sum().item()  # number of 1's in the mask

            # Average on-latents per token depends on dimensions
            if len(shape) == 1:
                # e.g., shape == [latent_dim]
                avg_on_per_token = total_on  # only one token
            elif len(shape) == 2:
                # e.g., shape == [seq, latent_dim]
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

    def save(self, save_dir, file_name="sae_masks.pt"):
        """
        Serialize hook_points and their masks to a file for later reuse.

        Args:
            save_dir (str): Directory to write into (created if missing).
            file_name (str): Name of the checkpoint file.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, file_name)
        checkpoint = {"hook_points": self.hook_points, "masks": self.masks}
        torch.save(checkpoint, save_path)
        print(f"SAEMasks saved to {save_path}")

    @classmethod
    def load(cls, load_dir, file_name="sae_masks.pt"):
        """
        Load previously saved masks from disk.

        Args:
            load_dir (str): Directory holding the checkpoint.
            file_name (str): Which mask set to load.

        Returns:
            SAEMasks: Instance reconstructed from file.
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

    def get_num_latents(self):
        """Count how many latents across all masks are “on” (mask > 0)."""
        num_latents = 0
        for mask in self.masks:
            num_latents += (mask > 0).sum().item()
        return num_latents

def build_sae_hook_fn(
    sae,
    sequence,
    bos_token_id,
    circuit_mask: Optional[SAEMasks] = None,
    mean_mask=False,
    cache_masked_activations=False,
    cache_sae_activations=False,
    mean_ablate=False,  
    fake_activations=False,  
    calc_error=False,
    use_error=False,
    use_mean_error=False,
):
    """
    Construct a forward hook for an SAE: encode → optional mask/ablation → decode → merge.

    Args:
        sae (SAE): The SAE module whose encode/decode to use.
        sequence (Tensor): Token indices that define padding/BOS for masking.
        bos_token_id (int): ID marking start-of-sequence (excluded from mask).
        circuit_mask (SAEMasks | None): If provided, zero or mean-ablate selected latents.
        mean_mask (bool): If True, use `sae.mean_ablation` as “off” value in mask.
        cache_masked_activations (bool): Store masked acts in `sae.feature_acts`.
        cache_sae_activations (bool): Store raw SAE activations in `sae.feature_acts`.
        mean_ablate (bool): Always ablate to `sae.mean_ablation`.
        fake_activations (bool | tuple): If (layer, acts), replaces acts at that layer.
        calc_error (bool): Compute `sae.error_term = original − updated`.
        use_error (bool): Add error term back into the returned activations.
        use_mean_error (bool): Add stored `sae.mean_error` back into the returned activations.

    Returns:
        Callable: Hook function matching the model’s fwd_hooks API.
    """
    # make the mask for the sequence
    mask = torch.ones_like(sequence, dtype=torch.bool)
    mask[sequence == bos_token_id] = False  # where mask is false, keep original

    def sae_hook(value, hook):
        feature_acts = sae.encode(value)
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

        out = sae.decode(feature_acts)
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

def register_sae_hooks(model, saes, tokens, **hook_kwargs):
    """Return a ready‑to‑use hooks list for model.run_with_hooks()."""
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

def run_with_saes(model, saes, tokens, **hook_kwargs):
    """
    Convenience thin‑wrapper around model.run_with_hooks that uses
    register_sae_hooks() under the hood.
    """
    hooks = register_sae_hooks(model, saes, tokens, **hook_kwargs)
    return model.run_with_hooks(tokens, return_type="logits", fwd_hooks=hooks), saes


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
    *,
    ig_steps=10,
    save_and_use=True,   # <--- new,
    logstats=False,  # <--- new
):
    """
    Perform integrated-gradients attribution on SAE latents & errors.

    Steps:
    1. For each SAE, interpolate from “clean” to “corrupted” activations.
    2. Hook in interpolated acts, backprop a metric, gather grads.
    3. Compute per-latent effect via zero-attribution rule.
    4. Average across `ig_steps` and either save or return results.

    Args:
        model, base_saes, token_list: Core inputs.
        clean_sae_cache, clean_error_cache: Dicts of original acts & errors.
        corr_sae_cache, corr_error_cache: Zero tensors for corrupted state.
        labels (Tensor): Target token indices for metric.
        save_dir (str): Where to write `.pt` files if `save_and_use`.
        ig_steps (int): Number of integration points.
        save_and_use (bool): If True, write effects to disk; else return them.
        logstats (bool): If True, print per-step diagnostics.

    Returns:
        None|(dict, dict): Only if save_and_use=False, returns (sae_effects, err_effects).
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

def iter_topk_effects(
    res_sae_effects: Dict[str, torch.Tensor],
    saes:            List[Any],
    K:               int,
    *,
    mode: str = "neg"            # "neg" | "abs"
):
    """Yield (layer, token, latent, value) for the top‑K chosen effects."""
    assert mode in {"neg", "abs"}
    parts = []
    for l, sae in enumerate(saes):
        eff  = res_sae_effects[sae.cfg.hook_name]              # [L, S]
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
    big   = torch.cat(parts, dim=1)                                     # [4, N]
    vals  = -big[3] if mode == "neg" else torch.abs(big[3])
    top_v, top_i = torch.topk(vals, K)

    for v, idx in zip(top_v, top_i):
        yield (
            int(big[0, idx]),
            int(big[1, idx]),
            int(big[2, idx]),
            float(v),
        )

def _metric_for_topk(
    *,                                    # keyword‑only for clarity
    model,
    saes,
    res_sae_effects,
    all_effects,                          # [4, N] tensor
    tokens,
    label,
    K,
    mode,
    device,
):
    """Return summed prob for the target label after masking top‑K effects."""
    vals = -all_effects[3] if mode == "neg" else torch.abs(all_effects[3])
    _, idx = torch.topk(vals, K)

    sel_layer = all_effects[0, idx].long()
    sel_tok   = all_effects[1, idx].long()
    sel_lat   = all_effects[2, idx].long()

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
    probs  = F.softmax(logits[..., -1, :], dim=-1)
    return probs[..., torch.arange(probs.shape[-2]), label].sum().item()


def compute_k_metrics(                            
    model,
    saes,
    res_sae_effects,
    device,
    tokens,
    label,
    *,
    k_max=7001,
    k_step=500,
):
    """
    Sweep K=1…k_max by k_step, measuring model performance when masking
    the top-K negative or absolute effects. Prints each result.

    Args:
        model, saes, res_sae_effects, device, tokens, label: core args
        k_max (int): Maximum K to test.
        k_step (int): Step size between successive K values.

    Returns:
        (List[int], List[float], List[float]): K_vals, neg_metrics, abs_metrics.
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

def find_min_k_for_threshold(K_vals, metrics_negative, metrics_absolute, clean_probs_baseline, threshold=0.8):
    """
    Scan the two metric lists to find the smallest K that exceeds
    `threshold × clean_probs_baseline` for negative and absolute modes.

    Args:
        K_vals (List[int])
        metrics_negative (List[float])
        metrics_absolute (List[float])
        clean_probs_baseline (Tensor or float)
        threshold (float): Fraction of baseline to reach.

    Returns:
        (min_k_negative, min_k_absolute): Either int or None.
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

def discover_circuit(model, saes, inter_toks_BL, device, ig_steps=10, k_max=7001, k_step=500, k_thres=0.6):
    """
    Orchestrate a full circuit discovery pipeline:
    1. Run model with SAEs to get clean activations + error.
    2. Integrated gradients to get per-latent effects.
    3. Compute k-sweep metrics and find minimum K threshold.
    4. Return the top-K hits via `iter_topk_effects`.

    Args:
        model, saes, inter_toks_BL, device: Core inputs.
        ig_steps, k_max, k_step: IG and K-sweep hyperparams.
        k_thres (float): Fraction of baseline performance to aim for.

    Returns:
        List of (layer, token, latent, value) tuples, or None if none exceed threshold.
    """

    model.reset_hooks(including_permanent=True)
    with torch.no_grad():
        inter_logits_BLV, saes = run_with_saes(model, saes, inter_toks_BL, calc_error=True, use_error=True, cache_sae_activations=True)
    
    inter_label = inter_logits_BLV[0, -1, :].argmax(-1).item()
    clean_probs_baseline = F.softmax(inter_logits_BLV[0, -1, :], dim=-1)[inter_label] #.item()

    for sae in saes:
        sae.mean_error = sae.error_term.detach()

    del inter_logits_BLV
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
        labels=torch.tensor([inter_label]).to(device),
        ig_steps=ig_steps,
        save_dir="", 
        save_and_use=False,
        logstats=False,) 
    
    K_vals, metrics_negative, metrics_absolute = compute_k_metrics(
    model, saes, res_sae_effects, device,      # <- hook_names removed
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
        # print(f"Skipping as no K found")
        return None
    
    # we only need to store entries
    entries = list(itertools.islice(topk_iter, K)) # preserves order

    return entries


def build_saved_pair_dict_fastest(
    model,
    saes,
    trial_entries: Sequence[Tuple[int, int, int, float]],
    unique_token_ids: Sequence[int],
    *,
    tok_k_pos_logits: int = 15,
    batch_size: int = 4096,
) -> Dict[str, List[Tuple[int, int, List[int]]]]:
    """Ultra-fast saved_pair_dict building, full GPU, vectorized."""

    layer_to_latents: Dict[int, set[int]] = defaultdict(set)
    for l, t, lat, _ in trial_entries:
        layer_to_latents[l].add(lat)

    unique_token_tensor = torch.tensor(unique_token_ids, device=device)  # stay on GPU
    saved_pair_dict: Dict[str, List[Tuple[int, int, List[int]]]] = defaultdict(list)

    with torch.no_grad():
        W_U = model.W_U.float().to(device)

        for layer_i, latents in tqdm(layer_to_latents.items()):
            latents = sorted(latents)
            W_dec = saes[layer_i].W_dec.to(device)

            for start in range(0, len(latents), batch_size): #, desc=f"layer {layer_i} hit-mapping"):
                batch = latents[start : start + batch_size]
                dirs = W_dec[batch]            # [batch_size, d_model]

                logits = dirs @ W_U            # [batch_size, vocab_size]
                topk_scores, topk_idx = torch.topk(logits, tok_k_pos_logits, dim=1)  # [batch_size, topk]

                # ---- GPU intersection ----
                # For each latent's topk, find which match unique_token_ids
                # Create: [batch_size, topk] -> True/False if in unique_token_ids

                # broadcast: [batch_size, topk] vs [1, num_unique_ids]
                topk_idx_exp = topk_idx.unsqueeze(-1)                   # [batch, topk, 1]
                unique_tok_exp = unique_token_tensor.view(1, 1, -1)     # [1, 1, num_unique_ids]

                matches = (topk_idx_exp == unique_tok_exp).any(-1)      # [batch, topk] -> True/False
                matching_token_ids = topk_idx[matches]                  # flatten matches

                if matching_token_ids.numel() > 0:
                    matched_latents, matched_topk = torch.nonzero(matches, as_tuple=True)
                    for latent_batch_idx, topk_pos in zip(matched_latents.tolist(), matched_topk.tolist()):
                        latent_idx_in_batch = batch[latent_batch_idx]
                        matched_token_id = topk_idx[latent_batch_idx, topk_pos].item()
                        label_str = model.to_string([matched_token_id]).strip()

                        if not label_str:
                            continue

                        matching_toks = [
                            t for l, t, lat, _ in trial_entries
                            if l == layer_i and lat == latent_idx_in_batch
                        ]
                        saved_pair_dict[label_str].append(
                            (layer_i, latent_idx_in_batch, matching_toks)
                        )

                cleanup_cuda()

    return dict(saved_pair_dict)

def gather_unique_tokens(
    model,
    prompt: Union[str, torch.Tensor],
    *,
    stop_tok: int,
    device: str = "cuda",
) -> Tuple[Sequence[int], str]:
    """Return *(unique_token_ids, full_suffix_str)* for prompt."""
    if isinstance(prompt, str):
        toks = model.to_tokens(prompt).to(device)
    else:
        toks = prompt.to(device)
    out  = toks.clone()
    unique_ids: set[int] = set()

    while True:
        with torch.no_grad():
            logits = model(out)[:, -1, :]
            next_id = logits.argmax(-1).item()
        if next_id == stop_tok:
            break
        if model.to_string(next_id) == "<end_of_turn>" or model.to_string(next_id) == "<eos>":
            break
        unique_ids.add(next_id)
        out = torch.cat([out, torch.tensor([[next_id]], device=device)], dim=1)

    return list(unique_ids)

def find_logit_lens_clusters(model, saes, entries, inter_toks_BL, stop_tok, verbose=True):
    # FIXIT: maybe remove the current token being predicted to not be here 
    uniq_ids = gather_unique_tokens(model, inter_toks_BL, stop_tok=stop_tok)
    prompt_str = model.to_string(inter_toks_BL[0, :])
    filtered_uniq_ids = [tok for tok in uniq_ids if model.to_string(tok) not in prompt_str]
    if verbose:
        print(f"Found {len(filtered_uniq_ids)} tokens not in prompt: {[model.to_string(tok) for tok in filtered_uniq_ids]}")
    saved_pair_dict = build_saved_pair_dict_fastest(
        model,
        saes,
        entries,
        filtered_uniq_ids,
        tok_k_pos_logits=15,
        batch_size=4096
    )
    return saved_pair_dict

def _steer_dtype(activations: torch.Tensor) -> torch.dtype:
    """Return dtype used to apply steering (fp32 while debugging)."""
    # if USE_FP16_STEERING:
    #     return activations.dtype          # fp16 / bf16
    return torch.float32                  # debug‑mode: high precision

def steering_hook(
    activations, hook, sae, latent_idx: int,
    coeff: float, steering_token_index: Optional[int] = None
):
    """
    Apply coeff · W_dec[latent_idx] either to the whole sequence or
    a single position (steering_token_index), using a *single* high‑precision
    add to avoid rounding‑order artefacts.
    """
    tgt_dtype = _steer_dtype(activations)
    steer_vec = (
        sae.W_dec[latent_idx]
        .to(device=activations.device, dtype=tgt_dtype) * float(coeff)
    )

    # ——— cast activations to tgt_dtype only for the in‑place add ———
    act_view = activations.to(dtype=tgt_dtype)

    if steering_token_index is None:
        act_view += steer_vec
    else:
        act_view[:, steering_token_index] += steer_vec

    # In debug mode this casts back to original (fp16/bf16)   ↓↓↓
    return act_view.to(dtype=activations.dtype)

def steering_effect_on_next_token(
    model, inter_toks_BL, saes, interventions, coeff, stop_tok
):
    """
    Returns (changed: bool, new_id: int, baseline_id: int)
    without sampling the whole sequence.
    """
    # --- register hooks exactly once ------------------------------------
    model.reset_hooks(including_permanent=True)
    with torch.no_grad():
        logits = model(inter_toks_BL)[:, -1]
    baseline_id = logits.argmax(-1).item()

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

    with torch.no_grad():
        logits = model(inter_toks_BL)[:, -1]        
    new_id      = logits.argmax(-1).item()
    
    model.reset_hooks(including_permanent=True)
    return new_id != baseline_id, new_id, baseline_id

def generate_once(model, inter_toks_BL: torch.Tensor, stop_tok: int, hooks: Optional[List] = None, *, max_tokens: int = 256, device: str = "cuda") -> str:  # type: ignore
    """Same as before, but accepts an explicit *hooks* list (or None)."""
    toks = inter_toks_BL.to(device)
    out  = toks.clone()

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

    return model.to_string(out[0, toks.shape[-1]:])

def sweep_coefficients_multi(
    model,
    saes: Dict[int, "SAE"],
    interventions: List[Tuple[int, int, int]],   # (layer, token_pos, latent)
    coefficients: List[float],
    inter_toks_BL: torch.Tensor,
    stop_tok: int = 1917,
    device: str = "cuda",
    max_tokens: int = 100,
) -> Dict[float, str]:

    # ─── pre‑aggregate: (coeff  →  layer → {pos → steer_vec}) ─────────────
    d_model   = saes[0].W_dec.shape[1]
    act_dtype = next(model.parameters()).dtype    # fp16 / bf16 / fp32

    steer_by_coeff: Dict[float, Dict[int, Dict[int, torch.Tensor]]] = {}
    for c in coefficients:
        per_coeff = defaultdict(                       # layer →
            lambda: defaultdict(                       #   pos →
                lambda: torch.zeros(                   #     steer_vec
                    d_model, device=device,
                    dtype=_steer_dtype(torch.empty((),dtype=act_dtype))
                )
            )
        )
        for layer_idx, pos, latent_idx in interventions:
            vec = (
                saes[layer_idx].W_dec[latent_idx]
                .to(device=device, dtype=_steer_dtype(torch.empty((),dtype=act_dtype)))
                * float(c)
            )
            per_coeff[layer_idx][pos] += vec
        steer_by_coeff[c] = per_coeff

    # ─── do the per‑coeff generation ──────────────────────────────────────
    outputs: Dict[float, str] = {}
    for c in coefficients:
        changed, new_id, bl_id = steering_effect_on_next_token(
            model, inter_toks_BL, saes, interventions, c, STOP_TOKEN_ID
        )
        if not changed:
            continue 
        per_coeff = steer_by_coeff[c]

        # register ONE hook per layer that adds the *summed* vector
        for layer_idx, pos_dict in per_coeff.items():
            hook_name = saes[layer_idx].cfg.hook_name

            def make_layer_hook(pos_dict_local):
                def layer_hook(activations, hook):
                    for pos, vec in pos_dict_local.items():
                        if pos < activations.size(1):
                            act_view = activations.to(_steer_dtype(activations))
                            act_view[:, pos] += vec
                            # copy back (same dtype cast logic as above)
                            activations.copy_(act_view.to(dtype=activations.dtype))
                    return activations
                return layer_hook

            model.add_hook(hook_name, make_layer_hook(pos_dict))

        # generate
        gen_suffix = generate_once(
            model, inter_toks_BL=inter_toks_BL, stop_tok=stop_tok,
            hooks=None, max_tokens=max_tokens, device=device,
        )
        outputs[c] = gen_suffix
        model.reset_hooks(including_permanent=True)

    return outputs

def run_steering_sweep(
    model,
    saes,
    inter_toks_BL: torch.Tensor,
    saved_pair_dict: Dict[str, List[Tuple[int, int, List[int]]]],
    baseline_text: str,
    *,
    coeff_grid: Sequence[int],
    stop_tok: int,
    max_tokens: int = 100,
) -> Dict[str, Dict[str, Any]]:
    """Return per‑label sweep results without re-running baseline."""
    results: Dict[str, Dict[str, Any]] = {}

    # baseline_toks = baseline_text.split()

    for label, latent_infos in tqdm(saved_pair_dict.items()):
        # build (layer, tok_pos, latent) triples
        interventions = [
            (layer_i, tok_pos, latent_i)
            for layer_i, latent_i, tok_positions in latent_infos
            for tok_pos in tok_positions
        ]
        sweep_out = sweep_coefficients_multi(
            model=model,
            saes=saes,
            interventions=interventions,
            coefficients=list(coeff_grid),
            inter_toks_BL=inter_toks_BL,
            stop_tok=stop_tok,
            max_tokens=max_tokens,
        )

        label_metrics: List[Dict[str, Any]] = []
        for c in coeff_grid:
            txt  = sweep_out.get(c, "")
            # lev  = Levenshtein.normalized_distance(txt, baseline_text)
            # jw   = 1 - JaroWinkler.normalized_similarity(txt, baseline_text)
            # jac  = jaccard_dissim(txt.split(), baseline_toks)
            # label_metrics.append(dict(coeff=c, levenshtein=lev, jaro_winkler=jw, jaccard=jac, steered_text=txt))
            label_metrics.append(dict(coeff=c, steered_text=txt))

        results[label] = {
            "base_text": baseline_text,
            "steered": label_metrics
        }

    return results


# %% Loading and stuff 

device = "cuda"
model_name = "gemma-2-2b-it"
model = load_model(model_name, device=device, use_custom_cache=False, dtype=torch.bfloat16)

layers = list(range(model.cfg.n_layers))
saes = load_pretrained_saes(layers=layers, release="gemma-scope-2b-pt-mlp-canonical", width="16k", device=device, canon=True)

# %% params stuff and testing prompts to use for testing

max_generation_length = 150
STOP_TOKEN_ID = 1917
COEFF_GRID = list(range(-100, 0, 20))
IG_STEPS = 10 
K_MAX = 90001
K_STEP = 10000
K_THRES = 0.6

file_path = "../data/first_100_passing_examples.json"
with open(file_path, 'r') as f:
    data = json.load(f)

# %% testing full generation for a task
p_id = 24
entry = data[p_id]

prompt = (
    "You are an expert Python programmer, and here is your task: "
    f"{entry['prompt']} Your code should pass these tests:\n\n"
    + "\n".join(entry["test_list"]) + "\nWrite your code below starting with \"```python\" and ending with \"```\".\n```python\n"
)

# find the tid of ym
toks_BL = model.to_tokens(prompt).to(device)
out_BL = toks_BL.clone()

while out_BL.shape[-1] - toks_BL.shape[-1] < 150:
    with torch.no_grad():
        logits_V = model(out_BL)[0, -1]
    next_id = logits_V.argmax(-1).item()
    del logits_V
    cleanup_cuda()
    if next_id == STOP_TOKEN_ID:
        break
    out_BL = torch.cat([out_BL, torch.tensor([[next_id]], device=device)], dim=1)
print(model.to_string(out_BL[0, toks_BL.shape[-1]:]))

# %%
out_BL.shape

# %% selecting a specific forward pass

inter_tok_id = 180
with torch.no_grad():
    inter_logits_V = model(out_BL[:, :inter_tok_id])[0, -1]
inter_label = inter_logits_V.argmax(-1).item()
del inter_logits_V
cleanup_cuda()
print(f"Inter token ID: {inter_tok_id}, Inter label: {inter_label}, Inter token: {model.to_string(inter_label)}")

# %% STEP 1: CIRCUIT DISCOVERY

disc_circuit = True

if disc_circuit:
    # entries = [{layer, tok, latent, ie}, ...]
    entries = discover_circuit(
        model=model,
        saes=saes,
        inter_toks_BL=out_BL[:, :inter_tok_id], #toks_BL,
        device=device,
        ig_steps=IG_STEPS,
        k_max=K_MAX,
        k_step=K_STEP,
        k_thres=K_THRES
    )
    if entries is not None:
        save_path = f"../outputs/circuit_prompt{p_id}_intertok{inter_tok_id}_k{len(entries)}_v3.pt"
        torch.save(entries, save_path)
        print(f"Saved {len(entries)} hits to {save_path}")
else:
    # Load the saved circuit entries
    save_path = f"../outputs/circuit_prompt{p_id}_intertok{inter_tok_id}_k30001_v3.pt"
    entries = torch.load(save_path)
    print(f"Loaded {len(entries)} hits from {save_path}")

# %% STEP 2: LOGIT LENS

logit_lens = True

if logit_lens: 
    saved_pair_dict = find_logit_lens_clusters(model, saes, entries, out_BL[:, :inter_tok_id], STOP_TOKEN_ID, verbose=True)
    if saved_pair_dict is not None:
        save_path = f"../outputs/saved_pair_dict_prompt{p_id}_intertok{inter_tok_id}_v3.pt"
        torch.save(saved_pair_dict, save_path)
        print(f"Saved {len(saved_pair_dict)} pairs to {save_path}")
    else:
        print("No saved pair dict found")
else: 
    save_path = f"../outputs/saved_pair_dict_prompt{p_id}_intertok{inter_tok_id}_v3.pt"
    saved_pair_dict = torch.load(save_path)
    print(f"Loaded {len(saved_pair_dict)} pairs from {save_path}")

# %% STEP 3: STEERING SWEEP

# prompt = model.to_string(out_BL[0, :inter_tok_id])
# print(prompt)

baseline_suffix = model.to_string(out_BL[0, inter_tok_id:])
print(baseline_suffix)

# %%
sweep_metrics = run_steering_sweep(
    model,
    saes,
    # prompt,
    inter_toks_BL=out_BL[:, :inter_tok_id],
    saved_pair_dict=saved_pair_dict,
    baseline_text=baseline_suffix,
    coeff_grid=COEFF_GRID,
    stop_tok=STOP_TOKEN_ID,
    max_tokens=100,
)


import pprint
pprint.pprint(sweep_metrics)
# %%
out_trial = model(model.to_string(out_BL[0, :inter_tok_id]))
print(out_trial.shape)
print(model.to_string(out_trial[0, 0].argmax(-1).item()))
print(model.to_string(out_trial[0, -1].argmax(-1).item()))
# %%
out_trial = model(out_BL[0, :inter_tok_id])
print(out_trial.shape)
print(model.to_string(out_trial[0, 0].argmax(-1).item()))
print(model.to_string(out_trial[0, -1].argmax(-1).item()))
# %%
model.to_string(out_BL[0, :inter_tok_id])
# %%
out_BL[0, 0]
# %% earliest planning position

C = {(l, t) for l, t, *_ in entries}
saved_pair_dict = find_logit_lens_clusters(model, saes, entries, out_BL[:, :inter_tok_id], STOP_TOKEN_ID, verbose=True)
print(saved_pair_dict)

# %%

pair_to_latents: Dict[Tuple[int, int], List[int]] = {}
for layer_i, latent_i, tok_pos_list in saved_pair_dict["2"]:
    for tok_pos in tok_pos_list:
        pair_to_latents.setdefault((layer_i, tok_pos), []).append(latent_i)

print(pair_to_latents)

# %%

def _measure_effects(pair_to_latents: Dict[Tuple[int, int], List[int]], clean_toks: torch.Tensor, coeff: float=-200.0):
        toks_prefix = clean_toks
        model.reset_hooks(including_permanent=True)
        with torch.no_grad():
            logits = model(toks_prefix)[:, -1]
        baseline_id = logits.argmax(-1).item()
        baseline_prob = F.softmax(logits, dim=-1)[0, baseline_id]
        print(baseline_prob)
        pos_eff: Dict[Tuple[int, int], float] = {}
        for (layer, tok_pos), latents in pair_to_latents.items():
            sae = saes[layer]
            for latent_idx in latents:
                model.add_hook(
                    sae.cfg.hook_name,
                    partial(steering_hook, sae=sae, latent_idx=latent_idx, coeff=coeff, steering_token_index=tok_pos),
                )
            with torch.no_grad():
                logits = model(toks_prefix)[:, -1]
            prob = F.softmax(logits, dim=-1)[0, baseline_id]
            pos_eff[(layer, tok_pos)] = (prob - baseline_prob).item()
            model.reset_hooks(including_permanent=True)
        return pos_eff

# %%

thresh = 0.1
position_effect: Dict[Tuple[int, int], float] = {}
F_prime: Set[Tuple[int, int]] = set()
position_effect = _measure_effects(pair_to_latents, out_BL[0, :inter_tok_id], coeff=-200.0)
F_prime = {pos for pos, eff in position_effect.items() if abs(eff) > thresh}

print(F_prime)

# %%

# Sort pairs by token index (pair[1]) and take top 10
sorted_pairs = sorted(F_prime, key=lambda x: x[1])[:10]

for pair in sorted_pairs:
    print(f"Layer {pair[0]}, Token {pair[1]}")
    print(f"Token text: {model.to_string(out_BL[0, pair[1]])}")
    print(f"Effect: {position_effect[pair]:.4f}")
    print("-"*100)

# %%

# pair2test = (0,18)
model.reset_hooks(including_permanent=True)
coeff = -200.0

layer, tok_pos = (2, 22) #pair2test
sae = saes[layer]

# Create steering vector: sum of all latent directions scaled by coeff
d_model = sae.W_dec.shape[1]
act_dtype = next(model.parameters()).dtype
steer_vec = torch.zeros(d_model, device=device, dtype=_steer_dtype(torch.empty((), dtype=act_dtype)))

for latent_idx in pair_to_latents[(layer, tok_pos)]:
    vec = (
        sae.W_dec[latent_idx]
        .to(device=device, dtype=_steer_dtype(torch.empty((), dtype=act_dtype)))
        * float(coeff)
    )
    steer_vec += vec

# Define hook function
def steering_hook_single(activations, hook):
    if tok_pos < activations.size(1):
        act_view = activations.to(_steer_dtype(activations))
        act_view[:, tok_pos] += steer_vec
        return act_view.to(dtype=activations.dtype)
    return activations

# Register hook and generate
model.add_hook(sae.cfg.hook_name, steering_hook_single)

gen_suffix = generate_once(
            model, inter_toks_BL=out_BL[:, :inter_tok_id], stop_tok=STOP_TOKEN_ID,
            hooks=None, max_tokens=100, device=device,
        )
print(gen_suffix)


model.reset_hooks(including_permanent=True)


# %%
print(position_effect[(17,190)])
print(position_effect[(7,24)])
print(position_effect[(17,135)])
# %%

print(model.to_string(out_BL[0, 15:30]))
# model.to_string(out_BL[0,24])

# %%

@dataclass
class Config:
    # I/O
    data_path: Path = Path("../data/external/first_100_passing_examples.json")
    hits_root: Path = Path("../models/mbpp_task2_2b_mlp")

    # generation
    stop_token_id: int = 1917
    max_new_tokens: int = 150

    # steering
    coeff: float = -200.0

    # SAE
    sae_release: str = "gemma-scope-2b-pt-mlp-canonical"
    layers: Sequence[int] | None = None

    # runtime
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
# -----------------------------------------------------------------------------
#  CORE ANALYZER
# -----------------------------------------------------------------------------

class CircuitAnalyzer:
    """Analyze a single `(prompt_idx, token_pred_idx)` pair."""

    def __init__(self, model, saes, cfg: Config):
        self.model, self.saes, self.cfg = model, saes, cfg
        if cfg.layers is None:
            cfg.layers = list(range(model.cfg.n_layers))
        with open(cfg.data_path, "r") as f:
            self.dataset = json.load(f)

    # ---------------- PUBLIC API ----------------

    def run(
        self,
        prompt_idx: int,
        token_pred_idx: int,
        *,
        keys: Iterable[str] | None = None,
        thresh: float = 0.0,
    ) -> Dict[str, object]:
        """Compute C, F_all, F_prime for one prediction.

        *keys* – list of saved‑pair dict keys to evaluate.  
        ▸ `None` → evaluate **all** keys.  
        ▸ `[]` (empty) → skip evaluation, so F_all = F_prime = ∅.
        """
        # 1 Load circuit entries (C) & baseline prompt
        hits_path = self.cfg.hits_root / f"prompt_{prompt_idx}/pred_{token_pred_idx}/hits.pt"
        trial = torch.load(hits_path, weights_only=True)
        C = {(l, t) for l, t, *_ in trial["entries"]}
        clean_prompt = trial["prompt"].lstrip("<bos>")

        # 2 Build saved‑pair dict -----------------------------------------
        uniq_ids = gather_unique_tokens(self.model, clean_prompt, stop_tok=self.cfg.stop_token_id)
        prompt_ids = self.model.to_tokens(clean_prompt)[0].tolist()
        filtered = [tok for tok in uniq_ids if tok not in prompt_ids]

        saved_pair_dict = build_saved_pair_dict_fastest(
            self.model, self.saes, trial["entries"], filtered
        )
        hit_positions = set()
        for key in saved_pair_dict:
            for layer_i, _, token_pos_list in saved_pair_dict[key]:
                for token_pos in token_pos_list:
                    hit_positions.add((layer_i, token_pos))

        F_all = hit_positions
        # Early‑exit if user provided an *empty* key list ------------------
        if keys is not None and len(list(keys)) == 0:
            return {
                "prompt_idx": prompt_idx,
                "token_pred_idx": token_pred_idx,
                "keys": [],
                "C": C,
                "F_all": F_all,
                "F_prime": set(),
                "position_effect": {},
            }

        # Choose keys
        keys = list(saved_pair_dict.keys()) if keys is None else list(keys)

        # 3 Map (layer, token_pos) → latents
        pair_to_latents: Dict[Tuple[int, int], List[int]] = {}
        for k in keys:
            for layer_i, latent_i, tok_pos_list in saved_pair_dict[k]:
                for tok_pos in tok_pos_list:
                    pair_to_latents.setdefault((layer_i, tok_pos), []).append(latent_i)

        # 4 Compute steering effects on demand ----------------------------
        position_effect: Dict[Tuple[int, int], float] = {}
        F_prime: Set[Tuple[int, int]] = set()
        if F_all:
            position_effect = self._measure_effects(pair_to_latents, clean_prompt)
            F_prime = {pos for pos, eff in position_effect.items() if abs(eff) > thresh}

        return {
            "prompt_idx": prompt_idx,
            "token_pred_idx": token_pred_idx,
            "keys": keys,
            "C": C,
            "F_all": F_all,
            "F_prime": F_prime,
            "position_effect": position_effect,
        }

    # ---------------- INTERNAL ----------------

    def _measure_effects(self, pair_to_latents: Dict[Tuple[int, int], List[int]], clean_prompt: str):
        toks_prefix = self.model.to_tokens(clean_prompt)
        self.model.reset_hooks(including_permanent=True)
        with torch.no_grad():
            logits = self.model(toks_prefix)[:, -1]
        baseline_id = logits.argmax(-1).item()
        baseline_prob = F.softmax(logits, dim=-1)[0, baseline_id]

        pos_eff: Dict[Tuple[int, int], float] = {}
        for (layer, tok_pos), latents in pair_to_latents.items():
            sae = self.saes[layer]
            for latent_idx in latents:
                self.model.add_hook(
                    sae.cfg.hook_name,
                    partial(steering_hook, sae=sae, latent_idx=latent_idx, coeff=self.cfg.coeff, steering_token_index=tok_pos),
                )
            with torch.no_grad():
                logits = self.model(toks_prefix)[:, -1]
            prob = F.softmax(logits, dim=-1)[0, baseline_id]
            pos_eff[(layer, tok_pos)] = (prob - baseline_prob).item()
            self.model.reset_hooks(including_permanent=True)
        return pos_eff

# -----------------------------------------------------------------------------
#  BATCH‑LEVEL CONVENIENCE -----------------------------------------------------
# -----------------------------------------------------------------------------

def analyze_batch(
    analyzer: CircuitAnalyzer,
    prompt_idx: int,
    mapping: Dict[int, List[str]],  # token_pred_idx → key list (may be empty)
    *,
    thresh: float = 0.0,
):
    """Run analyzer over several predictions & return unified sets."""
    union_C: Set[Tuple[int, int]] = set()
    union_F: Set[Tuple[int, int]] = set()
    union_Fp: Set[Tuple[int, int]] = set()

    for token_pred_idx, key_list in mapping.items():
        res = analyzer.run(prompt_idx, token_pred_idx, keys=key_list, thresh=thresh)
        union_C |= res["C"]
        union_F |= res["F_all"]
        union_Fp |= res["F_prime"]

    return {
        "union_C": union_C,
        "union_F_all": union_F,
        "union_F_prime": union_Fp,
        "circuit_minus_future": union_C - union_F,
    }
