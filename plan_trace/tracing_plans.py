# %% 
# # for a given prompt and token index,
# discover circuit with >60% perf recovery
# cluster based on decoding directions logit lens 
# test steering effect of clusters
# test effect of each latent / position in cluster
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
import torch.nn.functional as F
from typing import List, Optional, Dict, Any
from sae_lens import SAE, HookedSAETransformer
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory

from dotenv import load_dotenv
from huggingface_hub import login


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
p_id = 15
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

# %% selecting a specific forward pass

inter_tok_id = 297
with torch.no_grad():
    inter_logits_V = model(out_BL[:, :inter_tok_id])[0, -1]
inter_label = inter_logits_V.argmax(-1).item()
del inter_logits_V
cleanup_cuda()
print(f"Inter token ID: {inter_tok_id}, Inter label: {inter_label}, Inter token: {model.to_string(inter_label)}")


# %% STEP 1: CIRCUIT DISCOVERY

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

entries = discover_circuit(
    model=model,
    saes=saes,
    inter_toks_BL=toks_BL,
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