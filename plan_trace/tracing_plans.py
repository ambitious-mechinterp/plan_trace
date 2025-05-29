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
    """
    Clears GPU memory cache and runs garbage collection.
    """
    gc.collect()
    torch.cuda.empty_cache()

def clear_memory(saes, model, mask_bool=False):
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
        """
        Prints statistics about each binary mask:
          - total number of elements (latents)
          - total number of 'on' latents (mask == 1)
          - average on-latents per token
            * If shape == [latent_dim], there's effectively 1 token
            * If shape == [seq, latent_dim], it's 'sum of on-latents / seq'
        """
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
        Saves hook_points and masks to a single file (file_name) within save_dir.
        If you want multiple mask sets in the same directory, call save() with
        different file_name values. The directory is created if it does not exist.
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
        Loads hook_points and masks from a single file (file_name) within load_dir,
        returning an instance of SAEMasks. If you stored multiple mask sets in the
        directory, specify the file_name to load the correct one.
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
        num_latents = 0
        for mask in self.masks:
            num_latents += (mask > 0).sum().item()
        return num_latents


def build_sae_hook_fn(
    # Core components
    sae,
    sequence,
    bos_token_id,
    # Masking options
    circuit_mask: Optional[SAEMasks] = None,
    mean_mask=False,
    cache_masked_activations=False,
    cache_sae_activations=False,
    # Ablation options
    mean_ablate=False,  # Controls mean ablation of the SAE
    fake_activations=False,  # Controls whether to use fake activations
    calc_error=False,
    use_error=False,
    use_mean_error=False,
):
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

def run_sae_hook_fn(
    model,
    saes,
    sequence,
    circuit_mask: Optional[SAEMasks] = None,
    mean_mask=False,
    cache_sae_activations=False,
    cache_masked_activations=False,
    mean_ablate=False,
    fake_activations=False,
    calc_error=False,
    use_error=False,
    use_mean_error=False,
):
    hooks = []
    bos_token_id = model.tokenizer.bos_token_id

    for sae in saes:
        hook_fn = build_sae_hook_fn(
            sae=sae,
            sequence=sequence,
            bos_token_id=bos_token_id,
            circuit_mask=circuit_mask,
            mean_mask=mean_mask,
            cache_sae_activations=cache_sae_activations,
            cache_masked_activations=cache_masked_activations,
            mean_ablate=mean_ablate,
            fake_activations=fake_activations,
            calc_error=calc_error,
            use_error=use_error,
            use_mean_error=use_mean_error,
        )
        hooks.append((sae.cfg.hook_name, hook_fn))

    return model.run_with_hooks(sequence, return_type="logits", fwd_hooks=hooks), saes


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
# file_path = os.path.join(destination_folder, filename)
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
    model.reset_hooks(including_permanent=True)
    with torch.no_grad():
        inter_logits_BLV, saes = run_sae_hook_fn(model, saes, inter_toks_BL, calc_error=True, use_error=True, cache_sae_activations=True)
    
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


    hook_names = [sae.cfg.hook_name for sae in saes]

    ### K METRICS
    K_vals, metrics_negative, metrics_absolute = compute_k_metrics(
        model, saes, res_sae_effects, device, hook_names, inter_toks_BL, inter_label, k_max=k_max, k_step=k_step
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
    save_path = f"../outputs/circuit_prompt{p_id}_intertok{inter_tok_id}_k{len(entries)}.pt"
    torch.save(entries, save_path)
    print(f"Saved {len(entries)} hits to {save_path}")



# %%
# from helpers.utils import (
#     load_model,
#     load_pretrained_saes,
# )
# from circuit_disc import discover_circuit, compute_metric
# from plan_criteria import get_ll_clusters, test_steering

# device = "cuda"
# model_name = "gemma-2-2b-it"
# model = load_model(model_name, device=device, use_custom_cache=False, dtype=torch.bfloat16)

# layers = list(range(model.cfg.n_layers))
# saes = load_pretrained_saes(layers=layers, release="gemma-scope-2b-pt-mlp-canonical", width="16k", device=device, canon=True)

# # Set parameters for the generation
# max_generation_length = 150
# STOP_TOK_ID = 1917
# COEFF_GRID = list(range(-100, 0, 20))
# ig_steps = 10  

# prompt = ""
# toks = model.to_tokens(prompt).to(device)
# changable_toks = toks.clone()


# while True:
#     with torch.no_grad():
#         logits_BLV = model(changable_toks)

#     logits_BV = logits_BLV[:, -1, :]
#     probs_BV = F.softmax(logits_BV, dim=-1)
#     label = torch.argmax(logits_BV).item()
#     clean_probs_baseline = probs_BV[0, label]
#     if model.to_string(label) == "<end_of_turn>" or model.to_string(label) == "<eos>" or label == STOP_TOK_ID:
#         break
#     if changable_toks.shape[-1] - toks.shape[-1] > max_generation_length:
#         break
#     prompt_current = model.to_string(changable_toks[0])

#     # (layer, latent, tok, effect)
#     circuit = discover_circuit(model,
#                                saes,
#                                changable_toks,
#                                label=label,
#                                device=device,
#                                ig_steps=ig_steps,
#                                k_max=7001,
#                                 k_step=500,
#                                 k_thres=0.6,) 

#     # {ym : (layer, latent, tok) 
#     logit_lens_cluster = get_ll_clusters()

#     steered_clusters = test_steering()

#     planning_positions = test_steering()


#     changable_toks = torch.cat([changable_toks, torch.tensor([[label]], device=device)], dim=1)
