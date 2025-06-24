"""
Utility functions for model loading, memory management, and basic operations.

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

import os
import gc
import re
import torch
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login
from sae_lens import SAE, HookedSAETransformer
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from typing import List


def load_model(
    model_name: str, 
    device: str, 
    use_custom_cache: bool = False, 
    dtype: torch.dtype = torch.bfloat16, 
    weights_dir: str = None
) -> HookedSAETransformer:
    """
    Load a pretrained HookedSAETransformer, optionally redirecting HuggingFace/Torch cache.

    Args:
        model_name: HF identifier of the model to load.
        device: Compute device, e.g. "cuda" or "cpu".
        use_custom_cache: If True, force both TORCH_HOME and HF_HOME to a local directory.
        dtype: Numeric precision for model weights.
        weights_dir: Alternate cache directory (if exists).

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


def cleanup_cuda() -> None:
    """Run Python GC and empty the CUDA cache to free GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()


def clear_memory(saes, model, mask_bool: bool = False) -> None:
    """
    Zero out all gradients on SAEs (and their masks if requested) and on the main model.

    Args:
        saes: Iterable of SAE modules whose gradients to clear.
        model: The language model to clear gradients on.
        mask_bool: If True, also clear grads on any mask modules attached to SAEs.
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


def load_pretrained_saes(
    layers: List[int], 
    release: str,
    width: str = "16k",
    device: str = "cuda", 
    canon: bool = True
) -> List[SAE]:
    """
    For each layer in `layers`, pick the median-l0 SAE variant and load it.

    Args:
        layers: Transformer layers to attach SAEs to.
        release: The scope/release name for SAEs (e.g. "gemma-scope-2b-pt-mlp-canonical").
        width: Width identifier in the SAE directory (default "16k").
        device: Target device for the loaded SAEs.
        canon: If True, use the canonical naming convention when matching.

    Returns:
        List of SAE instances, one per requested layer.
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