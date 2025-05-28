import os
import gc
import re
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from typing import List, Optional, Dict

from dotenv import load_dotenv
from huggingface_hub import login

from sae_lens import SAE, HookedSAETransformer
import transformer_lens
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory

def activate_autoreload():
    """
    Enables IPython autoreload to automatically reload modules when edited.
    """
    try:
        ipython = get_ipython()
        if ipython is not None:
            ipython.magic("load_ext autoreload")
            ipython.magic("autoreload 2")
            print("Set autoreload in IPython.")
        else:
            print("Not in IPython.")
    except NameError:
        print("`get_ipython` not available. This script is not running in IPython.")

# Call the function during script initialization
activate_autoreload()

def load_model(model_name: str, device: str, use_custom_cache: bool = False, dtype: torch.dtype = torch.bfloat16, weights_dir: str = None) -> HookedSAETransformer:
    """
    Load a pre-trained transformer model.

    :param model_name: Name of the model to load.
    :param device: Device to load the model onto (e.g., "cuda" or "cpu").
    :param use_custom_cache: Whether to use a custom cache directory for model weights.
    :param dtype : The data type to use when loading the model. This function now supports bfloat16 (torch.bfloat16) as well as other dtypes. Defaults to torch.bfloat16.
    :param weights_dir: Directory to use for custom cache. If None, defaults to the environment variable TORCH_HOME.
    :return: A HookedSAETransformer instance.
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
    """
    Clears GPU memory cache and runs garbage collection.
    """
    gc.collect()
    torch.cuda.empty_cache()

def clear_memory(saes, model, mask_bool=False):
    """
    Clears out the gradients from the SAEs and the main model
    to avoid accumulation or weird artifacts.

    Args:
        saes (List[SAE]): A list of SAE objects (or similarly structured modules).
        model (nn.Module): The main model whose gradients we want to clear.
        mask_bool (bool): Whether to also clear mask gradients if they exist.
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
    Retrieves metadata for all pretrained SAE models as a pandas DataFrame.
    
    Returns:
        pd.DataFrame: A DataFrame containing information about all available 
                     pretrained SAE models, with columns like model names, 
                     configurations, and other metadata.
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
    Load pretrained SAEs for specified layers.
    
    Args:
        model_name: Name of the base model
        layers: List of layer indices
        release: SAE release name
        width: Width of SAE
        device: Device to load SAEs on
    
    Returns:
        List of loaded SAE objects
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