# %%
"""Quick latent-token selectivity check using Neuronpedia top activations."""

import argparse
import json
import math
import sys
from typing import Any, Dict, List, Sequence, Tuple
import gzip
import os
import re
import glob
import requests
import torch
from sae_lens import HookedSAETransformer

sys.path.append("../")

from plan_trace.utils import cleanup_cuda, load_model

# %%

act_path_base = "/project/pi_jensen_umass_edu/jnainani_umass_edu/activations"
layer_wise_pattern = "L{layer}/raw/batch-{batch_idx}.jsonl.gz"

cluster_path = "/work/pi_jensen_umass_edu/jnainani_umass_edu/plan_trace/outputs/prompt_24/token_149/clusters.json"

# %%

def tokens_to_text(tokens: Sequence[str], strip: bool = False) -> str:
    text = "".join(token.replace("▁", " ").replace("<0x0A>", "\n") for token in tokens)
    return text.strip() if strip else text



# %% 
with open(cluster_path, "r") as f:
    cluster_dict = json.load(f)
cluster_dict["2"]
# %%
# get the unique (layer, latent) pairs
unique_layer_latent = set()
for layer, lat, toks in cluster_dict["2"]:
    unique_layer_latent.add((layer, lat))
# %%
unique_layer_latent
# %%
# helper to find all entries for a given (layer, latent) by scanning batches
def find_latent_entries_in_layer_batches(layer: int, latent: Any) -> Tuple[List[Dict[str, Any]], int]:
    raw_dir = os.path.join(act_path_base, f"L{layer}", "raw")
    batch_files = sorted(
        glob.glob(os.path.join(raw_dir, "batch-*.jsonl.gz")),
        key=lambda p: int(re.search(r"batch-(\\d+)\\.jsonl\\.gz$", p).group(1)) if re.search(r"batch-(\\d+)\\.jsonl\\.gz$", p) else -1,
    )

    target_str = str(latent)
    for path in batch_files:
        matched: List[Dict[str, Any]] = []
        with gzip.open(path, "rt") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                idx = record.get("index")
                if idx is None:
                    continue
                if str(idx) == target_str:
                    matched.append(record)
        if matched:
            m = re.search(r"batch-(\\d+)\\.jsonl\\.gz$", path)
            batch_idx = int(m.group(1)) if m else -1
            return matched, batch_idx
    return [], -1

# trial to get acts for a single (layer, latent) pair
layer, latent = list(unique_layer_latent)[0]
print(layer, latent)
act_dict, found_batch = find_latent_entries_in_layer_batches(layer, latent)
act_dict
# %%
act_dict[0]['tokens']
# %%
