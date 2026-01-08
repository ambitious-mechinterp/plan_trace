# Imports and configuration
import re
import sys 
import json
import time
import torch

from pathlib import Path
sys.path.append("../")
from plan_trace.utils import load_model, load_pretrained_saes, cleanup_cuda
from plan_trace.steering import run_steering_sweep
from plan_trace.ood_detect import label_steering_clusters

# Config (adjust as needed)
model_name = "gemma-2-2b-it"
device = "cuda"
use_custom_cache = True  # toggle as needed; mirrors --use-custom-cache


data_path = "../data/first_100_passing_examples.json"
prompt_idx = 15
inter_token_id = 297  # token_pred_idx to anchor prefix for analysis
stop_token_id = 1917  # token for ```
coeff_grid = list(range(-100, 0, 20))

def build_prompt(entry: dict) -> str:
    return (
        "You are an expert Python programmer, and here is your task: "
        f"{entry['prompt']} Your code should pass these tests:\n\n"
        + "\n".join(entry["test_list"]) + "\nWrite your code below starting with \"```python\" and ending with \"```\".\n```python\n"
    )

# Load model/SAEs and generate tokens
with open(data_path, 'r') as f:
    data = json.load(f)

entry = data[prompt_idx]
prompt = build_prompt(entry)

print("Loading model and SAEs...")
model = load_model(model_name, device=device, use_custom_cache=use_custom_cache, dtype=torch.bfloat16)
layers = list(range(model.cfg.n_layers))
saes = load_pretrained_saes(
    layers=layers,
    release="gemma-scope-2b-pt-mlp-canonical",
    width="16k",
    device=device,
    canon=True,
)


inter_token_id = 297
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

inter_toks_BL = out_BL[:, :inter_token_id]
baseline_suffix = model.to_string(out_BL[0, inter_token_id:])

# Load clusters and restrict to label '1'
clusters_path = Path("../outputs/per_pos/test/prompt_15/token_297/clusters.json")
with open(clusters_path, 'r') as f:
    saved_pair_dict = json.load(f)

print("Available labels:", list(saved_pair_dict.keys()))
if '1' not in saved_pair_dict:
    raise ValueError("Label '1' not found in clusters.json")

saved_pair_dict = {'1': saved_pair_dict['1']}
print("Using labels:", list(saved_pair_dict.keys()))

# Collect sorted unique token positions for label '1'
positions = sorted({
    tok_pos
    for (_, _, tok_positions) in saved_pair_dict['1']
    for tok_pos in tok_positions
})
print(f"Candidate token positions for label '1': {positions[:20]}{'...' if len(positions) > 20 else ''}")

# Per-position sweep from the end, print steered generations (strings) − fixed printing
n_positions = 100
coeff_grid_local = list(range(-400, 0, 100))
max_tokens_local = 50

assert list(saved_pair_dict.keys()) == ['1'], "saved_pair_dict must be filtered to label '1'"

positions_desc = sorted({
    tok_pos
    for (_, _, tok_positions) in saved_pair_dict['1']
    for tok_pos in tok_positions
}, reverse=False)

earliest_position = None
earliest_position_steering_results = None

for tok_pos in positions_desc[:n_positions]:
    # Build filtered dict limited to this exact position
    sub = []
    for (li, latent_i, tok_positions) in saved_pair_dict['1']:
        if tok_pos in tok_positions:
            sub.append((li, latent_i, [tok_pos]))
    if not sub:
        continue
    filtered = {'1': sub}

    # Show which (layer, latent) pairs will be steered
    layer_latents = [(li, latent_i) for (li, latent_i, _) in sub]
    preview_pairs = layer_latents if len(layer_latents) <= 20 else layer_latents[:20] + [("...", "...")]
    print(f"\n=== Position {tok_pos} ===")
    print(f"Steering {len(layer_latents)} (layer, latent) pairs -> {preview_pairs}")

    # Run sweep with return_tokens=False to get strings
    pos_steering = run_steering_sweep(
        model=model,
        saes=saes,
        inter_toks_BL=inter_toks_BL,
        saved_pair_dict=filtered,
        baseline_text=baseline_suffix,
        coeff_grid=coeff_grid_local,
        stop_tok=stop_token_id,
        max_tokens=max_tokens_local,
        return_tokens=False,
    )

    # Label results
    pos_labels = label_steering_clusters(pos_steering, model=model, prefix_tokens_2d=inter_toks_BL)
    label_map = {k: v['final_label'] for k, v in pos_labels.items()}
    print(f"Labels: {label_map}")

    # Print baseline and steered generations (precompute previews to avoid backslashes in f-strings)
    base_text = pos_steering.get('1', {}).get('base_text', '')
    base_preview = (base_text[:200] if isinstance(base_text, str) else str(base_text)) \
        .replace("\n", " ")
    print("Baseline (preview): " + base_preview)

    for item in pos_steering.get('1', {}).get('steered', []):
        coeff = item.get('coeff')
        steered_text = item.get('steered_text', '')
        if hasattr(steered_text, "tolist"):
            try:
                steered_text = model.to_string(steered_text.tolist())
            except Exception:
                steered_text = str(steered_text)
        steered_preview = (steered_text[:200] if isinstance(steered_text, str) else str(steered_text)) \
            .replace("\n", " ")
        print(f"  coeff {coeff:>4}: " + steered_preview)

    if any(v == "Plan" for v in label_map.values()):
        earliest_position = tok_pos
        earliest_position_steering_results = pos_steering
        print(f"--> Found earliest Plan at position {tok_pos}")
        break

print(f"\nEarliest planning position (label '1'): {earliest_position}")

import random
## randomly sample 1, 2, 5, 10, 20 and then all latents of `1` and check the result of steering them with a scale of -100 to -400.
def _randomly_sample_latents(data, n = 5, label = "1"):
    flat_li = []
    for layer, latent, toks in data["1"]:
        for tok in toks:
            flat_li.append((layer, latent, tok))
    
    ### sample 'n' latents from here.
    x = min(n, len(flat_li))
    random_li = random.sample(flat_li, x)

    unflat_dict = {}
    for layer, latent, tok_pos in random_li:
        if (layer, latent) not in unflat_dict:
            unflat_dict[(layer, latent)] = []
        unflat_dict[(layer, latent)].append(tok_pos)
    
    filtered_pair_dict = {
        label: []
    }
    for (layer, latent), value in unflat_dict.items():
        filtered_pair_dict[label].append([
            layer,
            latent,
            value
        ])

    return filtered_pair_dict

labels = ["1", "x", "key"]
n_vals = [1, 2, 5, 10, 20, 100]

# for label in labels:
#     for n in n_vals:
for i in range(10):
    random_filtered_dict = _randomly_sample_latents(saved_pair_dict, 5, "1")
    pos_steering = run_steering_sweep(
        model=model,
        saes=saes,
        inter_toks_BL=inter_toks_BL,
        saved_pair_dict=random_filtered_dict,
        baseline_text=baseline_suffix,
        coeff_grid=coeff_grid_local,
        stop_tok=stop_token_id,
        max_tokens=max_tokens_local,
        return_tokens=False,
    )

    # print(f"=== Randomly sampled latent check n = {n}, label = {label} ===")
    print(f"=== Randomly sampled latent check n = 5, label = \"1\" ===")
    # for item in pos_steering.get(label, {}).get('steered', []):
    for item in pos_steering.get("1", {}).get('steered', []):
        coeff = item.get('coeff')
        steered_text = item.get('steered_text', '')
        if hasattr(steered_text, "tolist"):
            try:
                steered_text = model.to_string(steered_text.tolist())
            except Exception:
                steered_text = str(steered_text)
        steered_preview = (steered_text[:200] if isinstance(steered_text, str) else str(steered_text)) \
            .replace("\n", " ")
        print(f"  coeff {coeff:>4}: " + steered_preview)