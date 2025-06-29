#%%
# Detecting planning without the knowledge of future tokens

"""# Steps: 
1. Discover circuit for y_n
2. Get unique latents in the circuit 
3. Find the planning positions old fashioned, specifically the earliest layer
4. Do logit lens for all features in that layer 
5. Get the prompt string, string of current token being generated. 
6. Find the feature with logit lens token not in prompt or current token. 
"""

# Simple fix for imports when running interactively
import sys
import os
# Add parent directory to path so plan_trace package can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict

from plan_trace.utils import load_model, load_pretrained_saes, cleanup_cuda
from plan_trace.circuit_discovery import discover_circuit
from plan_trace.logit_lens import gather_unique_tokens, build_saved_pair_dict_fastest, find_logit_lens_clusters
from plan_trace.steering import run_steering_sweep

#%% Setup and Configuration
print("=== FTE Prime: Detecting Planning Without Future Knowledge ===")

# Parameters
prompt_idx = 15
inter_token_id = 297  # y_n position
predicted_token = "1"  # y_m - the actual token being predicted
model_name = "gemma-2-2b-it"
device = "cuda"

verbose = True

print(f"Prompt: {prompt_idx}, Token position: {inter_token_id}, Predicted token: '{predicted_token}'")

#%% Load Model and Data
print("\n=== Loading model and SAEs ===")
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
# Load and prepare data
data_path = "../data/first_100_passing_examples.json"
with open(data_path, 'r') as f:
    data = json.load(f)

entry = data[prompt_idx]
prompt = (
    "You are an expert Python programmer, and here is your task: "
    f"{entry['prompt']} Your code should pass these tests:\n\n"
    + "\n".join(entry["test_list"]) + "\nWrite your code below starting with \"```python\" and ending with \"```\".\n```python\n"
)

print(f"Model loaded: {model_name}")
print(f"SAEs loaded for {len(saes)} layers")
print(f"Prompt length: {len(prompt)} characters")

#%% Generate Sequence to Target Position
print("\n=== Generating sequence up to prediction position ===")
toks_BL = model.to_tokens(prompt).to(device)
out_BL = toks_BL.clone()

print(f"Initial prompt tokens: {toks_BL.shape[-1]}")
print(f"Target position: {inter_token_id}")

# Generate up to inter_token_id position
while out_BL.shape[-1] < inter_token_id:
    with torch.no_grad():
        logits_V = model(out_BL)[0, -1]
    next_id = logits_V.argmax(-1).item()
    del logits_V
    cleanup_cuda()
    out_BL = torch.cat([out_BL, torch.tensor([[next_id]], device=device)], dim=1)

# Extract tokens up to y_n position
inter_toks_BL = out_BL[:, :inter_token_id]

print(f"Generated sequence length: {out_BL.shape[-1]}")
print(f"Context up to position {inter_token_id}: {inter_toks_BL.shape[-1]} tokens")

# %%

print(model.to_string(inter_toks_BL[0]))
#%% Step 1: Discover Circuit for y_n
print(f"\n=== Step 1: Discovering circuit for token position {inter_token_id} ===")
circuit_entries = discover_circuit(
    model=model,
    saes=saes,
    inter_toks_BL=inter_toks_BL,
    device=device,
    ig_steps=10,
    k_max=70001,  # Reduced for faster computation
    k_step=10000,
    k_thres=0.6
)

if circuit_entries is None:
    print("❌ No circuit found meeting threshold criteria")
    raise RuntimeError("No circuit found - cannot proceed with analysis")

print(f"✅ Found circuit with {len(circuit_entries)} entries")
print("First few circuit entries:")
for i, (layer_idx, token_pos, latent_idx, effect_value) in enumerate(circuit_entries[:5]):
    print(f"  {i+1}. Layer {layer_idx}, Token {token_pos}, Latent {latent_idx}, Effect: {effect_value:.4f}")

#%% Step 2: Get Unique Latents in Circuit
print(f"\n=== Step 2: Extracting unique latents from circuit ===")

unique_latents_by_layer = defaultdict(set)
circuit_positions = set()
# Also keep track of which positions each latent appears at
latent_positions_by_layer = defaultdict(lambda: defaultdict(list))

for layer_idx, token_pos, latent_idx, effect_value in circuit_entries:
    unique_latents_by_layer[layer_idx].add(latent_idx)  # Just the latent index
    circuit_positions.add((layer_idx, token_pos))
    latent_positions_by_layer[layer_idx][latent_idx].append((token_pos, effect_value))

print(f"Circuit spans {len(unique_latents_by_layer)} layers with {len(circuit_positions)} positions")
for layer in sorted(unique_latents_by_layer.keys()):
    print(f"  Layer {layer}: {len(unique_latents_by_layer[layer])} unique latents")

# %% Step 3: find actual planning positions
stop_token_id = 1917
baseline_suffix = model.to_string(out_BL[0, inter_token_id:])
coeff_grid = list(range(-100, 0, 20))
saved_pair_dict = find_logit_lens_clusters(
        model, saes, circuit_entries, inter_toks_BL, stop_token_id, verbose=verbose
    )
steering_results = run_steering_sweep(
        model=model,
        saes=saes,
        inter_toks_BL=inter_toks_BL,
        saved_pair_dict=saved_pair_dict,
        baseline_text=baseline_suffix,
        coeff_grid=coeff_grid,
        stop_tok=stop_token_id,
        max_tokens=100,
    )

# %%

saved_pair_dict['1']

# %% Find unique latents in earliest layer for token "1"
# Extract all latents that predict "1" 
token_1_latents = saved_pair_dict['1']

# Group by layer
latents_by_layer = defaultdict(set)
for layer, latent_idx, token_positions in token_1_latents:
    latents_by_layer[layer].add(latent_idx)

# Find earliest layer
earliest_layer = min(latents_by_layer.keys())
earliest_layer_latents = latents_by_layer[earliest_layer]

print(f"Earliest layer with '1' predictions: {earliest_layer}")
print(f"Unique latents in layer {earliest_layer}: {sorted(earliest_layer_latents)}")
print(f"Number of unique latents: {len(earliest_layer_latents)}")

# Show details for each latent in the earliest layer
print(f"\nDetails for each latent in layer {earliest_layer}:")
for layer, latent_idx, token_positions in token_1_latents:
    if layer == earliest_layer:
        print(f"  Latent {latent_idx}: appears at positions {token_positions}")

# %% Test logit lens for one latent from earliest layer

latent2test = list(earliest_layer_latents)[0]
sae2test = saes[earliest_layer]
W_U = model.W_U.float().to(device)
W_dec = sae2test.W_dec.to(device)

print(f"Testing latent {latent2test} from layer {earliest_layer}")

# Get decoder direction for this latent
dirs = W_dec[latent2test]  # Shape: [d_model]
print(f"Decoder direction shape: {dirs.shape}")

# Compute logits via logit lens  
logits = dirs @ W_U  # Shape: [vocab_size]
print(f"Logits shape: {logits.shape}")

# Get top 15 predicted tokens
topk_scores, topk_idx = torch.topk(logits, 15, dim=0)  # dim=0 for 1D tensor
print(f"Top 15 token IDs: {topk_idx.tolist()}")

print(f"\nTop 15 predicted tokens:")
for i, (score, idx) in enumerate(zip(topk_scores, topk_idx)):
    token_str = model.to_string([idx.item()])
    print(f"{i+1:2}. '{token_str}' (ID: {idx.item()}, score: {score:.3f})")

# %%
toks_BL = model.to_tokens(prompt).to(device)
out_BL = toks_BL.clone()

print(f"Initial prompt tokens: {toks_BL.shape[-1]}")
print(f"Target position: {inter_token_id}")

# Generate up to inter_token_id position
while out_BL.shape[-1] - toks_BL.shape[-1] < 150:
    with torch.no_grad():
        logits_V = model(out_BL)[0, -1]
    next_id = logits_V.argmax(-1).item()
    if next_id == 1917:
        break
    del logits_V
    cleanup_cuda()
    out_BL = torch.cat([out_BL, torch.tensor([[next_id]], device=device)], dim=1)

print(model.to_string(out_BL[0, toks_BL.shape[-1]:]))

# %% Testing the FTE prime rule on this latent 
# G(f) = D(f) \ (P ∪ N)

G_f = 0
prompt_text = model.to_string(inter_toks_BL[0])
current_token_text = model.to_string(out_BL[0, inter_token_id])

for i, (score, idx) in enumerate(zip(topk_scores, topk_idx)):
    token_str = model.to_string([idx.item()])
    if token_str not in prompt_text and token_str != current_token_text:
        G_f += 1

print(f"G(f) = {G_f}")

# %% Calculate G(f) scores for ALL latents in the earliest layer
print(f"=== G(f) Scores for All Latents in Layer {earliest_layer} ===")

# Setup
sae2test = saes[earliest_layer]
W_U = model.W_U.float().to(device)
W_dec = sae2test.W_dec.to(device)
prompt_text = model.to_string(inter_toks_BL[0])
current_token_text = model.to_string(out_BL[0, inter_token_id])

print(f"Prompt length: {len(prompt_text)} chars")
print(f"Current token at position {inter_token_id}: '{current_token_text}'")
print(f"Analyzing {len(earliest_layer_latents)} latents in layer {earliest_layer}")
print()

# Calculate G(f) for each latent
latent_scores = []
top_k = 10  # Number of top predictions to consider

for latent_idx in earliest_layer_latents:
    # Get decoder direction for this latent
    dirs = W_dec[latent_idx]  # Shape: [d_model]
    
    # Compute logits via logit lens  
    logits = dirs @ W_U  # Shape: [vocab_size]
    
    # Get top predictions
    topk_scores, topk_idx = torch.topk(logits, top_k, dim=0)
    
    # Calculate G(f) = tokens predicted but not in prompt or current token
    G_f = 0
    planning_tokens = []
    
    for score, idx in zip(topk_scores, topk_idx):
        token_str = model.to_string([idx.item()])
        if token_str not in prompt_text and token_str != current_token_text:
            G_f += 1
            planning_tokens.append((token_str, score.item()))
    
    latent_scores.append({
        'latent_idx': latent_idx,
        'G_f_score': G_f,
        'planning_tokens': planning_tokens,
        'total_predictions': top_k
    })

# Sort by G(f) score (highest first)
latent_scores.sort(key=lambda x: x['G_f_score'], reverse=True)

# Display results
print(f"{'Rank':<4} {'Latent':<8} {'G(f)':<5} {'Planning Tokens'}")
print("-" * 70)

for rank, result in enumerate(latent_scores, 1):
    latent_idx = result['latent_idx']
    G_f_score = result['G_f_score']
    planning_tokens = result['planning_tokens'][:5]  # Show top 5 planning tokens
    
    token_strs = [f"'{t[0]}'" for t in planning_tokens]
    tokens_display = ", ".join(token_strs) if token_strs else "None"
    
    print(f"{rank:<4} {latent_idx:<8} {G_f_score:<5} {tokens_display}")

# Show detailed view of top 3 latents
print(f"\n=== Detailed View: Top 3 Planning Latents ===")
for i, result in enumerate(latent_scores[:3]):
    print(f"\n{i+1}. Latent {result['latent_idx']} (G(f) = {result['G_f_score']})")
    print(f"   Planning tokens (not in context):")
    for token, score in result['planning_tokens']:
        print(f"     '{token}' (score: {score:.3f})")

# Statistics
total_latents = len(latent_scores)
planning_latents = sum(1 for r in latent_scores if r['G_f_score'] > 0)
max_G_f = max(r['G_f_score'] for r in latent_scores)
avg_G_f = sum(r['G_f_score'] for r in latent_scores) / total_latents

print(f"\n=== Summary Statistics ===")
print(f"Total latents analyzed: {total_latents}")
print(f"Latents with planning evidence (G(f) > 0): {planning_latents}")
print(f"Maximum G(f) score: {max_G_f}")
print(f"Average G(f) score: {avg_G_f:.2f}")
print(f"Planning percentage: {planning_latents/total_latents*100:.1f}%")


# %% Function to calculate G-scores for any list of latents
def calculate_g_scores_for_latents(latent_list, layer_idx, model, saes, inter_toks_BL, out_BL, inter_token_id, top_k=10):
    """
    Calculate G(f) scores for a list of latents in a specific layer.
    
    Args:
        latent_list: List of latent indices
        layer_idx: Layer index
        model: The language model
        saes: List of SAE objects
        inter_toks_BL: Input tokens up to prediction position
        out_BL: Full generated sequence
        inter_token_id: Token position being analyzed
        top_k: Number of top predictions to consider
        
    Returns:
        List of dicts with latent analysis results
    """
    # Setup
    sae = saes[layer_idx]
    W_U = model.W_U.float().to(device)
    W_dec = sae.W_dec.to(device)
    prompt_text = model.to_string(inter_toks_BL[0])
    current_token_text = model.to_string(out_BL[0, inter_token_id])
    
    results = []
    
    for latent_idx in latent_list:
        # Get decoder direction for this latent
        dirs = W_dec[latent_idx]  # Shape: [d_model]
        
        # Compute logits via logit lens  
        logits = dirs @ W_U  # Shape: [vocab_size]
        
        # Get top predictions
        topk_scores, topk_idx = torch.topk(logits, top_k, dim=0)
        
        # Store all top 10 tokens directly
        top_tokens = []
        planning_tokens = []
        G_f = 0
        
        for score, idx in zip(topk_scores, topk_idx):
            token_str = model.to_string([idx.item()])
            token_info = {
                'token': token_str,
                'token_id': idx.item(),
                'score': score.item()
            }
            top_tokens.append(token_info)
            
            # Check if it's a planning token (not in current context)
            if token_str not in prompt_text and token_str != current_token_text:
                G_f += 1
                planning_tokens.append(token_info)
        
        results.append({
            'latent_idx': latent_idx,
            'layer_idx': layer_idx,
            'G_f_score': G_f,
            'top_tokens': top_tokens,  # All top 10 tokens with full info
            'planning_tokens': planning_tokens,  # Only planning tokens
            'total_predictions': top_k
        })
    
    return results

# %% Apply G-score analysis to ALL layers from saved_pair_dict['1']
print(f"\n=== G(f) Analysis for ALL Layers with '1' Predictions ===")

# Get all latents that predict '1', grouped by layer (from earlier analysis)
token_1_latents = saved_pair_dict['1']
latents_by_layer = defaultdict(set)
for layer, latent_idx, token_positions in token_1_latents:
    latents_by_layer[layer].add(latent_idx)

print(f"Found '1'-predicting latents in {len(latents_by_layer)} layers")

# Analyze each layer
all_layer_results = {}
for layer_idx in sorted(latents_by_layer.keys()):
    latent_list = list(latents_by_layer[layer_idx])
    print(f"\nAnalyzing Layer {layer_idx}: {len(latent_list)} latents")
    
    # Calculate G-scores for this layer
    layer_results = calculate_g_scores_for_latents(
        latent_list, layer_idx, model, saes, inter_toks_BL, out_BL, inter_token_id
    )
    
    # Sort by G(f) score
    layer_results.sort(key=lambda x: x['G_f_score'], reverse=True)
    all_layer_results[layer_idx] = layer_results
    
    # Show summary for this layer
    planning_count = sum(1 for r in layer_results if r['G_f_score'] > 0)
    max_g = max(r['G_f_score'] for r in layer_results)
    avg_g = sum(r['G_f_score'] for r in layer_results) / len(layer_results)
    
    print(f"  Planning latents: {planning_count}/{len(latent_list)} ({planning_count/len(latent_list)*100:.1f}%)")
    print(f"  Max G(f): {max_g}, Avg G(f): {avg_g:.2f}")
    
    # Show top 3 planning latents for this layer
    top_planners = [r for r in layer_results if r['G_f_score'] > 0][:3]
    for i, result in enumerate(top_planners):
        planning_tokens_str = ", ".join([f"'{t['token']}'" for t in result['planning_tokens'][:3]])
        print(f"    {i+1}. Latent {result['latent_idx']} (G={result['G_f_score']}): {planning_tokens_str}")


# %% Cross-layer comparison and best planning latents
print(f"\n=== Cross-Layer Planning Analysis ===")

# Find best planning latents across all layers
all_planning_latents = []
for layer_idx, layer_results in all_layer_results.items():
    for result in layer_results:
        if result['G_f_score'] > 0:
            all_planning_latents.append(result)

# Sort by G(f) score globally
all_planning_latents.sort(key=lambda x: x['G_f_score'], reverse=True)

print(f"Total planning latents across all layers: {len(all_planning_latents)}")
print(f"\nTop Planning Latents (all layers):")
print(f"{'Rank':<4} {'Layer':<6} {'Latent':<8} {'G(f)':<5} {'Top Planning Tokens'}")
print("-" * 80)

for rank, result in enumerate(all_planning_latents[:20], 1):
    planning_tokens_str = ", ".join([f"'{t['token']}'" for t in result['planning_tokens'][:7]])
    print(f"{rank:<4} {result['layer_idx']:<6} {result['latent_idx']:<8} {result['G_f_score']:<5} {planning_tokens_str}")

# # Layer-wise statistics
# print(f"\nLayer-wise Planning Statistics:")
# for layer_idx in sorted(all_layer_results.keys()):
#     layer_results = all_layer_results[layer_idx]
#     planning_count = sum(1 for r in layer_results if r['G_f_score'] > 0)
#     total_count = len(layer_results)
#     avg_g = sum(r['G_f_score'] for r in layer_results) / total_count
#     print(f"Layer {layer_idx:2}: {planning_count:2}/{total_count:2} planning ({planning_count/total_count*100:5.1f}%), avg G(f)={avg_g:.2f}")

# Save results for further analysis
layer_analysis_results = all_layer_results



# %% Cohesion Analysis - Find Monosemantic Planning Features

# Get embedding matrix
try:
    EMB = model.W_E  # Try this first
    print(f"Using model.W_E: {EMB.shape}")
except:
    EMB = model.embed_tokens.weight  # Fallback
    print(f"Using model.embed_tokens.weight: {EMB.shape}")

def cohesion(topk_ids):
    """Calculate cohesion score for a list of token IDs using cosine similarity."""
    if len(topk_ids) < 2:
        return 0.0
    vecs = EMB[topk_ids]                              # [K, d_model] 
    μ = vecs.mean(dim=0, keepdim=True)               # [1, d_model] - centroid
    sim = torch.nn.functional.cosine_similarity(vecs, μ, dim=1)  # [K] - similarity to centroid
    return sim.mean().item()                         # Average similarity (1 = perfectly cohesive)

COH_THR = 0.6  # Threshold for "cohesive" features

print(f"Calculating cohesion for all planning latents...")

# Add cohesion scores to all planning latents
cohesive_planning_latents = []
for result in all_planning_latents:
    # Get token IDs from top 10 predictions
    top_token_ids = torch.tensor([t['token_id'] for t in result['top_tokens']])
    
    # Calculate cohesion
    coh_score = cohesion(top_token_ids)
    
    # Add cohesion to result
    result_with_cohesion = result.copy()
    result_with_cohesion['cohesion_score'] = coh_score
    result_with_cohesion['is_cohesive'] = coh_score >= COH_THR
    
    cohesive_planning_latents.append(result_with_cohesion)

# Sort by cohesion score (highest first)
cohesive_planning_latents.sort(key=lambda x: x['cohesion_score'], reverse=True)

# Show results with both G(f) and cohesion scores
print(f"\n=== Planning Latents with G(f) and Cohesion Scores ===")
print(f"{'Rank':<4} {'Layer':<6} {'Latent':<8} {'G(f)':<5} {'Cohesion':<9} {'Cohesive':<9} {'Top Tokens'}")
print("-" * 100)

for rank, result in enumerate(cohesive_planning_latents[:25], 1):
    top_tokens_str = ", ".join([f"'{t['token']}'" for t in result['top_tokens'][:5]])
    cohesive_marker = "✅" if result['is_cohesive'] else "❌"
    
    print(f"{rank:<4} {result['layer_idx']:<6} {result['latent_idx']:<8} {result['G_f_score']:<5} "
          f"{result['cohesion_score']:.3f}{'':>4} {cohesive_marker:<9} {top_tokens_str}")

# Filter for high cohesion features
high_cohesion_planners = [r for r in cohesive_planning_latents if r['is_cohesive']]
print(f"\n=== High Cohesion Planning Features (cohesion >= {COH_THR}) ===")
print(f"Found {len(high_cohesion_planners)} cohesive planning latents out of {len(cohesive_planning_latents)} total")


# %%
if high_cohesion_planners:
    # Sort by combination of G(f) and cohesion
    high_cohesion_planners.sort(key=lambda x: 0.03*x['G_f_score'] + x['cohesion_score'], reverse=True)
    
    print(f"\nTop 10 Cohesive Planning Features (ranked by 0.03G(f) + Cohesion):")
    print(f"{'Rank':<4} {'Layer':<6} {'Latent':<8} {'G(f)':<5} {'Cohesion':<9} {'G×C':<8} {'Top Tokens'}")
    print("-" * 100)
    
    for rank, result in enumerate(high_cohesion_planners[:10], 1):
        top_tokens_str = ", ".join([f"'{t['token']}'" for t in result['top_tokens'][:6]])
        combined_score = 0.03*result['G_f_score'] + result['cohesion_score']
        
        print(f"{rank:<4} {result['layer_idx']:<6} {result['latent_idx']:<8} {result['G_f_score']:<5} "
              f"{result['cohesion_score']:.3f}{'':>4} {combined_score:.2f}{'':>3} {top_tokens_str}")
    
    # Detailed view of top 3 cohesive planners
    print(f"\n=== Detailed View: Top 10 Cohesive Planning Features ===")
    for i, result in enumerate(high_cohesion_planners[:10]):
        print(f"\n{i+1}. Layer {result['layer_idx']}, Latent {result['latent_idx']}")
        print(f"   G(f) Score: {result['G_f_score']}, Cohesion: {result['cohesion_score']:.3f}")
        print(f"   All top 10 tokens:")
        for j, token_info in enumerate(result['top_tokens']):
            print(f"     {j+1:2}. '{token_info['token']}' (score: {token_info['score']:.3f})")
        print(f"   Planning tokens (not in context):")
        for token_info in result['planning_tokens']:
            print(f"     • '{token_info['token']}' (score: {token_info['score']:.3f})")

# Statistics
total_planners = len(cohesive_planning_latents)
cohesive_count = len(high_cohesion_planners)
avg_cohesion = sum(r['cohesion_score'] for r in cohesive_planning_latents) / total_planners

print(f"\n=== Cohesion Statistics ===")
print(f"Total planning latents: {total_planners}")
print(f"Cohesive planning latents (>= {COH_THR}): {cohesive_count} ({cohesive_count/total_planners*100:.1f}%)")
print(f"Average cohesion score: {avg_cohesion:.3f}")

# Save cohesive results
cohesive_results = {
    'all_results': cohesive_planning_latents,
    'high_cohesion': high_cohesion_planners,
    'threshold': COH_THR,
    'stats': {
        'total_planners': total_planners,
        'cohesive_count': cohesive_count,
        'avg_cohesion': avg_cohesion
    }
}

# %%
