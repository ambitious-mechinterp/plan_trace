# for a given prompt, 
# we want to iterate through each forward pass
# discover circuit with >60% perf recovery
# cluster based on decoding directions logit lens 
# test steering effect of clusters
# test effect of each latent / position in cluster

import torch
import torch.nn.functional as F
from helpers.utils import (
    load_model,
    load_pretrained_saes,
)
from circuit_disc import discover_circuit, compute_metric
from plan_criteria import get_ll_clusters, test_steering

device = "cuda"
model_name = "gemma-2-2b-it"
model = load_model(model_name, device=device, use_custom_cache=False, dtype=torch.bfloat16)

layers = list(range(model.cfg.n_layers))
saes = load_pretrained_saes(layers=layers, release="gemma-scope-2b-pt-mlp-canonical", width="16k", device=device, canon=True)

# Set parameters for the generation
max_generation_length = 150
STOP_TOK_ID = 1917
COEFF_GRID = list(range(-100, 0, 20))
ig_steps = 10  

prompt = ""
toks = model.to_tokens(prompt).to(device)
changable_toks = toks.clone()


while True:
    with torch.no_grad():
        logits_BLV = model(changable_toks)

    logits_BV = logits_BLV[:, -1, :]
    probs_BV = F.softmax(logits_BV, dim=-1)
    label = torch.argmax(logits_BV).item()
    clean_probs_baseline = probs_BV[0, label]
    if model.to_string(label) == "<end_of_turn>" or model.to_string(label) == "<eos>" or label == STOP_TOK_ID:
        break
    if changable_toks.shape[-1] - toks.shape[-1] > max_generation_length:
        break
    prompt_current = model.to_string(changable_toks[0])

    # (layer, latent, tok, effect)
    circuit = discover_circuit(model,
                               saes,
                               changable_toks,
                               label=label,
                               device=device,
                               ig_steps=ig_steps,
                               k_max=7001,
                                k_step=500,
                                k_thres=0.6,) 

    # {ym : (layer, latent, tok) 
    logit_lens_cluster = get_ll_clusters()

    steered_clusters = test_steering()

    planning_positions = test_steering()


    changable_toks = torch.cat([changable_toks, torch.tensor([[label]], device=device)], dim=1)
