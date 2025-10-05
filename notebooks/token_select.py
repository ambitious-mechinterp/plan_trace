# %%
"""
1. Select latent under investigation L, and token X 
2. Sample 1000 random latents from width of 16k 
3. Collect top acts and sequences from the 1000
4. Filter sequences that have and dont have token X 
5. Make a function that generates activation vector for a given sae and sequence
6. Do auroc on the pos and negative set 
"""

# %%

import requests
import json 
import os 
import sys 
import math
import html
import numpy as np
import torch 
from torch import Tensor
from typing import Dict, Sequence, Tuple, Union
from IPython.display import HTML, display

sys.path.append("../")

from plan_trace.utils import load_model, load_pretrained_saes, cleanup_cuda
from plan_trace.hooks import run_with_saes, register_sae_hooks
from sae_lens import HookedSAETransformer, SAE

# %% Setup 

model_name = "gemma-2-2b"
device = "cuda"
model = load_model(model_name, device=device, use_custom_cache=True, dtype=torch.bfloat16)
layers = list(range(model.cfg.n_layers))
saes = load_pretrained_saes(
    layers=layers, 
    release="gemma-scope-2b-pt-mlp-canonical", 
    width="16k", 
    device=device, 
    canon=True
)

url = "https://www.neuronpedia.org/api/activation/get" 

# %% 
def generate_sae_act(
    model: HookedSAETransformer,
    saes: Sequence[SAE],
    input_sequence: Union[str, Sequence[str], Tensor],
    device: Union[str, torch.device] = device,
) -> Dict[str, Tensor]:
    if isinstance(input_sequence, Tensor):
        tokens = input_sequence.to(device)
    else:
        tokens = model.to_tokens(input_sequence).to(device)
    model.reset_hooks(including_permanent=True)
    with torch.no_grad():
        run_with_saes(model, saes, tokens, cache_sae_activations=True)
    return {sae.cfg.hook_name: sae.feature_acts for sae in saes}

def get_neuronpedia_info(latent_index, layer_index, model_name="gemma-2-2b", release_format="{layer}-gemmascope-mlp-16k", url=url):
    source = release_format.format(layer=layer_index)
    payload = {
        "modelId": model_name,
        "source": source,
        "index": str(latent_index),
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to fetch Neuronpedia info: {exc}") from exc

    if isinstance(data, dict):
        status = str(data.get("status", "")).lower()
        if status and status not in {"ok", "success"}:
            message = data.get("message") or data
            raise RuntimeError(f"Neuronpedia returned status '{status}' for payload {payload}: {message}")
    return data


def neuronpedia_json_to_latent_activations(
    neuronpedia_json,
    latent_spec: Dict[str, int],
    model: HookedSAETransformer,
    saes: Sequence[SAE],
    device: Union[str, torch.device] = device,
) -> Tuple[Sequence[str], Tensor]:
    description_tokens = neuronpedia_json[0]["tokens"]
    description_text = "".join(
        token.replace("▁", " ").replace("<0x0A>", "\n") for token in description_tokens
    )
    if not description_text:
        description_text = "".join(description_tokens)

    tokens = model.to_tokens(description_text).to(device)
    sae_for_layer = next((sae for sae in saes if sae.cfg.hook_layer == latent_spec["layer"]), None)
    if sae_for_layer is None:
        raise ValueError(f"Unable to locate SAE for layer {latent_spec['layer']}")

    layer_acts = generate_sae_act(model, [sae_for_layer], tokens)[sae_for_layer.cfg.hook_name][0]
    latent_index = latent_spec["latent"]
    if latent_index >= layer_acts.shape[-1]:
        raise ValueError(f"Latent index {latent_index} exceeds SAE width {layer_acts.shape[-1]}")

    latent_activations = layer_acts[:, latent_index].detach().float().cpu()
    str_tokens = model.to_str_tokens(description_text)
    seq_len = min(len(str_tokens), latent_activations.shape[0])
    return str_tokens[:seq_len], latent_activations[:seq_len]


def render_activation_heatmap_html(tokens: Sequence[str], activations: Tensor) -> str:
    activations_tensor = torch.as_tensor(activations, dtype=torch.float32)
    if activations_tensor.ndim == 0:
        activations_tensor = activations_tensor.unsqueeze(0)

    seq_len = min(len(tokens), activations_tensor.shape[0])
    tokens = tokens[:seq_len]
    activations_tensor = activations_tensor[:seq_len]

    max_abs_activation = float(activations_tensor.abs().max().item())
    if max_abs_activation == 0.0:
        max_abs_activation = 1e-6

    token_spans = []
    for token, activation in zip(tokens, activations_tensor):
        display_token = token.replace("▁", " ").replace("<0x0A>", "\n")
        escaped_token = html.escape(display_token).replace("\n", "<br/>").replace(" ", "&nbsp;")
        if not escaped_token:
            escaped_token = "&nbsp;"
        value = float(activation.item())
        normalized = max(-1.0, min(1.0, value / max_abs_activation))
        opacity = abs(normalized)
        color = "rgba(255, 87, 51, {:.4f})".format(opacity) if normalized >= 0 else "rgba(66, 135, 245, {:.4f})".format(opacity)
        tooltip = html.escape(f"{value:.4f}")
        token_spans.append(
            f"<span title=\"{tooltip}\" style=\"background:{color}; padding:2px 4px; margin:1px; display:inline-block; border-radius:4px; font-family:monospace;\">{escaped_token}</span>"
        )

    return "<div style='display:flex; flex-wrap:wrap; align-items:flex-start;'>" + "".join(token_spans) + "</div>"


def top_percent_mean(values: Tensor, percent: float = 0.01) -> float:
    if percent <= 0:
        raise ValueError("percent must be positive")

    flattened = torch.as_tensor(values, dtype=torch.float32).flatten()
    numel = flattened.numel()
    if numel == 0:
        return float("nan")

    k = max(1, int(math.ceil(numel * percent)))
    topk_vals, _ = torch.topk(flattened, k)
    return float(topk_vals.mean().item())


# %% Latent of interest 
latentL = {"layer":19, "latent":15238}
stringX = "cat"
tokenX = model.to_tokens(stringX)[:, -1]

L_desp = get_neuronpedia_info(latentL["latent"], latentL["layer"])
print("".join([txt.replace("▁", " ") for txt in L_desp[0]['tokens']]))

# %%
tokens_with_str, latent_activation_vector = neuronpedia_json_to_latent_activations(
    L_desp, latentL, model, saes, device=device
)
heatmap_html = render_activation_heatmap_html(tokens_with_str, latent_activation_vector)
display(HTML(heatmap_html))
print(f"Top 1% mean activation: {top_percent_mean(latent_activation_vector):.4f}")

# %%
rng = np.random.default_rng()
sampled_latents = rng.choice(16384, size=1000, replace=False).astype(np.int32)
sampled_latents[sampled_latents >= latentL["latent"]] += 1
sampled_latents = sampled_latents.tolist()


# %%
from concurrent.futures import ThreadPoolExecutor, as_completed


def _fetch_latent_info(latent_index: int):
    return latent_index, get_neuronpedia_info(latent_index, latentL["layer"])


max_workers = min(32, (os.cpu_count() or 4) * 2)
sampled_latent_infos = {}
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_map = {executor.submit(_fetch_latent_info, idx): idx for idx in sampled_latents}
    for future in as_completed(future_map):
        latent_index, info = future.result()
        sampled_latent_infos[latent_index] = info



# %%
def tokens_to_string(tokens):
    return "".join(
        token.replace("▁", " ").replace("<0x0A>", "\n") for token in tokens
    ).strip()


positive_sequences = []
negative_sequences = []
target_substring = stringX

for latent_index in sampled_latents:
    contexts = sampled_latent_infos.get(latent_index, []) or []
    for context in contexts:
        tokens = context.get("tokens") if isinstance(context, dict) else None
        if not tokens:
            continue
        text_sequence = tokens_to_string(tokens)
        if not text_sequence:
            continue
        record = {"latent": latent_index, "sequence": text_sequence}
        if target_substring in text_sequence:
            positive_sequences.append(record)
        else:
            negative_sequences.append(record)


def _print_examples(label, sequences):
    print(f"{label} examples (showing up to 5):")
    for example in sequences[:5]:
        print(f"[{example['latent']}] {example['sequence']}")
    if not sequences:
        print("None")


_print_examples("Positive", positive_sequences)
_print_examples("Negative", negative_sequences)


# %%
from collections import defaultdict


def _sample_negative_sequences(records, target_count, rng_inst):
    if target_count <= 0:
        return []
    by_latent = defaultdict(list)
    for record in records:
        by_latent[record["latent"]].append(record["sequence"])
    latent_ids = list(by_latent.keys())
    rng_inst.shuffle(latent_ids)
    for seqs in by_latent.values():
        rng_inst.shuffle(seqs)

    sampled = []
    active_latents = latent_ids
    while active_latents and len(sampled) < target_count:
        next_round = []
        for latent_id in active_latents:
            bucket = by_latent[latent_id]
            if not bucket:
                continue
            sampled.append({"latent": latent_id, "sequence": bucket.pop()})
            if len(sampled) >= target_count:
                break
            if bucket:
                next_round.append(latent_id)
        active_latents = next_round
    return sampled


def _sequence_top_percent_mean(sequence_text, sae_model, latent_index):
    tokens = model.to_tokens(sequence_text).to(device)
    acts = generate_sae_act(model, [sae_model], tokens)[sae_model.cfg.hook_name][0]
    latent_vector = acts[:, latent_index]
    return top_percent_mean(latent_vector)


def compute_auroc(pos_scores, neg_scores):
    pos_scores = np.asarray(pos_scores, dtype=np.float64)
    neg_scores = np.asarray(neg_scores, dtype=np.float64)
    if pos_scores.size == 0 or neg_scores.size == 0:
        return float("nan")
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    order = np.argsort(scores)
    ranked_labels = labels[order]
    ranks = np.arange(1, scores.size + 1, dtype=np.float64)
    pos_rank_sum = ranks[ranked_labels == 1].sum()
    pos_count = float(pos_scores.size)
    neg_count = float(neg_scores.size)
    return (pos_rank_sum - pos_count * (pos_count + 1) / 2.0) / (pos_count * neg_count)


if not positive_sequences:
    print("No positive sequences available; skipping AUROC computation.")
else:
    target_negative_count = min(len(negative_sequences), len(positive_sequences) * 3)
    sampled_negative_sequences = _sample_negative_sequences(
        negative_sequences, target_negative_count, rng
    )

    if not sampled_negative_sequences:
        print("No negative sequences sampled; cannot compute AUROC.")
    else:
        sae_for_layer = next(
            (sae for sae in saes if sae.cfg.hook_layer == latentL["layer"]),
            None,
        )
        if sae_for_layer is None:
            raise ValueError(f"Unable to locate SAE for layer {latentL['layer']}")

        positive_scores = []
        for record in positive_sequences:
            score = _sequence_top_percent_mean(
                record["sequence"], sae_for_layer, latentL["latent"]
            )
            positive_scores.append(score)

        negative_scores = []
        for record in sampled_negative_sequences:
            score = _sequence_top_percent_mean(
                record["sequence"], sae_for_layer, latentL["latent"]
            )
            negative_scores.append(score)

        positive_scores_arr = np.asarray(positive_scores, dtype=np.float64)
        negative_scores_arr = np.asarray(negative_scores, dtype=np.float64)

        auroc = compute_auroc(positive_scores_arr, negative_scores_arr)
        print(f"Positive sequences: {len(positive_scores_arr)}")
        print(f"Sampled negative sequences: {len(negative_scores_arr)}")
        print(
            f"Top 1% mean — positive mean: {positive_scores_arr.mean():.4f}, "
            f"negative mean: {negative_scores_arr.mean():.4f}"
        )
        print(f"AUROC: {auroc:.4f}")

# %%

records = []

for latent_index in sampled_latents:
    contexts = sampled_latent_infos.get(latent_index, []) or []
    for context in contexts:
        tokens = context.get("tokens") if isinstance(context, dict) else None
        if not tokens:
            continue
        text_sequence = tokens_to_string(tokens)
        if not text_sequence:
            continue
        record = {"latent": latent_index, "sequence": text_sequence}
        records.append(record)

# Build one giant text blob from all sequences
large_text = "\n".join(r["sequence"] for r in records) if records else ""

print(f"collected {len(records)} sequences; large_text length = {len(large_text)} chars")
import re
from collections import Counter

tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", large_text.lower())
stop = {"the","a","an","and","or","but","if","then","else","for","to","in","of","on",
        "at","by","with","from","as","is","are","was","were","be","been","being",
        "this","that","these","those","it","its","i","you","he","she","we","they",
        "my","your","his","her","our","their","me","him","her","us","them"}
tokens = [t for t in tokens if t not in stop and len(t) >= 3]
freqs = Counter(tokens)
# %% 

for w, c in freqs.most_common(50):
    print(w, c)



# %%
loi_seq_list = []
for i in range(len(L_desp)):
    loi_seq_list.append(tokens_to_string(L_desp[i]["tokens"]))
loi_seq = "\n".join(loi_seq_list)

tokens_loi = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", loi_seq.lower())
stop = {"the","a","an","and","or","but","if","then","else","for","to","in","of","on",
        "at","by","with","from","as","is","are","was","were","be","been","being",
        "this","that","these","those","it","its","i","you","he","she","we","they",
        "my","your","his","her","our","their","me","him","her","us","them"}
tokens_l = [t for t in tokens_loi if t not in stop and len(t) >= 3]
freqs_l = Counter(tokens_l)
# %% 

TOP_K = 1000
top_k_global_words = {w for w, _ in freqs.most_common(TOP_K)}
print(f"\nWords in L_desp not in top {TOP_K} global words:")
for w, c in freqs_l.most_common(100):
    if w not in top_k_global_words:
        print(w, c)
# %%

# %% AUROC per candidate word (re-filter per word with leading space)
from tqdm import tqdm

sae_for_layer = next(
    (sae for sae in saes if sae.cfg.hook_layer == latentL["layer"]),
    None,
)
if sae_for_layer is None:
    raise ValueError(f"Unable to locate SAE for layer {latentL['layer']}")

word_candidates = [
    w for w, _ in freqs_l.most_common(100) if w not in top_k_global_words
]

def _safe_auroc(value):
    try:
        v = float(value)
        return -1.0 if math.isnan(v) else v
    except Exception:
        return -1.0

auroc_results = []
for w in tqdm(word_candidates):
    target_substring = " " + w

    pos_records = []
    neg_records = []
    for rec in records:
        seq = rec.get("sequence", "")
        if not seq:
            continue
        seq_lower = seq.lower()
        if target_substring in seq_lower:
            pos_records.append(rec)
        else:
            neg_records.append(rec)

    if not pos_records or not neg_records:
        continue

    # Subsample positives to a max of 100 diverse examples
    target_positive_count = min(len(pos_records), 100)
    sampled_pos_records = _sample_negative_sequences(pos_records, target_positive_count, rng)
    if not sampled_pos_records:
        continue

    # Set negatives to up to 3x sampled positives
    target_negative_count = min(len(neg_records), len(sampled_pos_records) * 3)
    sampled_neg_records = _sample_negative_sequences(neg_records, target_negative_count, rng)
    if not sampled_neg_records:
        continue

    positive_scores = []
    for rec in sampled_pos_records:
        score = _sequence_top_percent_mean(rec["sequence"], sae_for_layer, latentL["latent"])
        positive_scores.append(score)

    negative_scores = []
    for rec in sampled_neg_records:
        score = _sequence_top_percent_mean(rec["sequence"], sae_for_layer, latentL["latent"])
        negative_scores.append(score)

    auroc = compute_auroc(positive_scores, negative_scores)
    auroc_results.append(
        {
            "word": w,
            "auroc": float(auroc),
            "pos": len(positive_scores),
            "neg": len(negative_scores),
        }
    )

auroc_results.sort(key=lambda x: _safe_auroc(x["auroc"]), reverse=True)
print("\nAUROC per candidate word (leading-space match):")
for item in auroc_results[:50]:
    print(f"{item['word']}: AUROC={item['auroc']:.4f} (pos={item['pos']}, neg={item['neg']})")
# %%
auroc_results.sort(key=lambda x: _safe_auroc(x["auroc"]), reverse=True)
print("\nAUROC per candidate word (leading-space match):")
for item in auroc_results[:50]:
    print(f"{item['word']}: AUROC={item['auroc']:.4f} (pos={item['pos']}, neg={item['neg']})")
# %%
