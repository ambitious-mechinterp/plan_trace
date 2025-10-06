"""
Logit lens clustering for grouping SAE latents by their decoding directions or
by checking Neuronpedia top activations.

Shape Suffix Definition: 
- B: batch size 
- L: Num of Input Tokens 
- V: vocabulary size
- S: Number of SAE neurons in a layer
"""

import json
from collections import defaultdict
from typing import Any, Dict, List, MutableMapping, Sequence, Tuple, Union

import requests
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .utils import cleanup_cuda

API_URL = "https://www.neuronpedia.org/api/activation/get"
DEFAULT_NEURONPEDIA_MODEL = "gemma-2-2b"
DEFAULT_NEURONPEDIA_SOURCE = "{layer}-gemmascope-mlp-16k"
DEFAULT_API_TOPK = 20
API_TIMEOUT = 30.0

_NEURONPEDIA_CACHE: Dict[Tuple[str, int, int, str, int], List[Dict[str, Any]]] = {}

def _resolve_neuronpedia_model_id(model, api_model: str | None) -> str:
    """
    Resolve the appropriate Neuronpedia model identifier.

    Many HuggingFace identifiers include instruction-tuning suffixes (e.g.,
    'gemma-2-2b-it') that are not valid on Neuronpedia. This function
    normalizes known variants to the expected Neuronpedia model id.
    """
    if api_model:
        return api_model

    candidate = getattr(getattr(model, "cfg", object), "model_name", DEFAULT_NEURONPEDIA_MODEL)
    lower = str(candidate).lower()

    # Normalize common Gemma variants
    if "gemma-2-2b" in lower:
        return "gemma-2-2b"

    # Fallback to provided candidate as-is
    return candidate


def _tokens_to_text(tokens: Sequence[str]) -> str:
    return "".join(token.replace("▁", " ").replace("<0x0A>", "\n") for token in tokens)


def _has_boundary_match(hay: str, needle: str) -> bool:
    if not needle:
        return False
    if any(ch.isalnum() for ch in needle):
        idx = 0
        hay_len = len(hay)
        needle_len = len(needle)
        while True:
            pos = hay.find(needle, idx)
            if pos == -1:
                return False
            left_ok = pos == 0 or not hay[pos - 1].isalnum()
            right_ok = (pos + needle_len) >= hay_len or not hay[pos + needle_len].isalnum()
            if left_ok and right_ok:
                return True
            idx = pos + 1
    return needle in hay


def _normalize_contexts(response: Any) -> List[Dict[str, Any]]:
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ValueError("Neuronpedia response string could not be parsed as JSON.") from exc

    if isinstance(response, list):
        return [item for item in response if isinstance(item, dict)]

    if isinstance(response, dict):
        for key in ("contexts", "data", "records"):
            candidates = response.get(key)
            if isinstance(candidates, list):
                return [item for item in candidates if isinstance(item, dict)]

    raise ValueError("Unexpected Neuronpedia response format; expected list of context dicts.")


def _fetch_neuronpedia_contexts(
    *,
    latent_index: int,
    layer_index: int,
    model_id: str,
    release_format: str,
    top_k: int,
    cache: MutableMapping[Tuple[str, int, int, str, int], List[Dict[str, Any]]],
    timeout: float,
) -> List[Dict[str, Any]]:
    cache_key = (model_id, layer_index, latent_index, release_format, top_k)
    if cache_key in cache:
        return cache[cache_key]

    source = release_format.format(layer=layer_index)
    payload = {"modelId": model_id, "source": source, "index": str(latent_index)}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to fetch Neuronpedia info for latent {latent_index} (layer {layer_index}): {exc}") from exc

    contexts = _normalize_contexts(data)
    cache[cache_key] = contexts[:top_k] if top_k > 0 else contexts
    return cache[cache_key]


def gather_unique_tokens(
    model,
    prompt: Union[str, torch.Tensor],
    *,
    stop_tok: int,
    device: str = "cuda",
) -> List[int]:
    """
    Generate from prompt until stop_tok and return unique token IDs encountered.
    
    Args:
        model: The language model
        prompt: Input prompt string or token tensor [1, L]
        stop_tok: Token ID to stop generation at
        device: Device for computation
        
    Returns:
        List of unique token IDs encountered during generation
    """
    if isinstance(prompt, str):
        toks = model.to_tokens(prompt).to(device)
    else:
        toks = prompt.to(device)
    out = toks.clone()
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


def build_saved_pair_dict_fastest(
    model,
    saes: List,
    trial_entries: Sequence[Tuple[int, int, int, float]],
    unique_token_ids: Sequence[int],
    *,
    tok_k_pos_logits: int = 15,
    batch_size: int = 4096,
    device: str = "cuda",
    # Scoring and thresholding controls
    score_threshold: float | None = None,
    count_weight: float = 0.03,
    min_match_count: int = 1,
) -> Dict[str, List[Tuple[int, int, List[int]]]]:
    """
    Ultra-fast saved_pair_dict building using vectorized GPU operations.
    
    Maps each SAE latent's decoding direction to tokens it most strongly predicts,
    then groups trial entries by these predicted tokens.
    
    Args:
        model: The language model (for W_U and tokenizer)
        saes: List of SAE objects
        trial_entries: Circuit entries [(layer, token_pos, latent_idx, effect_value)]
        unique_token_ids: Token IDs to check for matches
        tok_k_pos_logits: Number of top logits to consider per latent
        batch_size: Batch size for processing latents
        device: Device for computation
        
    Returns:
        Dict mapping token_string -> [(layer, latent_idx, [token_positions])]
    """
    layer_to_latents: Dict[int, set[int]] = defaultdict(set)
    for l, t, lat, _ in trial_entries:
        layer_to_latents[l].add(lat)

    unique_token_tensor = torch.tensor(unique_token_ids, device=device)
    saved_pair_dict: Dict[str, List[Tuple[int, int, List[int]]]] = defaultdict(list)

    with torch.no_grad():
        W_U = model.W_U.float().to(device)  # [D_model, V]

        for layer_i, latents in tqdm(layer_to_latents.items()):
            latents = sorted(latents)
            W_dec = saes[layer_i].W_dec.to(device)  # [S, D_model]

            for start in range(0, len(latents), batch_size):
                batch = latents[start : start + batch_size]
                dirs = W_dec[batch]            # [batch_size, D_model]

                logits = dirs @ W_U            # [batch_size, V]
                topk_scores, topk_idx = torch.topk(logits, tok_k_pos_logits, dim=1)  # [batch_size, tok_k_pos_logits]

                # GPU intersection: find which topk tokens match unique_token_ids
                # broadcast: [batch_size, tok_k_pos_logits] vs [1, num_unique_ids]
                topk_idx_exp = topk_idx.unsqueeze(-1)                   # [batch, tok_k_pos_logits, 1]
                unique_tok_exp = unique_token_tensor.view(1, 1, -1)     # [1, 1, num_unique_ids]

                matches = (topk_idx_exp == unique_tok_exp).any(-1)      # [batch, tok_k_pos_logits] -> True/False
                matching_token_ids = topk_idx[matches]                  # flatten matches

                # Compute per-latent cohesion over top-k token embeddings
                # Shape: [batch_size, tok_k_pos_logits, d_model]
                token_embs = model.W_E.float().to(device)[topk_idx]
                centroid = token_embs.mean(dim=1, keepdim=True)  # [batch_size, 1, d_model]
                cos_per_token = F.cosine_similarity(token_embs, centroid, dim=-1)  # [batch_size, tok_k_pos_logits]
                cohesion_per_latent = cos_per_token.mean(dim=1)  # [batch_size]

                # Count how many of the top-k are in the unique set (per latent)
                match_counts = matches.sum(dim=1)  # [batch_size]

                # Final score = cohesion + count_weight * count
                final_scores = cohesion_per_latent + count_weight * match_counts.float()

                if matching_token_ids.numel() > 0:
                    matched_latents, matched_topk = torch.nonzero(matches, as_tuple=True)
                    for latent_batch_idx, topk_pos in zip(matched_latents.tolist(), matched_topk.tolist()):
                        # Thresholding: optionally require min matches and score threshold
                        if score_threshold is not None:
                            if match_counts[latent_batch_idx].item() < min_match_count:
                                continue
                            if final_scores[latent_batch_idx].item() < score_threshold:
                                continue
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


def build_saved_pair_dict_neuronpedia(
    model,
    trial_entries: Sequence[Tuple[int, int, int, float]],
    unique_token_ids: Sequence[int],
    *,
    top_contexts: int,
    model_id: str,
    release_format: str,
    cache: MutableMapping[Tuple[str, int, int, str, int], List[Dict[str, Any]]],
    timeout: float,
) -> Dict[str, List[Tuple[int, int, List[int]]]]:
    layer_to_latents: Dict[int, set[int]] = defaultdict(set)
    for layer_idx, _, latent_idx, _ in trial_entries:
        layer_to_latents[layer_idx].add(latent_idx)

    token_labels: Dict[int, str] = {}
    for tok_id in unique_token_ids:
        label = model.to_string(tok_id)
        if not isinstance(label, str):
            continue
        label = label.strip()
        if label:
            token_labels[tok_id] = label

    if not token_labels:
        return {}

    # Precompute position buckets for faster lookup later
    entry_positions: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for layer_idx, token_pos, latent_idx, _ in trial_entries:
        entry_positions[(layer_idx, latent_idx)].append(token_pos)

    saved_pair_dict: Dict[str, List[Tuple[int, int, List[int]]]] = defaultdict(list)

    for layer_idx, latents in layer_to_latents.items():
        for latent_idx in sorted(latents):
            try:
                contexts = _fetch_neuronpedia_contexts(
                    latent_index=latent_idx,
                    layer_index=layer_idx,
                    model_id=model_id,
                    release_format=release_format,
                    top_k=top_contexts,
                    cache=cache,
                    timeout=timeout,
                )
            except RuntimeError as exc:
                # Surface the issue but continue with other latents
                print(exc)
                continue

            context_texts: List[str] = []
            for entry in contexts:
                tokens = entry.get("tokens") if isinstance(entry, dict) else None
                if not isinstance(tokens, list) or not tokens:
                    continue
                text = _tokens_to_text(tokens)
                if text:
                    # Normalize whitespace for easier substring matches
                    normalized = " ".join(text.replace("\r", " ").replace("\n", " ").split())
                    context_texts.append(normalized)

            if not context_texts:
                continue

            for tok_id, label in token_labels.items():
                if any(_has_boundary_match(text, label) for text in context_texts):
                    positions = entry_positions.get((layer_idx, latent_idx), [])
                    saved_pair_dict[label].append((layer_idx, latent_idx, positions))
                    break

    return dict(saved_pair_dict)


def find_logit_lens_clusters(
    model, 
    saes: List, 
    entries: List[Tuple[int, int, int, float]], 
    inter_toks_BL: torch.Tensor, 
    stop_tok: int, 
    verbose: bool = True,
    # Optional monosemantic filtering controls (see build_saved_pair_dict_fastest)
    score_threshold: float | None = None,
    count_weight: float = 1,
    min_match_count: int = 1,
    *,
    mode: str = "logit_lens",
    api_model: str | None = None,
    api_source: str = DEFAULT_NEURONPEDIA_SOURCE,
    api_topk: int = DEFAULT_API_TOPK,
    api_cache: MutableMapping[Tuple[str, int, int, str, int], List[Dict[str, Any]]] | None = None,
    api_timeout: float = API_TIMEOUT,
) -> Dict[str, List[Tuple[int, int, List[int]]]]:
    """
    Find clusters of SAE latents based on their decoding directions or Neuronpedia top contexts.

    Mode ``logit_lens`` (default) reproduces the previous cosine-similarity heuristic.
    Mode ``neuronpedia_topk`` performs API lookups for each latent and checks whether
    any of the top-k activation contexts contain candidate tokens.
    """
    # Generate unique tokens not in the original prompt
    uniq_ids = gather_unique_tokens(model, inter_toks_BL, stop_tok=stop_tok)
    prompt_tokens = inter_toks_BL[0]
    prompt_id_set = set(prompt_tokens.detach().cpu().tolist())
    prompt_str = model.to_string(prompt_tokens)

    def token_in_prompt(tok_id: int) -> bool:
        if tok_id in prompt_id_set:
            return True
        label = model.to_string(tok_id).strip()
        return bool(label) and _has_boundary_match(prompt_str, label)

    filtered_uniq_ids = [tok for tok in uniq_ids if not token_in_prompt(tok)]

    if verbose:
        print(
            "Found {} tokens not in prompt: {}".format(
                len(filtered_uniq_ids), [model.to_string(tok) for tok in filtered_uniq_ids]
            )
        )

    if mode == "logit_lens":
        saved_pair_dict = build_saved_pair_dict_fastest(
            model,
            saes,
            entries,
            filtered_uniq_ids,
            tok_k_pos_logits=15,
            batch_size=4096,
            score_threshold=score_threshold,
            count_weight=count_weight,
            min_match_count=min_match_count,
        )
    elif mode == "neuronpedia_topk":
        cache = api_cache if api_cache is not None else _NEURONPEDIA_CACHE
        resolved_model = _resolve_neuronpedia_model_id(model, api_model)
        saved_pair_dict = build_saved_pair_dict_neuronpedia(
            model,
            entries,
            filtered_uniq_ids,
            top_contexts=api_topk,
            model_id=resolved_model,
            release_format=api_source,
            cache=cache,
            timeout=api_timeout,
        )
    else:
        raise ValueError(f"Unknown clustering mode '{mode}'.")

    return saved_pair_dict
