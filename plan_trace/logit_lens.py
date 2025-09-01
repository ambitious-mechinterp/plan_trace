"""
Logit lens clustering for grouping SAE latents by their decoding directions.

Shape Suffix Definition: 
- B: batch size 
- L: Num of Input Tokens 
- V: vocabulary size
- S: Number of SAE neurons in a layer
"""

import torch
from tqdm import tqdm
from collections import defaultdict
from typing import List, Tuple, Dict, Union, Sequence
import torch.nn.functional as F
from .utils import cleanup_cuda


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


def find_logit_lens_clusters(
    model, 
    saes: List, 
    entries: List[Tuple[int, int, int, float]], 
    inter_toks_BL: torch.Tensor, 
    stop_tok: int, 
    verbose: bool = True,
    # Optional monosemantic filtering controls (see build_saved_pair_dict_fastest)
    score_threshold: float | None = None,
    count_weight: float = 0.1,
    min_match_count: int = 1,
) -> Dict[str, List[Tuple[int, int, List[int]]]]:
    """
    Find clusters of SAE latents based on their logit lens decoding directions.
    
    This function:
    1. Generates from the input tokens to get unique continuation tokens
    2. Filters out tokens already in the prompt
    3. Maps SAE latents to the tokens they most strongly predict
    4. Groups trial entries by these predicted tokens
    
    Args:
        model: The language model
        saes: List of SAE objects  
        entries: Circuit entries [(layer, token_pos, latent_idx, effect_value)]
        inter_toks_BL: Input tokens [B, L] 
        stop_tok: Token ID to stop generation at
        verbose: Whether to print debugging info
        
    Returns:
        Dict mapping predicted_token -> [(layer, latent_idx, [token_positions])]
    """
    # Generate unique tokens not in the original prompt
    uniq_ids = gather_unique_tokens(model, inter_toks_BL, stop_tok=stop_tok)
    # Combined filter: drop tokens that appear in prompt by ID or robust string boundaries
    prompt_tokens = inter_toks_BL[0]
    prompt_id_set = set(prompt_tokens.detach().cpu().tolist())
    prompt_str = model.to_string(prompt_tokens)

    def _has_boundary_match(hay: str, needle: str) -> bool:
        if not needle:
            return False
        # If the token contains any alphanumerics, require word boundaries to avoid substring hits.
        if any(ch.isalnum() for ch in needle):
            idx = 0
            L = len(hay)
            nL = len(needle)
            while True:
                pos = hay.find(needle, idx)
                if pos == -1:
                    return False
                left_ok = pos == 0 or not hay[pos - 1].isalnum()
                right_ok = (pos + nL) >= L or not hay[pos + nL].isalnum()
                if left_ok and right_ok:
                    return True
                idx = pos + 1
        # Pure punctuation/symbol: match anywhere
        return needle in hay

    def token_in_prompt(tok_id: int) -> bool:
        if tok_id in prompt_id_set:
            return True
        label = model.to_string(tok_id).strip()
        return bool(label) and _has_boundary_match(prompt_str, label)

    filtered_uniq_ids = [tok for tok in uniq_ids if not token_in_prompt(tok)]

    # Fallback: if everything filtered (common in code prompts), keep original uniq_ids
    # if len(filtered_uniq_ids) == 0:
    #     filtered_uniq_ids = uniq_ids
    
    if verbose:
        print(f"Found {len(filtered_uniq_ids)} tokens not in prompt: {[model.to_string(tok) for tok in filtered_uniq_ids]}")
    
    # Build the mapping from predicted tokens to circuit entries
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
    
    return saved_pair_dict 