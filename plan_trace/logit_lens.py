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
    device: str = "cuda"
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

                if matching_token_ids.numel() > 0:
                    matched_latents, matched_topk = torch.nonzero(matches, as_tuple=True)
                    for latent_batch_idx, topk_pos in zip(matched_latents.tolist(), matched_topk.tolist()):
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
    verbose: bool = True
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
    prompt_str = model.to_string(inter_toks_BL[0, :])
    filtered_uniq_ids = [tok for tok in uniq_ids if model.to_string(tok) not in prompt_str]
    
    if verbose:
        print(f"Found {len(filtered_uniq_ids)} tokens not in prompt: {[model.to_string(tok) for tok in filtered_uniq_ids]}")
    
    # Build the mapping from predicted tokens to circuit entries
    saved_pair_dict = build_saved_pair_dict_fastest(
        model,
        saes,
        entries,
        filtered_uniq_ids,
        tok_k_pos_logits=15,
        batch_size=4096
    )
    
    return saved_pair_dict 