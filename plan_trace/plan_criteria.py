from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from functools import partial
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from rapidfuzz.distance import JaroWinkler, Levenshtein

from helpers.utils import cleanup_cuda

STOP_TOKEN_ID = 1917
COEFF_GRID = list(range(-100, 0, 20))
USE_FP16_STEERING = True

def extract_all_generated_token_indices(model, out, toks):
    """
    Given a model, its output tokens, and the prompt tokens,
    returns the list of all generated‐suffix token indices
    (i.e. where “nothing is filtered out”).
    
    Args:
        model: TransformerLens model or similar with to_string and to_tokens methods.
        out:   Tensor containing model outputs (B, L_total).
        toks:  Tensor containing prompt tokens (B, L_prompt).

    Returns:
        List of token‐positions corresponding to the entire generated suffix.
    """
    # Total length and prompt length
    total_len  = out.shape[-1]
    prompt_len = toks.shape[-1]
    # The generated suffix runs from prompt_len up to total_len - 1
    return list(range(prompt_len, total_len))

def extract_non_generic_token_indices(model, out, toks):
    """
    Given a model, its output tokens, and the prompt tokens,
    identifies the non-generic (real body) token indices in the generated output.
    
    Args:
        model: TransformerLens model or similar with to_string and to_tokens methods.
        out: Tensor containing model outputs (B, L).
        toks: Tensor containing prompt tokens (B, L_prompt).

    Returns:
        List of token indices corresponding to the non-generic function body.
    """
    # 1. Extract the suffix string (generated part)
    suffix = model.to_string(out[0, toks.shape[-1]:])

    # 2. Split into lines and filter out scaffold lines
    lines = suffix.splitlines()
    non_generic_lines = []
    in_docstring = False

    for line in lines:
        stripped = line.strip()

        # Skip function definition line(s)
        if stripped.startswith("def "):
            continue

        # Detect docstring start/end (handles multiline docstrings)
        if re.match(r'^[ruRU]{0,2}"""', stripped) or re.match(r"^[ruRU]{0,2}'''", stripped):
            in_docstring = not in_docstring
            continue

        if in_docstring or stripped == "":
            continue

        non_generic_lines.append(line)

    # 3. Join lines to get the body text
    body_text = "\n".join(non_generic_lines)

    # 4. Tokenize the body text
    body_tokens = model.to_tokens(body_text, prepend_bos=False)[0]

    # 5. Find matching span in generated tokens
    generated_suffix_tokens = out[0, toks.shape[-1]:]
    non_generic_indices = []

    for i in range(len(generated_suffix_tokens) - len(body_tokens) + 1):
        if torch.all(generated_suffix_tokens[i : i + len(body_tokens)] == body_tokens):
            non_generic_indices = list(range(i + toks.shape[-1], i + toks.shape[-1] + len(body_tokens)))
            break

    return non_generic_indices

def remove_bos(text):
    if text.startswith("<bos>"):
        return text[len("<bos>"):].lstrip()
    return text

def generate_once(model, prompt: str, stop_tok: int, hooks: Optional[List] = None, *, max_tokens: int = 256, device: str = "cuda") -> str:  # type: ignore
    """Same as before, but accepts an explicit *hooks* list (or None)."""
    toks = model.to_tokens(prompt).to(device)
    out  = toks.clone()

    if hooks:
        for hook_name, fn in hooks:
            model.add_hook(hook_name, fn)

    while out.shape[-1] - toks.shape[-1] < max_tokens:
        with torch.no_grad():
            logits = model(out)[:, -1]
        next_id = logits.argmax(-1).item()
        if next_id == stop_tok:
            break
        if model.to_string(next_id) == "<end_of_turn>" or model.to_string(next_id) == "<eos>":
            break
        out = torch.cat([out, torch.tensor([[next_id]], device=device)], dim=1)

    if hooks:
        model.reset_hooks(including_permanent=True)

    return model.to_string(out[0, toks.shape[-1]:])

def generate_baseline(model, prompt: str, *, stop_tok: int ) -> str:
    """Generate until *stop_tok* and return the raw suffix."""
    return generate_once(model, prompt=prompt, stop_tok=stop_tok, hooks=None)

def gather_unique_tokens(
    model,
    prompt: str,
    *,
    stop_tok: int,
    device: str = "cuda",
) -> Tuple[Sequence[int], str]:
    """Return *(unique_token_ids, full_suffix_str)* for prompt."""
    toks = model.to_tokens(prompt).to(device)
    out  = toks.clone()
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

    return list(sorted(unique_ids)), model.to_string(out[0, toks.shape[-1]:])

def build_saved_pair_dict_fastest(
    model,
    saes,
    trial_entries: Sequence[Tuple[int, int, int, float]],
    unique_token_ids: Sequence[int],
    *,
    tok_k_pos_logits: int = 15,
    batch_size: int = 4096,
    device: str = "cuda",
) -> Dict[str, List[Tuple[int, int, List[int]]]]:
    """Ultra-fast saved_pair_dict building, full GPU, vectorized."""

    layer_to_latents: Dict[int, set[int]] = defaultdict(set)
    for l, t, lat, _ in trial_entries:
        layer_to_latents[l].add(lat)

    unique_token_tensor = torch.tensor(unique_token_ids, device=device)  # stay on GPU
    saved_pair_dict: Dict[str, List[Tuple[int, int, List[int]]]] = defaultdict(list)

    with torch.no_grad():
        W_U = model.W_U.float().to(device)

        for layer_i, latents in tqdm(layer_to_latents.items()):
            latents = sorted(latents)
            W_dec = saes[layer_i].W_dec.to(device)

            for start in range(0, len(latents), batch_size): #, desc=f"layer {layer_i} hit-mapping"):
                batch = latents[start : start + batch_size]
                dirs = W_dec[batch]            # [batch_size, d_model]

                logits = dirs @ W_U            # [batch_size, vocab_size]
                topk_scores, topk_idx = torch.topk(logits, tok_k_pos_logits, dim=1)  # [batch_size, topk]

                # ---- GPU intersection ----
                # For each latent's topk, find which match unique_token_ids
                # Create: [batch_size, topk] -> True/False if in unique_token_ids

                # broadcast: [batch_size, topk] vs [1, num_unique_ids]
                topk_idx_exp = topk_idx.unsqueeze(-1)                   # [batch, topk, 1]
                unique_tok_exp = unique_token_tensor.view(1, 1, -1)     # [1, 1, num_unique_ids]

                matches = (topk_idx_exp == unique_tok_exp).any(-1)      # [batch, topk] -> True/False
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


def jaccard_dissim(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    return 1 - len(sa & sb) / len(sa | sb) if sa or sb else 0.0

def _steer_dtype(activations: torch.Tensor) -> torch.dtype:
    """Return dtype used to apply steering (fp32 while debugging)."""
    if USE_FP16_STEERING:
        return activations.dtype          # fp16 / bf16
    return torch.float32                  # debug‑mode: high precision

def steering_hook(
    activations, hook, sae, latent_idx: int,
    coeff: float, steering_token_index: Optional[int] = None
):
    """
    Apply coeff · W_dec[latent_idx] either to the whole sequence or
    a single position (steering_token_index), using a *single* high‑precision
    add to avoid rounding‑order artefacts.
    """
    tgt_dtype = _steer_dtype(activations)
    steer_vec = (
        sae.W_dec[latent_idx]
        .to(device=activations.device, dtype=tgt_dtype) * float(coeff)
    )

    # ——— cast activations to tgt_dtype only for the in‑place add ———
    act_view = activations.to(dtype=tgt_dtype)

    if steering_token_index is None:
        act_view += steer_vec
    else:
        act_view[:, steering_token_index] += steer_vec

    # In debug mode this casts back to original (fp16/bf16)   ↓↓↓
    return act_view.to(dtype=activations.dtype)

def steering_effect_on_next_token(
    model, toks_prefix, saes, interventions, coeff, stop_tok
):
    """
    Returns (changed: bool, new_id: int, baseline_id: int)
    without sampling the whole sequence.
    """
    # --- register hooks exactly once ------------------------------------
    model.reset_hooks(including_permanent=True)
    with torch.no_grad():
        logits = model(toks_prefix)[:, -1]
    baseline_id = logits.argmax(-1).item()

    for layer_idx, tok_pos, latent_idx in interventions:
        sae = saes[layer_idx]
        model.add_hook(
            sae.cfg.hook_name,
            partial(
                steering_hook, sae=sae,
                latent_idx=latent_idx, coeff=coeff,
                steering_token_index=tok_pos,
            ),
        )

    with torch.no_grad():
        logits = model(toks_prefix)[:, -1]        
    new_id      = logits.argmax(-1).item()
    
    model.reset_hooks(including_permanent=True)
    return new_id != baseline_id, new_id, baseline_id

def sweep_coefficients_multi(
    model,
    saes: Dict[int, "SAE"],
    interventions: List[Tuple[int, int, int]],   # (layer, token_pos, latent)
    coefficients: List[float],
    prompt: str,
    stop_tok: int = 1917,
    device: str = "cuda",
    max_tokens: int = 100,
) -> Dict[float, str]:

    # ─── pre‑aggregate: (coeff  →  layer → {pos → steer_vec}) ─────────────
    d_model   = saes[0].W_dec.shape[1]
    act_dtype = next(model.parameters()).dtype    # fp16 / bf16 / fp32

    steer_by_coeff: Dict[float, Dict[int, Dict[int, torch.Tensor]]] = {}
    for c in coefficients:
        per_coeff = defaultdict(                       # layer →
            lambda: defaultdict(                       #   pos →
                lambda: torch.zeros(                   #     steer_vec
                    d_model, device=device,
                    dtype=_steer_dtype(torch.empty((),dtype=act_dtype))
                )
            )
        )
        for layer_idx, pos, latent_idx in interventions:
            vec = (
                saes[layer_idx].W_dec[latent_idx]
                .to(device=device, dtype=_steer_dtype(torch.empty((),dtype=act_dtype)))
                * float(c)
            )
            per_coeff[layer_idx][pos] += vec
        steer_by_coeff[c] = per_coeff

    # ─── do the per‑coeff generation ──────────────────────────────────────
    outputs: Dict[float, str] = {}
    for c in coefficients:
        changed, new_id, bl_id = steering_effect_on_next_token(
            model, prompt, saes, interventions, c, STOP_TOKEN_ID
        )
        if not changed:
            continue 
        per_coeff = steer_by_coeff[c]

        # register ONE hook per layer that adds the *summed* vector
        for layer_idx, pos_dict in per_coeff.items():
            hook_name = saes[layer_idx].cfg.hook_name

            def make_layer_hook(pos_dict_local):
                def layer_hook(activations, hook):
                    for pos, vec in pos_dict_local.items():
                        if pos < activations.size(1):
                            act_view = activations.to(_steer_dtype(activations))
                            act_view[:, pos] += vec
                            # copy back (same dtype cast logic as above)
                            activations.copy_(act_view.to(dtype=activations.dtype))
                    return activations
                return layer_hook

            model.add_hook(hook_name, make_layer_hook(pos_dict))

        # generate
        gen_suffix = generate_once(
            model, prompt=prompt, stop_tok=stop_tok,
            hooks=None, max_tokens=max_tokens, device=device,
        )
        outputs[c] = gen_suffix
        model.reset_hooks(including_permanent=True)

    return outputs

def run_steering_sweep(
    model,
    saes,
    prompt: str,
    saved_pair_dict: Dict[str, List[Tuple[int, int, List[int]]]],
    baseline_text: str,
    *,
    coeff_grid: Sequence[int] = COEFF_GRID,
    stop_tok: int = STOP_TOKEN_ID,
    max_tokens: int = 100,
) -> Dict[str, Dict[str, Any]]:
    """Return per‑label sweep results without re-running baseline."""
    results: Dict[str, Dict[str, Any]] = {}

    baseline_toks = baseline_text.split()

    for label, latent_infos in tqdm(saved_pair_dict.items()):
        # build (layer, tok_pos, latent) triples
        interventions = [
            (layer_i, tok_pos, latent_i)
            for layer_i, latent_i, tok_positions in latent_infos
            for tok_pos in tok_positions
        ]
        sweep_out = sweep_coefficients_multi(
            model=model,
            saes=saes,
            interventions=interventions,
            coefficients=list(coeff_grid),
            prompt=prompt,
            stop_tok=stop_tok,
            max_tokens=max_tokens,
        )

        label_metrics: List[Dict[str, Any]] = []
        for c in coeff_grid:
            txt  = sweep_out.get(c, "")
            lev  = Levenshtein.normalized_distance(txt, baseline_text)
            jw   = 1 - JaroWinkler.normalized_similarity(txt, baseline_text)
            jac  = jaccard_dissim(txt.split(), baseline_toks)
            label_metrics.append(dict(coeff=c, levenshtein=lev, jaro_winkler=jw, jaccard=jac, steered_text=txt))

        results[label] = {
            "base_text": baseline_text,
            "steered": label_metrics
        }

    return results




def get_ll_clusters():
    return

def test_steering():
    return