#!/usr/bin/env python
"""
Analyze a single SAE latent's token selectivity with AUC + MWU + Bonferroni.

Pipeline:
1. Load prompts (use first N sequences from prompts.json).
2. Tokenize with Gemma-2-2B, build token frequencies (sequence-level presence).
3. Load a Gemma Scope SAE and pick a latent (layer, index).
4. For each sequence, compute a top-q% mean activation summary for that latent.
5. For each token:
   - Compare activations in sequences that contain vs don't contain the token.
   - Compute Mann–Whitney U test, p-value, and AUC.
6. Apply frequency filters + Bonferroni correction.
7. Print top tokens by |AUC - 0.5| among those passing Bonferroni.

Usage example (fill in sae_release / sae_id from Neuronpedia config):

    python analyze_latent_tokens.py \
        --prompts-path data/prompts.json \
        --num-seqs 10000 \
        --latent-layer 7 \
        --latent-index 7643 \
        --sae-release gemma-scope-2b-pt-mlp \
        --sae-id "layer_7/width_16k/average_l0_86"

"""

import argparse
import json
import math
import sys
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sae_lens import SAE, HookedSAETransformer
from scipy.stats import mannwhitneyu  # pip install scipy

sys.path.append("../")
from plan_trace.utils import load_model, load_pretrained_saes, cleanup_cuda
from plan_trace.hooks import run_with_saes, register_sae_hooks

# -----------------------------
# Helpers
# -----------------------------


def tokens_to_text(tokens: Sequence[str]) -> str:
    """Reconstruct text from Neuronpedia-style token strings."""
    return "".join(
        t.replace("▁", " ").replace("<0x0A>", "\n") for t in tokens
    )


def decode_token_id(model: HookedSAETransformer, token_id: int) -> str:
    """Decode a single token id to a readable string."""
    try:
        return model.tokenizer.decode([token_id])
    except Exception:
        return f"<id:{token_id}>"


def compute_top_q_summary(
    latent_acts: torch.Tensor,
    top_q: float,
) -> torch.Tensor:
    """
    latent_acts: [B, L] tensor of latent activations per position.
    top_q: fraction (0 < top_q <= 1). E.g. 0.01 for top 1%.

    Returns: [B] sequence-level summaries (mean of top-q activations).
    """
    B, L = latent_acts.shape
    k = max(1, int(math.ceil(L * top_q)))
    top_vals, _ = torch.topk(latent_acts, k=k, dim=1)
    return top_vals.mean(dim=1)


def build_token_to_seq_index(
    all_token_ids: List[List[int]],
    special_ids: Sequence[int],
) -> Dict[int, List[int]]:
    """
    Build inverted index: token_id -> list of sequence indices where it appears.
    Presence is sequence-level (once per sequence).
    """
    special_set = set(special_ids)
    token_to_seqs: Dict[int, List[int]] = defaultdict(list)

    for seq_idx, ids in enumerate(all_token_ids):
        seen = set()
        for tid in ids:
            if tid in special_set:
                continue
            if tid in seen:
                continue
            seen.add(tid)
            token_to_seqs[tid].append(seq_idx)

    return token_to_seqs


# -----------------------------
# Main analysis
# -----------------------------


def analyze_latent_tokens(
    prompts_path: str,
    num_seqs: int,
    model_name: str,
    sae_release: str,
    sae_id: str,
    latent_layer: int,
    latent_index: int,
    device: str = "cuda",
    top_q: float = 0.01,
    min_freq: int = 20,
    max_freq_frac: float = 0.9,
    alpha: float = 0.05,
    batch_size: int = 32,
    max_print: int = 50,
) -> None:
    # 1. Load prompts
    with open(prompts_path, "r") as f:
        prompts = json.load(f)

    if num_seqs > len(prompts):
        print(f"[WARN] Requested num_seqs={num_seqs} > available={len(prompts)}; using all.")
        num_seqs = len(prompts)

    prompts = prompts[:num_seqs]
    print(f"[INFO] Loaded {len(prompts)} prompts from {prompts_path}")

    # Convert token lists -> raw text strings
    texts: List[str] = [tokens_to_text(p) for p in prompts]

    # 2. Load model (Gemma-2-2B)
    print(f"[INFO] Loading model: {model_name}")
    model: HookedSAETransformer = load_model(
        model_name,
        device=device,
        use_custom_cache=True,
        dtype=torch.bfloat16,
    )

    # 3. Load SAE for the desired layer
    print(f"[INFO] Loading SAE: release={sae_release}, sae_id={sae_id}")
    # sae, cfg_dict, sparsity = SAE.from_pretrained(
    #     release=sae_release,
    #     sae_id=sae_id,
    #     device=device,
    # )
    saes = load_pretrained_saes(
    layers=[latent_layer], 
    release=sae_release, 
    width="16k", 
    device=device, 
    canon=True
    )
    sae = saes[0]

    print(
        f"[INFO] SAE loaded: hook_name={sae.cfg.hook_name}, "
        f"layer={sae.cfg.hook_layer}, d_sae={sae.W_dec.shape[1]}, sparsity≈{sparsity:.1f}"
    )

    if latent_index >= sae.W_dec.shape[1]:
        raise ValueError(
            f"Latent index {latent_index} out of range for SAE width {sae.W_dec.shape[1]}"
        )

    # 4. Tokenize prompts + collect token ids + latent summaries
    print("[INFO] Tokenizing prompts and computing latent activations...")
    model.eval()
    all_seq_token_ids: List[List[int]] = []
    all_seq_summaries: List[torch.Tensor] = []

    # Try to get special token ids to ignore
    special_ids = []
    for attr in ("bos_id", "eos_id", "pad_id"):
        if hasattr(model.tokenizer, attr):
            v = getattr(model.tokenizer, attr)
            if v is not None:
                special_ids.append(int(v))

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]

            # [B, L]
            tokens = model.to_tokens(batch_texts).to(device)
            B, L = tokens.shape

            # Run model with SAE, cache feature activations
            model.reset_hooks(including_permanent=True)
            _ = run_with_saes(model, saes=[sae], tokens=tokens, cache_sae_activations=True)

            # sae.feature_acts: [B, L, D_sae]
            feature_acts = sae.feature_acts  # type: ignore
            if feature_acts is None:
                raise RuntimeError("SAE feature_acts not populated; check run_with_saes call.")

            latent_acts = feature_acts[..., latent_index]  # [B, L]
            seq_summary_batch = compute_top_q_summary(latent_acts, top_q=top_q)  # [B]

            # Save
            all_seq_summaries.append(seq_summary_batch.cpu())
            all_seq_token_ids.extend(tokens.cpu().tolist())

    seq_summaries = torch.cat(all_seq_summaries, dim=0).numpy()  # [N]
    N = len(seq_summaries)
    assert N == len(all_seq_token_ids)
    print(f"[INFO] Got sequence summaries for N={N} sequences")

    # 5. Build token -> sequence index mapping, compute frequencies
    print("[INFO] Building token->sequence index mapping...")
    token_to_seqs = build_token_to_seq_index(all_seq_token_ids, special_ids)
    print(f"[INFO] Found {len(token_to_seqs)} distinct non-special tokens")

    # 6. Frequency-based token filtering
    print(
        f"[INFO] Applying frequency filters: min_freq={min_freq}, "
        f"max_freq_frac={max_freq_frac}"
    )
    candidates = []
    for tid, seq_indices in token_to_seqs.items():
        f = len(seq_indices)
        if f < min_freq:
            continue
        if f > max_freq_frac * N:
            continue
        candidates.append((tid, seq_indices))

    print(f"[INFO] {len(candidates)} tokens remain after frequency filtering")

    # 7. MWU + AUC per token
    all_results: List[Tuple[int, int, float, float]] = []
    all_indices = np.arange(N)

    print("[INFO] Computing MWU & AUC for each candidate token...")
    for tid, pos_seqs in candidates:
        pos_idx = np.array(pos_seqs, dtype=int)
        neg_mask = np.ones(N, dtype=bool)
        neg_mask[pos_idx] = False
        neg_idx = all_indices[neg_mask]

        pos = seq_summaries[pos_idx]
        neg = seq_summaries[neg_idx]

        if len(pos) == 0 or len(neg) == 0:
            continue

        # Mann–Whitney U test
        U, p = mannwhitneyu(pos, neg, alternative="two-sided")

        # AUC: U / (n_pos * n_neg)
        n_pos = len(pos)
        n_neg = len(neg)
        auc = U / (n_pos * n_neg)

        all_results.append((tid, n_pos, auc, p))

    if not all_results:
        print("[WARN] No tokens had valid MWU statistics. Try loosening filters.")
        return

    # 8. Bonferroni correction and ranking
    M = len(all_results)
    print(f"[INFO] Applying Bonferroni correction over M={M} tests; alpha={alpha}")
    corrected_results = []
    for tid, freq, auc, p in all_results:
        p_bonf = min(p * M, 1.0)
        corrected_results.append((tid, freq, auc, p, p_bonf))

    # Keep only those passing Bonferroni and with non-trivial AUC
    filtered = [
        r for r in corrected_results
        if r[4] <= alpha and not math.isnan(r[2])
    ]
    if not filtered:
        print("[WARN] No tokens survived Bonferroni at this alpha. Try relaxing alpha or filters.")
        return

    # Sort by |AUC - 0.5| descending (strongest association first)
    filtered.sort(key=lambda r: abs(r[2] - 0.5), reverse=True)

    # 9. Print results
    print(
        f"\n=== Top tokens for latent (layer={latent_layer}, index={latent_index}) "
        f"===\n"
    )
    print(
        f"{'rank':>4}  {'id':>8}  {'freq':>6}  {'AUC':>8}  {'p_bonf':>10}  token\n"
        + "-" * 80
    )

    for rank, (tid, freq, auc, p_raw, p_bonf) in enumerate(filtered[:max_print], start=1):
        tok_str = decode_token_id(model, tid)
        tok_str = tok_str.replace("\n", "\\n")
        print(
            f"{rank:4d}  {tid:8d}  {freq:6d}  {auc:8.3f}  {p_bonf:10.2e}  {tok_str}"
        )

    print("\n[INFO] Done.")
    cleanup_cuda()


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze token selectivity of a single SAE latent (Gemma Scope)."
    )
    parser.add_argument(
        "--prompts-path",
        type=str,
        default="../data/prompts.json",
        help="Path to prompts.json (list of token lists).",
    )
    parser.add_argument(
        "--num-seqs",
        type=int,
        default=10000,
        help="Number of sequences to use (from the start of prompts).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gemma-2-2b",
        help="Model name passed to load_model.",
    )
    parser.add_argument(
        "--sae-release",
        type=str,
        default="gemma-scope-2b-pt-mlp",
        help="SAE release name for SAE.from_pretrained (see Gemma Scope HF page).",
    )
    parser.add_argument(
        "--sae-id",
        type=str,
        default="layer_7/width_16k/average_l0_86",
        help="SAE ID string, e.g. 'layer_7/width_16k/average_l0_86'. "
             "Override with the exact value from Neuronpedia/HF.",
    )
    parser.add_argument(
        "--latent-layer",
        type=int,
        default=7,
        help="Layer index of latent (for logging only).",
    )
    parser.add_argument(
        "--latent-index",
        type=int,
        default=7643,
        help="Latent index within the SAE.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (e.g. 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--top-q",
        type=float,
        default=0.01,
        help="Top-q fraction for activation summarization (e.g., 0.01 = top 1%).",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=20,
        help="Minimum #sequences a token must appear in to be tested.",
    )
    parser.add_argument(
        "--max-freq-frac",
        type=float,
        default=0.9,
        help="Maximum fraction of sequences a token may appear in to be tested.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for Bonferroni-corrected p-values.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for model forward passes.",
    )
    parser.add_argument(
        "--max-print",
        type=int,
        default=50,
        help="Maximum number of tokens to print.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze_latent_tokens(
        prompts_path=args.prompts_path,
        num_seqs=args.num_seqs,
        model_name=args.model_name,
        sae_release=args.sae_release,
        sae_id=args.sae_id,
        latent_layer=args.latent_layer,
        latent_index=args.latent_index,
        device=args.device,
        top_q=args.top_q,
        min_freq=args.min_freq,
        max_freq_frac=args.max_freq_frac,
        alpha=args.alpha,
        batch_size=args.batch_size,
        max_print=args.max_print,
    )
