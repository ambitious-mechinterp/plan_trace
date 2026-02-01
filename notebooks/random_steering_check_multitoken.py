# %%
from __future__ import annotations

import json
import os
import sys
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
sys.path.append("../")
from plan_trace.steering import sweep_coefficients_multi, run_steering_sweep
from plan_trace.utils import load_model, load_pretrained_saes
from plan_trace.ood_detect import label_steering_clusters

# %%
# Configuration
ROOT_DIR = Path("/home/jnainani_umass_edu/w/plan_trace")
PARENT_DIR = Path("/work/pi_jensen_umass_edu/abhishekmish_umass_edu/plan_trace/")

# Path for updated_planning_analysis.json (verdict file)
CLASSIFIED_ROOT = PARENT_DIR / "outputs" / "planning_scale_classified_e" / "instruct"

# Path for clusters.json, circuit_entries.pt, metadata.json, earliest_position*, etc.
DATA_ROOT = PARENT_DIR / "outputs" / "planning_scale" / "instruct"

# Prompt data file (same as original script)
PROMPT_DATA_PATH = ROOT_DIR / "data" / "external" / "all_examples_og_prompt_with_position_info_and_success_V2.json"

MODEL_NAME = "gemma-2-2b-it"
DEVICE = "cuda"

TARGET_COEFF = -200
SEED = 0
MAX_TOKENS = 100

# %%
# Load model and SAEs
rng = random.Random(SEED)
model = load_model(MODEL_NAME, device=DEVICE, use_custom_cache=True, dtype=torch.bfloat16)
layers = list(range(model.cfg.n_layers))
saes = load_pretrained_saes(
    layers=layers,
    release="gemma-scope-2b-pt-mlp-canonical",
    width="16k",
    device=DEVICE,
    canon=True,
)

# %%
# Output naming
UPDATED_ANALYSIS_NAME = "updated_planning_analysis.json"
AUGMENTED_ANALYSIS_NAME = "updated_planning_analysis_random_steer.json"

# Steering parameters
STOP_TOKEN_ID = 1917
GEN_LIMIT = 150
POSITION_SAMPLE_RATIO = 0.2
PER_POSITION_LATENTS_M = 5
LATENT_RESAMPLES = 3
STEP_COEFF = 25
RESAMPLE_FOR_MIN_COEFF = False

# Processing limits (set to None to process all)
MAX_PROMPTS: Optional[int] = None
MAX_TOKENS_PER_PROMPT: Optional[int] = None
START_PROMPT_IDX: Optional[int] = 10
PROGRESS_EVERY = 10

# %%
# Helper functions

def _load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _discover_prompt_token_pairs() -> List[Tuple[int, int, Path, Path]]:
    """
    Discover all (prompt_idx, token_idx) pairs that have:
      - updated_planning_analysis.json in CLASSIFIED_ROOT
      - Required data files (clusters.json, circuit_entries.pt) in DATA_ROOT

    Returns list of (prompt_idx, token_idx, classified_token_dir, data_token_dir)
    """
    pairs = []

    # Find all prompt folders in classified root
    prompt_pattern = re.compile(r"prompt_(\d+)")
    token_pattern = re.compile(r"token_(\d+)")

    if not CLASSIFIED_ROOT.exists():
        print(f"[error] CLASSIFIED_ROOT does not exist: {CLASSIFIED_ROOT}")
        return pairs

    for prompt_dir in sorted(CLASSIFIED_ROOT.iterdir()):
        if not prompt_dir.is_dir():
            continue
        match = prompt_pattern.match(prompt_dir.name)
        if not match:
            continue
        prompt_idx = int(match.group(1))

        # Find all token folders
        for token_dir in sorted(prompt_dir.iterdir()):
            if not token_dir.is_dir():
                continue
            match = token_pattern.match(token_dir.name)
            if not match:
                continue
            token_idx = int(match.group(1))

            classified_token_dir = token_dir
            data_token_dir = DATA_ROOT / f"prompt_{prompt_idx}" / f"token_{token_idx}"

            # Check required files exist
            analysis_path = classified_token_dir / UPDATED_ANALYSIS_NAME
            circuit_path = data_token_dir / "circuit_entries.pt"
            clusters_path = data_token_dir / "clusters.json"
            metadata_path = data_token_dir / "metadata.json"

            if analysis_path.exists() and circuit_path.exists() and clusters_path.exists() and metadata_path.exists():
                pairs.append((prompt_idx, token_idx, classified_token_dir, data_token_dir))

    return pairs


def _steered_text_to_str(val: Any, *, model) -> str:
    if hasattr(val, "tolist"):
        return model.to_string(val.tolist())
    if isinstance(val, list):
        return model.to_string(val)
    return str(val)


def _serialize_latents(latents: Sequence[Tuple[int, int]]) -> List[List[int]]:
    return [[int(li), int(latent_i)] for (li, latent_i) in latents]


def _serialize_removed_future(
    removed_future: Sequence[Tuple[int, Sequence[Tuple[int, int]], str]],
    *,
    text_limit: int = 1000,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for pos, latents, text in removed_future:
        out.append({
            "pos": int(pos),
            "latents": _serialize_latents(latents),
            "steered_text": text[:text_limit],
        })
    return out


def _build_prompt(entry: Dict[str, Any]) -> str:
    """Build the instruct prompt from a prompt data entry (same as original script)."""
    return (
        "You are an expert Python programmer, and here is your task: "
        f"{entry['prompt']} Your code should pass these tests:\n\n"
        + "\n".join(entry["test_list"])
        + "\nWrite your code, without docstrings, below starting with \"```python\" and ending with \"```\".\n```python\n"
    )


def _get_prompt_cache(
    prompt_idx: int,
    *,
    model,
    device: str,
    data: Sequence[Dict[str, Any]],
    cache: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Get or create cached prompt data with autoregressive generation.
    Same approach as the original script's _get_prompt_cache.
    """
    if prompt_idx in cache:
        return cache[prompt_idx]

    entry = data[prompt_idx]
    prompt = _build_prompt(entry)
    toks_BL = model.to_tokens(prompt).to(device)
    out_BL = toks_BL.clone()

    # Autoregressive generation until stop token or limit
    while out_BL.shape[-1] - toks_BL.shape[-1] < GEN_LIMIT:
        with torch.no_grad():
            logits_V = model(out_BL)[0, -1]
        next_id = logits_V.argmax(-1).item()
        del logits_V
        if next_id == STOP_TOKEN_ID:
            break
        out_BL = torch.cat([out_BL, torch.tensor([[next_id]], device=device)], dim=1)

    cache[prompt_idx] = {"entry": entry, "prompt": prompt, "out_BL": out_BL}
    return cache[prompt_idx]


# %%
def _find_earliest_planning_position(
    *,
    model,
    saes,
    out_BL: torch.Tensor,
    baseline_suffix: str,
    token_i: int,
    selected_label: str,
    clusters: Dict[str, Any],
) -> Tuple[Optional[int], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if selected_label not in clusters:
        return None, None, None

    pairs_for_label = clusters[selected_label]
    positions = sorted({
        tok_pos
        for (_li, _latent_i, tok_positions) in pairs_for_label
        for tok_pos in tok_positions
    })
    inter_toks_BL = out_BL[:, :token_i]

    for tok_pos in positions:
        filtered = {selected_label: []}
        for li, latent_i, tok_positions in pairs_for_label:
            if tok_pos in tok_positions:
                filtered[selected_label].append([li, latent_i, [tok_pos]])

        pos_steering = run_steering_sweep(
            model=model,
            saes=saes,
            inter_toks_BL=inter_toks_BL,
            saved_pair_dict=filtered,
            baseline_text=baseline_suffix,
            coeff_grid=[TARGET_COEFF],
            stop_tok=STOP_TOKEN_ID,
            max_tokens=MAX_TOKENS,
            return_tokens=True,
        )
        pos_labels = label_steering_clusters(pos_steering, model=model, prefix_tokens_2d=inter_toks_BL)
        final_label = pos_labels.get(selected_label, {}).get("final_label", "Can't say")
        if final_label == "Plan":
            return tok_pos, pos_steering, pos_labels

    return None, None, None


# %%
def run_random_steer_eval_and_min_coeff(
    *,
    model,
    saes,
    out_BL: torch.Tensor,
    baseline_suffix: str,
    selected_label: str,
    target_coeff: int,
    stop_tok: int,
    max_tokens: int,
    token_i: int,
    circuit_entries: Sequence[Any],
    earliest_position_found: int,
    earliest_latents: Sequence[Tuple[int, int]],
    reference_steered_text: str,
    rng_obj: random.Random,
    position_sample_ratio: Optional[float] = None,
    per_position_latents_m: int = 5,
    latent_resamples: int = 1,
    step_coeff: int = 25,
    resample_for_min_coeff: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Random steering evaluation and minimal coefficient search.
    """
    # Build position -> [(layer, latent)] from circuit_entries
    pos_to_latents: Dict[int, List[Tuple[int, int]]] = {}
    for entry in circuit_entries:
        try:
            layer_i, tok_pos, latent_i, _val = entry
        except Exception:
            if isinstance(entry, dict):
                layer_i = int(entry.get("layer", 0))
                tok_pos = int(entry.get("token", 0))
                latent_i = int(entry.get("latent", 0))
            else:
                layer_i = int(entry[0])
                tok_pos = int(entry[1])
                latent_i = int(entry[2])
        pos_to_latents.setdefault(int(tok_pos), []).append((int(layer_i), int(latent_i)))

    all_positions = sorted(pos_to_latents.keys())
    positions_wo_earliest = [p for p in all_positions if p != earliest_position_found]

    # Determine sample size
    if position_sample_ratio is not None and 0.0 < position_sample_ratio <= 1.0:
        sample_n = int(round(len(positions_wo_earliest) * position_sample_ratio))
        sample_n = max(0, min(sample_n, len(positions_wo_earliest)))
    else:
        sample_n = min(5, len(positions_wo_earliest))
    sampled_positions = rng_obj.sample(positions_wo_earliest, sample_n) if sample_n > 0 else []
    selected_positions = [earliest_position_found] + sampled_positions

    if verbose:
        print(f"\nSelected positions (including earliest): {selected_positions}")

    def latents_for_position(p: int) -> List[Tuple[int, int]]:
        return pos_to_latents.get(int(p), [])

    matches_ref: List[Tuple[int, List[Tuple[int, int]]]] = []
    removed_future: List[Tuple[int, List[Tuple[int, int]], str]] = []

    # Evaluate random steers
    for pos in selected_positions:
        available_latents = latents_for_position(pos)
        if not available_latents:
            if verbose:
                print(f"Position {pos}: no latents available, skipping.")
            continue
        m = min(per_position_latents_m, len(available_latents))
        for _ in range(max(1, latent_resamples)):
            sampled_latents = rng_obj.sample(available_latents, m)

            filtered = {selected_label: [[li, latent_i, [pos]] for (li, latent_i) in sampled_latents]}
            pos_steering = run_steering_sweep(
                model=model,
                saes=saes,
                inter_toks_BL=out_BL[:, :token_i],
                saved_pair_dict=filtered,
                baseline_text=baseline_suffix,
                coeff_grid=[target_coeff],
                stop_tok=stop_tok,
                max_tokens=max_tokens,
                return_tokens=True,
            )

            entries = pos_steering[selected_label]["steered"]
            base_text = pos_steering[selected_label]["base_text"]
            if not entries:
                continue
            e = entries[0]
            val = e.get("steered_text")
            if hasattr(val, "tolist"):
                steered_text = model.to_string(val.tolist())
            elif isinstance(val, list):
                steered_text = model.to_string(val)
            else:
                steered_text = str(val)

            # Treat empty as no change
            if steered_text is None or steered_text.strip() == "":
                continue

            contains_future = (selected_label in steered_text)
            if steered_text == reference_steered_text:
                if verbose:
                    print(f"Match to reference at position {pos} using {len(sampled_latents)} latents")
                matches_ref.append((pos, sampled_latents))
            elif (steered_text != base_text) and (not contains_future):
                if verbose:
                    print(f"Future '{selected_label}' removed at position {pos} using {len(sampled_latents)} latents")
                removed_future.append((pos, sampled_latents, steered_text))

    if verbose:
        if matches_ref:
            print("\nReproduced reference steered generation at:")
            for (pos, lat_list) in matches_ref:
                print(f"  pos={pos}, latents={lat_list[:5]}{'...' if len(lat_list) > 5 else ''}")
        else:
            print("\nNo matches to the reference steered generation were found among sampled latents.")

        if removed_future:
            print(f"\nSteered generations where future token '{selected_label}' is removed:")
            for (pos, lat_list, steered_txt) in removed_future:
                print(f"  pos={pos}, latents={lat_list[:5]}{'...' if len(lat_list) > 5 else ''}")
                print(steered_txt[:1000])

    # Minimal coefficient search
    coeff_candidates: List[int]
    if target_coeff < 0:
        coeff_candidates = [0] + [-c for c in range(step_coeff, abs(target_coeff) + step_coeff, step_coeff)]
    else:
        coeff_candidates = [0] + [c for c in range(step_coeff, abs(target_coeff) + step_coeff, step_coeff)]
    coeff_candidates = [c for c in coeff_candidates if abs(c) <= abs(target_coeff)]

    earliest_filtered = {selected_label: [[li, latent_i, [earliest_position_found]] for (li, latent_i) in earliest_latents]}

    found_coeff: Optional[int] = None
    earliest_txt_at_found: Optional[str] = None

    if removed_future or matches_ref:
        if verbose:
            print("\nStarting minimal-coefficient search for earliest position (no random effects)...")
        for cand in coeff_candidates:
            # 1) earliest check
            earliest_sweep = run_steering_sweep(
                model=model,
                saes=saes,
                inter_toks_BL=out_BL[:, :token_i],
                saved_pair_dict=earliest_filtered,
                baseline_text=baseline_suffix,
                coeff_grid=[cand],
                stop_tok=stop_tok,
                max_tokens=max_tokens,
                return_tokens=True,
            )
            earliest_entries = earliest_sweep[selected_label]["steered"]
            if not earliest_entries:
                continue
            val = earliest_entries[0].get("steered_text")
            if hasattr(val, "tolist"):
                earliest_txt = model.to_string(val.tolist())
            elif isinstance(val, list):
                earliest_txt = model.to_string(val)
            else:
                earliest_txt = str(val)
            if earliest_txt is None or earliest_txt.strip() == "":
                continue
            if earliest_txt != reference_steered_text:
                continue

            # 2) random effects check
            ok_random = True
            for (pos, lat_list, _prev_txt) in removed_future:
                if resample_for_min_coeff:
                    available_latents = latents_for_position(pos)
                    if not available_latents:
                        continue
                    m = min(per_position_latents_m, len(available_latents))
                    lat_list = rng_obj.sample(available_latents, m)
                filtered = {selected_label: [[li, latent_i, [pos]] for (li, latent_i) in lat_list]}
                rnd_sweep = run_steering_sweep(
                    model=model,
                    saes=saes,
                    inter_toks_BL=out_BL[:, :token_i],
                    saved_pair_dict=filtered,
                    baseline_text=baseline_suffix,
                    coeff_grid=[cand],
                    stop_tok=stop_tok,
                    max_tokens=max_tokens,
                    return_tokens=True,
                )
                rnd_entries = rnd_sweep[selected_label]["steered"]
                if not rnd_entries:
                    continue
                rnd_val = rnd_entries[0].get("steered_text")
                if hasattr(rnd_val, "tolist"):
                    rnd_txt = model.to_string(rnd_val.tolist())
                elif isinstance(rnd_val, list):
                    rnd_txt = model.to_string(rnd_val)
                else:
                    rnd_txt = str(rnd_val)
                base_txt = rnd_sweep[selected_label]["base_text"]
                # no random effect if empty, equals base, or future token STILL present
                if rnd_txt is None or rnd_txt.strip() == "" or rnd_txt == base_txt or (selected_label in rnd_txt):
                    continue
                ok_random = False
                break

            if ok_random:
                found_coeff = cand
                earliest_txt_at_found = earliest_txt
                if verbose:
                    print(f"Found minimal coefficient with no random effects: {found_coeff}")
                    print("Earliest steered text (truncated):")
                    print(earliest_txt[:1000])
                break
    else:
        if verbose:
            print("\nSkipping minimal-coefficient search (no passing random attempts detected).")

    return {
        "selected_positions": selected_positions,
        "matches_ref": matches_ref,  # list of (pos, latents)
        "removed_future": removed_future,  # list of (pos, latents, steered_text)
        "coeff_candidates": coeff_candidates,
        "found_min_coeff": found_coeff,
        "earliest_text_at_found": earliest_txt_at_found,
        "latent_resamples": latent_resamples,
        "resample_for_min_coeff": resample_for_min_coeff,
    }


# %%
# Load prompt data (same file used by original script)
print("[info] Loading prompt data...")
if not PROMPT_DATA_PATH.exists():
    raise FileNotFoundError(f"Could not locate prompt data file: {PROMPT_DATA_PATH}")
prompt_data = _load_json(PROMPT_DATA_PATH)
print(f"[info] Loaded {len(prompt_data)} prompt entries from {PROMPT_DATA_PATH}")

# %%
# Discover all prompt/token pairs
print("[info] Discovering prompt/token pairs...")
all_pairs = _discover_prompt_token_pairs()
print(f"[info] Found {len(all_pairs)} prompt/token pairs with required files")

# Group by prompt
pairs_by_prompt: Dict[int, List[Tuple[int, int, Path, Path]]] = {}
for prompt_idx, token_idx, classified_dir, data_dir in all_pairs:
    pairs_by_prompt.setdefault(prompt_idx, []).append((prompt_idx, token_idx, classified_dir, data_dir))

prompt_indices = sorted(pairs_by_prompt.keys())
print(f"[info] {len(prompt_indices)} unique prompts")

# %%
# Show sample of discovered data
if all_pairs:
    sample_prompt_idx, sample_token_idx, sample_classified_dir, sample_data_dir = all_pairs[0]
    print(f"\n[sample] First pair: prompt_{sample_prompt_idx}/token_{sample_token_idx}")
    print(f"  classified_dir: {sample_classified_dir}")
    print(f"  data_dir: {sample_data_dir}")

    # Show analysis file content
    analysis_path = sample_classified_dir / UPDATED_ANALYSIS_NAME
    if analysis_path.exists():
        analysis = _load_json(analysis_path)
        print(f"\n  updated_planning_analysis.json keys: {list(analysis.keys())}")
        for k, v in list(analysis.items())[:2]:
            print(f"    {k}: {v}")

    # Show metadata
    metadata_path = sample_data_dir / "metadata.json"
    if metadata_path.exists():
        metadata = _load_json(metadata_path)
        print(f"\n  metadata.json keys: {list(metadata.keys())[:10]}...")
        print(f"    prompt_idx: {metadata.get('prompt_idx')}")
        print(f"    inter_token_id: {metadata.get('inter_token_id')}")
        print(f"    baseline_text preview: {metadata.get('baseline_text', '')[:100]}...")

# %%
# Main processing loop
stats = {
    "prompts_total": len(prompt_indices),
    "tokens_total": len(all_pairs),
    "tokens_processed": 0,
    "tokens_skipped_no_futures": 0,
    "tokens_skipped_out_of_range": 0,
    "tokens_skipped_prompt_not_found": 0,
    "futures_total": 0,
    "futures_evaluated": 0,
    "futures_skipped": 0,
    "futures_not_planning": 0,
}
print(f"\n[config] Starting processing with stats: {stats}")

# Cache for prompt data (full dict with entry, prompt, out_BL)
prompt_cache: Dict[int, Dict[str, Any]] = {}

# %%
# Process all pairs
processed_prompts = 0
start_idx = START_PROMPT_IDX or 0

for prompt_idx in prompt_indices[start_idx:]:
    if MAX_PROMPTS is not None and processed_prompts >= MAX_PROMPTS:
        break

    prompt_pairs = pairs_by_prompt[prompt_idx]

    # Apply token limit if set
    if MAX_TOKENS_PER_PROMPT is not None:
        prompt_pairs = prompt_pairs[:MAX_TOKENS_PER_PROMPT]

    print(f"\n[processing] prompt_{prompt_idx} with {len(prompt_pairs)} tokens")

    # Check prompt_idx is valid in prompt_data
    if prompt_idx < 0 or prompt_idx >= len(prompt_data):
        print(f"  [skip] prompt_idx={prompt_idx} not found in prompt_data (len={len(prompt_data)})")
        stats["tokens_skipped_prompt_not_found"] += len(prompt_pairs)
        stats["tokens_processed"] += len(prompt_pairs)
        processed_prompts += 1
        continue

    # Get or generate out_BL for this prompt using autoregressive generation
    cache_entry = _get_prompt_cache(
        prompt_idx,
        model=model,
        device=DEVICE,
        data=prompt_data,
        cache=prompt_cache,
    )
    out_BL = cache_entry["out_BL"]

    for _, token_idx, classified_token_dir, data_token_dir in prompt_pairs:
        # Load analysis file (verdicts)
        analysis_path = classified_token_dir / UPDATED_ANALYSIS_NAME
        analysis = _load_json(analysis_path)

        # Filter futures to evaluate (Plan or Can't say)
        futures = []
        for future_tok, verdicts in analysis.items():
            if not isinstance(verdicts, dict):
                continue
            v0 = verdicts.get("original_verdict")
            v1 = verdicts.get("new_verdict")
            if (v0 in {"Plan", "Can't say"}) or (v1 in {"Plan", "Can't say"}):
                futures.append(future_tok)

        stats["futures_total"] += len(futures)

        if not futures:
            stats["tokens_skipped_no_futures"] += 1
            stats["tokens_processed"] += 1
            continue

        # Check token_idx is in range
        if token_idx >= out_BL.shape[-1]:
            print(f"  [skip] token_{token_idx}: out of range (out_BL has {out_BL.shape[-1]} tokens)")
            stats["tokens_skipped_out_of_range"] += 1
            stats["tokens_processed"] += 1
            continue

        # Load required data files
        circuit_path = data_token_dir / "circuit_entries.pt"
        clusters_path = data_token_dir / "clusters.json"

        circuit_entries = torch.load(circuit_path, map_location="cpu", weights_only=True)
        clusters = _load_json(clusters_path)

        baseline_suffix = model.to_string(out_BL[0, token_idx:])

        print(f"  [token_{token_idx}] {len(futures)} futures to evaluate: {futures[:5]}{'...' if len(futures) > 5 else ''}")

        for future_tok in futures:
            stats["futures_evaluated"] += 1
            selected_label = future_tok

            earliest_position_found, earliest_pos_steering, _pos_labels = _find_earliest_planning_position(
                model=model,
                saes=saes,
                out_BL=out_BL,
                baseline_suffix=baseline_suffix,
                token_i=token_idx,
                selected_label=selected_label,
                clusters=clusters,
            )

            if earliest_position_found is None or earliest_pos_steering is None:
                analysis.setdefault(selected_label, {})["random_steer_debug"] = {
                    "skipped_reason": "no_earliest_planning_position",
                }
                stats["futures_skipped"] += 1
                continue

            pairs_for_label = clusters.get(selected_label, [])
            earliest_latents: List[Tuple[int, int]] = []
            for li, latent_i, tok_positions in pairs_for_label:
                if earliest_position_found in tok_positions:
                    earliest_latents.append((int(li), int(latent_i)))

            if not earliest_latents:
                analysis.setdefault(selected_label, {})["random_steer_debug"] = {
                    "skipped_reason": "no_latents_at_earliest_position",
                }
                stats["futures_skipped"] += 1
                continue

            ref_entries = earliest_pos_steering[selected_label]["steered"]
            ref_entry = next((e for e in ref_entries if e.get("coeff") == TARGET_COEFF), None)
            if ref_entry is None and ref_entries:
                ref_entry = ref_entries[0]
            if ref_entry is None:
                analysis.setdefault(selected_label, {})["random_steer_debug"] = {
                    "skipped_reason": "no_reference_steered_entry",
                }
                stats["futures_skipped"] += 1
                continue

            reference_steered_text = _steered_text_to_str(ref_entry.get("steered_text"), model=model)

            random_eval = run_random_steer_eval_and_min_coeff(
                model=model,
                saes=saes,
                out_BL=out_BL,
                baseline_suffix=baseline_suffix,
                selected_label=selected_label,
                target_coeff=TARGET_COEFF,
                stop_tok=STOP_TOKEN_ID,
                max_tokens=MAX_TOKENS,
                token_i=token_idx,
                circuit_entries=circuit_entries,
                earliest_position_found=earliest_position_found,
                earliest_latents=earliest_latents,
                reference_steered_text=reference_steered_text,
                rng_obj=rng,
                position_sample_ratio=POSITION_SAMPLE_RATIO,
                per_position_latents_m=PER_POSITION_LATENTS_M,
                latent_resamples=LATENT_RESAMPLES,
                step_coeff=STEP_COEFF,
                resample_for_min_coeff=RESAMPLE_FOR_MIN_COEFF,
                verbose=False,
            )

            existing_verdict = analysis.get(selected_label, {}).get("new_verdict")
            if existing_verdict is None:
                existing_verdict = analysis.get(selected_label, {}).get("original_verdict")
            new_new_verdict = existing_verdict

            if random_eval.get("removed_future") and random_eval.get("found_min_coeff") is None:
                new_new_verdict = "Not planning"
                stats["futures_not_planning"] += 1

            analysis.setdefault(selected_label, {})["new_new_verdict"] = new_new_verdict
            analysis.setdefault(selected_label, {})["random_steer_debug"] = {
                "earliest_position_found": int(earliest_position_found),
                "earliest_latents_count": len(earliest_latents),
                "selected_positions": [int(p) for p in random_eval.get("selected_positions", [])],
                "matches_ref": [
                    {"pos": int(pos), "latents": _serialize_latents(lat_list)}
                    for (pos, lat_list) in random_eval.get("matches_ref", [])
                ],
                "removed_future": _serialize_removed_future(random_eval.get("removed_future", [])),
                "found_min_coeff": random_eval.get("found_min_coeff"),
            }

        # Save augmented analysis to the classified_token_dir
        output_path = classified_token_dir / AUGMENTED_ANALYSIS_NAME
        with open(output_path, "w") as f:
            json.dump(analysis, f, indent=2)

        stats["tokens_processed"] += 1

        if PROGRESS_EVERY > 0 and stats["tokens_processed"] % PROGRESS_EVERY == 0:
            print(
                f"[progress] {stats['tokens_processed']}/{stats['tokens_total']} tokens | "
                f"futures evaluated={stats['futures_evaluated']} | "
                f"not_planning={stats['futures_not_planning']}"
            )

    processed_prompts += 1

# %%
# Final stats
print("\n" + "=" * 60)
print("Random steering multi-token analysis complete!")
print("=" * 60)
print(f"Final stats: {json.dumps(stats, indent=2)}")

# %%
