# %%
from __future__ import annotations

import json
import os
import sys
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
sys.path.append("../")
from plan_trace.steering import sweep_coefficients_multi
from plan_trace.utils import load_model, load_pretrained_saes

# %%
# Configuration
ROOT_DIR = Path("/home/jnainani_umass_edu/w/plan_trace")

PARENT_DIR = Path("/work/pi_jensen_umass_edu/abhishekmish_umass_edu/plan_trace/")
OUTPUT_ROOT = PARENT_DIR / "outputs" / "all_scale" / "instruct"
MODEL_NAME = "gemma-2-2b-it"
DEVICE = "cuda"

TARGET_COEFF = -200
NUM_RANDOM_TRIALS = 100
SEED = 0
MAX_TOKENS = 100

# Which files to read per token directory.
PLANNING_FILES = ("earliest_position_planning_analysis.json", "planning_analysis.json")
STEERING_FILES = ("earliest_position.json", "steering_results.json")


# %% 

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
    step_coeff: int = 25,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Step 3+ logic as a reusable function:
      - sample random positions from original circuit (ratio-based or fixed cap)
      - sample latents per position
      - run steering at target_coeff and count only changes where the future token is removed
      - search the smallest |coeff| from 0 toward target_coeff that reproduces earliest reference and avoids random effects
    
    Returns an informative dict with selections, results and the chosen minimal coefficient (if any).
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
    }


# %%
"""
TODOs for scaling this experiment across many prompts/tokens:

1) Iterate over driver JSON to pick (prompt_idx, token_idx) pairs
   - Source: /work/pi_jensen_umass_edu/jnainani_umass_edu/plan_trace/data/base_vs_instruct_oracle_c200.json
   - For each record: read {mode, prompt_idx, token_idx}, where mode ∈ {"instruct","base"}

2) Check for per-token analysis file existence
   - Path pattern:
     /work/pi_jensen_umass_edu/jnainani_umass_edu/plan_trace/outputs/scale_bvi_c200_first_change/{mode}/prompt_{prompt_idx}/token_{token_idx}/updated_planning_analysis.json
   - If exists, load the JSON. It maps future-token → verdicts:
       {
         "<future_token>": { "original_verdict": "...", "new_verdict": "..." },
         ...
       }

3) Filter futures to evaluate
   - Keep futures where any verdict is "Plan" or "Can't say" (original or new)
   - For each kept future: set selected_label = future token string (e.g., "+")
   - Prepare model/saes/out_BL/baseline and circuit artifacts (like in this notebook)

4) Run random-steer evaluation
   - Call run_random_steer_eval_and_min_coeff(...) with selected params
   - If random check removes the future at any sampled position AND we cannot find a smaller coefficient
     that preserves the earliest reference while removing random effects:
       → set "new_new_verdict" for that future to "Not planning"
   - Otherwise leave verdicts unchanged

5) Save augmented results
   - Write an updated JSON beside updated_planning_analysis.json (or add a new key into it), e.g.:
       "new_new_verdict": "Not planning" | "Plan" | "Can't say"
   - Keep logs: selected positions, removed_future list, and found_min_coeff for debugging

6) Parallelization/efficiency
   - Reuse model/SAE loads
   - Cache tokens/out_BL per prompt_idx
   - Consider smaller step_coeff for finer search near 0 if needed
"""
# %% 

# Scaling implementation for the TODOs above.
from plan_trace.steering import run_steering_sweep
from plan_trace.ood_detect import label_steering_clusters

DRIVER_JSON_CANDIDATES = [
    ROOT_DIR / "data" / "base_vs_instruct_oracle_c200.json",
    Path("/work/pi_jensen_umass_edu/jnainani_umass_edu/plan_trace/data/base_vs_instruct_oracle_c200.json"),
]
PROMPT_DATA_CANDIDATES = [
    ROOT_DIR / "data" / "external" / "sanitized-mbpp.json",
    Path("/work/pi_jensen_umass_edu/jnainani_umass_edu/plan_trace/data/external/sanitized-mbpp.json"),
]
OG_PARENT_DIR = Path("/work/pi_jensen_umass_edu/jnainani_umass_edu/plan_trace/")
SCALE_OUTPUT_ROOT = OG_PARENT_DIR / "outputs" / "scale_bvi_c200_first_change"
SCALE_DATA_OUTPUT_ROOT = PARENT_DIR / "outputs" / "all_scale"
UPDATED_ANALYSIS_NAME = "updated_planning_analysis.json"
AUGMENTED_ANALYSIS_NAME = "updated_planning_analysis_random_steer.json"

STOP_TOKEN_ID = 1917
GEN_LIMIT = 150
POSITION_SAMPLE_RATIO = 0.2
PER_POSITION_LATENTS_M = 5
STEP_COEFF = 25

MAX_RECORDS: Optional[int] = 100 # None  # set to an int for a quick smoke run
MAX_FUTURES_PER_TOKEN: Optional[int] = None  # cap futures evaluated per token
PRINT_FIRST_N = 3

# %%
def _resolve_first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _build_prompt(entry: Dict[str, Any]) -> str:
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
    if prompt_idx in cache:
        return cache[prompt_idx]

    entry = data[prompt_idx + 1]
    prompt = _build_prompt(entry)
    toks_BL = model.to_tokens(prompt).to(device)
    out_BL = toks_BL.clone()

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

# %%
driver_path = _resolve_first_existing(DRIVER_JSON_CANDIDATES)
if driver_path is None:
    raise FileNotFoundError("Could not locate driver JSON for base_vs_instruct_oracle_c200.")

prompt_data_path = _resolve_first_existing(PROMPT_DATA_CANDIDATES)
if prompt_data_path is None:
    raise FileNotFoundError("Could not locate sanitized-mbpp.json for prompts.")

driver_records = _load_json(driver_path)
if isinstance(driver_records, dict) and "records" in driver_records:
    driver_records = driver_records["records"]

prompt_data = _load_json(prompt_data_path)
prompt_cache: Dict[int, Dict[str, Any]] = {}
task_id_to_index: Dict[int, int] = {}
for idx, entry in enumerate(prompt_data):
    try:
        task_id = int(entry.get("task_id"))
    except Exception:
        continue
    # Keep the first index if duplicates exist.
    # NOTE: outputs were created with PROMPT_IDX where entry = data[PROMPT_IDX + 1],
    # so prompt_idx is (data_index - 1).
    task_id_to_index.setdefault(task_id, idx - 1)


# %%

stats = {
    "records_total": len(driver_records),
    "records_skipped_no_analysis": 0,
    "records_processed": 0,
    "futures_total": 0,
    "futures_evaluated": 0,
    "futures_skipped": 0,
    "futures_not_planning": 0,
}
printed_samples = 0
for rec_i, rec in enumerate(driver_records):
    if MAX_RECORDS is not None and rec_i >= MAX_RECORDS:
        break

    mode = rec.get("mode", "instruct")
    if mode != "instruct":
        continue

    task_id = rec.get("task_id")
    if task_id is None:
        print(f"Skipping record with missing task_id: {rec}")
        continue
    task_id = int(task_id)
    prompt_idx = task_id_to_index.get(task_id)
    if prompt_idx is None:
        print(f"Skipping task_id={task_id}: not found in sanitized-mbpp.json")
        continue
    if prompt_idx < 0:
        print(f"Skipping task_id={task_id}: mapped prompt_idx={prompt_idx} invalid")
        continue

    position_info = rec.get("position_info", {})
    token_idx = position_info.get("instruct_token_pos")
    if token_idx is None:
        print(f"Skipping task_id={task_id}: missing instruct_token_pos")
        continue
    token_idx = int(token_idx)

    token_dir = SCALE_OUTPUT_ROOT / mode / f"prompt_{prompt_idx}" / f"token_{token_idx}"
    token_data_dir = SCALE_DATA_OUTPUT_ROOT / mode / f"prompt_{prompt_idx}" / f"token_{token_idx}"
    analysis_path = token_dir / UPDATED_ANALYSIS_NAME
    circuit_path = token_data_dir / "circuit_entries.pt"
    if not analysis_path.exists() or not circuit_path.exists():
        stats["records_skipped_no_analysis"] += 1
        continue

    analysis = _load_json(analysis_path)
    futures = []
    for future_tok, verdicts in analysis.items():
        if not isinstance(verdicts, dict):
            continue
        v0 = verdicts.get("original_verdict")
        v1 = verdicts.get("new_verdict")
        if (v0 in {"Plan", "Can't say"}) or (v1 in {"Plan", "Can't say"}):
            futures.append(future_tok)

    stats["futures_total"] += len(futures)
    if MAX_FUTURES_PER_TOKEN is not None:
        futures = futures[:MAX_FUTURES_PER_TOKEN]

    if not futures:
        stats["records_processed"] += 1
        continue

    if printed_samples < PRINT_FIRST_N:
        try:
            sample_entry = prompt_data[prompt_idx]
            sample_prompt = sample_entry.get("prompt", "")
        except Exception:
            sample_prompt = ""
        print(
            f"[sample {rec_i}] task_id={task_id} prompt_idx={prompt_idx} "
            f"token_idx={token_idx} prompt='{sample_prompt[:120]}'"
        )
        printed_samples += 1

    cache_entry = _get_prompt_cache(
        prompt_idx,
        model=model,
        device=DEVICE,
        data=prompt_data,
        cache=prompt_cache,
    )
    out_BL = cache_entry["out_BL"]

    if token_idx >= out_BL.shape[-1]:
        print(f"Skipping prompt {prompt_idx}, token {token_idx}: token_idx out of range.")
        stats["records_processed"] += 1
        continue

    baseline_suffix = model.to_string(out_BL[0, token_idx:])

    circuit_path = token_data_dir / "circuit_entries.pt"
    clusters_path = token_data_dir / "clusters.json"
    if (not circuit_path.exists()) or (not clusters_path.exists()):
        print(f"Skipping prompt {prompt_idx}, token {token_idx}: missing circuit/clusters.")
        stats["records_processed"] += 1
        continue

    circuit_entries = torch.load(circuit_path, map_location="cpu")
    clusters = _load_json(clusters_path)

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
            step_coeff=STEP_COEFF,
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

    output_path = token_dir / AUGMENTED_ANALYSIS_NAME
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)

    stats["records_processed"] += 1

print("Random steering scale run stats:", stats)

# %%
"""
Code from below was used to run experiments on a single prompt and token and optimize code and make it reusable for the rest of the experiments.
"""

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

"""
prompt_idx = 7 
step_i = 248 
g_i = "rows"
step_j1 = 317
g_j1 = "+" 
g_j2 = "[]"
earliest_position = 21

"""
"""
1. load the prompt idx 
2. load the token i
3. prepare the input and print the baseline generation
4. load circuit, clusters, metadata for token i
5. rerun per position steering sweep for token i and reproduce the planning analysis for g_j1 
"""

# %%
# Step 1 & 2: Load indices (prompt idx and token i) and target cluster
PROMPT_IDX: int = 7
TOKEN_I: int = 248
SELECTED_LABEL: str = "+"
STOP_TOKEN_ID: int = 1917  # matches pipeline's stop token (``` token)
GEN_LIMIT: int = 150       # number of tokens to generate beyond prompt (like pipeline)


# %%
# Step 3: Prepare the input and print the baseline generation
from plan_trace.steering import run_steering_sweep
from plan_trace.ood_detect import label_steering_clusters

data_candidates: List[Path] = [
    ROOT_DIR / "data" / "external" / "sanitized-mbpp.json",
    # Path("/work/pi_jensen_umass_edu/jnainani_umass_edu/plan_trace/data/first_100_passing_examples.json"),
    # Path("../data/first_100_passing_examples.json").resolve(),
]

DATA_PATH: Optional[Path] = next((p for p in data_candidates if p.exists()), None)
if DATA_PATH is None:
    raise FileNotFoundError("Could not locate data/first_100_passing_examples.json")

with open(DATA_PATH, "r") as f:
    data = json.load(f)

entry = data[PROMPT_IDX+1]
print(entry)

# %%
# prompt = (
#     "You are an expert Python programmer, and here is your task: "
#     f"{entry['prompt']} Your code should pass these tests:\n\n"
#     + "\n".join(entry["test_list"])
#     + "\nWrite your code below starting with \"```python\" and ending with \"```\".\n```python\n"
# )
prompt = (
    "You are an expert Python programmer, and here is your task: "
    f"{entry['prompt']} Your code should pass these tests:\n\n"
    + "\n".join(entry["test_list"]) + "\nWrite your code, without docstrings, below starting with \"```python\" and ending with \"```\".\n```python\n"
)

toks_BL = model.to_tokens(prompt).to(DEVICE)
out_BL = toks_BL.clone()

while out_BL.shape[-1] - toks_BL.shape[-1] < GEN_LIMIT:
    with torch.no_grad():
        logits_V = model(out_BL)[0, -1]
    next_id = logits_V.argmax(-1).item()
    del logits_V
    if next_id == STOP_TOKEN_ID:
        break
    out_BL = torch.cat([out_BL, torch.tensor([[next_id]], device=DEVICE)], dim=1)

full_response = model.to_string(out_BL[0])
baseline_suffix = model.to_string(out_BL[0, TOKEN_I:])
print("Full generated response:\n", full_response)
print("\nBaseline continuation from TOKEN_I", TOKEN_I, ":\n", baseline_suffix[:300], "...")


# %%
# Step 4: Load circuit, clusters, metadata for token i from the fixed parent dir
token_dir = OUTPUT_ROOT / f"prompt_{PROMPT_IDX}" / f"token_{TOKEN_I}"
print("Loading artifacts from:", token_dir)

circuit_entries = torch.load(token_dir / "circuit_entries.pt", map_location="cpu")
with open(token_dir / "clusters.json", "r") as f:
    clusters: Dict[str, Any] = json.load(f)
with open(token_dir / "metadata.json", "r") as f:
    metadata: Dict[str, Any] = json.load(f)
with open(token_dir / "steering_results.json", "r") as f:
    steering_results_saved: Dict[str, Any] = json.load(f)

print("Artifacts loaded:",
      f"\n- circuit_entries: {len(circuit_entries)} entries",
      f"\n- clusters: {list(clusters.keys())}",
      f"\n- metadata keys: {list(metadata.keys())}",
      f"\n- steering_results (labels): {list(steering_results_saved.keys())}")


# %%
# Step 5: Rerun per-position steering sweep for TOKEN_I and reproduce planning analysis for SELECTED_LABEL
if SELECTED_LABEL not in clusters:
    raise KeyError(f"Cluster label '{SELECTED_LABEL}' not found in clusters.json")

pairs_for_label: List[List[Any]] = clusters[SELECTED_LABEL]
positions: List[int] = sorted({
    tok_pos
    for (_li, _latent_i, tok_positions) in pairs_for_label
    for tok_pos in tok_positions
})

print(f"Running per-position sweep for label '{SELECTED_LABEL}' across {len(positions)} positions...")
inter_toks_BL = out_BL[:, :TOKEN_I]

earliest_position_found: Optional[int] = None
earliest_pos_steering: Optional[Dict[str, Any]] = None
earliest_pos_labels: Optional[Dict[str, Any]] = None

for tok_pos in positions:
    # Build filtered dict for this single token position, only for the selected label
    filtered = {SELECTED_LABEL: []}
    for li, latent_i, tok_positions in pairs_for_label:
        if tok_pos in tok_positions:
            filtered[SELECTED_LABEL].append([li, latent_i, [tok_pos]])

    # Run steering sweep at this position
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

    # Analyze planning label for this position
    pos_labels = label_steering_clusters(pos_steering, model=model, prefix_tokens_2d=inter_toks_BL)
    label_info = pos_labels.get(SELECTED_LABEL, {})
    final_label = label_info.get("final_label", "Can't say")
    print(f"pos={tok_pos}: final_label for '{SELECTED_LABEL}' -> {final_label}")

    if final_label == "Plan":
        earliest_position_found = tok_pos
        earliest_pos_steering = pos_steering
        earliest_pos_labels = pos_labels
        break

if earliest_position_found is not None:
    print(f"\nEarliest planning position for label '{SELECTED_LABEL}': {earliest_position_found}")
else:
    print(f"\nNo planning position found for label '{SELECTED_LABEL}' in positions: {positions}")

# %%
"""
1. load the latents in the earliest position and print the layer, latent and the string of the token at the earliest position. 
2. print the steered generation for the above check and save it in a variable
3. load the circuit and sample N=5 random token positions along with the earliest position from above.
3a. sample M=5 random latents that are present in the circuit for each token position in step 3 
3b. run the steering sweep for the latents and check if any of the steered generations are the same as the steered generation for the above check.

"""

# %%
# Step 1: Load latents in the earliest position and print layer, latent and token string
if 'earliest_position_found' not in globals() or earliest_position_found is None:
    raise RuntimeError("earliest_position_found is not available. Run the earlier cells first.")

if 'clusters' not in globals():
    raise RuntimeError("clusters not loaded. Run the earlier cells that load artifacts.")

pairs_for_label_step2: List[List[Any]] = clusters[SELECTED_LABEL]
earliest_latents: List[Tuple[int, int]] = []
for li, latent_i, tok_positions in pairs_for_label_step2:
    if earliest_position_found in tok_positions:
        earliest_latents.append((int(li), int(latent_i)))

prefix_to_earliest = model.to_string(out_BL[0, : earliest_position_found + 1])
token_str_list = model.to_str_tokens(prefix_to_earliest)
token_at_earliest = token_str_list[-1] if token_str_list else ""

print(f"Earliest position: {earliest_position_found}")
print(f"Token at earliest position: {token_at_earliest!r}")
print(f"Num latents at earliest position for '{SELECTED_LABEL}': {len(earliest_latents)}")
for (li, latent_i) in earliest_latents:
    print(f"  layer={li}, latent={latent_i}")


# %%
# Step 2: Print the steered generation for the above check (earliest position) and save it
if 'earliest_pos_steering' not in globals() or earliest_pos_steering is None:
    raise RuntimeError("earliest_pos_steering not found. Run the per-position sweep cell first.")

ref_entries = earliest_pos_steering[SELECTED_LABEL]["steered"]
ref_entry = None
for e in ref_entries:
    if e.get("coeff", None) == TARGET_COEFF:
        ref_entry = e
        break
if ref_entry is None:
    ref_entry = ref_entries[0]

val = ref_entry.get("steered_text")
reference_steered_text: str
if hasattr(val, "tolist"):
    reference_steered_text = model.to_string(val.tolist())
elif isinstance(val, list):
    reference_steered_text = model.to_string(val)
else:
    reference_steered_text = str(val)

print("\nReference steered generation (earliest position):")
print(reference_steered_text[:1000])


# %%
# Step 3, 3a, 3b:
# - Sample N=5 random positions (plus earliest) from the ORIGINAL CIRCUIT (not clusters)
# - For each position, sample M=5 random latents present in the circuit for that position
# - Run a steering sweep and check if any steered generations match the reference
# Build position -> [(layer, latent)] mapping from circuit_entries (layer, token, latent, value)

POSITION_SAMPLE_RATIO = 0.2
circuit_pos_to_latents: Dict[int, List[Tuple[int, int]]] = {}
for entry in circuit_entries:
    try:
        layer_i, tok_pos, latent_i, _val = entry
    except Exception:
        # Fallback if entry is not a 4-tuple
        # Try to coerce common shapes like dicts or longer tuples
        if isinstance(entry, dict):
            layer_i = int(entry.get("layer", 0))
            tok_pos = int(entry.get("token", 0))
            latent_i = int(entry.get("latent", 0))
        else:
            layer_i = int(entry[0])
            tok_pos = int(entry[1])
            latent_i = int(entry[2])
    circuit_pos_to_latents.setdefault(int(tok_pos), []).append((int(layer_i), int(latent_i)))

all_positions: List[int] = sorted(circuit_pos_to_latents.keys())

positions_wo_earliest = [p for p in all_positions if p != earliest_position_found]
# Optional ratio-based sampling: define POSITION_SAMPLE_RATIO (e.g., 0.3) in a prior cell to override fixed N
if 'POSITION_SAMPLE_RATIO' in globals() and isinstance(POSITION_SAMPLE_RATIO, float) and 0.0 < POSITION_SAMPLE_RATIO <= 1.0:
    sample_n = int(round(len(positions_wo_earliest) * POSITION_SAMPLE_RATIO))
    sample_n = max(0, min(sample_n, len(positions_wo_earliest)))
else:
    sample_n = min(5, len(positions_wo_earliest))
sampled_positions = rng.sample(positions_wo_earliest, sample_n) if sample_n > 0 else []
selected_positions: List[int] = [earliest_position_found] + sampled_positions

print(f"\nSelected positions (including earliest): {selected_positions}")

def latents_for_position(p: int) -> List[Tuple[int, int]]:
    return circuit_pos_to_latents.get(int(p), [])

matches_ref: List[Tuple[int, List[Tuple[int, int]]]] = []          # (pos, sampled_latents) that reproduced reference
changes_nonref: List[Tuple[int, List[Tuple[int, int]], str]] = []  # (pos, sampled_latents, steered_text) changed vs base but != reference

for pos in selected_positions:
    available_latents = latents_for_position(pos)
    if not available_latents:
        print(f"Position {pos}: no latents available, skipping.")
        continue
    m = min(5, len(available_latents))
    sampled_latents = rng.sample(available_latents, m)

    # Build filtered dict for this position with sampled latents
    filtered = {SELECTED_LABEL: []}
    for li, latent_i in sampled_latents:
        filtered[SELECTED_LABEL].append([li, latent_i, [pos]])

    pos_steering = run_steering_sweep(
        model=model,
        saes=saes,
        inter_toks_BL=out_BL[:, :TOKEN_I],
        saved_pair_dict=filtered,
        baseline_text=baseline_suffix,
        coeff_grid=[TARGET_COEFF],
        stop_tok=STOP_TOKEN_ID,
        max_tokens=MAX_TOKENS,
        return_tokens=True,
    )

    entries = pos_steering[SELECTED_LABEL]["steered"]
    base_text = pos_steering[SELECTED_LABEL]["base_text"]
    # With a single coefficient, we expect one entry
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

    # Treat empty steered output as "no change"
    if steered_text is None or steered_text == "" or steered_text.strip() == "":
        # optional: uncomment to log no-change
        # print(f"No change at position {pos} (empty steered output)")
        continue

    # Early exit: treat empty as no-change
    if steered_text is None or steered_text.strip() == "":
        continue

    # Helper: does steered contain the future token label?
    contains_future = (SELECTED_LABEL in steered_text)

    if steered_text == reference_steered_text:
        print(f"Match to reference at position {pos} using {len(sampled_latents)} latents")
        matches_ref.append((pos, sampled_latents))
    # Count only if FUTURE TOKEN IS REMOVED (important change)
    elif (steered_text != base_text) and (not contains_future):
        print(f"Future '{SELECTED_LABEL}' removed at position {pos} using {len(sampled_latents)} latents")
        changes_nonref.append((pos, sampled_latents, steered_text))

if matches_ref:
    print("\nReproduced reference steered generation at:")
    for (pos, lat_list) in matches_ref:
        print(f"  pos={pos}, latents={lat_list[:5]}{'...' if len(lat_list) > 5 else ''}")
else:
    print("\nNo matches to the reference steered generation were found among sampled latents.")

if changes_nonref:
    print(f"\nSteered generations where future token '{SELECTED_LABEL}' is removed:")
    for (pos, lat_list, steered_txt) in changes_nonref:
        print(f"  pos={pos}, latents={lat_list[:5]}{'...' if len(lat_list) > 5 else ''}")
        print(steered_txt[:1000])


# %%
# Coefficient search: find smallest |coeff| from 0 toward TARGET_COEFF that
# reproduces earliest reference but causes no random effects on previously changed trials.
if ('changes_nonref' in globals() and changes_nonref) or ('matches_ref' in globals() and matches_ref):
    print("\nStarting minimal-coefficient search for earliest position (no random effects)...")
    # Build candidate coefficients from 0 toward TARGET_COEFF
    step = 25
    if TARGET_COEFF < 0:
        coeff_candidates = [0] + [ -c for c in range(step, abs(TARGET_COEFF) + step, step) ]
    else:
        coeff_candidates = [0] + [ c for c in range(step, abs(TARGET_COEFF) + step, step) ]
    # Restrict to within original magnitude
    coeff_candidates = [c for c in coeff_candidates if abs(c) <= abs(TARGET_COEFF)]

    # Build earliest-position filtered latents for the selected label
    earliest_filtered = {SELECTED_LABEL: []}
    for li, latent_i in earliest_latents:
        earliest_filtered[SELECTED_LABEL].append([li, latent_i, [earliest_position_found]])

    found_coeff = None
    for cand in coeff_candidates:
        # 1) Check earliest reproduces reference
        earliest_sweep = run_steering_sweep(
            model=model,
            saes=saes,
            inter_toks_BL=out_BL[:, :TOKEN_I],
            saved_pair_dict=earliest_filtered,
            baseline_text=baseline_suffix,
            coeff_grid=[cand],
            stop_tok=STOP_TOKEN_ID,
            max_tokens=MAX_TOKENS,
            return_tokens=True,
        )
        earliest_entries = earliest_sweep[SELECTED_LABEL]["steered"]
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

        # 2) Check no random effects on previously changed trials
        ok_random = True
        for (pos, lat_list, _prev_txt) in changes_nonref:
            filtered = {SELECTED_LABEL: [[li, latent_i, [pos]] for (li, latent_i) in lat_list]}
            rnd_sweep = run_steering_sweep(
                model=model,
                saes=saes,
                inter_toks_BL=out_BL[:, :TOKEN_I],
                saved_pair_dict=filtered,
                baseline_text=baseline_suffix,
                coeff_grid=[cand],
                stop_tok=STOP_TOKEN_ID,
                max_tokens=MAX_TOKENS,
                return_tokens=True,
            )
            rnd_entries = rnd_sweep[SELECTED_LABEL]["steered"]
            if not rnd_entries:
                continue
            rnd_val = rnd_entries[0].get("steered_text")
            if hasattr(rnd_val, "tolist"):
                rnd_txt = model.to_string(rnd_val.tolist())
            elif isinstance(rnd_val, list):
                rnd_txt = model.to_string(rnd_val)
            else:
                rnd_txt = str(rnd_val)
            base_txt = rnd_sweep[SELECTED_LABEL]["base_text"]
            # Treat as no random effect if empty, equals base, or FUTURE TOKEN STILL PRESENT
            if rnd_txt is None or rnd_txt.strip() == "" or rnd_txt == base_txt or (SELECTED_LABEL in rnd_txt):
                continue
            ok_random = False
            break

        if ok_random:
            found_coeff = cand
            print(f"Found minimal coefficient with no random effects: {found_coeff}")
            print("Earliest steered text (truncated):")
            print(earliest_txt[:1000])
            break

    if found_coeff is None:
        print("No coefficient found (within |TARGET_COEFF|) that preserves earliest reference and removes random effects.")
else:
    print("\nSkipping minimal-coefficient search (no passing random attempts detected).")


"""



"""

# %%
