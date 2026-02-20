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
MODEL_NAME = "gemma-2-2b-it"
DEVICE = "cuda"

RUN_MODE = "base" #os.getenv("RUN_MODE", "instruct")
if RUN_MODE not in {"instruct", "base"}:
    raise ValueError(f"RUN_MODE must be 'instruct' or 'base', got: {RUN_MODE!r}")
TOKEN_POS_KEY = "instruct_token_pos" if RUN_MODE == "instruct" else "base_token_pos"
DRIVER_CODE_KEY = "instruct_code" if RUN_MODE == "instruct" else "model_output"
OUTPUT_ROOT = PARENT_DIR / "outputs" / "all_scale" / RUN_MODE
MODEL_NAME = "gemma-2-2b-it" if RUN_MODE == "instruct" else "gemma-2-2b"

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
    latent_resamples: int = 1,
    step_coeff: int = 25,
    resample_for_min_coeff: bool = False,
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
LATENT_RESAMPLES = 3
STEP_COEFF = 25
RESAMPLE_FOR_MIN_COEFF = False

MAX_RECORDS: Optional[int] = None  # set to an int for a quick smoke run
START_RECORD_IDX: Optional[int] = 0 # None  # set to an int for a quick smoke run
MAX_FUTURES_PER_TOKEN: Optional[int] = None  # cap futures evaluated per token
PRINT_FIRST_N = 100
PROGRESS_EVERY = 25

# %%
def _resolve_first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _build_prompt(entry: Dict[str, Any], mode: str = "instruct") -> str:
    if mode == "instruct":
        return (
        "You are an expert Python programmer, and here is your task: "
        f"{entry['prompt']} Your code should pass these tests:\n\n"
        + "\n".join(entry["test_list"])
        + "\nWrite your code, without docstrings, below starting with \"```python\" and ending with \"```\".\n```python\n"
        )
    elif mode == "base":
        return (
        "You are an expert Python programmer, and here is your task: "
        f"{entry['prompt']} Your code should pass these tests:\n\n"
        + "\n".join(entry["test_list"])
        + "\nWrite your code below starting with \"```python\" and ending with \"```\".\n```python\n"
        )

def _get_prompt_cache(
    prompt_idx: int,
    *,
    model,
    device: str,
    data: Sequence[Dict[str, Any]],
    cache: Dict[int, Dict[str, Any]],
    mode: str = "instruct",
) -> Dict[str, Any]:
    if prompt_idx in cache:
        return cache[prompt_idx]

    entry = data[prompt_idx] #data[prompt_idx + 1]
    prompt = _build_prompt(entry, mode=RUN_MODE)
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

print(
    f"[config] RUN_MODE={RUN_MODE} TOKEN_POS_KEY={TOKEN_POS_KEY} "
    f"DRIVER_CODE_KEY={DRIVER_CODE_KEY} MODEL_NAME={MODEL_NAME}"
)
print(f"[config] driver_path={driver_path}")
print(f"[config] prompt_data_path={prompt_data_path}")

driver_records = _load_json(driver_path)
if isinstance(driver_records, dict) and "records" in driver_records:
    driver_records = driver_records["records"]

new_prompt_path = "/home/jnainani_umass_edu/w/plan_trace/data/external/all_examples_og_prompt_with_position_info_and_success_V2.json"
prompt_data = _load_json(new_prompt_path)
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
    task_id_to_index.setdefault(task_id, idx)


# %%

stats = {
    "records_total": len(driver_records),
    "records_skipped_no_analysis": 0,
    "records_skipped_mode": 0,
    "records_skipped_missing_task_id": 0,
    "records_skipped_prompt_idx_missing": 0,
    "records_skipped_prompt_idx_invalid": 0,
    "records_skipped_missing_token_pos": 0,
    "records_processed": 0,
    "futures_total": 0,
    "futures_evaluated": 0,
    "futures_skipped": 0,
    "futures_not_planning": 0,
}
print(stats)
printed_samples = 0
start_idx = START_RECORD_IDX or 0
records_iter = driver_records[start_idx:] if START_RECORD_IDX is not None else driver_records
for rec_i, rec in enumerate(records_iter, start=start_idx):

    if MAX_RECORDS is not None and rec_i >= MAX_RECORDS + START_RECORD_IDX:
        break

    # mode = rec.get("mode", RUN_MODE)
    mode = RUN_MODE
    # if mode != RUN_MODE:
    #     stats["records_skipped_mode"] += 1
    #     continue

    task_id = rec.get("task_id")
    if task_id is None:
        print(f"Skipping record with missing task_id: {rec}")
        stats["records_skipped_missing_task_id"] += 1
        continue
    task_id = int(task_id)
    prompt_idx = task_id_to_index.get(task_id)
    if prompt_idx is None:
        print(f"Skipping task_id={task_id}, prompt_idx={prompt_idx}: not found in sanitized-mbpp.json")
        stats["records_skipped_prompt_idx_missing"] += 1
        continue
    if prompt_idx < 0:
        print(f"Skipping task_id={task_id}, prompt_idx={prompt_idx}: mapped prompt_idx={prompt_idx} invalid")
        stats["records_skipped_prompt_idx_invalid"] += 1
        continue

    position_info = rec.get("position_info", {})
    token_idx = position_info.get(TOKEN_POS_KEY)
    if token_idx is None:
        print(f"Skipping task_id={task_id}, prompt_idx={prompt_idx}: missing {TOKEN_POS_KEY}")
        stats["records_skipped_missing_token_pos"] += 1
        continue
    token_idx = int(token_idx)

    token_dir = SCALE_OUTPUT_ROOT / mode / f"prompt_{prompt_idx}" / f"token_{token_idx}"
    token_data_dir = SCALE_DATA_OUTPUT_ROOT / mode / f"prompt_{prompt_idx}" / f"token_{token_idx}"
    analysis_path = token_dir / UPDATED_ANALYSIS_NAME
    circuit_path = token_data_dir / "circuit_entries.pt"
    if not analysis_path.exists() or not circuit_path.exists():
        print(f"Skipping prompt {prompt_idx}, token {token_idx}: missing analysis/circuit.")
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
        sample_code = rec.get(DRIVER_CODE_KEY, "")
        print(
            f"[sample {rec_i}] task_id={task_id} prompt_idx={prompt_idx} "
            f"token_idx={token_idx} prompt='{sample_prompt[:120]}'"
        )
        if sample_code:
            print(f"[sample {rec_i}] {DRIVER_CODE_KEY}='{sample_code[:120]}'")
        printed_samples += 1

    cache_entry = _get_prompt_cache(
        prompt_idx,
        model=model,
        device=DEVICE,
        data=prompt_data,
        cache=prompt_cache,
        mode=RUN_MODE,
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

    circuit_entries = torch.load(circuit_path, map_location="cpu", weights_only=True)
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

    output_path = token_dir / AUGMENTED_ANALYSIS_NAME
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)

    stats["records_processed"] += 1
    if PROGRESS_EVERY > 0 and stats["records_processed"] % PROGRESS_EVERY == 0:
        total_records = len(driver_records)
        print(
            f"[progress] {stats['records_processed']}/{total_records} records processed | "
            f"futures evaluated={stats['futures_evaluated']} | "
            f"not_planning={stats['futures_not_planning']} | "
            f"skipped_no_analysis={stats['records_skipped_no_analysis']}"
        )

print("Random steering scale run stats:", stats)

# %%