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
OUTPUT_ROOT = PARENT_DIR / "outputs" / "planning_scale" / "instruct"
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
def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _find_first_existing(token_dir: Path, names: Sequence[str]) -> Optional[Path]:
    for name in names:
        candidate = token_dir / name
        if candidate.exists():
            return candidate
    return None


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip()
    return str(value)


def _extract_verdict(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("final_label", "new_verdict", "original_verdict"):
            if key in value and isinstance(value[key], str):
                return value[key]
    return None


def _normalize_tokens(value: Any) -> Optional[Tuple[int, ...]]:
    if isinstance(value, list) and all(isinstance(v, int) for v in value):
        return tuple(value)
    return None


def _extract_target_entry(
    steering_results: Dict[str, Any],
    label: str,
    coeff: int,
) -> Optional[Dict[str, Any]]:
    entries = steering_results.get(label, {}).get("steered", [])
    exact = next((e for e in entries if e.get("coeff") == coeff), None)
    if exact is not None:
        return exact
    if not entries:
        return None
    return min(entries, key=lambda e: abs(e.get("coeff", 0) - coeff))


def _get_position(metadata: Dict[str, Any]) -> Optional[int]:
    if metadata.get("earliest_position") is not None:
        return int(metadata["earliest_position"])
    if metadata.get("inter_token_id") is not None:
        return int(metadata["inter_token_id"])
    return None


def _label_latents_at_position(
    clusters: Dict[str, List[Tuple[int, int, List[int]]]],
    label: str,
    position: Optional[int],
) -> List[Tuple[int, int, int]]:
    latents: List[Tuple[int, int, int]] = []
    for layer_i, latent_i, tok_positions in clusters.get(label, []):
        if position is None:
            for tok_pos in tok_positions:
                latents.append((layer_i, tok_pos, latent_i))
        elif position in tok_positions:
            latents.append((layer_i, position, latent_i))
    return latents


def _sample_random_interventions(
    circuit_entries: Sequence[Tuple[int, int, int, float]],
    position: Optional[int],
    n_latents: int,
    rng: random.Random,
) -> List[Tuple[int, int, int]]:
    if n_latents <= 0:
        return []

    candidates: List[Tuple[int, int, int]] = []
    for layer_i, tok_pos, latent_i, _ in circuit_entries:
        if position is None or tok_pos == position:
            candidates.append((layer_i, tok_pos, latent_i))

    if len(candidates) < n_latents:
        candidates = [
            (layer_i, tok_pos, latent_i)
            for layer_i, tok_pos, latent_i, _ in circuit_entries
        ]

    unique_candidates = list({c for c in candidates})
    if not unique_candidates:
        return []

    n_latents = min(n_latents, len(unique_candidates))
    return rng.sample(unique_candidates, n_latents)


def _compare_outputs(
    target_tokens: Optional[Tuple[int, ...]],
    target_text: Optional[str],
    random_tokens: Optional[Tuple[int, ...]],
    random_text: Optional[str],
) -> bool:
    if target_tokens is not None and random_tokens is not None:
        return target_tokens == random_tokens
    if target_text is not None and random_text is not None:
        return target_text == random_text
    return False


# %%
@dataclass
class RandomNullResult:
    label: str
    original_verdict: str
    new_verdict: str
    null_due_to_random: bool
    n_latents: int
    coeff: int
    matched_trial: Optional[int]


# %%
def run_random_null_check(
    *,
    token_dir: Path,
    model,
    saes: List[Any],
    rng: random.Random,
    target_coeff: int,
    num_trials: int,
    max_tokens: int,
    device: str,
) -> List[RandomNullResult]:
    planning_path = _find_first_existing(token_dir, PLANNING_FILES)
    steering_path = _find_first_existing(token_dir, STEERING_FILES)
    clusters_path = token_dir / "clusters.json"
    metadata_path = token_dir / "metadata.json"
    circuit_path = token_dir / "circuit_entries.pt"

    if not planning_path or not steering_path:
        return []
    if not (clusters_path.exists() and metadata_path.exists() and circuit_path.exists()):
        return []

    planning_analysis = _load_json(planning_path)
    steering_results = _load_json(steering_path)
    clusters = _load_json(clusters_path)
    metadata = _load_json(metadata_path)
    circuit_entries = torch.load(circuit_path, map_location="cpu")

    position = _get_position(metadata)
    prefix_text = metadata.get("input_prefix_text")
    if not prefix_text:
        return []

    inter_toks_BL = model.to_tokens(prefix_text).to(device)
    saes_dict = {i: sae for i, sae in enumerate(saes)}

    results: List[RandomNullResult] = []
    for label, verdict_raw in planning_analysis.items():
        verdict = _extract_verdict(verdict_raw)
        if verdict not in {"Plan", "Can't say"}:
            results.append(
                RandomNullResult(
                    label=label,
                    original_verdict=verdict or "Unknown",
                    new_verdict=verdict or "Unknown",
                    null_due_to_random=False,
                    n_latents=0,
                    coeff=target_coeff,
                    matched_trial=None,
                )
            )
            continue

        label_latents = _label_latents_at_position(clusters, label, position)
        n_latents = len(label_latents)
        if n_latents == 0:
            results.append(
                RandomNullResult(
                    label=label,
                    original_verdict=verdict,
                    new_verdict=verdict,
                    null_due_to_random=False,
                    n_latents=0,
                    coeff=target_coeff,
                    matched_trial=None,
                )
            )
            continue

        target_entry = _extract_target_entry(steering_results, label, target_coeff)
        if target_entry is None:
            results.append(
                RandomNullResult(
                    label=label,
                    original_verdict=verdict,
                    new_verdict=verdict,
                    null_due_to_random=False,
                    n_latents=n_latents,
                    coeff=target_coeff,
                    matched_trial=None,
                )
            )
            continue

        target_tokens = _normalize_tokens(target_entry.get("steered_text"))
        target_text = _normalize_text(
            target_entry.get("decoded_text") or target_entry.get("steered_text")
        )

        matched_trial: Optional[int] = None
        for trial_idx in range(num_trials):
            interventions = _sample_random_interventions(
                circuit_entries=circuit_entries,
                position=position,
                n_latents=n_latents,
                rng=rng,
            )
            if not interventions:
                break

            sweep_out = sweep_coefficients_multi(
                model=model,
                saes=saes_dict,
                interventions=interventions,
                coefficients=[target_coeff],
                inter_toks_BL=inter_toks_BL,
                stop_tok=1917,
                device=device,
                max_tokens=max_tokens,
                return_tokens=True,
            )
            random_tokens = _normalize_tokens(sweep_out.get(target_coeff))
            random_text = None
            if random_tokens is not None:
                random_text = _normalize_text(model.to_string(list(random_tokens)))

            if _compare_outputs(target_tokens, target_text, random_tokens, random_text):
                matched_trial = trial_idx
                break

        null_due_to_random = matched_trial is not None
        new_verdict = "Null" if null_due_to_random else verdict
        results.append(
            RandomNullResult(
                label=label,
                original_verdict=verdict,
                new_verdict=new_verdict,
                null_due_to_random=null_due_to_random,
                n_latents=n_latents,
                coeff=target_coeff,
                matched_trial=matched_trial,
            )
        )

    return results


# %%
def iter_token_dirs(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    for prompt_dir in sorted(root.glob("prompt_*")):
        if not prompt_dir.is_dir():
            continue
        for token_dir in sorted(prompt_dir.glob("token_*")):
            if token_dir.is_dir():
                yield token_dir


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
summary_rows = []
for token_dir in iter_token_dirs(OUTPUT_ROOT):
    results = run_random_null_check(
        token_dir=token_dir,
        model=model,
        saes=saes,
        rng=rng,
        target_coeff=TARGET_COEFF,
        num_trials=NUM_RANDOM_TRIALS,
        max_tokens=MAX_TOKENS,
        device=DEVICE,
    )
    if not results:
        continue
    print(f"Processed {token_dir}")
    print(f"Results: {results}")
    break
    out_path = token_dir / "random_nullified_planning_analysis.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                r.label: {
                    "original_verdict": r.original_verdict,
                    "new_verdict": r.new_verdict,
                    "null_due_to_random": r.null_due_to_random,
                    "n_latents": r.n_latents,
                    "coeff": r.coeff,
                    "matched_trial": r.matched_trial,
                }
                for r in results
            },
            f,
            indent=2,
        )

    summary_rows.extend(
        {
            "token_dir": str(token_dir),
            "label": r.label,
            "original_verdict": r.original_verdict,
            "new_verdict": r.new_verdict,
            "null_due_to_random": r.null_due_to_random,
            "n_latents": r.n_latents,
            "coeff": r.coeff,
            "matched_trial": r.matched_trial,
        }
        for r in results
    )

# summary_path = OUTPUT_ROOT / "random_nullified_summary.json"
# if summary_rows:
#     with open(summary_path, "w") as f:
#         json.dump(summary_rows, f, indent=2)

# %%

# %%
# %%