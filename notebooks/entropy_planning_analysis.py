# %%
"""
Entropy vs Planning Analysis

Hypotheses:
1. Entropy for tokens that were "planned for" at some earlier position will be LOWER
   than tokens that were not planned for.
2. Entropy for "affected" tokens (where plan_count > 0 at prediction time) will be HIGHER
   than tokens where we see no planning evidence.
"""
from __future__ import annotations

import json
import glob
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# %%
# Configuration
ENTROPY_ROOT = Path("/home/jnainani_umass_edu/w/plan_trace/notebooks/outputs/entropy")
PLANNING_ROOT = Path("/work/pi_jensen_umass_edu/abhishekmish_umass_edu/plan_trace/outputs/planning_scale_classified_e/instruct")
DATA_ROOT = Path("/work/pi_jensen_umass_edu/abhishekmish_umass_edu/plan_trace/outputs/planning_scale/instruct")

# %%
# Helper functions

def load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def discover_entropy_prompts() -> List[int]:
    """Find all prompt indices that have entropy results."""
    prompts = []
    for folder in ENTROPY_ROOT.iterdir():
        if folder.is_dir() and folder.name.startswith("prompt_"):
            match = re.match(r"prompt_(\d+)", folder.name)
            if match:
                prompts.append(int(match.group(1)))
    return sorted(prompts)


def discover_planning_tokens(prompt_idx: int) -> List[int]:
    """Find all token indices with planning analysis for a prompt."""
    prompt_dir = PLANNING_ROOT / f"prompt_{prompt_idx}"
    if not prompt_dir.exists():
        return []
    tokens = []
    for folder in prompt_dir.iterdir():
        if folder.is_dir() and folder.name.startswith("token_"):
            match = re.match(r"token_(\d+)", folder.name)
            if match:
                tokens.append(int(match.group(1)))
    return sorted(tokens)


def get_planning_verdict(planning_analysis: Dict[str, Any], token_str: str) -> Optional[str]:
    """Get the verdict for a token from planning analysis."""
    if token_str not in planning_analysis:
        return None
    entry = planning_analysis[token_str]
    if isinstance(entry, dict):
        return entry.get("new_verdict") or entry.get("original_verdict")
    return None


# %%
# Load all entropy data
print("[info] Discovering entropy results...")
entropy_prompts = discover_entropy_prompts()
print(f"[info] Found entropy results for {len(entropy_prompts)} prompts")

# %%
# Load entropy data for each prompt
entropy_data: Dict[int, Dict[str, Any]] = {}
for prompt_idx in entropy_prompts:
    entropy_path = ENTROPY_ROOT / f"prompt_{prompt_idx}" / "entropy_token_results.json"
    if entropy_path.exists():
        entropy_data[prompt_idx] = load_json(entropy_path)

print(f"[info] Loaded entropy data for {len(entropy_data)} prompts")

# %%
# Show sample entropy data structure
if entropy_data:
    sample_prompt = list(entropy_data.keys())[0]
    sample = entropy_data[sample_prompt]
    print(f"\n[sample] Prompt {sample_prompt} entropy data keys: {list(sample.keys())}")
    print(f"  analyzed_range: {sample.get('analyzed_range')}")
    print(f"  num token_results: {len(sample.get('token_results', {}))}")

    # Show first token result
    token_results = sample.get("token_results", {})
    if token_results:
        first_key = list(token_results.keys())[0]
        first_result = token_results[first_key]
        print(f"\n  Sample token result (token_idx={first_key}):")
        print(f"    predicted_token_str: {first_result.get('predicted_token_str')}")
        print(f"    entropy: {first_result.get('entropy')}")

# %%
# Build mapping: for each prompt, track which tokens were "planned for" at which positions
# planned_tokens_map[prompt_idx][token_str] = list of positions where this token was planned

print("\n[info] Building planned tokens map...")

planned_tokens_map: Dict[int, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
affected_positions: Dict[int, Dict[int, bool]] = defaultdict(dict)  # prompt -> token_idx -> is_affected
position_plan_counts: Dict[int, Dict[int, int]] = defaultdict(dict)  # prompt -> token_idx -> plan_count

for prompt_idx in entropy_prompts:
    planning_tokens = discover_planning_tokens(prompt_idx)

    for token_idx in planning_tokens:
        analysis_path = PLANNING_ROOT / f"prompt_{prompt_idx}" / f"token_{token_idx}" / "updated_planning_analysis.json"
        if not analysis_path.exists():
            continue

        planning_analysis = load_json(analysis_path)
        plan_count = 0

        for token_str, verdict_data in planning_analysis.items():
            if not isinstance(verdict_data, dict):
                continue
            verdict = verdict_data.get("new_verdict") or verdict_data.get("original_verdict")
            if verdict == "Plan":
                plan_count += 1
                planned_tokens_map[prompt_idx][token_str].append(token_idx)

        # Track if this position is "affected" (has any planning)
        affected_positions[prompt_idx][token_idx] = plan_count > 0
        position_plan_counts[prompt_idx][token_idx] = plan_count

print(f"[info] Processed planning data for {len(planned_tokens_map)} prompts")

# %%
# Show sample planned tokens
if planned_tokens_map:
    sample_prompt = list(planned_tokens_map.keys())[0]
    sample_planned = dict(list(planned_tokens_map[sample_prompt].items())[:5])
    print(f"\n[sample] Planned tokens for prompt {sample_prompt}:")
    for token_str, positions in sample_planned.items():
        print(f"  '{token_str}' planned at positions: {positions[:10]}{'...' if len(positions) > 10 else ''}")

# %%
# Hypothesis 1: Entropy for tokens that were "planned for" vs not
# For each predicted token, check if it was planned at any earlier position

print("\n[info] Testing Hypothesis 1: Planned-for tokens have lower entropy")

entropy_planned_for: List[float] = []
entropy_not_planned_for: List[float] = []
planned_for_details: List[Dict[str, Any]] = []

for prompt_idx, entropy_result in entropy_data.items():
    token_results = entropy_result.get("token_results", {})
    planned_map = planned_tokens_map.get(prompt_idx, {})

    for token_idx_str, result in token_results.items():
        token_idx = int(token_idx_str)
        entropy = result.get("entropy")
        predicted_str = result.get("predicted_token_str")

        if entropy is None or predicted_str is None:
            continue

        # Check if this token was planned at any EARLIER position
        planning_positions = planned_map.get(predicted_str, [])
        earlier_planning = [pos for pos in planning_positions if pos < token_idx]

        was_planned_for = len(earlier_planning) > 0

        if was_planned_for:
            entropy_planned_for.append(entropy)
            planned_for_details.append({
                "prompt_idx": prompt_idx,
                "token_idx": token_idx,
                "predicted_str": predicted_str,
                "entropy": entropy,
                "planned_at_positions": earlier_planning,
            })
        else:
            entropy_not_planned_for.append(entropy)

print(f"\n  Tokens that were planned for: {len(entropy_planned_for)}")
print(f"  Tokens that were NOT planned for: {len(entropy_not_planned_for)}")

if entropy_planned_for and entropy_not_planned_for:
    mean_planned = np.mean(entropy_planned_for)
    mean_not_planned = np.mean(entropy_not_planned_for)
    std_planned = np.std(entropy_planned_for)
    std_not_planned = np.std(entropy_not_planned_for)

    print(f"\n  Mean entropy (planned for):     {mean_planned:.4f} (std: {std_planned:.4f})")
    print(f"  Mean entropy (NOT planned for): {mean_not_planned:.4f} (std: {std_not_planned:.4f})")
    print(f"  Difference: {mean_planned - mean_not_planned:.4f}")

    # Statistical test
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(entropy_planned_for, entropy_not_planned_for)
    print(f"\n  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")

    if p_value < 0.05:
        if mean_planned < mean_not_planned:
            print("  => SUPPORTS Hypothesis 1: Planned-for tokens have significantly LOWER entropy")
        else:
            print("  => CONTRADICTS Hypothesis 1: Planned-for tokens have significantly HIGHER entropy")
    else:
        print("  => No significant difference (p > 0.05)")

# %%
# Hypothesis 2: Entropy at "affected" positions (plan_count > 0) vs not

print("\n[info] Testing Hypothesis 2: Affected positions have higher entropy")

entropy_affected: List[float] = []
entropy_not_affected: List[float] = []
affected_details: List[Dict[str, Any]] = []

for prompt_idx, entropy_result in entropy_data.items():
    token_results = entropy_result.get("token_results", {})
    affected_map = affected_positions.get(prompt_idx, {})
    plan_counts = position_plan_counts.get(prompt_idx, {})

    for token_idx_str, result in token_results.items():
        token_idx = int(token_idx_str)
        entropy = result.get("entropy")

        if entropy is None:
            continue

        # Check if this position is "affected" (has planning evidence)
        is_affected = affected_map.get(token_idx, False)
        plan_count = plan_counts.get(token_idx, 0)

        if is_affected:
            entropy_affected.append(entropy)
            affected_details.append({
                "prompt_idx": prompt_idx,
                "token_idx": token_idx,
                "entropy": entropy,
                "plan_count": plan_count,
            })
        else:
            entropy_not_affected.append(entropy)

print(f"\n  Affected positions (plan_count > 0): {len(entropy_affected)}")
print(f"  Not affected positions: {len(entropy_not_affected)}")

if entropy_affected and entropy_not_affected:
    mean_affected = np.mean(entropy_affected)
    mean_not_affected = np.mean(entropy_not_affected)
    std_affected = np.std(entropy_affected)
    std_not_affected = np.std(entropy_not_affected)

    print(f"\n  Mean entropy (affected):     {mean_affected:.4f} (std: {std_affected:.4f})")
    print(f"  Mean entropy (NOT affected): {mean_not_affected:.4f} (std: {std_not_affected:.4f})")
    print(f"  Difference: {mean_affected - mean_not_affected:.4f}")

    # Statistical test
    t_stat, p_value = stats.ttest_ind(entropy_affected, entropy_not_affected)
    print(f"\n  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")

    if p_value < 0.05:
        if mean_affected > mean_not_affected:
            print("  => SUPPORTS Hypothesis 2: Affected positions have significantly HIGHER entropy")
        else:
            print("  => CONTRADICTS Hypothesis 2: Affected positions have significantly LOWER entropy")
    else:
        print("  => No significant difference (p > 0.05)")

# %%
# Additional analysis: Correlation between plan_count and entropy

print("\n[info] Correlation analysis: plan_count vs entropy")

plan_counts_list: List[int] = []
entropies_list: List[float] = []

for detail in affected_details:
    plan_counts_list.append(detail["plan_count"])
    entropies_list.append(detail["entropy"])

if plan_counts_list and entropies_list:
    correlation, corr_p_value = stats.pearsonr(plan_counts_list, entropies_list)
    print(f"\n  Pearson correlation (plan_count vs entropy): {correlation:.4f}")
    print(f"  p-value: {corr_p_value:.6f}")

    spearman_corr, spearman_p = stats.spearmanr(plan_counts_list, entropies_list)
    print(f"\n  Spearman correlation: {spearman_corr:.4f}")
    print(f"  p-value: {spearman_p:.6f}")

# %%
# Save detailed results
output_dir = Path("outputs/entropy_planning_analysis")
output_dir.mkdir(parents=True, exist_ok=True)

results = {
    "hypothesis_1": {
        "description": "Planned-for tokens have lower entropy",
        "n_planned_for": len(entropy_planned_for),
        "n_not_planned_for": len(entropy_not_planned_for),
        "mean_entropy_planned_for": float(np.mean(entropy_planned_for)) if entropy_planned_for else None,
        "mean_entropy_not_planned_for": float(np.mean(entropy_not_planned_for)) if entropy_not_planned_for else None,
        "std_planned_for": float(np.std(entropy_planned_for)) if entropy_planned_for else None,
        "std_not_planned_for": float(np.std(entropy_not_planned_for)) if entropy_not_planned_for else None,
    },
    "hypothesis_2": {
        "description": "Affected positions have higher entropy",
        "n_affected": len(entropy_affected),
        "n_not_affected": len(entropy_not_affected),
        "mean_entropy_affected": float(np.mean(entropy_affected)) if entropy_affected else None,
        "mean_entropy_not_affected": float(np.mean(entropy_not_affected)) if entropy_not_affected else None,
        "std_affected": float(np.std(entropy_affected)) if entropy_affected else None,
        "std_not_affected": float(np.std(entropy_not_affected)) if entropy_not_affected else None,
    },
    "entropy_planned_for": entropy_planned_for,
    "entropy_not_planned_for": entropy_not_planned_for,
    "entropy_affected": entropy_affected,
    "entropy_not_affected": entropy_not_affected,
}

with open(output_dir / "entropy_planning_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n[info] Results saved to {output_dir / 'entropy_planning_results.json'}")

# %%
# Visualization
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Hypothesis 1: Box plot
    ax1 = axes[0]
    data_h1 = [entropy_planned_for, entropy_not_planned_for]
    labels_h1 = [f"Planned for\n(n={len(entropy_planned_for)})",
                 f"Not planned for\n(n={len(entropy_not_planned_for)})"]
    bp1 = ax1.boxplot(data_h1, labels=labels_h1, patch_artist=True)
    bp1['boxes'][0].set_facecolor('#4C78A8')
    bp1['boxes'][1].set_facecolor('#F58518')
    ax1.set_ylabel("Entropy")
    ax1.set_title("Hypothesis 1: Entropy of Planned-for vs Not Planned-for Tokens")

    # Hypothesis 2: Box plot
    ax2 = axes[1]
    data_h2 = [entropy_affected, entropy_not_affected]
    labels_h2 = [f"Affected\n(n={len(entropy_affected)})",
                 f"Not affected\n(n={len(entropy_not_affected)})"]
    bp2 = ax2.boxplot(data_h2, labels=labels_h2, patch_artist=True)
    bp2['boxes'][0].set_facecolor('#4C78A8')
    bp2['boxes'][1].set_facecolor('#F58518')
    ax2.set_ylabel("Entropy")
    ax2.set_title("Hypothesis 2: Entropy at Affected vs Not Affected Positions")

    plt.tight_layout()
    plt.savefig(output_dir / "entropy_planning_boxplots.png", dpi=200)
    plt.show()
    print(f"[info] Plot saved to {output_dir / 'entropy_planning_boxplots.png'}")

except ImportError:
    print("[warning] matplotlib not available, skipping visualization")

# %%
# Distribution histograms
try:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # H1 distributions
    axes[0, 0].hist(entropy_planned_for, bins=30, alpha=0.7, color='#4C78A8', edgecolor='black')
    axes[0, 0].axvline(np.mean(entropy_planned_for), color='red', linestyle='--', label=f'Mean: {np.mean(entropy_planned_for):.3f}')
    axes[0, 0].set_xlabel("Entropy")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Entropy Distribution: Planned-for Tokens")
    axes[0, 0].legend()

    axes[0, 1].hist(entropy_not_planned_for, bins=30, alpha=0.7, color='#F58518', edgecolor='black')
    axes[0, 1].axvline(np.mean(entropy_not_planned_for), color='red', linestyle='--', label=f'Mean: {np.mean(entropy_not_planned_for):.3f}')
    axes[0, 1].set_xlabel("Entropy")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Entropy Distribution: Not Planned-for Tokens")
    axes[0, 1].legend()

    # H2 distributions
    axes[1, 0].hist(entropy_affected, bins=30, alpha=0.7, color='#4C78A8', edgecolor='black')
    axes[1, 0].axvline(np.mean(entropy_affected), color='red', linestyle='--', label=f'Mean: {np.mean(entropy_affected):.3f}')
    axes[1, 0].set_xlabel("Entropy")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Entropy Distribution: Affected Positions")
    axes[1, 0].legend()

    axes[1, 1].hist(entropy_not_affected, bins=30, alpha=0.7, color='#F58518', edgecolor='black')
    axes[1, 1].axvline(np.mean(entropy_not_affected), color='red', linestyle='--', label=f'Mean: {np.mean(entropy_not_affected):.3f}')
    axes[1, 1].set_xlabel("Entropy")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Entropy Distribution: Not Affected Positions")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "entropy_planning_histograms.png", dpi=200)
    plt.show()
    print(f"[info] Histograms saved to {output_dir / 'entropy_planning_histograms.png'}")

except ImportError:
    pass

# %%
# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print("\nHypothesis 1: Tokens that were planned for have LOWER entropy")
print("-" * 60)
if entropy_planned_for and entropy_not_planned_for:
    diff_h1 = np.mean(entropy_planned_for) - np.mean(entropy_not_planned_for)
    print(f"  Planned-for mean: {np.mean(entropy_planned_for):.4f}")
    print(f"  Not planned mean: {np.mean(entropy_not_planned_for):.4f}")
    print(f"  Difference:       {diff_h1:.4f} ({'lower' if diff_h1 < 0 else 'higher'} for planned)")

print("\nHypothesis 2: Affected positions have HIGHER entropy")
print("-" * 60)
if entropy_affected and entropy_not_affected:
    diff_h2 = np.mean(entropy_affected) - np.mean(entropy_not_affected)
    print(f"  Affected mean:     {np.mean(entropy_affected):.4f}")
    print(f"  Not affected mean: {np.mean(entropy_not_affected):.4f}")
    print(f"  Difference:        {diff_h2:.4f} ({'higher' if diff_h2 > 0 else 'lower'} for affected)")

# %%
