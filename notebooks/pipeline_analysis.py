# %%
"""
so each token_x will have planning_analysis.json and metadata.json

1. We can first count how many forward passes have at least one planning and show across all three prompts 
2. we want the horizon of planning - when a token is being planned for in token_x, we have to find for which token_y forward is it the first token in the "baseline_text"
"""

# %%
import os
import json
from pathlib import Path

# %%
# Config
PROMPTS = [11, 15, 24]
OUTPUTS_ROOT = Path("../outputs")

# %%
def count_planning_for_prompt(prompt_idx: int) -> tuple[int, int]:
    """Return (num_with_plan, total_positions) for a given prompt directory."""
    prompt_dir = OUTPUTS_ROOT / f"prompt_{prompt_idx}"
    if not prompt_dir.exists() or not prompt_dir.is_dir():
        return 0, 0
    num_with_plan = 0
    total = 0
    for token_dir in sorted(prompt_dir.iterdir()):
        if not token_dir.is_dir():
            continue
        planning_path = token_dir / "planning_analysis.json"
        if not planning_path.exists():
            continue
        total += 1
        try:
            with open(planning_path, "r") as f:
                planning = json.load(f)
        except Exception:
            continue
        has_plan = any(status == "Plan" for status in planning.values())
        if has_plan:
            num_with_plan += 1
    return num_with_plan, total

# %%
per_prompt = {}
grand_plan = 0
grand_total = 0
for p in PROMPTS:
    plan_cnt, tot = count_planning_for_prompt(p)
    per_prompt[p] = (plan_cnt, tot)
    grand_plan += plan_cnt
    grand_total += tot

# %%
print("Planning counts per prompt (num_with_plan / total_positions):")
for p in PROMPTS:
    plan_cnt, tot = per_prompt[p]
    print(f"- prompt_{p}: {plan_cnt} / {tot}")
print("-")
print(f"Grand total: {grand_plan} / {grand_total}")

# %%
# Planning horizon analysis
from typing import Dict, Tuple, List, Optional

def _load_per_token_data(prompt_idx: int) -> Tuple[List[int], Dict[int, Dict[str, str]], Dict[int, str]]:
    """
    Returns:
      positions: sorted list of token ids (y)
      planning_map: y -> planning_analysis dict[label->status]
      baseline_map: y -> baseline_text string
    """
    prompt_dir = OUTPUTS_ROOT / f"prompt_{prompt_idx}"
    positions: List[int] = []
    planning_map: Dict[int, Dict[str, str]] = {}
    baseline_map: Dict[int, str] = {}
    if not prompt_dir.exists():
        return positions, planning_map, baseline_map
    for token_dir in prompt_dir.iterdir():
        if not token_dir.is_dir():
            continue
        name = token_dir.name
        if not name.startswith("token_"):
            continue
        try:
            y = int(name.split("_")[-1])
        except Exception:
            continue
        planning_path = token_dir / "planning_analysis.json"
        metadata_path = token_dir / "metadata.json"
        if not planning_path.exists() or not metadata_path.exists():
            continue
        try:
            with open(planning_path, "r") as f:
                planning = json.load(f)
            with open(metadata_path, "r") as f:
                meta = json.load(f)
        except Exception:
            continue
        positions.append(y)
        planning_map[y] = planning
        baseline_map[y] = meta.get("baseline_text", "")
    positions.sort()
    return positions, planning_map, baseline_map


def _label_at_start(text: str, label: str) -> bool:
    if not isinstance(text, str) or not isinstance(label, str) or not label:
        return False
    s = text.lstrip()
    if not s.startswith(label):
        return False
    # Boundary on the right if alphanumeric label
    if any(ch.isalnum() for ch in label):
        L = len(s)
        nL = len(label)
        if nL < L and s[nL].isalnum():
            return False
    return True


def compute_planning_horizons(prompt_idx: int) -> List[int]:
    """
    For each token_x with at least one "Plan" label, iterate over ALL future token_y
    directories (y > x), load token_y/metadata.json, and check whether
    metadata["baseline_text"] starts with that label (with boundary). Record the
    first such y per (x, label) and return the horizon distances (y - x).
    """
    prompt_dir = OUTPUTS_ROOT / f"prompt_{prompt_idx}"
    if not prompt_dir.exists() or not prompt_dir.is_dir():
        return []

    # Collect all token_y directories and sort by y
    all_tokens: List[int] = []
    token_dir_map: Dict[int, Path] = {}
    for token_dir in prompt_dir.iterdir():
        if token_dir.is_dir() and token_dir.name.startswith("token_"):
            try:
                y = int(token_dir.name.split("_")[-1])
            except Exception:
                continue
            all_tokens.append(y)
            token_dir_map[y] = token_dir
    all_tokens.sort()
    index_of = {y: i for i, y in enumerate(all_tokens)}

    # Identify token_x that have planning_analysis.json and their Plan labels
    horizons: List[int] = []
    for x in all_tokens:
        x_dir = token_dir_map[x]
        planning_path = x_dir / "planning_analysis.json"
        if not planning_path.exists():
            continue
        try:
            with open(planning_path, "r") as f:
                planning = json.load(f)
        except Exception:
            continue
        plan_labels = [lbl for lbl, status in planning.items() if status == "Plan"]
        if not plan_labels:
            continue

        # For each label planned at x, scan forward y > x and find first baseline start match
        for lbl in plan_labels:
            print(f"Planning for {lbl} at {x}")
            horizon_found: Optional[int] = None
            for j in range(index_of[x] + 1, len(all_tokens)):
                y = all_tokens[j]
                y_dir = token_dir_map[y]
                meta_path = y_dir / "metadata.json"
                if not meta_path.exists():
                    continue
                try:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                except Exception:
                    continue
                base = meta.get("baseline_text", "")
                if _label_at_start(base, lbl):
                    horizon_found = y - x
                    print(f"Found horizon for {lbl} at {y}")
                    break
            if horizon_found is not None:
                horizons.append(horizon_found)
    return horizons


# %%
print("\nPlanning horizon (distance in tokens to first baseline start match):")
for p in PROMPTS:
    hs = compute_planning_horizons(p)
    if hs:
        avg = sum(hs) / len(hs)
        print(f"- prompt_{p}: N={len(hs)}, avg={avg:.2f}, min={min(hs)}, max={max(hs)}")
    else:
        print(f"- prompt_{p}: N=0")

# %%
# Visualization: counts and horizons
import matplotlib.pyplot as plt
import seaborn as sns

ANALYSIS_DIR = OUTPUTS_ROOT / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def plot_planning_counts(per_prompt: Dict[int, tuple[int, int]]):
    labels = [f"prompt_{p}" for p in PROMPTS]
    plan_counts = [per_prompt[p][0] for p in PROMPTS]
    totals = [per_prompt[p][1] for p in PROMPTS]
    nonplan = [tot - pl for pl, tot in zip(plan_counts, totals)]

    x = range(len(labels))
    plt.figure(figsize=(8, 5), dpi=150)
    plt.bar(x, nonplan, label="No Plan", color="#d9d9d9")
    plt.bar(x, plan_counts, bottom=nonplan, label="Plan", color="#4c78a8")
    plt.xticks(x, labels)
    plt.ylabel("Count of positions")
    plt.title("Planning counts per prompt")
    plt.legend()
    for i, (pl, tot) in enumerate(zip(plan_counts, totals)):
        pct = (pl / tot * 100.0) if tot > 0 else 0.0
        plt.text(i, tot + max(tot * 0.02, 0.5), f"{pl}/{tot} ({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
    out_path = ANALYSIS_DIR / "planning_counts.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()
    print(f"Saved: {out_path}")


def plot_planning_horizons(hmap: Dict[int, List[int]]):
    # Histogram per prompt
    plt.figure(figsize=(9, 5), dpi=150)
    bins = 20
    for p, hs in hmap.items():
        if hs:
            sns.histplot(hs, bins=bins, kde=False, stat="count", label=f"prompt_{p}", element="step", fill=False)
    plt.xlabel("Horizon (tokens)")
    plt.ylabel("Count")
    plt.title("Planning horizon distribution")
    plt.legend()
    out_path = ANALYSIS_DIR / "planning_horizon_hist.png"
    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()
    print(f"Saved: {out_path}")

    # Boxplot across prompts
    plt.figure(figsize=(8, 5), dpi=150)
    data = []
    labels = []
    for p in PROMPTS:
        hs = hmap.get(p, [])
        if hs:
            data.append(hs)
            labels.append(f"prompt_{p}")
    if data:
        sns.boxplot(data=data)
        plt.xticks(range(len(labels)), labels)
        plt.ylabel("Horizon (tokens)")
        plt.title("Planning horizon summary")
        out_path2 = ANALYSIS_DIR / "planning_horizon_box.png"
        plt.tight_layout()
        plt.savefig(out_path2)
        plt.show()
        print(f"Saved: {out_path2}")


# %%
plot_planning_counts(per_prompt)
horizon_map = {p: compute_planning_horizons(p) for p in PROMPTS}
plot_planning_horizons(horizon_map)

# %%

