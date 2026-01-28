import argparse
import glob
import json
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_DATA_PATH = (
    "/work/pi_jensen_umass_edu/abhishekmish_umass_edu/plan_trace/data/external/"
    "all_examples_og_prompt_with_position_info_and_success_V2.json"
)
DEFAULT_OUTPUT_BASE = "/home/jnainani_umass_edu/w/plan_trace/outputs"
DEFAULT_SCALE_OUTPUT_ROOT = (
    "/work/pi_jensen_umass_edu/jnainani_umass_edu/plan_trace/outputs/scale_bvi_c200_first_change"
)
DEFAULT_PLOT_NAME = "plan_grid_e_c200_first_change_random_steer.png"
DEFAULT_ANALYSIS_NAME = "updated_planning_analysis_random_steer.json"


def load_json(file):
    with open(file, "r") as f:
        return json.load(f)


def dump_json(filename, my_dict):
    with open(filename, "w") as f:
        json.dump(my_dict, f, indent=2)


def _parse_prompt_token(token_planning_folder):
    parts = token_planning_folder.split("/")
    if len(parts) < 3:
        return None, None
    prompt_part = parts[-2]
    token_part = parts[-1]
    if not prompt_part.startswith("prompt_") or not token_part.startswith("token_"):
        return None, None
    try:
        prompt_idx = int(prompt_part.split("_", 1)[1])
        token_idx = int(token_part.split("_", 1)[1])
    except ValueError:
        return None, None
    return prompt_idx, token_idx


def _build_position_token_map(data):
    token_map = {}
    for idx, entry in enumerate(data):
        position_info = entry.get("position_info")
        if not position_info:
            continue
        base_pos = position_info.get("base_token_pos")
        instruct_pos = position_info.get("instruct_token_pos")
        if base_pos is None or instruct_pos is None:
            continue
        token_map[idx] = {"base": int(base_pos), "instruct": int(instruct_pos)}
    return token_map


def _filter_tokens_first_change(all_tokens, token_map):
    filtered = []
    for token_planning_folder in all_tokens:
        prompt_idx, token_idx = _parse_prompt_token(token_planning_folder)
        if prompt_idx is None:
            continue
        bucket = token_map.get(prompt_idx)
        if not bucket:
            continue
        if "/base/" in token_planning_folder and token_idx == bucket["base"]:
            filtered.append(token_planning_folder)
        elif "/instruct/" in token_planning_folder and token_idx == bucket["instruct"]:
            filtered.append(token_planning_folder)
    return filtered


def _build_allowed_token_dirs(output_root, token_map):
    all_tokens = [
        *glob.glob(os.path.join(output_root, "base", "*", "*")),
        *glob.glob(os.path.join(output_root, "instruct", "*", "*")),
    ]
    return set(_filter_tokens_first_change(all_tokens, token_map))


def _resolve_verdict(verdicts, verdict_key):
    if not isinstance(verdicts, dict):
        return None
    if verdict_key in verdicts and verdicts[verdict_key] is not None:
        return verdicts[verdict_key]
    if "new_verdict" in verdicts and verdicts["new_verdict"] is not None:
        return verdicts["new_verdict"]
    return verdicts.get("original_verdict")


def _detect_planning(planning_dict, verdict_key):
    plans = []
    for key, verdicts in planning_dict.items():
        verdict = _resolve_verdict(verdicts, verdict_key)
        if verdict == "Plan":
            plans.append(key)
    return plans


def _detect_cant_says(planning_dict, verdict_key):
    for verdicts in planning_dict.values():
        verdict = _resolve_verdict(verdicts, verdict_key)
        if verdict == "Can't say":
            return True
    return False


def _get_ym_plans(folder, analysis_name, verdict_key, allowed_token_dirs):
    token_dirs = glob.glob(os.path.join(folder, "*"))
    if allowed_token_dirs is not None:
        token_dirs = [d for d in token_dirs if d in allowed_token_dirs]
    token_planning_files = []
    for token_dir in token_dirs:
        preferred = os.path.join(token_dir, analysis_name)
        fallback = os.path.join(token_dir, "updated_planning_analysis.json")
        if os.path.exists(preferred):
            token_planning_files.append(preferred)
        elif os.path.exists(fallback):
            token_planning_files.append(fallback)
    planning_datas = [load_json(f) for f in token_planning_files]
    y_ms = []
    for data in planning_datas:
        y_ms.extend(_detect_planning(data, verdict_key))
    return list(set(y_ms))


def _detect_cs(folder, analysis_name, verdict_key, allowed_token_dirs):
    token_dirs = glob.glob(os.path.join(folder, "*"))
    if allowed_token_dirs is not None:
        token_dirs = [d for d in token_dirs if d in allowed_token_dirs]
    token_planning_files = []
    for token_dir in token_dirs:
        preferred = os.path.join(token_dir, analysis_name)
        fallback = os.path.join(token_dir, "updated_planning_analysis.json")
        if os.path.exists(preferred):
            token_planning_files.append(preferred)
        elif os.path.exists(fallback):
            token_planning_files.append(fallback)
    planning_datas = [load_json(f) for f in token_planning_files]
    return any(_detect_cant_says(data, verdict_key) for data in planning_datas)


def _get_base_and_instruct_plans(
    iter_idx, root_folder, analysis_name, verdict_key, allowed_token_dirs
):
    base_folder = os.path.join(root_folder, "base", f"prompt_{iter_idx}")
    instruct_folder = os.path.join(root_folder, "instruct", f"prompt_{iter_idx}")
    base_yms = (
        _get_ym_plans(base_folder, analysis_name, verdict_key, allowed_token_dirs)
        if os.path.exists(base_folder)
        else None
    )
    instruct_yms = (
        _get_ym_plans(instruct_folder, analysis_name, verdict_key, allowed_token_dirs)
        if os.path.exists(instruct_folder)
        else None
    )
    return {"base": base_yms, "instruct": instruct_yms}


def _get_base_and_instruct_cantsays(
    iter_idx, root_folder, analysis_name, verdict_key, allowed_token_dirs
):
    base_folder = os.path.join(root_folder, "base", f"prompt_{iter_idx}")
    instruct_folder = os.path.join(root_folder, "instruct", f"prompt_{iter_idx}")
    base_cs = (
        _detect_cs(base_folder, analysis_name, verdict_key, allowed_token_dirs)
        if os.path.exists(base_folder)
        else False
    )
    instruct_cs = (
        _detect_cs(instruct_folder, analysis_name, verdict_key, allowed_token_dirs)
        if os.path.exists(instruct_folder)
        else False
    )
    return {"base": base_cs, "instruct": instruct_cs}


def add_earliest_plan_fields(data, output_root, analysis_name, verdict_key, allowed_token_dirs):
    for idx, entry in enumerate(data):
        ym_plans = _get_base_and_instruct_plans(
            idx,
            root_folder=output_root,
            analysis_name=analysis_name,
            verdict_key=verdict_key,
            allowed_token_dirs=allowed_token_dirs,
        )
        entry["base_plans_e"] = ym_plans["base"]
        entry["instruct_plans_e"] = ym_plans["instruct"]
        cs = _get_base_and_instruct_cantsays(
            idx,
            root_folder=output_root,
            analysis_name=analysis_name,
            verdict_key=verdict_key,
            allowed_token_dirs=allowed_token_dirs,
        )
        entry["base_cs_e"] = cs["base"]
        entry["instruct_cs_e"] = cs["instruct"]


def _get_2x2_grid(selected_cases, with_e=False):
    suffix = "_e" if with_e else ""
    instruct_plan_base_no_plan = []
    base_plan_instruct_no_plan = []
    both_plans = []
    no_plans = []

    for entry in selected_cases:
        base_planning, instruct_planning = False, False

        if entry.get("base_plans" + suffix) and len(entry["base_plans" + suffix]) > 0:
            base_planning = True
        elif entry.get("base_cs" + suffix) is True:
            continue

        if entry.get("instruct_plans" + suffix) and len(entry["instruct_plans" + suffix]) > 0:
            instruct_planning = True
        elif entry.get("instruct_cs" + suffix) is True:
            continue

        if base_planning:
            if instruct_planning:
                both_plans.append(entry)
            else:
                base_plan_instruct_no_plan.append(entry)
        elif instruct_planning:
            instruct_plan_base_no_plan.append(entry)
        else:
            no_plans.append(entry)

    return {
        "Both": both_plans,
        "Only instruct": instruct_plan_base_no_plan,
        "Only base": base_plan_instruct_no_plan,
        "None": no_plans,
    }


def compute_4x4_counts(data, with_e=False):
    quadrants = [
        ((True, True), (0, 0), "Instruct Pass / Base Pass"),
        ((True, False), (0, 2), "Instruct Pass / Base Fail"),
        ((False, True), (2, 0), "Instruct Fail / Base Pass"),
        ((False, False), (2, 2), "Instruct Fail / Base Fail"),
    ]

    counts = np.zeros((4, 4), dtype=int)

    for (ip, bp), (r0, c0), _title in quadrants:
        selected = [
            case for case in data if case.get("instruct_pass") == ip and case.get("base_pass") == bp
        ]
        buckets = _get_2x2_grid(selected, with_e=with_e)

        counts[r0 + 0, c0 + 0] = len(buckets["Both"])
        counts[r0 + 0, c0 + 1] = len(buckets["Only instruct"])
        counts[r0 + 1, c0 + 0] = len(buckets["Only base"])
        counts[r0 + 1, c0 + 1] = len(buckets["None"])

    return counts


def plot_4x4_plan_grid(data, with_e=False, ax=None, cmap="Blues", title=None):
    counts = compute_4x4_counts(data, with_e=with_e)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

    im = ax.imshow(counts, cmap=cmap)

    vmax = counts.max() if counts.size else 1
    for r in range(4):
        for c in range(4):
            ax.text(
                c,
                r,
                str(counts[r, c]),
                ha="center",
                va="center",
                fontsize=12,
                color="white" if counts[r, c] > 0.6 * vmax else "black",
            )

    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=1)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    ax.axhline(1.5, linewidth=3)
    ax.axhline(3.5, linewidth=3)
    ax.axvline(1.5, linewidth=3)
    ax.axvline(3.5, linewidth=3)

    ax.axhline(-0.5, linewidth=3)
    ax.axvline(-0.5, linewidth=3)

    inner_labels = {
        (0, 0): "Both Plans",
        (0, 1): "Only Instruct Plans",
        (1, 0): "Only Base Plans",
        (1, 1): "None Plan",
    }
    for r0, c0 in [(0, 0), (0, 2), (2, 0), (2, 2)]:
        for (dr, dc), lab in inner_labels.items():
            r = r0 + dr
            c = c0 + dc
            ax.text(c, r - 0.32, lab, ha="center", va="center", fontsize=8)

    ax.text(0.5, -0.95, "Base Pass", ha="center", va="center", fontsize=14)
    ax.text(2.5, -0.95, "Base Fail", ha="center", va="center", fontsize=14)

    ax.text(-0.95, 0.5, "Instruct Pass", ha="center", va="center", fontsize=14, rotation=90)
    ax.text(-0.95, 2.5, "Instruct Fail", ha="center", va="center", fontsize=14, rotation=90)

    quad_titles = {
        (0.5, 0.5): "IP / BP",
        (0.5, 2.5): "IP / BF",
        (2.5, 0.5): "IF / BP",
        (2.5, 2.5): "IF / BF",
    }
    for (y, x), t in quad_titles.items():
        ax.text(x, y, t, ha="center", va="center", fontsize=12, fontweight="bold")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")
    ax.set_xlim(-1.2, 3.5)
    ax.set_ylim(3.5, -1.2)
    if title is None:
        title = "4x4 Grid: Pass/Fail quadrants x Planning categories"
    ax.set_title(title, fontsize=14)
    return ax


def _build_output_paths(output_base, plot_path, desc):
    suffix = f"_{desc}" if desc else ""
    resolved_plot_path = plot_path or os.path.join(output_base, "plots", DEFAULT_PLOT_NAME.replace(".png", f"{suffix}.png"))
    return resolved_plot_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot 4x4 grid using random-steering verdicts (first change only)."
    )
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-base", default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--output-root", default=DEFAULT_SCALE_OUTPUT_ROOT)
    parser.add_argument("--plot-path", default=None)
    parser.add_argument("--save-data-path", default=None)
    parser.add_argument("--analysis-name", default=DEFAULT_ANALYSIS_NAME)
    parser.add_argument("--verdict-key", default="new_new_verdict")
    parser.add_argument("--desc", default="")
    args = parser.parse_args()

    data = load_json(args.data_path)
    token_map = _build_position_token_map(data)
    allowed_token_dirs = _build_allowed_token_dirs(args.output_root, token_map)

    add_earliest_plan_fields(
        data,
        output_root=args.output_root,
        analysis_name=args.analysis_name,
        verdict_key=args.verdict_key,
        allowed_token_dirs=allowed_token_dirs,
    )

    title_bits = ["random_steer", "first_change", f"verdict={args.verdict_key}"]
    if args.desc:
        title_bits.append(args.desc)
    plot_title = "4x4 Grid (" + ", ".join(title_bits) + ")"

    plot_path = _build_output_paths(args.output_base, args.plot_path, args.desc)
    ax = plot_4x4_plan_grid(data, with_e=True, title=plot_title)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    ax.figure.savefig(plot_path, bbox_inches="tight", dpi=200)

    if args.save_data_path:
        os.makedirs(os.path.dirname(args.save_data_path), exist_ok=True)
        dump_json(args.save_data_path, data)


if __name__ == "__main__":
    main()
