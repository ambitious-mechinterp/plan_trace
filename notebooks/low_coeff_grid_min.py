import argparse
import ast
import glob
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_DATA_PATH = (
    "/work/pi_jensen_umass_edu/abhishekmish_umass_edu/plan_trace/data/external/"
    "all_examples_og_prompt_with_position_info_and_success_V2.json"
)
DEFAULT_SCALE_FOLDER = "/work/pi_jensen_umass_edu/abhishekmish_umass_edu/plan_trace/outputs/all_scale"
DEFAULT_OUTPUT_BASE = "/home/jnainani_umass_edu/w/plan_trace/outputs"
DEFAULT_MAX_COEFF = -200


def dump_json(filename, my_dict):
    with open(filename, "w") as f:
        json.dump(my_dict, f, indent=2)


def load_json(file):
    with open(file, "r") as f:
        return json.load(f)


def any_repeated_substring(s, min_repeats=5, max_sub_len=10):
    n = len(s)
    L = max_sub_len
    if n < L:
        return False
    counts = Counter(s[i : i + L] for i in range(n - L + 1))
    return any(v >= min_repeats for v in counts.values())


class Normalizer(ast.NodeTransformer):
    def __init__(self):
        self.var_map = {}
        self.func_map = {}
        self.class_map = {}

    def _rename(self, name, mapping):
        if name not in mapping:
            mapping[name] = f"id_{len(mapping)}"
        return mapping[name]

    def visit_Name(self, node):
        node.id = self._rename(node.id, self.var_map)
        return node

    def visit_arg(self, node):
        node.arg = self._rename(node.arg, self.var_map)
        return node

    def visit_FunctionDef(self, node):
        node.name = self._rename(node.name, self.func_map)
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node):
        node.name = self._rename(node.name, self.class_map)
        self.generic_visit(node)
        return node


def normalize_ast(code):
    try:
        tree = ast.parse(code)
        norm = Normalizer().visit(tree)
        ast.fix_missing_locations(norm)
        return norm
    except Exception:
        return None


def _is_ast_possible(code):
    try:
        extract = "def" + code.split("```python\n")[-1].split("def")[1]
        ast_tree = normalize_ast(extract)
        return ast_tree is not None
    except Exception:
        return False


def _downgrade_plan_to_degenerate(steered_text, original_ym=None):
    if original_ym is not None and original_ym in steered_text:
        return True

    stripped_text = steered_text.strip()
    if stripped_text == "":
        return False
    if not _is_ast_possible(steered_text):
        return True
    if " " not in stripped_text and len(stripped_text) >= 10:
        return True
    if any_repeated_substring(steered_text) and (
        original_ym == "return" or (original_ym != "return" and "return" not in steered_text)
    ):
        return True
    if original_ym == "return":
        return True
    return False


def _not_planning(steered_text, original_ym=None):
    if original_ym is not None and original_ym in steered_text:
        return True
    stripped_text = steered_text.strip()
    if stripped_text == "" or (" " not in stripped_text and len(stripped_text) >= 10):
        return True
    return False


def _classify_earliest_position_as_planning(steering_results, planning_analysis, max_coeff):
    ym_keys = list(planning_analysis.keys())
    new_planning_analysis = {}
    for y_m in ym_keys:
        new_verdict = planning_analysis[y_m]["final_label"]
        base_suffix = steering_results[y_m].get("base_text", "")

        relevant_steered = [
            e for e in steering_results[y_m]["steered"] if e.get("coeff", -999999) >= max_coeff
        ]
        all_degenerate = all(
            _downgrade_plan_to_degenerate(e.get("decoded_text", ""), y_m) for e in relevant_steered
        )
        all_not_planning = all(
            _not_planning(e.get("decoded_text", ""), y_m) for e in relevant_steered
        )
        base_ok = not base_suffix.startswith(y_m)

        if planning_analysis[y_m]["final_label"] == "Plan":
            if base_ok and not all_degenerate:
                new_verdict = "Not planning" if all_not_planning else "Plan"
            else:
                new_verdict = "Can't say"
        elif planning_analysis[y_m]["final_label"] == "Can't say":
            if base_ok and not all_degenerate and all_not_planning:
                new_verdict = "Not planning"

        new_planning_analysis[y_m] = {
            "original_verdict": planning_analysis[y_m]["final_label"],
            "new_verdict": new_verdict,
        }

    return new_planning_analysis


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


def _filter_tokens_first_two_changes(all_tokens, token_map):
    filtered = []
    for token_planning_folder in all_tokens:
        prompt_idx, token_idx = _parse_prompt_token(token_planning_folder)
        if prompt_idx is None:
            continue
        bucket = token_map.get(prompt_idx)
        if not bucket:
            continue
        if "/base/" in token_planning_folder and token_idx in (
            bucket["base"],
            bucket["base"] + 1,
        ):
            filtered.append(token_planning_folder)
        elif "/instruct/" in token_planning_folder and token_idx in (
            bucket["instruct"],
            bucket["instruct"] + 1,
        ):
            filtered.append(token_planning_folder)
    return filtered


def update_earliest_position_planning(
    scale_folder, output_root, max_coeff, token_map=None, position_mode="all"
):
    all_tokens = [
        *glob.glob(os.path.join(scale_folder, "base", "*", "*")),
        *glob.glob(os.path.join(scale_folder, "instruct", "*", "*")),
    ]
    if token_map is not None:
        if position_mode == "first_change":
            all_tokens = _filter_tokens_first_change(all_tokens, token_map)
        elif position_mode == "first_two_changes":
            all_tokens = _filter_tokens_first_two_changes(all_tokens, token_map)

    for token_planning_folder in all_tokens:
        metadata = load_json(os.path.join(token_planning_folder, "metadata.json"))
        new_planning_analysis = {}
        if metadata.get("earliest_position") is not None:
            planning_analysis = load_json(
                os.path.join(token_planning_folder, "earliest_position_planning_analysis.json")
            )
            steering_results = load_json(
                os.path.join(token_planning_folder, "earliest_position.json")
            )
            new_planning_analysis = _classify_earliest_position_as_planning(
                steering_results, planning_analysis, max_coeff
            )

        updated_folder = os.path.join(output_root, *token_planning_folder.split("/")[-3:])
        os.makedirs(updated_folder, exist_ok=True)
        dump_json(os.path.join(updated_folder, "updated_planning_analysis.json"), new_planning_analysis)


def _detect_planning(planning_dict):
    return [key for key, value in planning_dict.items() if value["new_verdict"] == "Plan"]


def _detect_cant_says(planning_dict):
    return any(value["new_verdict"] == "Can't say" for value in planning_dict.values())


def _get_ym_plans(folder):
    token_planning_files = glob.glob(os.path.join(folder, "*", "*.json"))
    planning_datas = [load_json(f) for f in token_planning_files]
    y_ms = []
    for data in planning_datas:
        y_ms.extend(_detect_planning(data))
    return list(set(y_ms))


def _detect_cs(folder):
    token_planning_files = glob.glob(os.path.join(folder, "*", "*.json"))
    planning_datas = [load_json(f) for f in token_planning_files]
    return any(_detect_cant_says(data) for data in planning_datas)


def _get_base_and_instruct_plans(iter_idx, root_folder):
    base_folder = os.path.join(root_folder, "base", f"prompt_{iter_idx}")
    instruct_folder = os.path.join(root_folder, "instruct", f"prompt_{iter_idx}")
    base_yms = _get_ym_plans(base_folder) if os.path.exists(base_folder) else None
    instruct_yms = _get_ym_plans(instruct_folder) if os.path.exists(instruct_folder) else None
    return {"base": base_yms, "instruct": instruct_yms}


def _get_base_and_instruct_cantsays(iter_idx, root_folder):
    base_folder = os.path.join(root_folder, "base", f"prompt_{iter_idx}")
    instruct_folder = os.path.join(root_folder, "instruct", f"prompt_{iter_idx}")
    base_cs = _detect_cs(base_folder) if os.path.exists(base_folder) else False
    instruct_cs = _detect_cs(instruct_folder) if os.path.exists(instruct_folder) else False
    return {"base": base_cs, "instruct": instruct_cs}


def add_earliest_plan_fields(data, output_root):
    for idx, entry in enumerate(data):
        ym_plans = _get_base_and_instruct_plans(idx, root_folder=output_root)
        entry["base_plans_e"] = ym_plans["base"]
        entry["instruct_plans_e"] = ym_plans["instruct"]
        cs = _get_base_and_instruct_cantsays(idx, root_folder=output_root)
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


def _build_output_paths(output_base, max_coeff, desc, output_root, plot_path, position_mode):
    coeff_tag = f"c{abs(max_coeff)}"
    mode_tag = ""
    if position_mode == "first_change":
        mode_tag = "_first_change"
    elif position_mode == "first_two_changes":
        mode_tag = "_first_two_changes"
    suffix = f"_{desc}" if desc else ""
    resolved_output_root = output_root or os.path.join(
        output_base, f"scale_bvi_{coeff_tag}{mode_tag}{suffix}"
    )
    resolved_plot_path = plot_path or os.path.join(
        output_base, "plots", f"plan_grid_e_{coeff_tag}{mode_tag}{suffix}.png"
    )
    return resolved_output_root, resolved_plot_path


def main():
    parser = argparse.ArgumentParser(
        description="Classify earliest-position planning and save 4x4 grid plot."
    )
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--scale-folder", default=DEFAULT_SCALE_FOLDER)
    parser.add_argument("--output-base", default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--plot-path", default=None)
    parser.add_argument("--save-data-path", default=None)
    parser.add_argument("--max-coeff", type=int, default=DEFAULT_MAX_COEFF)
    parser.add_argument("--desc", default="")
    parser.add_argument(
        "--position-mode",
        choices=["all", "first_change", "first_two_changes"],
        default="all",
        help=(
            "Use all tokens, only the first base/instruct token positions, "
            "or the first two positions."
        ),
    )
    args = parser.parse_args()

    output_root, plot_path = _build_output_paths(
        args.output_base,
        args.max_coeff,
        args.desc,
        args.output_root,
        args.plot_path,
        args.position_mode,
    )

    data = load_json(args.data_path)
    token_map = None
    if args.position_mode in {"first_change", "first_two_changes"}:
        token_map = _build_position_token_map(data)

    update_earliest_position_planning(
        args.scale_folder,
        output_root,
        args.max_coeff,
        token_map=token_map,
        position_mode=args.position_mode,
    )

    add_earliest_plan_fields(data, output_root)

    title_bits = [f"max_coeff={args.max_coeff}"]
    if args.position_mode == "first_change":
        title_bits.append("first_change")
    elif args.position_mode == "first_two_changes":
        title_bits.append("first_two_changes")
    if args.desc:
        title_bits.append(args.desc)
    plot_title = "4x4 Grid (" + ", ".join(title_bits) + ")"

    ax = plot_4x4_plan_grid(data, with_e=True, title=plot_title)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    ax.figure.savefig(plot_path, bbox_inches="tight", dpi=200)

    if args.save_data_path:
        os.makedirs(os.path.dirname(args.save_data_path), exist_ok=True)
        dump_json(args.save_data_path, data)


if __name__ == "__main__":
    main()
