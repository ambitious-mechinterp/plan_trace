#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import builtins
import csv
import glob
import io
import json
import keyword
import math
import os
import re
import tokenize
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_PARENT = "/work/pi_jensen_umass_edu/abhishekmish_umass_edu/plan_trace"
DEFAULT_PROMPT_DATA = os.path.join(
    DEFAULT_PARENT,
    "data",
    "external",
    "all_examples_og_prompt_with_position_info_and_success_V2.json",
)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def extract_token_index(token_path: str) -> Optional[int]:
    match = re.search(r"token_(\d+)", token_path)
    return int(match.group(1)) if match else None


def get_predicted_token(metadata: Dict[str, Any]) -> Optional[str]:
    baseline_tokens = metadata.get("baseline_token_strings")
    if not isinstance(baseline_tokens, list) or len(baseline_tokens) < 2:
        return None
    return baseline_tokens[1]


def escape_token(token: Optional[str]) -> Optional[str]:
    if token is None:
        return None
    escaped = []
    for ch in token:
        if ch == "\t":
            escaped.append("\\t")
        elif ch == "\n":
            escaped.append("\\n")
        elif ch == "\r":
            escaped.append("\\r")
        elif ch == " ":
            escaped.append("\\s")
        else:
            escaped.append(ch)
    return "".join(escaped)


def load_prompt_data(path: str) -> List[Dict[str, Any]]:
    data = load_json(path)
    return data if isinstance(data, list) else []


def get_instruct_code(
    prompt_data: Sequence[Dict[str, Any]], prompt_idx: int
) -> Optional[str]:
    if 0 <= prompt_idx < len(prompt_data):
        code = prompt_data[prompt_idx].get("instruct_code")
        return code if isinstance(code, str) else None
    return None


def sanitize_token_list(tokens: Any) -> List[str]:
    if not isinstance(tokens, list):
        return []
    if not tokens:
        return []
    return [t for t in tokens[1:] if isinstance(t, str)]


def select_code_tokens(
    code: str, input_prefix_tokens: List[str], baseline_tokens: List[str]
) -> Tuple[List[str], int, str]:
    combined_tokens = input_prefix_tokens + baseline_tokens
    combined_text = "".join(combined_tokens)
    if combined_text == code:
        return combined_tokens, len(input_prefix_tokens), "combined"
    baseline_text = "".join(baseline_tokens)
    if baseline_text == code:
        return baseline_tokens, 0, "baseline"
    return combined_tokens, len(input_prefix_tokens), "combined_unverified"


def align_tokens_to_code(
    code: str, tokens: Sequence[str]
) -> List[Optional[Tuple[int, int]]]:
    spans: List[Optional[Tuple[int, int]]] = []
    offset = 0
    for token in tokens:
        if token == "":
            spans.append(None)
            continue
        if code.startswith(token, offset):
            start = offset
            end = offset + len(token)
            spans.append((start, end))
            offset = end
            continue
        found = code.find(token, offset)
        if found == -1:
            spans.append(None)
            continue
        start = found
        end = found + len(token)
        spans.append((start, end))
        offset = end
    return spans


def get_line_offsets(code: str) -> List[int]:
    offsets = [0]
    for line in code.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


DELIMITERS = {",", ":", "(", ")", "[", "]", "{", "}", "."}
SKIP_ROLE_TYPES = {
    tokenize.NL,
    tokenize.NEWLINE,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.COMMENT,
}
BUILTIN_CALLABLES = {
    name for name, value in builtins.__dict__.items() if callable(value)
}


def map_lex_tag(token_type: int, token_str: str) -> Optional[str]:
    if token_type == tokenize.NAME:
        return "KW" if keyword.iskeyword(token_str) else "ID"
    if token_type == tokenize.NUMBER:
        return "NUM"
    if token_type == tokenize.STRING:
        return "STR"
    if token_type == tokenize.OP:
        return "DL" if token_str in DELIMITERS else "OP"
    if token_type == tokenize.INDENT:
        return "IND"
    if token_type == tokenize.DEDENT:
        return "DED"
    if token_type == tokenize.COMMENT:
        return "CMT"
    return None


def tokenize_python(code: str) -> List[Dict[str, Any]]:
    offsets = get_line_offsets(code)
    tokens: List[Dict[str, Any]] = []
    try:
        stream = tokenize.generate_tokens(io.StringIO(code).readline)
        for tok in stream:
            if tok.type in {tokenize.ENCODING, tokenize.ENDMARKER}:
                continue
            start = offsets[tok.start[0] - 1] + tok.start[1]
            end = offsets[tok.end[0] - 1] + tok.end[1]
            tokens.append(
                {
                    "string": tok.string,
                    "start": start,
                    "end": end,
                    "type": tok.type,
                    "lex_tag": map_lex_tag(tok.type, tok.string),
                    "role_tag": None,
                    "role_context": None,
                }
            )
    except tokenize.TokenError:
        return tokens

    significant = [
        idx for idx, tok in enumerate(tokens) if tok["type"] not in SKIP_ROLE_TYPES
    ]
    for pos, idx in enumerate(significant):
        tok = tokens[idx]
        if tok["lex_tag"] != "ID":
            continue
        prev_tok = tokens[significant[pos - 1]] if pos > 0 else None
        next_tok = tokens[significant[pos + 1]] if pos + 1 < len(significant) else None
        after_dot = prev_tok is not None and prev_tok["string"] == "."
        followed_by_call = next_tok is not None and next_tok["string"] == "("
        if after_dot:
            tok["role_context"] = "attr_call" if followed_by_call else "attr_access"
            tok["role_tag"] = "MTH" if followed_by_call else "ATTR"
        elif followed_by_call:
            tok["role_context"] = "call"
            tok["role_tag"] = "BLT" if tok["string"] in BUILTIN_CALLABLES else "FN"
    return tokens


def classify_token_string(token_str: Optional[str]) -> Optional[str]:
    if token_str is None:
        return None
    if token_str.strip() == "":
        return None
    text = token_str.lstrip()
    try:
        stream = tokenize.generate_tokens(io.StringIO(text).readline)
        significant = []
        for tok in stream:
            if tok.type in {tokenize.ENCODING, tokenize.ENDMARKER}:
                continue
            if tok.type in {tokenize.NL, tokenize.NEWLINE}:
                continue
            significant.append(tok)
    except tokenize.TokenError:
        return None
    if len(significant) != 1:
        return None
    return map_lex_tag(significant[0].type, significant[0].string)


def classify_planned_token(
    token_str: Optional[str], role_context: Optional[str]
) -> Tuple[Optional[str], Optional[str]]:
    lex_tag = classify_token_string(token_str)
    role_tag = None
    if lex_tag == "ID":
        if role_context == "call":
            role_tag = "BLT" if token_str in BUILTIN_CALLABLES else "FN"
        elif role_context == "attr_call":
            role_tag = "MTH"
        elif role_context == "attr_access":
            role_tag = "ATTR"
        elif token_str in BUILTIN_CALLABLES:
            role_tag = "BLT"
    return lex_tag, role_tag


def find_best_python_token(
    python_tokens: Sequence[Dict[str, Any]], span: Optional[Tuple[int, int]]
) -> Optional[Dict[str, Any]]:
    if span is None:
        return None
    start, end = span
    best = None
    best_overlap = 0
    for tok in python_tokens:
        overlap = min(end, tok["end"]) - max(start, tok["start"])
        if overlap <= 0:
            continue
        if overlap > best_overlap:
            best_overlap = overlap
            best = tok
    return best


def plot_tag_distribution(
    tag_counts: Counter, title: str, out_path: str, xlabel: str, ylabel: str
) -> None:
    if not tag_counts:
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ImportError:
        print("matplotlib is required to write plots; skipping plot output.")
        return
    labels, values = zip(*tag_counts.most_common())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
        tick.set_ha("right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_probability_matrix(
    matrix: List[List[float]],
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    out_path: str,
) -> None:
    if not matrix or not row_labels or not col_labels:
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ImportError:
        print("matplotlib is required to write plots; skipping matrix output.")
        return
    masked = None
    try:
        import numpy as np  # type: ignore[import-not-found]

        masked = np.array(matrix, dtype=float)
        masked = np.ma.masked_invalid(masked)
    except ImportError:
        masked = matrix
    fig, ax = plt.subplots(figsize=(max(6, len(col_labels) * 0.6), 5))
    im = ax.imshow(masked, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Predicted tag")
    ax.set_ylabel("Planned tag")
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)
    fig.colorbar(im, ax=ax, label="Probability")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def normalize_counts(tag_counts: Counter) -> Counter:
    total = sum(tag_counts.values())
    if total == 0:
        return Counter()
    return Counter({tag: count / total for tag, count in tag_counts.items()})


def ratio_counts(numerator: Counter, denominator: Counter) -> Counter:
    ratios = Counter()
    for tag in set(numerator) | set(denominator):
        denom = denominator.get(tag, 0)
        if denom:
            ratios[tag] = numerator.get(tag, 0) / denom
    return ratios


def wilson_interval(
    count: int, total: int, z: float = 1.96
) -> Tuple[Optional[float], Optional[float]]:
    if total <= 0:
        return None, None
    phat = count / total
    denom = 1 + (z**2) / total
    center = (phat + (z**2) / (2 * total)) / denom
    margin = (
        z
        * math.sqrt((phat * (1 - phat) / total) + (z**2) / (4 * total**2))
        / denom
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def tag_for_plot(lex_tag: Optional[str], role_tag: Optional[str]) -> Optional[str]:
    if role_tag in {"BLT", "FN", "MTH", "ATTR"}:
        return role_tag
    return lex_tag


def build_ast_spans(code: str) -> List[Tuple[int, int, str]]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    offsets = get_line_offsets(code)
    spans: List[Tuple[int, int, str]] = []
    for node in ast.walk(tree):
        lineno = getattr(node, "lineno", None)
        end_lineno = getattr(node, "end_lineno", None)
        col = getattr(node, "col_offset", None)
        end_col = getattr(node, "end_col_offset", None)
        if (
            lineno is None
            or end_lineno is None
            or col is None
            or end_col is None
        ):
            continue
        start = offsets[lineno - 1] + col
        end = offsets[end_lineno - 1] + end_col
        if end < start:
            continue
        spans.append((start, end, node.__class__.__name__))
    spans.sort(key=lambda item: item[1] - item[0])
    return spans


def find_smallest_covering_node(
    ast_spans: Sequence[Tuple[int, int, str]], start: int, end: int
) -> Optional[str]:
    for node_start, node_end, node_type in ast_spans:
        if node_start <= start and node_end >= end:
            return node_type
    return None


def map_token_idx_to_code_idx(
    token_idx: Optional[int], prefix_len: int, mode: str, code_tokens_len: int
) -> Optional[int]:
    if token_idx is None:
        return None
    if mode == "baseline":
        if token_idx < prefix_len:
            return None
        code_idx = token_idx - prefix_len
    else:
        code_idx = token_idx
    if code_idx < 0 or code_idx >= code_tokens_len:
        return None
    return code_idx


def analyze_tokens(
    parent_folder: str,
    out_json: str,
    out_csv: Optional[str],
    prompt_data_path: str,
    min_relational_total: int,
) -> None:
    folder = os.path.join(parent_folder, "outputs", "planning_scale", "instruct", "*")
    rows: List[Dict[str, Any]] = []
    prompt_data = load_prompt_data(prompt_data_path)
    prompt_cache: Dict[int, Dict[str, Any]] = {}
    baseline_tag_counts = Counter()
    baseline_tag_counts_per_prompt: Dict[int, Counter] = {}
    planned_candidate_tag_counts = Counter()
    not_planned_candidate_tag_counts = Counter()
    cs_candidate_tag_counts = Counter()
    affected_predicted_tag_counts = Counter()
    not_affected_predicted_tag_counts = Counter()
    baseline_prompts_seen = set()

    for fi in glob.glob(folder):
        token_files = glob.glob(fi + "/*")
        prompt_idx = fi.split("_")[-1]
        for tf in token_files:
            planning_path = tf.replace(
                "planning_scale", "planning_scale_classified_e"
            ) + "/updated_planning_analysis.json"
            if not os.path.exists(planning_path):
                print(f"Missing file: {planning_path}")
                continue

            planning_analysis = load_json(planning_path)
            plan_tokens: List[str] = []
            plan_count = 0
            cs_count = 0
            np_count = 0
            for key, value in planning_analysis.items():
                verdict = value.get("new_verdict") if isinstance(value, dict) else value
                lex_tag, role_tag = classify_planned_token(key, None)
                tag = tag_for_plot(lex_tag, role_tag)
                if verdict == "Plan":
                    plan_tokens.append(key)
                    plan_count += 1
                    if tag:
                        planned_candidate_tag_counts[tag] += 1
                elif verdict == "Can't say":
                    cs_count += 1
                    if tag:
                        cs_candidate_tag_counts[tag] += 1
                elif verdict == "Not planning":
                    np_count += 1
                    if tag:
                        not_planned_candidate_tag_counts[tag] += 1

            metadata_path = os.path.join(tf, "metadata.json")
            metadata = load_json(metadata_path) if os.path.exists(metadata_path) else {}
            predicted_token = get_predicted_token(metadata)

            prompt_id = int(prompt_idx)
            instruct_code = get_instruct_code(prompt_data, prompt_id)
            input_prefix_tokens = sanitize_token_list(
                metadata.get("input_prefix_token_strings")
            )
            baseline_tokens = sanitize_token_list(
                metadata.get("baseline_token_strings")
            )
            code_tokens: List[str] = []
            token_spans: List[Optional[Tuple[int, int]]] = []
            ast_spans: List[Tuple[int, int, str]] = []
            token_idx = extract_token_index(tf)
            planned_tokens_ast_groups: List[Optional[str]] = []
            predicted_token_ast_group: Optional[str] = None
            predicted_lex_tag: Optional[str] = None
            predicted_role_tag: Optional[str] = None
            planned_tokens_lex_tags: List[Optional[str]] = []
            planned_tokens_role_tags: List[Optional[str]] = []
            if instruct_code:
                cache_key = prompt_id
                if cache_key not in prompt_cache:
                    code_tokens, prefix_len, mode = select_code_tokens(
                        instruct_code, input_prefix_tokens, baseline_tokens
                    )
                    token_spans = align_tokens_to_code(instruct_code, code_tokens)
                    ast_spans = build_ast_spans(instruct_code)
                    python_tokens = tokenize_python(instruct_code)
                    prompt_tag_counts = Counter()
                    for token in python_tokens:
                        tag = tag_for_plot(token["lex_tag"], token["role_tag"])
                        if tag:
                            prompt_tag_counts[tag] += 1
                    prompt_cache[cache_key] = {
                        "code_tokens": code_tokens,
                        "token_spans": token_spans,
                        "ast_spans": ast_spans,
                        "python_tokens": python_tokens,
                        "prompt_tag_counts": prompt_tag_counts,
                        "prefix_len": prefix_len,
                        "mode": mode,
                    }
                if cache_key not in baseline_prompts_seen:
                    baseline_prompts_seen.add(cache_key)
                    prompt_counts = prompt_cache[cache_key]["prompt_tag_counts"]
                    baseline_tag_counts.update(prompt_counts)
                    baseline_tag_counts_per_prompt[cache_key] = prompt_counts.copy()
                cached = prompt_cache[cache_key]
                code_tokens = cached["code_tokens"]
                token_spans = cached["token_spans"]
                ast_spans = cached["ast_spans"]
                python_tokens = cached["python_tokens"]
                code_idx = map_token_idx_to_code_idx(
                    token_idx, cached["prefix_len"], cached["mode"], len(code_tokens)
                )
                matched_python_token = None
                if code_idx is not None and token_spans[code_idx] is not None:
                    start, end = token_spans[code_idx]
                    predicted_token_ast_group = find_smallest_covering_node(
                        ast_spans, start, end
                    )
                    matched_python_token = find_best_python_token(
                        python_tokens, (start, end)
                    )
                role_context = (
                    matched_python_token["role_context"]
                    if matched_python_token is not None
                    else None
                )
                predicted_lex_tag = (
                    matched_python_token["lex_tag"]
                    if matched_python_token is not None
                    else classify_token_string(predicted_token)
                )
                predicted_role_tag = (
                    matched_python_token["role_tag"] if matched_python_token else None
                )
            if (
                predicted_role_tag is None
                and predicted_lex_tag == "ID"
                and predicted_token in BUILTIN_CALLABLES
            ):
                predicted_role_tag = "BLT"

            if plan_tokens:
                planned_tokens_ast_groups = [
                    predicted_token_ast_group for _ in plan_tokens
                ]
                for token_str in plan_tokens:
                    lex_tag, role_tag = classify_planned_token(
                        token_str, role_context
                    )
                    planned_tokens_lex_tags.append(lex_tag)
                    planned_tokens_role_tags.append(role_tag)
            else:
                planned_tokens_ast_groups = [None] * len(plan_tokens)
                planned_tokens_lex_tags = []
                planned_tokens_role_tags = []
                for token_str in plan_tokens:
                    lex_tag, role_tag = classify_planned_token(token_str, None)
                    planned_tokens_lex_tags.append(lex_tag)
                    planned_tokens_role_tags.append(role_tag)

            predicted_tag = tag_for_plot(predicted_lex_tag, predicted_role_tag)
            if predicted_tag:
                if plan_count > 0:
                    affected_predicted_tag_counts[predicted_tag] += 1
                else:
                    not_affected_predicted_tag_counts[predicted_tag] += 1

            rows.append(
                {
                    "token_path": tf,
                    "prompt": prompt_id,
                    "token_idx": token_idx,
                    "plan_count": plan_count,
                    "cs_count": cs_count,
                    "np_count": np_count,
                    "any_plan": plan_count > 0,
                    "planned_tokens": plan_tokens,
                    "planned_tokens_ast_groups": planned_tokens_ast_groups,
                    "planned_tokens_lex_tags": planned_tokens_lex_tags,
                    "planned_tokens_role_tags": planned_tokens_role_tags,
                    "predicted_token": predicted_token,
                    "predicted_token_ast_group": predicted_token_ast_group,
                    "predicted_token_lex_tag": predicted_lex_tag,
                    "predicted_token_role_tag": predicted_role_tag,
                    "predicted_in_planned": predicted_token in plan_tokens
                    if predicted_token
                    else False,
                }
            )

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)

    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(
                [
                    "token_path",
                    "prompt",
                    "token_idx",
                    "plan_count",
                    "cs_count",
                    "np_count",
                    "any_plan",
                    "planned_tokens",
                    "planned_tokens_ast_groups",
                    "planned_tokens_lex_tags",
                    "planned_tokens_role_tags",
                    "predicted_token",
                    "predicted_token_ast_group",
                    "predicted_token_lex_tag",
                    "predicted_token_role_tag",
                    "predicted_in_planned",
                ]
            )
            for row in rows:
                escaped_planned_tokens = [
                    escape_token(token) for token in row["planned_tokens"]
                ]
                escaped_planned_groups = [
                    escape_token(token) for token in row["planned_tokens_ast_groups"]
                ]
                escaped_planned_lex_tags = [
                    escape_token(tag) for tag in row["planned_tokens_lex_tags"]
                ]
                escaped_planned_role_tags = [
                    escape_token(tag) for tag in row["planned_tokens_role_tags"]
                ]
                escaped_predicted_token = escape_token(row["predicted_token"])
                writer.writerow(
                    [
                        row["token_path"],
                        row["prompt"],
                        row["token_idx"],
                        row["plan_count"],
                        row["cs_count"],
                        row["np_count"],
                        row["any_plan"],
                        json.dumps(escaped_planned_tokens),
                        json.dumps(escaped_planned_groups),
                        json.dumps(escaped_planned_lex_tags),
                        json.dumps(escaped_planned_role_tags),
                        escaped_predicted_token,
                        escape_token(row["predicted_token_ast_group"]),
                        escape_token(row["predicted_token_lex_tag"]),
                        escape_token(row["predicted_token_role_tag"]),
                        row["predicted_in_planned"],
                    ]
                )

    planned_tag_counts = Counter()
    for row in rows:
        for lex_tag, role_tag in zip(
            row["planned_tokens_lex_tags"], row["planned_tokens_role_tags"]
        ):
            tag = tag_for_plot(lex_tag, role_tag)
            if tag:
                planned_tag_counts[tag] += 1

    predicted_tag_counts = Counter()
    for row in rows:
        tag = tag_for_plot(
            row["predicted_token_lex_tag"], row["predicted_token_role_tag"]
        )
        if tag:
            predicted_tag_counts[tag] += 1

    joint_planned_given_pred = Counter()
    planned_total_by_pred = Counter()
    for row in rows:
        if row["plan_count"] <= 0:
            continue
        predicted_tag = tag_for_plot(
            row["predicted_token_lex_tag"], row["predicted_token_role_tag"]
        )
        if not predicted_tag:
            continue
        planned_tags = []
        for lex_tag, role_tag in zip(
            row["planned_tokens_lex_tags"], row["planned_tokens_role_tags"]
        ):
            tag = tag_for_plot(lex_tag, role_tag)
            if tag:
                planned_tags.append(tag)
        if not planned_tags:
            continue
        planned_total_by_pred[predicted_tag] += len(planned_tags)
        for planned_tag in planned_tags:
            joint_planned_given_pred[(planned_tag, predicted_tag)] += 1

    plot_tag_distribution(
        planned_tag_counts,
        "Planned token tag distribution (role overrides lex)",
        "outputs/analysis/planned_tokens_tags.png",
        "Tag",
        "Count",
    )
    plot_tag_distribution(
        predicted_tag_counts,
        "Predicted token tag distribution (role overrides lex)",
        "outputs/analysis/predicted_token_tags.png",
        "Tag",
        "Count",
    )
    plot_tag_distribution(
        baseline_tag_counts,
        "Baseline tag distribution (all generated tokens)",
        "outputs/analysis/baseline_token_tags.png",
        "Tag",
        "Count",
    )

    planned_tag_freq = normalize_counts(planned_tag_counts)
    predicted_tag_freq = normalize_counts(predicted_tag_counts)
    baseline_tag_freq = normalize_counts(baseline_tag_counts)
    plot_tag_distribution(
        planned_tag_freq,
        "Planned token tag distribution (normalized)",
        "outputs/analysis/planned_tokens_tags_normalized.png",
        "Tag",
        "Share",
    )
    plot_tag_distribution(
        predicted_tag_freq,
        "Predicted token tag distribution (normalized)",
        "outputs/analysis/predicted_token_tags_normalized.png",
        "Tag",
        "Share",
    )
    plot_tag_distribution(
        baseline_tag_freq,
        "Baseline tag distribution (normalized)",
        "outputs/analysis/baseline_token_tags_normalized.png",
        "Tag",
        "Share",
    )

    planned_candidate_rate = normalize_counts(planned_candidate_tag_counts)
    not_planned_candidate_rate = normalize_counts(not_planned_candidate_tag_counts)
    plan_vs_not_ratio = ratio_counts(planned_candidate_rate, not_planned_candidate_rate)
    plot_tag_distribution(
        plan_vs_not_ratio,
        "Planned vs not-planned tag ratio (candidate rates)",
        "outputs/analysis/planned_vs_not_planned_ratio.png",
        "Tag",
        "Planned/Not planned",
    )

    affected_predicted_rate = normalize_counts(affected_predicted_tag_counts)
    not_affected_predicted_rate = normalize_counts(not_affected_predicted_tag_counts)
    affected_vs_not_ratio = ratio_counts(
        affected_predicted_rate, not_affected_predicted_rate
    )
    plot_tag_distribution(
        affected_vs_not_ratio,
        "Affected vs not-affected tag ratio (predicted rates)",
        "outputs/analysis/affected_vs_not_affected_ratio.png",
        "Tag",
        "Affected/Not affected",
    )

    os.makedirs("outputs/analysis", exist_ok=True)
    with open("outputs/analysis/tag_control_baseline.json", "w") as f:
        json.dump(
            {
                "baseline_tag_counts": dict(baseline_tag_counts),
                "baseline_tag_rates": dict(baseline_tag_freq),
                "baseline_tag_counts_per_prompt": {
                    str(k): dict(v) for k, v in baseline_tag_counts_per_prompt.items()
                },
                "planned_candidate_tag_counts": dict(planned_candidate_tag_counts),
                "not_planned_candidate_tag_counts": dict(
                    not_planned_candidate_tag_counts
                ),
                "cant_say_candidate_tag_counts": dict(cs_candidate_tag_counts),
                "planned_candidate_tag_rates": dict(planned_candidate_rate),
                "not_planned_candidate_tag_rates": dict(not_planned_candidate_rate),
                "planned_vs_not_planned_ratio": dict(plan_vs_not_ratio),
                "affected_predicted_tag_counts": dict(affected_predicted_tag_counts),
                "not_affected_predicted_tag_counts": dict(
                    not_affected_predicted_tag_counts
                ),
                "affected_predicted_tag_rates": dict(affected_predicted_rate),
                "not_affected_predicted_tag_rates": dict(not_affected_predicted_rate),
                "affected_vs_not_affected_ratio": dict(affected_vs_not_ratio),
                "planned_given_predicted_counts": {
                    f"{planned}|{predicted}": count
                    for (planned, predicted), count in joint_planned_given_pred.items()
                },
                "planned_given_predicted_total": dict(planned_total_by_pred),
            },
            f,
            indent=2,
        )

    relational_rows = []
    for (planned_tag, predicted_tag), count in joint_planned_given_pred.items():
        total = planned_total_by_pred.get(predicted_tag, 0)
        prob = count / total if total else None
        lower, upper = wilson_interval(count, total)
        relational_rows.append(
            {
                "planned_tag": planned_tag,
                "predicted_tag": predicted_tag,
                "count": count,
                "total_planned_for_predicted": total,
                "probability": prob,
                "ci_lower": lower,
                "ci_upper": upper,
            }
        )
    relational_rows.sort(
        key=lambda item: (
            item["predicted_tag"],
            -(item["probability"] or 0),
            item["planned_tag"],
        )
    )

    predicted_tags = sorted(planned_total_by_pred.keys())
    planned_tags = sorted(
        {planned for planned, _ in joint_planned_given_pred.keys()}
    )
    prob_matrix = []
    for planned_tag in planned_tags:
        row = []
        for predicted_tag in predicted_tags:
            total = planned_total_by_pred.get(predicted_tag, 0)
            count = joint_planned_given_pred.get((planned_tag, predicted_tag), 0)
            if total and total >= min_relational_total:
                row.append(count / total)
            else:
                row.append(float("nan"))
        prob_matrix.append(row)
    plot_probability_matrix(
        prob_matrix,
        planned_tags,
        predicted_tags,
        "P(planned tag | planning, predicted tag)",
        "outputs/analysis/planned_given_predicted_matrix.png",
    )
    with open("outputs/analysis/planned_given_predicted.json", "w") as f:
        json.dump(relational_rows, f, indent=2)
    with open("outputs/analysis/planned_given_predicted.csv", "w", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(
            [
                "planned_tag",
                "predicted_tag",
                "count",
                "total_planned_for_predicted",
                "probability",
                "ci_lower",
                "ci_upper",
            ]
        )
        for row in relational_rows:
            writer.writerow(
                [
                    row["planned_tag"],
                    row["predicted_tag"],
                    row["count"],
                    row["total_planned_for_predicted"],
                    row["probability"],
                    row["ci_lower"],
                    row["ci_upper"],
                ]
            )
    with open(
        "outputs/analysis/planned_given_predicted_filtered.csv", "w", newline=""
    ) as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(
            [
                "planned_tag",
                "predicted_tag",
                "count",
                "total_planned_for_predicted",
                "probability",
                "ci_lower",
                "ci_upper",
            ]
        )
        for row in relational_rows:
            if row["total_planned_for_predicted"] < min_relational_total:
                continue
            writer.writerow(
                [
                    row["planned_tag"],
                    row["predicted_tag"],
                    row["count"],
                    row["total_planned_for_predicted"],
                    row["probability"],
                    row["ci_lower"],
                    row["ci_upper"],
                ]
            )

    with_pred = sum(1 for r in rows if r["predicted_token"] is not None)
    in_planned = sum(1 for r in rows if r["predicted_in_planned"])

    print(f"Total token rows: {len(rows)}")
    print(f"With predicted token available: {with_pred}")
    if with_pred:
        print(f"Predicted token in planned set: {in_planned}/{with_pred}")
    print(f"Wrote JSON: {out_json}")
    if out_csv:
        print(f"Wrote CSV: {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze planned tokens vs predicted tokens using the same"
            " classified outputs as plan_scale_analysis_v2.ipynb."
        )
    )
    parser.add_argument(
        "--parent-folder",
        default=DEFAULT_PARENT,
        help="Root plan_trace folder used in the notebook",
    )
    parser.add_argument(
        "--out-json",
        default="outputs/plan_token_prediction_analysis.json",
        help="Path to write JSON results",
    )
    parser.add_argument(
        "--out-csv",
        default="outputs/plan_token_prediction_analysis.csv",
        help="Path to write CSV results. Use empty string to skip.",
    )
    parser.add_argument(
        "--prompt-data",
        default=DEFAULT_PROMPT_DATA,
        help="Path to prompt dataset containing instruct_code.",
    )
    parser.add_argument(
        "--min-relational-total",
        type=int,
        default=5,
        help="Minimum total count for relational probabilities and matrix.",
    )

    args = parser.parse_args()
    out_csv = args.out_csv if args.out_csv else None
    analyze_tokens(
        args.parent_folder,
        args.out_json,
        out_csv,
        args.prompt_data,
        args.min_relational_total,
    )


if __name__ == "__main__":
    main()
