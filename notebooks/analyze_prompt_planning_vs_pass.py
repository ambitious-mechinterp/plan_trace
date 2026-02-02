#!/usr/bin/env python3
"""
Analyze prompt-level planning rate vs pass rate.
Computes "any plan" rate per prompt and creates ROC curve for pass prediction.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


DEFAULT_PARENT = "/work/pi_jensen_umass_edu/abhishekmish_umass_edu/plan_trace"
DEFAULT_PROMPT_DATA = os.path.join(
    "/work/pi_jensen_umass_edu/jnainani_umass_edu/plan_trace",
    "data",
    "external",
    "all_examples_og_prompt_with_position_info_and_success_V2.json",
)


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def compute_prompt_planning_metrics(
    parent_folder: str,
    use_random_steer: bool = True,
    verdict_key: str = "new_new_verdict",
) -> List[Dict[str, Any]]:
    """
    Compute planning metrics for each prompt.

    Returns list of dicts with:
    - prompt_idx: int
    - n_tokens: int (total tokens for this prompt)
    - tokens_with_any_plan: int (tokens with at least 1 Plan verdict)
    - any_plan_rate: float (fraction of tokens with planning)
    - total_plan_events: int (total Plan verdicts across all tokens)
    - plan_events_per_token: float (average Plan verdicts per token)
    """
    folder = os.path.join(parent_folder, "outputs", "planning_scale", "instruct", "*")

    prompt_metrics: Dict[int, Dict[str, Any]] = {}

    for fi in glob.glob(folder):
        # Skip if not a directory
        if not os.path.isdir(fi):
            continue

        # Extract prompt index from folder name (e.g., "prompt_42")
        folder_name = os.path.basename(fi)
        if not folder_name.startswith("prompt_"):
            continue

        try:
            prompt_idx = int(folder_name.split("_")[-1])
        except ValueError:
            print(f"Warning: Could not parse prompt index from {fi}")
            continue

        token_files = glob.glob(os.path.join(fi, "*"))

        if prompt_idx not in prompt_metrics:
            prompt_metrics[prompt_idx] = {
                "prompt_idx": prompt_idx,
                "n_tokens": len(token_files),
                "n_tokens_with_files": 0,  # Track how many tokens have planning files
                "tokens_with_any_plan": 0,
                "total_plan_events": 0,
                "total_cs_events": 0,
                "total_np_events": 0,
            }

        for tf in token_files:
            # Check for random steer file first, fall back to original
            base_path = tf.replace("planning_scale", "planning_scale_classified_e")
            random_steer_path = os.path.join(
                base_path, "updated_planning_analysis_random_steer.json"
            )
            original_path = os.path.join(base_path, "updated_planning_analysis.json")

            planning_path = None
            if use_random_steer and os.path.exists(random_steer_path):
                planning_path = random_steer_path
            elif os.path.exists(original_path):
                planning_path = original_path

            if planning_path is None:
                continue

            planning_analysis = load_json(planning_path)
            prompt_metrics[prompt_idx]["n_tokens_with_files"] += 1

            # Count events using specified verdict key with fallback
            token_has_plan = False
            for key, value in planning_analysis.items():
                if isinstance(value, dict):
                    verdict = value.get(verdict_key)
                    if verdict is None:
                        verdict = value.get("new_verdict")
                    if verdict is None:
                        verdict = value.get("original_verdict")
                else:
                    verdict = value

                if verdict == "Plan":
                    prompt_metrics[prompt_idx]["total_plan_events"] += 1
                    token_has_plan = True
                elif verdict == "Can't say":
                    prompt_metrics[prompt_idx]["total_cs_events"] += 1
                elif verdict == "Not planning":
                    prompt_metrics[prompt_idx]["total_np_events"] += 1

            if token_has_plan:
                prompt_metrics[prompt_idx]["tokens_with_any_plan"] += 1

    # Compute rates
    results = []
    for prompt_idx, metrics in sorted(prompt_metrics.items()):
        n_tokens = metrics["n_tokens"]
        metrics["any_plan_rate"] = (
            metrics["tokens_with_any_plan"] / n_tokens if n_tokens > 0 else 0.0
        )
        metrics["plan_events_per_token"] = (
            metrics["total_plan_events"] / n_tokens if n_tokens > 0 else 0.0
        )
        results.append(metrics)

    return results


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_name: str,
    out_path: str,
) -> Tuple[float, float]:
    """Plot ROC curve and return AUC and best threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Find best threshold (maximize Youden's J statistic)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    best_tpr = tpr[best_idx]
    best_fpr = fpr[best_idx]

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    plt.scatter(
        [best_fpr],
        [best_tpr],
        color="red",
        s=100,
        zorder=5,
        label=f"Best threshold = {best_threshold:.3f}",
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {metric_name} predicting Pass")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return roc_auc, best_threshold


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_name: str,
    out_path: str,
) -> float:
    """Plot precision-recall curve and return average precision."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    avg_precision = average_precision_score(y_true, y_score)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="darkorange", lw=2, label=f"PR curve (AP = {avg_precision:.3f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve: {metric_name} predicting Pass")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return avg_precision


def plot_scatter(
    metrics: List[Dict[str, Any]],
    pass_status: Dict[int, bool],
    metric_key: str,
    metric_name: str,
    out_path: str,
) -> None:
    """Create scatter plot of metric vs pass/fail."""
    pass_values = []
    fail_values = []

    for m in metrics:
        prompt_idx = m["prompt_idx"]
        value = m[metric_key]
        if pass_status.get(prompt_idx):
            pass_values.append(value)
        else:
            fail_values.append(value)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        range(len(pass_values)),
        pass_values,
        color="green",
        alpha=0.6,
        label=f"Pass (n={len(pass_values)})",
    )
    plt.scatter(
        range(len(fail_values)),
        fail_values,
        color="red",
        alpha=0.6,
        label=f"Fail (n={len(fail_values)})",
    )
    plt.xlabel("Prompt index (sorted by pass/fail)")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} by Pass/Fail Status")
    plt.legend()
    plt.grid(alpha=0.3)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def analyze_planning_vs_pass(
    parent_folder: str,
    prompt_data_path: str,
    out_dir: str,
    use_random_steer: bool = True,
    verdict_key: str = "new_new_verdict",
) -> None:
    """Main analysis function."""
    print(f"Computing planning metrics per prompt...")
    print(f"  use_random_steer: {use_random_steer}")
    print(f"  verdict_key: {verdict_key}")

    metrics = compute_prompt_planning_metrics(
        parent_folder,
        use_random_steer=use_random_steer,
        verdict_key=verdict_key,
    )

    print(f"  Found metrics for {len(metrics)} prompts")

    # Load pass/fail status
    print(f"\nLoading prompt data from {prompt_data_path}...")
    prompt_data = load_json(prompt_data_path)
    pass_status = {i: entry.get("instruct_pass", False) for i, entry in enumerate(prompt_data)}

    # Filter to only prompts we have metrics for AND that have planning data
    # (at least one token with planning analysis file)
    metrics_with_data = [
        m for m in metrics
        if m["n_tokens_with_files"] > 0  # Has at least one token with planning file
    ]

    print(f"  Prompts with planning data: {len(metrics_with_data)} / {len(metrics)}")

    y_true = np.array([pass_status.get(m["prompt_idx"], False) for m in metrics_with_data])

    n_pass = np.sum(y_true)
    n_fail = len(y_true) - n_pass
    print(f"  Pass prompts with data: {n_pass}")
    print(f"  Fail prompts with data: {n_fail}")

    # Update metrics to only include those with data
    metrics = metrics_with_data

    # Analyze both metrics
    metrics_to_analyze = [
        ("any_plan_rate", "Any-Plan Rate"),
        ("plan_events_per_token", "Plan Events Per Token"),
    ]

    os.makedirs(out_dir, exist_ok=True)

    results_summary = []

    for metric_key, metric_name in metrics_to_analyze:
        print(f"\n=== {metric_name} ===")

        y_score = np.array([m[metric_key] for m in metrics])

        # Statistics
        pass_values = y_score[y_true == 1]
        fail_values = y_score[y_true == 0]

        print(f"  Pass mean: {np.mean(pass_values):.4f} ± {np.std(pass_values):.4f}")
        print(f"  Fail mean: {np.mean(fail_values):.4f} ± {np.std(fail_values):.4f}")
        print(f"  Difference: {np.mean(pass_values) - np.mean(fail_values):.4f}")

        # ROC curve
        roc_auc, best_threshold = plot_roc_curve(
            y_true,
            y_score,
            metric_name,
            os.path.join(out_dir, f"roc_{metric_key}.png"),
        )
        print(f"  ROC AUC: {roc_auc:.4f}")
        print(f"  Best threshold: {best_threshold:.4f}")

        # Precision-Recall curve
        avg_precision = plot_precision_recall_curve(
            y_true,
            y_score,
            metric_name,
            os.path.join(out_dir, f"pr_{metric_key}.png"),
        )
        print(f"  Average Precision: {avg_precision:.4f}")

        # Scatter plot
        plot_scatter(
            metrics,
            pass_status,
            metric_key,
            metric_name,
            os.path.join(out_dir, f"scatter_{metric_key}.png"),
        )

        results_summary.append({
            "metric": metric_name,
            "metric_key": metric_key,
            "pass_mean": float(np.mean(pass_values)),
            "pass_std": float(np.std(pass_values)),
            "fail_mean": float(np.mean(fail_values)),
            "fail_std": float(np.std(fail_values)),
            "mean_diff": float(np.mean(pass_values) - np.mean(fail_values)),
            "roc_auc": float(roc_auc),
            "best_threshold": float(best_threshold),
            "avg_precision": float(avg_precision),
        })

    # Save results
    with open(os.path.join(out_dir, "metrics_summary.json"), "w") as f:
        json.dump(
            {
                "prompt_metrics": metrics,
                "analysis_summary": results_summary,
                "n_prompts": len(metrics),
                "n_pass": int(n_pass),
                "n_fail": int(n_fail),
            },
            f,
            indent=2,
        )

    print(f"\n=== Summary ===")
    print(f"Results saved to {out_dir}")
    print(f"  ROC curves: roc_*.png")
    print(f"  PR curves: pr_*.png")
    print(f"  Scatter plots: scatter_*.png")
    print(f"  Metrics: metrics_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze prompt-level planning rate vs pass rate with ROC curves."
    )
    parser.add_argument(
        "--parent-folder",
        default=DEFAULT_PARENT,
        help="Root plan_trace folder",
    )
    parser.add_argument(
        "--prompt-data",
        default=DEFAULT_PROMPT_DATA,
        help="Path to prompt dataset",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/planning_vs_pass_analysis",
        help="Output directory for plots and results",
    )
    parser.add_argument(
        "--use-random-steer",
        action="store_true",
        default=True,
        help="Prefer random steering results if available",
    )
    parser.add_argument(
        "--no-random-steer",
        action="store_false",
        dest="use_random_steer",
        help="Do not use random steering results",
    )
    parser.add_argument(
        "--verdict-key",
        default="new_new_verdict",
        help="Verdict key to use (default: new_new_verdict for random steer)",
    )

    args = parser.parse_args()

    analyze_planning_vs_pass(
        args.parent_folder,
        args.prompt_data,
        args.out_dir,
        use_random_steer=args.use_random_steer,
        verdict_key=args.verdict_key,
    )


if __name__ == "__main__":
    main()
