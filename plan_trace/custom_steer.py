"""
    Motivation of the file:
    - Use `steering.py` and a JSON file to custom steer a model and then, record the output.

    `json` file contains:
        - the list of latents hand picked.

    args:
        - cap: the max token position upto which intervention would be possible. 
        - latent_file: the file which contains the list of latents.
        - output_dir: the output directory of where to store the latents.
        - model: model name to use for steering.
        - data_file: path to the data file containing prompts.
        - prompt_idx: index of the prompt to use from the data file.
"""

import argparse
import json
import os
import torch
from typing import List, Dict, Any, Optional, Tuple

def parse_args():
    parser = argparse.ArgumentParser(description="Custom steer a model using latents from a JSON file.")
    parser.add_argument("--cap", type=int, required=True, help="Max token position for intervention.")
    parser.add_argument("--latent-file", type=str, required=True, help="Path to the JSON file with latents.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to store outputs.")
    parser.add_argument("--model", type=str, required=True, help="Model name to use for steering.")
    parser.add_argument("--data-file", type=str, required=True, help="Path to the data file containing prompts.")
    parser.add_argument("--prompt-idx", type=int, required=True, help="Index of the prompt to use from the data file.")
    parser.add_argument("--mode", type=str, required=False, help="Mode indicating whether it is necessary to use the docstring")
    return parser.parse_args()

def filter_latents_by_cap(latents, cap):
    """
    Filters latents for all keys in the input dict.
    For each entry in latents[key], only latent indices <= cap are kept.
    Entries with no remaining latents are removed.
    Adds an 'all' key containing all filtered entries across all other keys.
    """
    filtered = {}
    ultimate_key_entries = []

    for key, entries in latents.items():
        new_entries = []
        for entry in entries:
            layer, token_pos, latent_list = entry
            filtered_latents = [x for x in latent_list if x <= cap]
            if filtered_latents:
                new_entry = [layer, token_pos, filtered_latents]
                new_entries.append(new_entry)
                ultimate_key_entries.append(new_entry)
        if new_entries:
            filtered[key] = new_entries

    # Combine all filtered entries into one "all" key
    filtered["all"] = ultimate_key_entries
    return filtered


# ...existing code...

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.latent_file, "r") as f:
        latents = json.load(f)

    filtered_latents = filter_latents_by_cap(latents, args.cap)

    output_path = os.path.join(args.output_dir, "filtered_latents.json")
    with open(output_path, "w") as f:
        json.dump(filtered_latents, f, indent=2)

    # Hardcoded parameters for steering sweep
    device = "cuda"
    stop_tok = 1917
    max_tokens = 100
    coeff_grid = list(range(100, -1, -20))

    # Load model and data (example, assumes correct imports)
    from plan_trace.utils import load_model, load_pretrained_saes
    from plan_trace.steering import run_steering_sweep

    model = load_model(args.model, device=device, use_custom_cache=False, dtype=torch.bfloat16)
    layers = list(range(model.cfg.n_layers))
    saes = load_pretrained_saes(
        layers=layers,
        release="gemma-scope-2b-pt-mlp-canonical",
        width="16k",
        device=device,
        canon=True
    )

    with open(args.data_file, "r") as f:
        data = json.load(f)
    entry = data[args.prompt_idx]

    if args.mode == "nodocstring":
        prompt = (
            "You are an expert Python programmer, and here is your task: "
            f"{entry["prompt"]} Your code should pass these tests:\n\n"
            + "\n".join(entry["test_list"]) + "\nWrite your code, without docstrings, below starting with \"```python\" and ending with \"```\".\n```python\n"
        )
    else:
        prompt = (
            "You are an expert Python programmer, and here is your task: "
            f"{entry["prompt"]} Your code should pass these tests:\n\n"
            + "\n".join(entry["test_list"]) + "\nWrite your code below starting with \"```python\" and ending with \"```\".\n```python\n"
        )

    toks_BL = model.to_tokens(prompt).to(device)
    inter_toks_BL = toks_BL  # You may want to slice this for your use case

    baseline_text = model.to_string(toks_BL[0, :])

    # Prepare saved_pair_dict from filtered_latents
    saved_pair_dict = filtered_latents  # This should match the expected format

    results = run_steering_sweep(
        model=model,
        saes=saes,
        inter_toks_BL=inter_toks_BL,
        saved_pair_dict=saved_pair_dict,
        baseline_text=baseline_text,
        coeff_grid=coeff_grid,
        stop_tok=stop_tok,
        max_tokens=max_tokens,
        return_tokens=True,
        skip_change=True
    )

    def _json_safe_steering(steering_results: Dict[str, Any]) -> Dict[str, Any]:
        safe: Dict[str, Any] = {}
        for label, data in steering_results.items():
            steered_list = []
            for entry in data.get("steered", []):
                coeff = entry.get("coeff")
                val = entry.get("steered_text")
                if hasattr(val, "tolist"):
                    tokens_list = val.tolist()
                    decoded_text = None
                    if model is not None:
                        try:
                            # model.to_string accepts list[int] or tensor
                            decoded_text = model.to_string(tokens_list)
                        except Exception:
                            decoded_text = None
                    steered_list.append({
                        "coeff": coeff,
                        "is_tokens": True,
                        "steered_text": tokens_list,
                        "decoded_text": decoded_text,
                    })
                else:
                    steered_list.append({
                        "coeff": coeff,
                        "is_tokens": False,
                        "steered_text": val,
                    })
            safe[label] = {
                "base_text": data.get("base_text", ""),
                "steered": steered_list,
            }
        return safe

    # Save results
    results_path = os.path.join(args.output_dir, "steering_results.json")
    with open(results_path, "w") as f:
        json.dump(_json_safe_steering(results), f, indent=2)

# ...existing code...
if __name__ == "__main__":
    main()