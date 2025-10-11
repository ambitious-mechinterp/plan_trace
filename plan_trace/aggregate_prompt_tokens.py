import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Set

import torch

from .utils import load_model, cleanup_cuda


def build_prompt_text(entry: Dict[str, object]) -> str:
    return (
        "You are an expert Python programmer, and here is your task: "
        f"{entry['prompt']} Your code should pass these tests:\n\n"
        + "\n".join(entry["test_list"]) + "\nWrite your code below starting with \"```python\" and ending with \"```\".\n```python\n"
    )


def find_prompt_token_targets(outputs_root: str) -> Dict[int, Set[int]]:
    mapping: Dict[int, Set[int]] = {}
    for root, dirs, files in os.walk(outputs_root):
        if "metadata.json" in files:
            p = Path(root)
            try:
                prompt_dir = p.parent.name  # prompt_{idx}
                token_dir = p.name          # token_{id}
                if not prompt_dir.startswith("prompt_"):
                    continue
                if not token_dir.startswith("token_"):
                    continue
                prompt_idx = int(prompt_dir.split("_")[-1])
                token_idx = int(token_dir.split("_")[-1])
            except Exception:
                continue
            mapping.setdefault(prompt_idx, set()).add(token_idx)
    return mapping


def tokenize_text_to_str_tokens(model, text: str) -> List[str]:
    return model.to_str_tokens(text)


def generate_full_sequence(model, prompt_text: str, device: str, stop_token_id: int, max_new_tokens: int) -> torch.Tensor:
    toks_BL = model.to_tokens(prompt_text).to(device)
    out_BL = toks_BL.clone()
    while out_BL.shape[-1] - toks_BL.shape[-1] < max_new_tokens:
        with torch.no_grad():
            logits_V = model(out_BL)[0, -1]
        next_id = logits_V.argmax(-1).item()
        del logits_V
        cleanup_cuda()
        if next_id == stop_token_id:
            break
        out_BL = torch.cat([out_BL, torch.tensor([[next_id]], device=device)], dim=1)
    return out_BL


def main():
    parser = argparse.ArgumentParser(description="Aggregate prompt-wise tokenized strings and baseline generation")
    parser.add_argument("--outputs-dir", default="outputs", help="Root outputs directory containing prompt_*/token_*/metadata.json")
    parser.add_argument("--data-path", default="data/first_100_passing_examples.json", help="Dataset path used to build prompts")
    parser.add_argument("--model", default="gemma-2-2b-it", help="Model name")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--stop-token-id", type=int, default=1917, help="Stop token id to halt generation")
    parser.add_argument("--max-new-tokens", type=int, default=150, help="Max new tokens to generate after prompt")
    parser.add_argument("--out-file", default="outputs/prompt_tokenized_map.json", help="Path to write aggregated JSON")
    args = parser.parse_args()

    # Discover prompts and token targets
    prompt_to_tokens = find_prompt_token_targets(args.outputs_dir)
    if not prompt_to_tokens:
        print(f"No metadata found under '{args.outputs_dir}'. Nothing to aggregate.")
        return

    # Load dataset
    with open(args.data_path, "r") as f:
        data = json.load(f)

    # Load model
    model = load_model(args.model, device=args.device, use_custom_cache=True, dtype=torch.bfloat16)

    aggregated: Dict[str, object] = {}

    for prompt_idx in sorted(prompt_to_tokens.keys()):
        entry = data[prompt_idx]
        prompt_text = build_prompt_text(entry)

        # Generate once per prompt
        out_BL = generate_full_sequence(
            model=model,
            prompt_text=prompt_text,
            device=args.device,
            stop_token_id=args.stop_token_id,
            max_new_tokens=args.max_new_tokens,
        )

        prompt_len = model.to_tokens(prompt_text).shape[-1]
        full_text = model.to_string(out_BL[0])
        prompt_text_str = prompt_text
        full_token_strings = tokenize_text_to_str_tokens(model, full_text)
        prompt_token_strings = tokenize_text_to_str_tokens(model, prompt_text_str)

        token_results: Dict[str, object] = {}
        for token_idx in sorted(prompt_to_tokens[prompt_idx]):
            if token_idx >= out_BL.shape[-1]:
                # Skip if token index beyond generated sequence
                continue
            input_prefix_text = model.to_string(out_BL[0, :token_idx])
            baseline_text = model.to_string(out_BL[0, token_idx:])
            input_prefix_token_strings = tokenize_text_to_str_tokens(model, input_prefix_text)
            baseline_token_strings = tokenize_text_to_str_tokens(model, baseline_text)

            token_results[str(token_idx)] = {
                "input_prefix_token_strings": input_prefix_token_strings,
                "baseline_token_strings": baseline_token_strings,
                "baseline_text": baseline_text,
            }

        aggregated[str(prompt_idx)] = {
            "prompt_text": prompt_text,
            "prompt_token_strings": prompt_token_strings,
            "full_out_token_strings": full_token_strings,
            "token_results": token_results,
        }

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"Wrote aggregated tokenization to: {out_path}")


if __name__ == "__main__":
    main()


