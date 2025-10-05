"""Quick latent-token selectivity check using Neuronpedia top activations."""

import argparse
import json
import math
import sys
from typing import Any, Dict, List, Sequence, Tuple

import requests
import torch
from sae_lens import HookedSAETransformer

sys.path.append("../")

from plan_trace.utils import cleanup_cuda, load_model

API_URL = "https://www.neuronpedia.org/api/activation/get"
DEFAULT_MODEL = "gemma-2-2b"
DEFAULT_RELEASE_FORMAT = "{layer}-gemmascope-mlp-16k"
DEFAULT_TOP_K = 50


def tokens_to_text(tokens: Sequence[str], strip: bool = False) -> str:
    text = "".join(token.replace("▁", " ").replace("<0x0A>", "\n") for token in tokens)
    return text.strip() if strip else text


def get_neuronpedia_info(
    latent_index: int,
    layer_index: int,
    *,
    model_name: str,
    release_format: str = DEFAULT_RELEASE_FORMAT,
    url: str = API_URL,
    timeout: float = 30.0,
) -> Any:
    source = release_format.format(layer=layer_index)
    payload = {"modelId": model_name, "source": source, "index": str(latent_index)}
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to fetch Neuronpedia info: {exc}") from exc

    if isinstance(data, dict):
        status = str(data.get("status", "")).lower()
        if status and status not in {"ok", "success"}:
            message = data.get("message") or data
            raise RuntimeError(
                f"Neuronpedia returned status '{status}' for payload {payload}: {message}"
            )
    return data


def normalize_contexts(response: Any) -> List[Dict[str, Any]]:
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ValueError("Neuronpedia response was a string that could not be parsed as JSON.") from exc

    if isinstance(response, list):
        return [item for item in response if isinstance(item, dict)]

    if isinstance(response, dict):
        for key in ("contexts", "data", "records"):
            candidates = response.get(key)
            if isinstance(candidates, list):
                return [item for item in candidates if isinstance(item, dict)]

    raise ValueError("Unexpected Neuronpedia response format; expected list of context dicts.")


def extract_top_contexts(raw_contexts: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    contexts: List[Dict[str, Any]] = []
    for entry in raw_contexts:
        tokens = entry.get("tokens")
        if not isinstance(tokens, list) or not tokens:
            continue

        joined_text = tokens_to_text(tokens, strip=False)
        max_value = entry.get("maxValue")
        if max_value is None:
            values = entry.get("values")
            if isinstance(values, list) and values:
                try:
                    max_value = max(values)
                except TypeError:
                    max_value = None

        try:
            numeric_max = float(max_value)
        except (TypeError, ValueError):
            numeric_max = float("-inf")

        display_text = " ".join(joined_text.split())
        contexts.append(
            {
                "entry": entry,
                "joined_text": joined_text,
                "display_text": display_text,
                "max_value": numeric_max,
            }
        )

    contexts.sort(key=lambda item: item["max_value"], reverse=True)
    if top_k > 0:
        contexts = contexts[:top_k]
    return contexts


def resolve_token(
    model: HookedSAETransformer,
    *,
    token_text: str | None = None,
    token_id: int | None = None,
) -> Tuple[int, str]:
    if (token_text is None) == (token_id is None):
        raise ValueError("Provide exactly one of --token or --token-id.")

    if token_text is not None:
        token_tensor = model.to_tokens(token_text)
        if token_tensor.numel() == 0:
            raise ValueError("Token text produced no tokens when tokenized.")
        token_id = int(token_tensor[0, -1].item())

    assert token_id is not None
    decoded = model.tokenizer.decode([int(token_id)])
    if decoded == "":
        raise ValueError(f"Decoded token string for id {token_id} is empty.")
    return int(token_id), decoded


def parse_args() -> argparse.Namespace:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(
        description="Cheap heuristic check for whether a latent is selective for a target token."
    )
    parser.add_argument("--layer", type=int, required=True, help="Latent layer index.")
    parser.add_argument("--latent", type=int, required=True, help="Latent index to inspect.")
    parser.add_argument(
        "--token",
        type=str,
        help="Text whose final token will be matched against the top activation contexts.",
    )
    parser.add_argument(
        "--token-id",
        type=int,
        help="Explicit token id to match; provide instead of --token.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of top activation contexts to inspect (default: {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name to load for tokenization (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=default_device,
        help=f"Device for model loading (default: {default_device}).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=DEFAULT_RELEASE_FORMAT,
        help=(
            "Format string for the Neuronpedia source. The literal '{layer}' will be replaced with the "
            "requested layer index."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.topk <= 0:
        raise ValueError("--topk must be positive.")

    model = load_model(
        args.model,
        device=args.device,
        use_custom_cache=True,
        dtype=torch.bfloat16,
    )

    try:
        token_id, token_string = resolve_token(
            model, token_text=args.token, token_id=args.token_id
        )

        response = get_neuronpedia_info(
            latent_index=args.latent,
            layer_index=args.layer,
            model_name=args.model,
            release_format=args.source,
        )
        raw_contexts = normalize_contexts(response)
        top_contexts = extract_top_contexts(raw_contexts, args.topk)

        if not top_contexts:
            print("No activation contexts returned for the requested latent.")
            return

        match_records: List[Tuple[int, Dict[str, Any]]] = []
        for rank, ctx in enumerate(top_contexts, start=1):
            if token_string in ctx["joined_text"]:
                match_records.append((rank, ctx))

        print(
            f"Checked top {len(top_contexts)} contexts for latent {args.latent} "
            f"at layer {args.layer}."
        )
        print(f"Target token id {token_id}: {token_string!r}")
        print(f"Token appears in {len(match_records)} of the top contexts.")

        if match_records:
            print("\nSequences containing the token:")
            for rank, ctx in match_records:
                max_val = ctx["max_value"]
                max_str = f"{max_val:.2f}" if math.isfinite(max_val) else "n/a"
                highlighted = ctx["joined_text"].replace(token_string, f"[{token_string}]")
                pretty = " ".join(highlighted.replace("\r", " ").replace("\n", " ").split())
                print(f"{rank:>3}. [max={max_str}] {pretty}")
        else:
            print("Token not found in the inspected contexts.")
    finally:
        if "model" in locals():
            del model
        if torch.cuda.is_available():
            cleanup_cuda()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - surface useful error message to CLI
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
