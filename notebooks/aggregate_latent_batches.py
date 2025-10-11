#!/usr/bin/env python3
"""Aggregate activation batch records into a single JSON indexed by layer, then latent.

Scans an activations root directory with per-layer subdirectories like:
  <act_base>/L{layer}/raw/batch-*.jsonl.gz

For every record in each JSONL.GZ batch file, groups by (layer, record["index"]).
Writes a single JSON file with structure:

{
  "<layer>": {
    "<latent>": [ { ... original record ... }, ... ]
  },
  ...
}

Notes:
- Keys are strings to ensure JSON compatibility.
- For large corpora use streaming modes to avoid high memory usage.
"""

import argparse
import gzip
import json
import os
import re
import sys
from typing import Dict, List, Any, Tuple, Optional, Set
import gc


def discover_layers(act_base: str) -> List[int]:
    """Return sorted list of layer numbers by scanning for L{n} directories."""
    layers: List[int] = []
    if not os.path.isdir(act_base):
        raise FileNotFoundError(f"Activations base not found: {act_base}")
    for name in os.listdir(act_base):
        m = re.fullmatch(r"L(\d+)", name)
        if m and os.path.isdir(os.path.join(act_base, name)):
            layers.append(int(m.group(1)))
    layers.sort()
    return layers


def iter_batch_files_for_layer(act_base: str, layer: int) -> List[str]:
    """Return sorted list of batch file paths for a layer."""
    raw_dir = os.path.join(act_base, f"L{layer}", "raw")
    if not os.path.isdir(raw_dir):
        return []
    batch_files = [
        os.path.join(raw_dir, f)
        for f in os.listdir(raw_dir)
        if re.fullmatch(r"batch-\d+\.jsonl\.gz", f)
    ]
    batch_files.sort(key=lambda p: int(re.search(r"batch-(\d+)\.jsonl\.gz$", p).group(1)))
    return batch_files


def aggregate_records_for_layer(
    act_base: str,
    layer: int,
    filter_latents: Optional[Set[str]] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], int, int]:
    """Aggregate records for a single layer, grouped by latent.

    Returns (layer_map, files_read, records_aggregated).
    """
    layer_map: Dict[str, List[Dict[str, Any]]] = {}
    files_read = 0
    records_aggregated = 0
    batch_files = iter_batch_files_for_layer(act_base, layer)
    print(f"Layer {layer}: {len(batch_files)} batch files")
    for path in batch_files:
        files_read += 1
        try:
            with gzip.open(path, "rt") as fh:
                for line in fh:
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue
                    latent = record.get("index")
                    if latent is None:
                        continue
                    latent_key = str(latent)
                    if filter_latents is not None and latent_key not in filter_latents:
                        continue
                    # Only keep the first top_k records per latent (default 20)
                    bucket = layer_map.setdefault(latent_key, [])
                    # We assume input already sorted by max activation; take first N seen
                    if len(bucket) < TOP_K_PER_LATENT:
                        bucket.append(record)
                    records_aggregated += 1
        except Exception as e:
            print(f"Warning: failed to read {path}: {e}")
    print(
        f"  Aggregated layer {layer}: {sum(len(v) for v in layer_map.values())} records across {len(layer_map)} latents"
    )
    return layer_map, files_read, records_aggregated


def load_latent_filter(path: Optional[str]) -> Optional[Set[str]]:
    """Load a set of latent ids (as strings) from a JSON file, if provided.

    Accepts either:
    - list/array of latent ids, or
    - dict mapping any keys to arrays of entries where each entry is (layer, latent, ...)
    """
    if not path:
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Warning: failed to load latent filter from {path}: {e}")
        return None
    latents: Set[str] = set()
    if isinstance(data, list):
        for item in data:
            latents.add(str(item))
    elif isinstance(data, dict):
        # Expect dict[str, list[tuple|list]] where entry[1] is latent id
        for _k, entries in data.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                try:
                    latent = entry[1]
                    latents.add(str(latent))
                except Exception:
                    continue
    else:
        print("Warning: unsupported latent filter JSON format; ignoring")
        return None
    print(f"Loaded latent filter with {len(latents)} entries")
    return latents


def write_streaming_single_json(
    act_base: str,
    layers: List[int],
    output_path: str,
    indent: int,
    gzip_output: bool,
    filter_latents: Optional[Set[str]] = None,
) -> None:
    """Stream a single JSON file, writing one layer at a time to limit memory."""
    open_fn = gzip.open if gzip_output else open
    with open_fn(output_path, "wt") as out:
        pretty = indent > 0
        if pretty:
            out.write("{\n")
        else:
            out.write("{")
        first_layer = True
        total_files = 0
        total_records = 0
        for layer in layers:
            layer_map, files_read, records_aggregated = aggregate_records_for_layer(
                act_base, layer, filter_latents
            )
            total_files += files_read
            total_records += records_aggregated
            if not first_layer:
                out.write("," + ("\n" if pretty else ""))
            else:
                first_layer = False
            layer_key = json.dumps(str(layer))
            layer_json = json.dumps(layer_map, indent=indent)
            if pretty:
                out.write(" " * indent + f"{layer_key}: {layer_json}")
            else:
                out.write(f"{layer_key}:{layer_json}")
            # release memory for this layer before next
            del layer_map
            gc.collect()
        if pretty:
            out.write("\n}")
        else:
            out.write("}")
    print(f"Done. Files read: {total_files}, records aggregated: {total_records}")


def write_per_layer_files(
    act_base: str,
    layers: List[int],
    output_dir: str,
    indent: int,
    gzip_output: bool,
    filter_latents: Optional[Set[str]] = None,
) -> None:
    """Write one JSON per layer. Great for very large datasets."""
    os.makedirs(output_dir, exist_ok=True)
    total_files = 0
    total_records = 0
    for layer in layers:
        layer_map, files_read, records_aggregated = aggregate_records_for_layer(
            act_base, layer, filter_latents
        )
        total_files += files_read
        total_records += records_aggregated
        suffix = ".json.gz" if gzip_output else ".json"
        out_path = os.path.join(output_dir, f"L{layer}.top{TOP_K_PER_LATENT}{suffix}")
        open_fn = gzip.open if gzip_output else open
        with open_fn(out_path, "wt") as f:
            json.dump(layer_map, f, indent=indent)
        print(f"Wrote layer {layer} to {out_path}")
        del layer_map
        gc.collect()
    print(f"Done. Files read: {total_files}, records aggregated: {total_records}")


def parse_layers_arg(layers_arg: str, act_base: str) -> List[int]:
    if layers_arg.lower() == "all":
        return discover_layers(act_base)
    parts = [p.strip() for p in layers_arg.split(",") if p.strip()]
    layers: List[int] = []
    for p in parts:
        if re.fullmatch(r"\d+", p):
            layers.append(int(p))
        elif re.fullmatch(r"\d+-\d+", p):
            a, b = map(int, p.split("-"))
            if a <= b:
                layers.extend(list(range(a, b + 1)))
            else:
                layers.extend(list(range(b, a + 1)))
        else:
            raise ValueError(f"Invalid layers spec: {p}")
    layers = sorted(set(layers))
    return layers


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Aggregate per-layer latent batch records with memory-friendly streaming options.")
    parser.add_argument("--act-base", required=True, help="Root path containing L{layer}/raw/batch-*.jsonl.gz")
    parser.add_argument("--layers", default="all", help="Layer selection: 'all', '0,2,5', or ranges '3-7'")
    parser.add_argument("--output", required=True, help="Path to write aggregated JSON output (used by single_json mode)")
    parser.add_argument("--output-dir", default="", help="Directory for per_layer mode outputs (defaults to dirname of --output)")
    parser.add_argument("--output-mode", choices=["single_json", "per_layer"], default="single_json", help="Write a single JSON (streamed) or one file per layer")
    parser.add_argument("--gzip-output", action="store_true", help="Gzip the output file(s)")
    parser.add_argument("--indent", type=int, default=0, help="JSON indent (0 for compact)")
    parser.add_argument("--filter-latents", default="", help="Optional path to JSON specifying latent ids to include")
    parser.add_argument("--top-k", type=int, default=20, help="Keep only first K records per latent (assumes pre-sorted)")
    args = parser.parse_args(argv)

    act_base = args.act_base
    layers = parse_layers_arg(args.layers, act_base)
    if not layers:
        print("No layers found to process.")
        return 1
    print(f"Activations base: {act_base}")
    print(f"Layers: {layers}")
    latent_filter = load_latent_filter(args.filter_latents) if args.filter_latents else None
    global TOP_K_PER_LATENT
    TOP_K_PER_LATENT = max(1, int(args.top_k))

    if args.output_mode == "single_json":
        out_dir = os.path.dirname(os.path.abspath(args.output))
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        write_streaming_single_json(
            act_base=act_base,
            layers=layers,
            output_path=args.output,
            indent=args.indent,
            gzip_output=args.gzip_output,
            filter_latents=latent_filter,
        )
        print(f"Wrote aggregated JSON to: {args.output}")
    else:
        out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.output)) or "."
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        write_per_layer_files(
            act_base=act_base,
            layers=layers,
            output_dir=out_dir,
            indent=args.indent,
            gzip_output=args.gzip_output,
            filter_latents=latent_filter,
        )
        print(f"Wrote per-layer files under: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


