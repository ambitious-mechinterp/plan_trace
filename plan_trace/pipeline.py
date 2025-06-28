"""
Complete pipeline demonstration for planning detection in language models.

This script shows how to use all the modular components together to:
1. Discover circuits using integrated gradients
2. Cluster latents by logit lens  
3. Test steering effects
4. Analyze planning positions

Shape Suffix Definition: 
- B: batch size 
- L: Num of Input Tokens 
- O: Num of Output Tokens
- V: vocabulary size
- S: Number of SAE neurons in a layer
"""

import json
import torch
import os
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from .utils import load_model, load_pretrained_saes, cleanup_cuda
from .circuit_discovery import discover_circuit
from .logit_lens import find_logit_lens_clusters  
from .steering import run_steering_sweep
from .analysis import CircuitAnalyzer, Config, analyze_batch


def save_pipeline_results(
    result: Dict[str, Any], 
    output_dir: str, 
    verbose: bool = True
) -> Path:
    """
    Save pipeline results to organized folder structure.
    
    Creates folder structure: output_dir/prompt_{idx}/token_{pos}/
    Saves:
    - circuit_entries.pt: Circuit discovery results
    - clusters.json: Logit lens clusters
    - steering_results.json: Steering sweep results  
    - metadata.json: Run metadata and configuration
    
    Args:
        result: Pipeline result dictionary
        output_dir: Base output directory
        verbose: Whether to print save locations
        
    Returns:
        Path to the created folder
    """
    # Create folder structure
    prompt_idx = result["prompt_idx"]
    inter_token_id = result["inter_token_id"]
    
    folder_path = Path(output_dir) / f"prompt_{prompt_idx}" / f"token_{inter_token_id}"
    folder_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().isoformat()
    
    # Save circuit entries as torch file
    if result["circuit_entries"] is not None:
        circuit_path = folder_path / "circuit_entries.pt"
        torch.save(result["circuit_entries"], circuit_path)
        if verbose:
            print(f"Saved circuit entries to: {circuit_path}")
    
    # Save clusters as JSON
    if result["clusters"] is not None:
        clusters_path = folder_path / "clusters.json"
        with open(clusters_path, 'w') as f:
            json.dump(result["clusters"], f, indent=2)
        if verbose:
            print(f"Saved clusters to: {clusters_path}")
    
    # Save steering results as JSON
    if result["steering_results"] is not None:
        steering_path = folder_path / "steering_results.json"
        with open(steering_path, 'w') as f:
            json.dump(result["steering_results"], f, indent=2)
        if verbose:
            print(f"Saved steering results to: {steering_path}")
    
    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "prompt_idx": prompt_idx,
        "inter_token_id": inter_token_id,
        "baseline_text": result["baseline_text"],
        "status": result["status"],
        "num_circuit_entries": len(result["circuit_entries"]) if result["circuit_entries"] else 0,
        "num_clusters": len(result["clusters"]) if result["clusters"] else 0,
        "cluster_labels": list(result["clusters"].keys()) if result["clusters"] else [],
    }
    
    metadata_path = folder_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    if verbose:
        print(f"Saved metadata to: {metadata_path}")
    
    return folder_path


def run_full_pipeline(
    prompt_idx: int,
    inter_token_id: int,
    model_name: str = "gemma-2-2b-it",
    device: str = "cuda",
    ig_steps: int = 10,
    k_max: int = 90001,
    k_step: int = 10000,
    k_thres: float = 0.6,
    coeff_grid: List[int] = None,
    stop_token_id: int = 1917,
    data_path: str = "data/first_100_passing_examples.json",
    save_outputs: bool = False,
    output_dir: str = "outputs",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the complete planning detection pipeline on a single example.
    
    Args:
        prompt_idx: Index into the dataset to analyze
        inter_token_id: Token position to analyze for planning
        model_name: HuggingFace model identifier
        device: Device for computation
        ig_steps: Number of integration steps for integrated gradients
        k_max: Maximum K to test in circuit discovery
        k_step: Step size for K sweep
        k_thres: Threshold for circuit discovery (fraction of baseline performance)
        coeff_grid: Steering coefficients to test
        stop_token_id: Token ID to stop generation at
        data_path: Path to dataset file
        save_outputs: Whether to save results to organized folders
        output_dir: Base directory for saving outputs
        verbose: Whether to print progress information
        
    Returns:
        Dict containing all pipeline results
    """
    if coeff_grid is None:
        coeff_grid = list(range(-100, 0, 20))
    
    if verbose:
        print(f"Starting pipeline for prompt {prompt_idx}, token position {inter_token_id}")
    
    # 1. Load model and SAEs
    if verbose:
        print("Loading model and SAEs...")
    model = load_model(model_name, device=device, use_custom_cache=False, dtype=torch.bfloat16)
    layers = list(range(model.cfg.n_layers))
    saes = load_pretrained_saes(
        layers=layers, 
        release="gemma-scope-2b-pt-mlp-canonical", 
        width="16k", 
        device=device, 
        canon=True
    )
    
    # 2. Load and process data
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    entry = data[prompt_idx]
    prompt = (
        "You are an expert Python programmer, and here is your task: "
        f"{entry['prompt']} Your code should pass these tests:\n\n"
        + "\n".join(entry["test_list"]) + "\nWrite your code below starting with \"```python\" and ending with \"```\".\n```python\n"
    )
    
    # Generate full sequence for analysis
    if verbose:
        print("Generating full sequence...")
    toks_BL = model.to_tokens(prompt).to(device)
    out_BL = toks_BL.clone()
    
    while out_BL.shape[-1] - toks_BL.shape[-1] < 150:
        with torch.no_grad():
            logits_V = model(out_BL)[0, -1]
        next_id = logits_V.argmax(-1).item()
        del logits_V
        cleanup_cuda()
        if next_id == stop_token_id:
            break
        out_BL = torch.cat([out_BL, torch.tensor([[next_id]], device=device)], dim=1)
    
    # Extract the specific prediction position
    inter_toks_BL = out_BL[:, :inter_token_id]
    baseline_suffix = model.to_string(out_BL[0, inter_token_id:])
    
    if verbose:
        print(f"Analyzing prediction at position {inter_token_id}")
        print(f"Baseline continuation: {baseline_suffix[:100]}...")
    
    # 3. Circuit Discovery
    if verbose:
        print("Discovering circuit...")
    entries = discover_circuit(
        model=model,
        saes=saes,
        inter_toks_BL=inter_toks_BL,
        device=device,
        ig_steps=ig_steps,
        k_max=k_max,
        k_step=k_step,
        k_thres=k_thres
    )
    
    if entries is None:
        if verbose:
            print("No circuit found meeting threshold criteria")
        return {
            "prompt_idx": prompt_idx,
            "inter_token_id": inter_token_id,
            "circuit_entries": None,
            "clusters": None,
            "steering_results": None,
            "status": "no_circuit_found"
        }
    
    if verbose:
        print(f"Found circuit with {len(entries)} entries")
    
    # 4. Logit Lens Clustering
    if verbose:
        print("Clustering latents by logit lens...")
    saved_pair_dict = find_logit_lens_clusters(
        model, saes, entries, inter_toks_BL, stop_token_id, verbose=verbose
    )
    
    if verbose:
        print(f"Found {len(saved_pair_dict)} clusters: {list(saved_pair_dict.keys())}")
    
    # 5. Steering Sweep
    if verbose:
        print("Running steering sweeps...")
    steering_results = run_steering_sweep(
        model=model,
        saes=saes,
        inter_toks_BL=inter_toks_BL,
        saved_pair_dict=saved_pair_dict,
        baseline_text=baseline_suffix,
        coeff_grid=coeff_grid,
        stop_tok=stop_token_id,
        max_tokens=100,
    )
    
    if verbose:
        print("Pipeline completed successfully!")
    
    result = {
        "prompt_idx": prompt_idx,
        "inter_token_id": inter_token_id,
        "circuit_entries": entries,
        "clusters": saved_pair_dict,
        "steering_results": steering_results,
        "baseline_text": baseline_suffix,
        "status": "success"
    }
    
    # Save outputs if requested
    if save_outputs:
        save_pipeline_results(result, output_dir, verbose=verbose)
    
    return result


def detect_docstring_tokens(
    model, 
    tokens: torch.Tensor, 
    start_search_idx: int = 0
) -> List[Tuple[int, int]]:
    """
    Detect docstring token ranges in the generated sequence.
    
    Looks for patterns like:
    - Triple quotes
    - Docstring-like content after function definitions
    
    Args:
        model: The language model (for token conversion)
        tokens: Token tensor [B, L]
        start_search_idx: Index to start searching from
        
    Returns:
        List of (start_idx, end_idx) tuples for docstring ranges
    """
    text = model.to_string(tokens[0])
    docstring_ranges = []
    
    # Look for triple quote patterns  
    triple_quotes = [chr(34)*3, chr(39)*3]  
    
    for quote in triple_quotes:
        start = 0
        while True:
            start_pos = text.find(quote, start)
            if start_pos == -1:
                break
            
            # Find matching closing quote
            end_pos = text.find(quote, start_pos + 3)
            if end_pos == -1:
                break
                
            # Convert text positions back to token positions (approximate)
            # This is a rough conversion - for exact mapping we'd need token offsets
            start_tokens = len(model.to_tokens(text[:start_pos])[0])
            end_tokens = len(model.to_tokens(text[:end_pos + 3])[0])
            
            if start_tokens >= start_search_idx:
                docstring_ranges.append((start_tokens, end_tokens))
            
            start = end_pos + 3
    
    return docstring_ranges


def is_token_in_docstring(
    token_idx: int, 
    docstring_ranges: List[Tuple[int, int]]
) -> bool:
    """
    Check if a token index falls within any docstring range.
    
    Args:
        token_idx: Token position to check
        docstring_ranges: List of (start, end) docstring ranges
        
    Returns:
        True if token is in a docstring
    """
    for start, end in docstring_ranges:
        if start <= token_idx <= end:
            return True
    return False


def analyze_planning_evidence(
    steering_results: Dict[str, Dict[str, Any]]
) -> Dict[str, str]:
    """
    Analyze steering results to detect planning evidence.
    
    Logic:
    - If steered text is non-empty and doesn't contain the predicted token -> "planning"
    - If all steered texts are empty -> "unsure"  
    - Otherwise -> "no_planning"
    
    Args:
        steering_results: Output from run_steering_sweep
        
    Returns:
        Dict mapping cluster_label -> planning_status
    """
    planning_analysis = {}
    
    for label, data in steering_results.items():
        steered_entries = data.get("steered", [])
        
        has_non_empty = False
        has_planning_evidence = False
        
        for entry in steered_entries:
            steered_text = entry.get("steered_text", "")
            
            if steered_text:  # Non-empty
                has_non_empty = True
                # Check if the predicted token (label) is NOT in the steered text
                if label.lower() not in steered_text.lower():
                    has_planning_evidence = True
                    break  # Found evidence, no need to check more
        
        if has_planning_evidence:
            planning_analysis[label] = "planning"
        elif not has_non_empty:
            planning_analysis[label] = "unsure"
        else:
            planning_analysis[label] = "no_planning"
    
    return planning_analysis


def run_automated_token_pipeline(
    prompt_idx: int,
    model_name: str = "gemma-2-2b-it",
    device: str = "cuda",
    skip_docstrings: bool = True,
    start_token_offset: int = 0,
    max_tokens_to_analyze: int = 50,
    ig_steps: int = 10,
    k_max: int = 90001,
    k_step: int = 10000,
    k_thres: float = 0.6,
    coeff_grid: List[int] = None,
    stop_token_id: int = 1917,
    data_path: str = "data/first_100_passing_examples.json",
    save_outputs: bool = False,
    output_dir: str = "outputs",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the pipeline automatically over multiple token positions.
    
    Args:
        prompt_idx: Index into the dataset to analyze
        model_name: HuggingFace model identifier
        device: Device for computation
        skip_docstrings: Whether to skip tokens inside docstrings
        start_token_offset: Token offset from prompt end to start analysis
        max_tokens_to_analyze: Maximum number of tokens to analyze
        ig_steps: Number of integration steps for integrated gradients
        k_max: Maximum K to test in circuit discovery
        k_step: Step size for K sweep
        k_thres: Threshold for circuit discovery
        coeff_grid: Steering coefficients to test
        stop_token_id: Token ID to stop generation at
        data_path: Path to dataset file
        save_outputs: Whether to save results
        output_dir: Base directory for saving outputs
        verbose: Whether to print progress
        
    Returns:
        Dict containing results for all analyzed positions
    """
    if coeff_grid is None:
        coeff_grid = list(range(-100, 0, 20))
    
    if verbose:
        print(f"Starting automated pipeline for prompt {prompt_idx}")
    
    # Load model and SAEs once
    model = load_model(model_name, device=device, use_custom_cache=False, dtype=torch.bfloat16)
    layers = list(range(model.cfg.n_layers))
    saes = load_pretrained_saes(
        layers=layers, 
        release="gemma-scope-2b-pt-mlp-canonical", 
        width="16k", 
        device=device, 
        canon=True
    )
    
    # Load and process data
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    entry = data[prompt_idx]
    prompt = (
        "You are an expert Python programmer, and here is your task: "
        f"{entry['prompt']} Your code should pass these tests:\n\n"
        + "\n".join(entry["test_list"]) + "\nWrite your code below starting with \"```python\" and ending with \"```\".\n```python\n"
    )
    
    # Generate full sequence
    if verbose:
        print("Generating full sequence...")
    toks_BL = model.to_tokens(prompt).to(device)
    out_BL = toks_BL.clone()
    
    while out_BL.shape[-1] - toks_BL.shape[-1] < 150:
        with torch.no_grad():
            logits_V = model(out_BL)[0, -1]
        next_id = logits_V.argmax(-1).item()
        del logits_V
        cleanup_cuda()
        if next_id == stop_token_id:
            break
        out_BL = torch.cat([out_BL, torch.tensor([[next_id]], device=device)], dim=1)
    
    # Detect docstrings and adjust start position if needed
    docstring_ranges = []
    docstring_end = 0
    if skip_docstrings:
        docstring_ranges = detect_docstring_tokens(model, out_BL, toks_BL.shape[-1])
        if docstring_ranges:
            # Find the end of the last docstring
            docstring_end = max(end for start, end in docstring_ranges)
            if verbose:
                print(f"Detected {len(docstring_ranges)} docstring ranges: {docstring_ranges}")
                print(f"Will start analysis after docstrings end at token {docstring_end}")
    
    # Determine token positions to analyze
    prompt_len = toks_BL.shape[-1]
    base_start = prompt_len + start_token_offset
    
    # Start analysis after docstrings if we're skipping them
    start_analysis = max(base_start, docstring_end + 1) if skip_docstrings and docstring_end > base_start else base_start
    end_analysis = min(out_BL.shape[-1] - 1, start_analysis + max_tokens_to_analyze)
    
    results = {
        "prompt_idx": prompt_idx,
        "prompt_length": prompt_len,
        "total_length": out_BL.shape[-1],
        "analyzed_range": (start_analysis, end_analysis),
        "docstring_ranges": docstring_ranges,
        "token_results": {},
        "planning_summary": {},
    }
    
    successful_analyses = 0
    
    for token_idx in range(start_analysis, end_analysis):
        if verbose:
            print(f"\n--- Analyzing token position {token_idx} ---")
        
        # Run single-token pipeline
        token_result = run_single_token_analysis(
            model=model,
            saes=saes,
            out_BL=out_BL,
            inter_token_id=token_idx,
            ig_steps=ig_steps,
            k_max=k_max,
            k_step=k_step,
            k_thres=k_thres,
            coeff_grid=coeff_grid,
            stop_token_id=stop_token_id,
            verbose=verbose
        )
        
        # Analyze planning evidence if successful
        if token_result["status"] == "success":
            planning_analysis = analyze_planning_evidence(token_result["steering_results"])
            token_result["planning_analysis"] = planning_analysis
            successful_analyses += 1
            
            if verbose:
                planning_tokens = [label for label, status in planning_analysis.items() if status == "planning"]
                if planning_tokens:
                    print(f"  Planning evidence found for: {planning_tokens}")
        
        results["token_results"][token_idx] = token_result
        
        # Save individual results if requested
        if save_outputs and token_result["status"] == "success":
            save_pipeline_results(token_result, output_dir, verbose=False)
    
    # Generate summary
    all_planning = {}
    for token_idx, token_result in results["token_results"].items():
        if token_result.get("planning_analysis"):
            for label, status in token_result["planning_analysis"].items():
                if status == "planning":
                    if label not in all_planning:
                        all_planning[label] = []
                    all_planning[label].append(token_idx)
    
    results["planning_summary"] = all_planning
    
    if verbose:
        print(f"\n=== SUMMARY ===")
        print(f"Analyzed {successful_analyses} positions successfully")
        print(f"Planning evidence found for {len(all_planning)} unique tokens:")
        for token, positions in all_planning.items():
            print(f"  '{token}': positions {positions}")
    
    return results


def run_single_token_analysis(
    model,
    saes,
    out_BL: torch.Tensor,
    inter_token_id: int,
    ig_steps: int = 10,
    k_max: int = 90001,
    k_step: int = 10000,
    k_thres: float = 0.6,
    coeff_grid: List[int] = None,
    stop_token_id: int = 1917,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run pipeline analysis for a single token position.
    
    This is the core analysis extracted from run_full_pipeline
    but without model loading and data processing.
    """
    if coeff_grid is None:
        coeff_grid = list(range(-100, 0, 20))
    
    # Extract the specific prediction position
    inter_toks_BL = out_BL[:, :inter_token_id]
    baseline_suffix = model.to_string(out_BL[0, inter_token_id:])
    
    if verbose:
        print(f"Baseline continuation: {baseline_suffix[:50]}...")
    
    # Circuit Discovery
    entries = discover_circuit(
        model=model,
        saes=saes,
        inter_toks_BL=inter_toks_BL,
        device=inter_toks_BL.device,
        ig_steps=ig_steps,
        k_max=k_max,
        k_step=k_step,
        k_thres=k_thres
    )
    
    if entries is None:
        return {
            "prompt_idx": -1,  # Will be filled by caller
            "inter_token_id": inter_token_id,
            "circuit_entries": None,
            "clusters": None,
            "steering_results": None,
            "baseline_text": baseline_suffix,
            "status": "no_circuit_found"
        }
    
    if verbose:
        print(f"Found circuit with {len(entries)} entries")
    
    # Logit Lens Clustering
    saved_pair_dict = find_logit_lens_clusters(
        model, saes, entries, inter_toks_BL, stop_token_id, verbose=verbose
    )
    
    if verbose:
        print(f"Found {len(saved_pair_dict)} clusters: {list(saved_pair_dict.keys())}")
    
    # Steering Sweep
    steering_results = run_steering_sweep(
        model=model,
        saes=saes,
        inter_toks_BL=inter_toks_BL,
        saved_pair_dict=saved_pair_dict,
        baseline_text=baseline_suffix,
        coeff_grid=coeff_grid,
        stop_tok=stop_token_id,
        max_tokens=100,
    )
    
    return {
        "prompt_idx": -1,  # Will be filled by caller
        "inter_token_id": inter_token_id,
        "circuit_entries": entries,
        "clusters": saved_pair_dict,
        "steering_results": steering_results,
        "baseline_text": baseline_suffix,
        "status": "success"
    }


def demo_circuit_analyzer():
    """
    Demonstrate usage of the CircuitAnalyzer class for batch analysis.
    """
    print("Demo: Circuit Analyzer")
    
    # Load model and SAEs
    model = load_model("gemma-2-2b-it", device="cuda", dtype=torch.bfloat16)
    layers = list(range(model.cfg.n_layers))
    saes = load_pretrained_saes(layers=layers, release="gemma-scope-2b-pt-mlp-canonical", width="16k", device="cuda")
    
    # Setup configuration
    cfg = Config(
        data_path="../data/first_100_passing_examples.json",
        hits_root="../models/mbpp_task2_2b_mlp",
        coeff=-200.0
    )
    
    # Create analyzer
    analyzer = CircuitAnalyzer(model, saes, cfg)
    
    # Analyze a single prediction
    result = analyzer.run(
        prompt_idx=24,
        token_pred_idx=180,
        keys=["2"],  # Analyze only latents that predict token "2"
        thresh=0.1   # 10% effect threshold
    )
    
    print(f"Circuit positions (C): {len(result['C'])}")
    print(f"Future planning positions (F_all): {len(result['F_all'])}")
    print(f"Significant planning positions (F_prime): {len(result['F_prime'])}")
    
    # Batch analysis across multiple predictions
    mapping = {
        180: ["2"],     # At position 180, analyze latents predicting "2"
        185: ["def"],   # At position 185, analyze latents predicting "def" 
        190: [],        # At position 190, skip evaluation
    }
    
    batch_result = analyze_batch(analyzer, prompt_idx=24, mapping=mapping, thresh=0.1)
    print(f"\nBatch analysis results:")
    print(f"Union circuit positions: {len(batch_result['union_C'])}")
    print(f"Union planning positions: {len(batch_result['union_F_prime'])}")


def main():
    """Command-line interface for the automated token pipeline."""
    parser = argparse.ArgumentParser(
        description="Run automated planning detection pipeline across multiple token positions"
    )
    
    # Required arguments
    parser.add_argument("prompt_idx", type=int, help="Index of prompt in dataset to analyze")
    
    # Model and device options
    parser.add_argument("--model", default="gemma-2-2b-it", help="HuggingFace model name")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    
    # Analysis range options
    parser.add_argument("--start-offset", type=int, default=0, 
                       help="Token offset from end of prompt to start analysis. If skipping docstrings, analysis will start after all docstrings if they extend beyond this offset (default: 0)")
    parser.add_argument("--max-tokens", type=int, default=50,
                       help="Maximum number of tokens to analyze (default: 50)")
    
    # Docstring handling
    parser.add_argument("--include-docstrings", action="store_true",
                       help="Include tokens inside docstrings (default: skip them)")
    
    # Circuit discovery parameters
    parser.add_argument("--ig-steps", type=int, default=10,
                       help="Integration steps for circuit discovery (default: 10)")
    parser.add_argument("--k-max", type=int, default=90001,
                       help="Maximum K for circuit discovery (default: 90001)")
    parser.add_argument("--k-step", type=int, default=10000,
                       help="K step size for circuit discovery (default: 10000)")
    parser.add_argument("--k-thres", type=float, default=0.6,
                       help="Performance threshold for circuit discovery (default: 0.6)")
    
    # Steering parameters
    parser.add_argument("--coeff-start", type=int, default=-100,
                       help="Starting steering coefficient (default: -100)")
    parser.add_argument("--coeff-end", type=int, default=0,
                       help="Ending steering coefficient (default: 0)")
    parser.add_argument("--coeff-step", type=int, default=20,
                       help="Steering coefficient step size (default: 20)")
    
    # Data and output options
    parser.add_argument("--data-path", default="data/first_100_passing_examples.json",
                       help="Path to dataset file")
    parser.add_argument("--output-dir", default="outputs",
                       help="Output directory for saving results")
    parser.add_argument("--save", action="store_true",
                       help="Save results to files")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress verbose output")
    
    args = parser.parse_args()
    
    # Build coefficient grid
    coeff_grid = list(range(args.coeff_start, args.coeff_end, args.coeff_step))
    
    print(f"Starting automated pipeline analysis:")
    print(f"  Prompt index: {args.prompt_idx}")
    print(f"  Model: {args.model}")
    print(f"  Requested start offset: {args.start_offset} tokens after prompt")
    print(f"  Max tokens to analyze: {args.max_tokens}")
    print(f"  Skip docstrings: {not args.include_docstrings}")
    print(f"  Steering coefficients: {coeff_grid}")
    print(f"  Save outputs: {args.save}")
    print()
    
    # Run the automated pipeline
    result = run_automated_token_pipeline(
        prompt_idx=args.prompt_idx,
        model_name=args.model,
        device=args.device,
        skip_docstrings=not args.include_docstrings,
        start_token_offset=args.start_offset,
        max_tokens_to_analyze=args.max_tokens,
        ig_steps=args.ig_steps,
        k_max=args.k_max,
        k_step=args.k_step,
        k_thres=args.k_thres,
        coeff_grid=coeff_grid,
        data_path=args.data_path,
        save_outputs=args.save,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Prompt {args.prompt_idx} analysis complete!")
    print(f"Requested range: tokens {result['prompt_length'] + args.start_offset} onwards")
    print(f"Actual analyzed range: tokens {result['analyzed_range'][0]} to {result['analyzed_range'][1]}")
    print(f"Total positions analyzed: {len([r for r in result['token_results'].values() if r['status'] == 'success'])}")
    
    if result['planning_summary']:
        print(f"\nPlanning evidence found:")
        for token, positions in result['planning_summary'].items():
            print(f"  Token '{token}': positions {positions}")
    else:
        print(f"\nNo planning evidence found in analyzed range.")
    
    if args.save:
        print(f"\nResults saved to: {args.output_dir}/")


if __name__ == "__main__":
    # Example usage - single token analysis
    # single_result = run_full_pipeline(
    #     prompt_idx=15,
    #     inter_token_id=297,
    #     save_outputs=True,
    #     output_dir="outputs",
    #     verbose=True
    # )
    
    # if single_result["status"] == "success":
    #     print(f"\nSingle Token Pipeline Results:")
    #     print(f"- Found {len(single_result['circuit_entries'])} circuit entries")
    #     print(f"- Identified {len(single_result['clusters'])} token clusters")
    #     print(f"- Tested steering on {len(single_result['steering_results'])} clusters")
    
    # print("\n" + "="*50)
    # print("Running automated token iteration pipeline...")
    # print("="*50)
    
    # # Example usage - automated token iteration
    # automated_result = run_automated_token_pipeline(
    #     prompt_idx=15,
    #     skip_docstrings=True,        # Skip docstring tokens
    #     start_token_offset=5,        # Start analysis 5 tokens after prompt
    #     max_tokens_to_analyze=20,    # Analyze up to 20 tokens
    #     save_outputs=True,
    #     output_dir="outputs",
    #     verbose=True
    # )
    
    # print(f"\nAutomated Pipeline Summary:")
    # print(f"- Analyzed range: {automated_result['analyzed_range']}")
    # print(f"- Total positions processed: {len(automated_result['token_results'])}")
    # print(f"- Planning evidence summary: {automated_result['planning_summary']}")
    
    # Uncomment to run analyzer demo
    # demo_circuit_analyzer()
    
    main() 