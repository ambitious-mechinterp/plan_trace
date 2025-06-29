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
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

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


if __name__ == "__main__":
    # Example usage
    result = run_full_pipeline(
        prompt_idx=15,
        inter_token_id=297,
        save_outputs=True,  # Enable output saving
        output_dir="outputs",  # Save to outputs/ directory
        verbose=True
    )
    
    if result["status"] == "success":
        print(f"\nPipeline Results:")
        print(f"- Found {len(result['circuit_entries'])} circuit entries")
        print(f"- Identified {len(result['clusters'])} token clusters")
        print(f"- Tested steering on {len(result['steering_results'])} clusters")
        
        # Show some steering results
        for label, data in list(result['steering_results'].items())[:3]:
            print(f"\nCluster '{label}':")
            successful_steers = [s for s in data['steered'] if s['steered_text']]
            print(f"  - {len(successful_steers)} successful steering interventions")
            if successful_steers:
                print(f"  - Example steered text: {successful_steers[0]['steered_text'][:50]}...")
    else:
        print("Pipeline failed - no circuit found")
    
    # Uncomment to run analyzer demo
    # demo_circuit_analyzer() 