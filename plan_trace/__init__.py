"""
Plan Trace: A toolkit for detecting planning in language models using SAE circuit analysis.

This package provides a complete pipeline for:
1. Circuit discovery using integrated gradients on SAE latents
2. Clustering latents by their logit lens decoding directions  
3. Testing steering effects to measure causal impacts
4. Analyzing planning positions across different prediction contexts

Main Components:
- utils: Model loading, memory management, SAE utilities
- hooks: SAE intervention hooks and masking
- circuit_discovery: Integrated gradients attribution and circuit discovery
- logit_lens: Logit lens clustering for grouping latents
- steering: Steering interventions and generation sweeps
- analysis: Circuit analysis tools and batch processing
- pipeline: End-to-end pipeline demonstrations
"""

# Core utilities
from .utils import (
    load_model,
    load_pretrained_saes, 
    cleanup_cuda,
    clear_memory,
    get_pretrained_saes_ids
)

# SAE hooks and masking
from .hooks import (
    SAEMasks,
    build_sae_hook_fn,
    register_sae_hooks,
    run_with_saes
)

# Circuit discovery
from .circuit_discovery import (
    discover_circuit,
    run_integrated_gradients,
    get_saes_cache,
    iter_topk_effects,
    compute_k_metrics,
    find_min_k_for_threshold
)

# Edge attribution functions
from .edge_attr import (
    iter_topk_negative_effects,
    iter_topk_absolute_effects,
    discover_circuit_edge_attr
)

# Logit lens clustering
from .logit_lens import (
    find_logit_lens_clusters,
    gather_unique_tokens,
    build_saved_pair_dict_fastest
)

# Steering interventions
from .steering import (
    steering_hook,
    steering_effect_on_next_token,
    generate_once,
    sweep_coefficients_multi,
    run_steering_sweep
)

# Analysis tools
from .analysis import (
    CircuitAnalyzer,
    Config,
    analyze_batch
)

# Pipeline
from .pipeline import (
    run_full_pipeline,
    save_pipeline_results,
    demo_circuit_analyzer
)

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = [
    # Utils
    "load_model",
    "load_pretrained_saes", 
    "cleanup_cuda",
    "clear_memory",
    "get_pretrained_saes_ids",
    
    # Hooks
    "SAEMasks",
    "build_sae_hook_fn", 
    "register_sae_hooks",
    "run_with_saes",
    
    # Circuit Discovery
    "discover_circuit",
    "run_integrated_gradients",
    "get_saes_cache",
    "iter_topk_effects",
    "compute_k_metrics",
    "find_min_k_for_threshold",
    
    # Edge Attribution
    "iter_topk_negative_effects",
    "iter_topk_absolute_effects", 
    "discover_circuit_edge_attr",
    
    # Logit Lens
    "find_logit_lens_clusters",
    "gather_unique_tokens",
    "build_saved_pair_dict_fastest",
    
    # Steering
    "steering_hook",
    "steering_effect_on_next_token", 
    "generate_once",
    "sweep_coefficients_multi",
    "run_steering_sweep",
    
    # Analysis
    "CircuitAnalyzer",
    "Config",
    "analyze_batch",
    
    # Pipeline
    "run_full_pipeline",
    "save_pipeline_results",
    "demo_circuit_analyzer",
] 