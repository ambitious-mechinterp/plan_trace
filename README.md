# Plan Trace: Planning Detection in Language Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive toolkit for detecting and analyzing planning behavior in language models using Sparse Autoencoder (SAE) circuit analysis.

## 🎯 Overview

Plan Trace implements a sophisticated 4-step pipeline to identify when and how language models engage in planning during text generation:

1. **🔍 Circuit Discovery**: Uses integrated gradients on SAE latents to find circuits responsible for specific predictions
2. **🎭 Logit Lens Clustering**: Groups SAE latents by their decoding directions to identify what tokens they predict
3. **🎛️ Steering Effects**: Tests causal impacts by steering latent activations and measuring generation changes  
4. **📊 Planning Analysis**: Identifies "planning positions" where future token predictions influence current outputs

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/plan_trace.git
cd plan_trace

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from plan_trace import run_full_pipeline

# Run the complete planning detection pipeline
result = run_full_pipeline(
    prompt_idx=15,          # Dataset example to analyze
    inter_token_id=297,     # Token position to investigate
    save_outputs=True,      # Save results to organized folders
    verbose=True            # Print progress information
)

# Check results
if result["status"] == "success":
    print(f"Found {len(result['circuit_entries'])} circuit entries")
    print(f"Identified {len(result['clusters'])} token clusters")
    print(f"Baseline text: {result['baseline_text'][:100]}...")
```

### Running as a Module

```bash
# Run the pipeline directly
python -m plan_trace.pipeline

# Or with custom parameters
python -c "
from plan_trace import run_full_pipeline
result = run_full_pipeline(prompt_idx=24, inter_token_id=180, verbose=True)
"
```

## 📁 Output Structure

When `save_outputs=True`, results are organized as:

```
outputs/
├── prompt_15/
│   └── token_297/
│       ├── circuit_entries.pt      # Circuit discovery results (PyTorch)
│       ├── clusters.json           # Logit lens clusters
│       ├── steering_results.json   # Steering sweep results
│       └── metadata.json          # Run metadata and statistics
└── prompt_24/
    └── token_180/
        ├── circuit_entries.pt
        ├── clusters.json
        ├── steering_results.json
        └── metadata.json
```

## 🛠️ Advanced Usage

### Circuit Analysis

```python
from plan_trace import CircuitAnalyzer, Config, load_model, load_pretrained_saes

# Load model and SAEs
model = load_model("gemma-2-2b-it", device="cuda")
saes = load_pretrained_saes(
    layers=list(range(26)), 
    release="gemma-scope-2b-pt-mlp-canonical"
)

# Create analyzer
cfg = Config(coeff=-200.0, device="cuda")
analyzer = CircuitAnalyzer(model, saes, cfg)

# Analyze specific prediction
result = analyzer.run(
    prompt_idx=24,
    token_pred_idx=180,
    keys=["def", "function"],  # Only analyze these predicted tokens
    thresh=0.1                 # 10% effect threshold
)

print(f"Circuit positions: {len(result['C'])}")
print(f"Planning positions: {len(result['F_prime'])}")
```

### Batch Analysis

```python
from plan_trace import analyze_batch

# Analyze multiple prediction positions
mapping = {
    180: ["def", "function"],   # At position 180, analyze these tokens
    185: ["return"],            # At position 185, analyze "return"
    190: [],                    # At position 190, skip evaluation
}

batch_result = analyze_batch(analyzer, prompt_idx=24, mapping=mapping, thresh=0.1)
print(f"Total planning positions: {len(batch_result['union_F_prime'])}")
```

### Custom Pipeline Components

```python
from plan_trace import (
    discover_circuit, 
    find_logit_lens_clusters, 
    run_steering_sweep
)

# Step-by-step pipeline
entries = discover_circuit(model, saes, tokens, device="cuda")
clusters = find_logit_lens_clusters(model, saes, entries, tokens, stop_tok=1917)
steering = run_steering_sweep(model, saes, tokens, clusters, baseline_text, coeff_grid=range(-100, 0, 20))
```

## 📊 Understanding Results

### Circuit Entries
Each circuit entry is a tuple: `(layer, token_position, latent_index, effect_value)`
- **layer**: Which transformer layer the SAE latent is in
- **token_position**: Which input token position this latent activates on
- **latent_index**: The specific SAE latent index
- **effect_value**: The magnitude of this latent's effect on the prediction

### Logit Lens Clusters
Clusters group SAE latents by what tokens they most strongly predict:
```json
{
  "def": [(layer, latent_idx, [token_positions])],
  "function": [(layer, latent_idx, [token_positions])],
  "return": [(layer, latent_idx, [token_positions])]
}
```

### Steering Results
For each cluster, shows how steering affects generation:
```json
{
  "def": {
    "base_text": "original generation...",
    "steered": [
      {"coeff": -80, "steered_text": "modified generation..."},
      {"coeff": -60, "steered_text": "different generation..."}
    ]
  }
}
```

## 🏗️ Project Structure

```
plan_trace/
├── __init__.py           # Package API and imports
├── utils.py              # Model loading, memory management
├── hooks.py              # SAE intervention hooks and masking
├── circuit_discovery.py  # Integrated gradients attribution
├── logit_lens.py         # Logit lens clustering
├── steering.py           # Steering interventions and sweeps
├── analysis.py           # CircuitAnalyzer and batch analysis
└── pipeline.py           # End-to-end pipeline orchestration
```

## ⚙️ Configuration

### Pipeline Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ig_steps` | 10 | Integration steps for integrated gradients |
| `k_max` | 90001 | Maximum circuit size to test |
| `k_step` | 10000 | Step size for circuit size sweep |
| `k_thres` | 0.6 | Performance threshold for circuit discovery |
| `coeff_grid` | `range(-100, 0, 20)` | Steering coefficients to test |
| `stop_token_id` | 1917 | Token ID to stop generation |

### Model Configuration

```python
# Default model loading
model = load_model(
    "gemma-2-2b-it", 
    device="cuda", 
    dtype=torch.bfloat16,
    use_custom_cache=False
)

# Default SAE loading  
saes = load_pretrained_saes(
    layers=list(range(26)),
    release="gemma-scope-2b-pt-mlp-canonical",
    width="16k",
    device="cuda"
)
```

## 🔬 Research Applications

This toolkit is designed for research into:

- **Planning Detection**: Identifying when models engage in multi-step reasoning
- **Circuit Analysis**: Understanding mechanistic pathways for specific behaviors  
- **Causal Intervention**: Testing causal relationships between representations and outputs
- **SAE Interpretability**: Analyzing sparse autoencoder latent functions

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Development Setup

```bash
# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/

# Check code style
black plan_trace/
flake8 plan_trace/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on [SAE Lens](https://github.com/jbloomAus/SAELens) for sparse autoencoder functionality
- Uses [TransformerLens](https://github.com/neelnanda-io/TransformerLens) for model interpretability
- Inspired by mechanistic interpretability research from Anthropic, Redwood Research, and others

## 📚 Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{plan_trace2024,
  title={Plan Trace: Planning Detection in Language Models},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/plan_trace}
}
```

---

For more examples and detailed documentation, see the [examples/](examples/) directory.
