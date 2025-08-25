## Plan Trace

Planning detection and analysis toolkit for language models using Sparse Autoencoder (SAE) circuits.

### Features
- Circuit discovery via integrated gradients on SAE latents
- Logit-lens-based clustering of latents
- Causal steering sweeps over latent clusters
- Automated multi-token analysis with optional docstring skipping

### Install
```bash
pip install -r requirements.txt
```

### Minimal usage
```bash
python -m plan_trace.pipeline --help
```

#### Run automated token analysis (recommended)
```bash
python -m plan_trace.pipeline \
  15 \
  --model gemma-2-2b-it \
  --device cuda \
  --start-offset 0 \
  --max-tokens 50 \
  --ig-steps 10 \
  --k-max 90001 \
  --k-step 10000 \
  --k-thres 0.6 \
  --coeff-start -100 \
  --coeff-end 0 \
  --coeff-step 20 \
  --data-path data/first_100_passing_examples.json \
  --output-dir outputs \
  --save
```

#### Programmatic single-token run
```python
from plan_trace.pipeline import run_full_pipeline

result = run_full_pipeline(
    prompt_idx=15,
    inter_token_id=297,
    model_name="gemma-2-2b-it",
    device="cuda",
    save_outputs=True,
    output_dir="outputs",
    verbose=True,
)
```

### CLI arguments
- prompt_idx (positional): dataset index to analyze
- --model: HuggingFace model id (default: gemma-2-2b-it)
- --device: cuda or cpu (default: cuda)
- --start-offset: tokens after prompt to start analysis (default: 0)
- --max-tokens: number of tokens to analyze (default: 50)
- --include-docstrings: include tokens inside docstrings (default: skip)
- --ig-steps: IG steps (default: 10)
- --k-max: max K for circuit discovery (default: 90001)
- --k-step: step for K sweep (default: 10000)
- --k-thres: performance threshold (default: 0.6)
- --coeff-start/--coeff-end/--coeff-step: steering coefficient grid
- --data-path: dataset json path
- --output-dir: directory to write results
- --save: write results to disk
- --quiet: suppress verbose logs

### Outputs
When saving is enabled, per-token results are written to `outputs/prompt_{idx}/token_{pos}/`:
- `circuit_entries.pt`: circuit discovery results (torch)
- `clusters.json`: logit-lens clusters
- `steering_results.json`: steering sweep results
- `metadata.json`: run metadata

### API surface
Key functions in `plan_trace.pipeline`:
- `run_full_pipeline(...)`: end-to-end for a single token position
- `run_automated_token_pipeline(...)`: iterate over multiple token positions
- `run_single_token_analysis(...)`: core analysis given precomputed sequence

### Notes
- Default model/SAE: `gemma-2-2b-it` with Gemma-Scope 2B MLP canonical 16k SAEs
- Uses bfloat16 and CUDA by default



