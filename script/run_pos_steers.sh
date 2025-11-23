#!/bin/bash
set -e  # stop on any error

echo "=== Running BASE steering runs first ==="

# ----- prompt 11: base with instruct latents -----
# echo "Running base -> instruct (prompt 11)..."
# python -m plan_trace.custom_steer --cap 69 --latent-file outputs/comp_exp/base-fail/instruct-comp/prompt_11/token_97/clusters.json --output-dir outputs/steer-base-11-with-97 --model gemma-2-2b --data-file data/external/first_100_failing_examples_without_docstrings_base_model_og_prompt_V2.json --prompt-idx 11
# python -m plan_trace.custom_steer --cap 69 --latent-file outputs/comp_exp/base-fail/instruct-comp/prompt_11/token_104/clusters.json --output-dir outputs/steer-base-11-with-104 --model gemma-2-2b --data-file data/external/first_100_failing_examples_without_docstrings_base_model_og_prompt_V2.json --prompt-idx 11
# python -m plan_trace.custom_steer --cap 69 --latent-file outputs/comp_exp/base-fail/instruct-comp/prompt_11/token_108/clusters.json --output-dir outputs/steer-base-11-with-108 --model gemma-2-2b --data-file data/external/first_100_failing_examples_without_docstrings_base_model_og_prompt_V2.json --prompt-idx 11
# python -m plan_trace.custom_steer --cap 69 --latent-file outputs/comp_exp/base-fail/instruct-comp/prompt_11/token_109/clusters.json --output-dir outputs/steer-base-11-with-109 --model gemma-2-2b --data-file data/external/first_100_failing_examples_without_docstrings_base_model_og_prompt_V2.json --prompt-idx 11
# python -m plan_trace.custom_steer --cap 69 --latent-file outputs/comp_exp/base-fail/instruct-comp/prompt_11/token_110/clusters.json --output-dir outputs/steer-base-11-with-110 --model gemma-2-2b --data-file data/external/first_100_failing_examples_without_docstrings_base_model_og_prompt_V2.json --prompt-idx 11

# # ----- prompt 8: base with instruct latents -----
# echo "Running base -> instruct (prompt 8)..."
# python -m plan_trace.custom_steer --cap 75 --latent-file outputs/comp_exp/base-fail/instruct-comp/prompt_8/token_107/clusters.json --output-dir outputs/steer-base-8-with-107 --model gemma-2-2b --data-file data/external/first_100_failing_examples_without_docstrings_base_model_og_prompt_V2.json --prompt-idx 8
# python -m plan_trace.custom_steer --cap 75 --latent-file outputs/comp_exp/base-fail/instruct-comp/prompt_8/token_121/clusters.json --output-dir outputs/steer-base-8-with-121 --model gemma-2-2b --data-file data/external/first_100_failing_examples_without_docstrings_base_model_og_prompt_V2.json --prompt-idx 8

# echo "=== BASE runs complete ==="
# echo

# echo "=== Running INSTRUCT steering runs ==="

# # ----- prompt 11: instruct with base latents -----
# echo "Running instruct -> base (prompt 11)..."
# python -m plan_trace.custom_steer --cap 69 --latent-file outputs/comp-exp/base-fail/base-comp/prompt_11/token_93/clusters.json --output-dir outputs/steer-instruct-11-with-93 --model gemma-2-2b-it --data-file data/external/first_100_failing_examples_without_docstrings_base_model_og_prompt_V2.json --prompt-idx 11 --mode nodocstring

# # ----- prompt 8: instruct with base latents -----
# echo "Running instruct -> base (prompt 8)..."
# python -m plan_trace.custom_steer --cap 75 --latent-file outputs/comp-exp/base-fail/base-comp/prompt_8/token_102/clusters.json --output-dir outputs/steer-instruct-8-with-102 --model gemma-2-2b-it --data-file data/external/first_100_failing_examples_without_docstrings_base_model_og_prompt_V2.json --prompt-idx 8 --mode nodocstring
# python -m plan_trace.custom_steer --cap 75 --latent-file outputs/comp-exp/base-fail/base-comp/prompt_8/token_103/clusters.json --output-dir outputs/steer-instruct-8-with-103 --model gemma-2-2b-it --data-file data/external/first_100_failing_examples_without_docstrings_base_model_og_prompt_V2.json --prompt-idx 8 --mode nodocstring
# python -m plan_trace.custom_steer --cap 75 --latent-file outputs/comp-exp/base-fail/base-comp/prompt_8/token_104/clusters.json --output-dir outputs/steer-instruct-8-with-104 --model gemma-2-2b-it --data-file data/external/first_100_failing_examples_without_docstrings_base_model_og_prompt_V2.json --prompt-idx 8 --mode nodocstring

# echo "=== INSTRUCT runs complete ==="
# echo "=== All steering runs finished successfully! ==="


# prompt-53, only instruct
# python -m plan_trace.custom_steer --cap 152 --latent-file outputs/comp-exp/base-pass/base-comp/prompt_53/token_182/clusters.json --output-dir outputs/steer-instruct-53-with-182 --model gemma-2-2b-it --data-file data/external/first_100_passing_examples_without_docstrings_base_model_og_prompt_V2.json --prompt-idx 53 --mode nodocstring
# python -m plan_trace.custom_steer --cap 152 --latent-file outputs/comp-exp/base-pass/base-comp/prompt_53/token_175/clusters.json --output-dir outputs/steer-instruct-53-with-175 --model gemma-2-2b-it --data-file data/external/first_100_passing_examples_without_docstrings_base_model_og_prompt_V2.json --prompt-idx 53 --mode nodocstring
# python -m plan_trace.custom_steer --cap 152 --latent-file outputs/comp-exp/base-pass/base-comp/prompt_53/token_176/clusters.json --output-dir outputs/steer-instruct-53-with-176 --model gemma-2-2b-it --data-file data/external/first_100_passing_examples_without_docstrings_base_model_og_prompt_V2.json --prompt-idx 53 --mode nodocstring
# python -m plan_trace.custom_steer --cap 152 --latent-file outputs/comp-exp/base-pass/base-comp/prompt_53/token_183/clusters.json --output-dir outputs/steer-instruct-53-with-183 --model gemma-2-2b-it --data-file data/external/first_100_passing_examples_without_docstrings_base_model_og_prompt_V2.json --prompt-idx 53 --mode nodocstring

# prompt-6 only base
python -m plan_trace.custom_steer --cap 106 --latent-file outputs/comp_exp/base-pass/instruct-comp/prompt_6/token_137/clusters.json --output-dir outputs/steer-base-6-with-137 --model gemma-2-2b --data-file data/external/first_100_passing_examples_without_docstrings_base_model_og_prompt_V2.json --prompt-idx 6
python -m plan_trace.custom_steer --cap 106 --latent-file outputs/comp_exp/base-pass/instruct-comp/prompt_6/token_139/clusters.json --output-dir outputs/steer-base-6-with-139 --model gemma-2-2b --data-file data/external/first_100_passing_examples_without_docstrings_base_model_og_prompt_V2.json --prompt-idx 6