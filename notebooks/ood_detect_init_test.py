# %%
"""
1. load everything 
2. setup task 24, with yn=180
3. generate steered gens for ym="(" and ym="2"
4. draft function to detect duplicates and calculate perplexity

"""

# %%

import sys 
import torch 
import json 

sys.path.append("../")

from plan_trace.pipeline import run_single_token_analysis
from plan_trace.utils import load_model, load_pretrained_saes, cleanup_cuda

# %% setup vars

model_name = "gemma-2-2b-it"
device = "cuda"
data_path = "../data/first_100_passing_examples.json"
prompt_idx = 24
yn = 180
stop_token_id = 1917
ig_steps = 10
k_max = 90001
k_step = 10000
k_thres = 0.6
coeff_grid = list(range(-100, 0, 20))

# %%
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

# %%

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
# %%
print(model.to_string(out_BL[0, toks_BL.shape[-1]:]))

# %%


token_result = run_single_token_analysis(
            model=model,
            saes=saes,
            out_BL=out_BL,
            inter_token_id=yn,
            ig_steps=ig_steps,
            k_max=k_max,
            k_step=k_step,
            k_thres=k_thres,
            coeff_grid=coeff_grid,
            stop_token_id=stop_token_id,
            verbose=True
        )
print(token_result)
# %%
print("Base text:")
print(token_result['steering_results']['(']['base_text'])

# %%
for steered_gen in token_result['steering_results']['(']['steered']:
    print(steered_gen['coeff'])
    print(steered_gen['steered_text'])
    print("-"*100)
# %%
for steered_gen in token_result['steering_results']['2']['steered']:
    print(steered_gen['coeff'])
    print(steered_gen['steered_text'])
    print("-"*100)
# %%

# %% Simple degeneracy checks (string-only, no probabilities)
import re
import zlib
from collections import Counter


def _split_words(text: str):
    if not text:
        return []
    # Split into alphanum words; punctuation treated separately by downstream metrics
    return re.findall(r"[A-Za-z0-9_]+", text)


def _max_run(sequence):
    if not sequence:
        return 0
    max_run = 1
    cur = 1
    for i in range(1, len(sequence)):
        if sequence[i] == sequence[i - 1]:
            cur += 1
            if cur > max_run:
                max_run = cur
        else:
            cur = 1
    return max_run


def _ngram_repeat_fraction(items, n: int):
    if n <= 0 or len(items) < 2 * n:
        return 0.0
    ngrams = [tuple(items[i : i + n]) for i in range(len(items) - n + 1)]
    counts = Counter(ngrams)
    total = len(ngrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / total if total > 0 else 0.0


def _type_token_ratio(items):
    return (len(set(items)) / len(items)) if items else 0.0


def _compression_ratio(text: str):
    if not text:
        return 1.0
    raw = text.encode("utf-8", errors="ignore")
    comp = zlib.compress(raw, level=9)
    return len(comp) / max(1, len(raw))


def compute_string_degeneracy_metrics(text: str):
    words = _split_words(text)
    # Character-level
    chars = list(text)
    char_max_repeat = _max_run(chars)
    # Word-level
    word_max_repeat = _max_run(words)
    word_ttr = _type_token_ratio(words)
    bigram_repeat = _ngram_repeat_fraction(words, 2)
    trigram_repeat = _ngram_repeat_fraction(words, 3)
    compression = _compression_ratio(text)

    # Simple boolean decision with conservative thresholds
    degenerate = (
        char_max_repeat >= 8
        or word_max_repeat >= 5
        or bigram_repeat > 0.25
        or trigram_repeat > 0.15
        or word_ttr < 0.25
        or compression < 0.45
    )

    return {
        "degenerate": degenerate,
        "char_max_repeat": char_max_repeat,
        "word_max_repeat": word_max_repeat,
        "word_ttr": round(word_ttr, 3),
        "bigram_repeat": round(bigram_repeat, 3),
        "trigram_repeat": round(trigram_repeat, 3),
        "compression_ratio": round(compression, 3),
    }


def _print_simple_report(label: str, text: str):
    metrics = compute_string_degeneracy_metrics(text)
    print(label)
    print(metrics)
    print("-" * 80)


base_text = token_result['steering_results']['(']['base_text']
_print_simple_report("BASE", base_text)

for steered_gen in token_result['steering_results']['(']['steered']:
    coeff = steered_gen['coeff']
    text = steered_gen['steered_text']
    print(text)
    _print_simple_report(f"( coeff={coeff} )", text)

# %%

for steered_gen in token_result['steering_results']['2']['steered']:
    coeff = steered_gen['coeff']
    text = steered_gen['steered_text']
    print(text)
    _print_simple_report(f"2 coeff={coeff}", text)


# %%
