# %%
"""
1. load everything 
2. setup task 24, with yn=180
3. generate steered gens for ym="(" and ym="2" with individual functions, not the whole pipeline
4. draft function to detect duplicates and calculate perplexity with tokens, not strings
"""

# %%

import sys 
import torch 
import json 

sys.path.append("../")

from plan_trace.pipeline import run_single_token_analysis, 
from plan_trace.utils import load_model, load_pretrained_saes, cleanup_cuda
from plan_trace.circuit_discovery import discover_circuit
from plan_trace.logit_lens import find_logit_lens_clusters, gather_unique_tokens
from plan_trace.steering import run_steering_sweep

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

# Extract the specific prediction position
inter_toks_BL = out_BL[:, :yn]
baseline_suffix = model.to_string(out_BL[0, yn:])

# %%
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

# %%

# Generate unique tokens not in the original prompt
uniq_ids = gather_unique_tokens(model, inter_toks_BL, stop_tok=stop_token_id)
# Combined filter: drop tokens that appear in prompt by ID or robust string boundaries
prompt_tokens = inter_toks_BL[0]
prompt_id_set = set(prompt_tokens.detach().cpu().tolist())
prompt_str = model.to_string(prompt_tokens)
# %%

filtered_uniq_ids = [tok for tok in uniq_ids if model.to_string(tok).strip() not in prompt_str]
print([model.to_string(fil_unq_id) for fil_unq_id in filtered_uniq_ids])
# %%





def _has_boundary_match(hay: str, needle: str) -> bool:
    if not needle:
        return False
    # If the token contains any alphanumerics, require word boundaries to avoid substring hits.
    if any(ch.isalnum() for ch in needle):
        idx = 0
        L = len(hay)
        nL = len(needle)
        while True:
            pos = hay.find(needle, idx)
            if pos == -1:
                return False
            left_ok = pos == 0 or not hay[pos - 1].isalnum()
            right_ok = (pos + nL) >= L or not hay[pos + nL].isalnum()
            if left_ok and right_ok:
                return True
            idx = pos + 1
    # Pure punctuation/symbol: match anywhere
    return needle in hay

def token_in_prompt(tok_id: int) -> bool:
    if tok_id in prompt_id_set:
        return True
    s = model.to_string(tok_id)
    # Try several string forms to handle leading spaces
    cands = [s]
    if s:
        cands.append(s.lstrip())
        cands.append(s.strip())
    for c in cands:
        if c and _has_boundary_match(prompt_str, c):
            return True
    return False

filtered_uniq_ids = [tok for tok in uniq_ids if not token_in_prompt(tok)]

print(len(filtered_uniq_ids))
# %%
for tok in uniq_ids:
    label = model.to_string(tok).strip()
    print(label)
    if tok in prompt_id_set:
        print("True")
        continue
    if _has_boundary_match(prompt_str, label):  # boundary-aware string check
        print("True")
        continue
    print("False")

# %%

_has_boundary_match(prompt_str, "(")

# %%

print([model.to_string(fil_unq_id) for fil_unq_id in filtered_uniq_ids])

# %%
print(model.to_string(out_BL[0, :yn]))

# %%
# Logit Lens Clustering
saved_pair_dict = find_logit_lens_clusters(
    model, saes, entries, inter_toks_BL, stop_token_id, verbose=False
)

# %%

# Filter saved_pair_dict to only include "(" and "2" keys
filtered_pair_dict = {k: v for k, v in saved_pair_dict.items() if k in ["(", "2"]}

# %%
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
    return_tokens=True
)

# %%

print("Base tokens:\n",out_BL[0, yn:])
print("Base text:\n",baseline_suffix)

print("( tokens:\n",steering_results['(']['steered'][0]['steered_text'])
print("( text:\n",model.to_string(steering_results['(']['steered'][0]['steered_text']))

print("2 tokens:\n",steering_results['2']['steered'][2]['steered_text'])
print("2 text:\n",model.to_string(steering_results['2']['steered'][2]['steered_text']))

# %%

print("( tokens:\n",steering_results['(']['steered'][2]['steered_text'])
print("( text:\n",model.to_string(steering_results['(']['steered'][2]['steered_text']))

# %%
steering_results_str = run_steering_sweep(
    model=model,
    saes=saes,
    inter_toks_BL=inter_toks_BL,
    saved_pair_dict=filtered_pair_dict,
    baseline_text=baseline_suffix,
    coeff_grid=coeff_grid,
    stop_tok=stop_token_id,
    max_tokens=100,
    return_tokens=False
)
print("2 tokens:\n",steering_results_str['2']['steered'][2]['steered_text'])

# %%

# Simple degeneracy detector on token continuations

def _compute_repetition_metrics(tokens_1d: torch.Tensor, n_list=(2, 3, 4), top_k: int = 3):
    """Compute basic repetition statistics on a 1D token id tensor."""
    if tokens_1d.ndim != 1:
        tokens_1d = tokens_1d.reshape(-1)
    total_len = int(tokens_1d.numel())
    if total_len == 0:
        return {
            "length": 0,
            "max_run_len": 0,
            "unique_ratio": 1.0,
            "top_k_ratio": 0.0,
            **{f"ngram_repeat_frac_{n}": 0.0 for n in n_list},
        }

    # Max run length of identical tokens
    max_run = 1
    current_run = 1
    for i in range(1, total_len):
        if tokens_1d[i].item() == tokens_1d[i - 1].item():
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 1

    # Unique ratio
    unique_ratio = float(torch.unique(tokens_1d).numel()) / float(total_len)

    # Top-k token mass ratio
    values, counts = torch.unique(tokens_1d, return_counts=True)
    top_counts, _ = torch.topk(counts.float(), k=min(top_k, counts.numel()))
    top_k_ratio = float(top_counts.sum().item() / total_len)

    # N-gram repeat fraction: fraction of n-grams that are repeated at least once
    ngram_repeat_fracs = {}
    for n in n_list:
        if total_len < n:
            ngram_repeat_fracs[f"ngram_repeat_frac_{n}"] = 0.0
            continue
        # Collect n-grams as tuples on CPU for hashing
        toks_cpu = tokens_1d.detach().cpu().tolist()
        counts_map = {}
        for i in range(0, total_len - n + 1):
            key = tuple(toks_cpu[i : i + n])
            counts_map[key] = counts_map.get(key, 0) + 1
        repeated = sum(1 for c in counts_map.values() if c > 1)
        ngram_repeat_fracs[f"ngram_repeat_frac_{n}"] = float(repeated) / float(len(counts_map))

    return {
        "length": total_len,
        "max_run_len": int(max_run),
        "unique_ratio": float(unique_ratio),
        "top_k_ratio": float(top_k_ratio),
        **ngram_repeat_fracs,
    }


def _compute_continuation_ppl(model, prefix_tokens_2d: torch.Tensor, continuation_tokens_1d: torch.Tensor):
    """Compute perplexity over the continuation conditioned on the prefix."""
    if continuation_tokens_1d.ndim != 1:
        continuation_tokens_1d = continuation_tokens_1d.reshape(-1)
    if prefix_tokens_2d.ndim != 2:
        raise ValueError("prefix_tokens_2d must be 2D [1, seq]")
    if prefix_tokens_2d.shape[0] != 1:
        raise ValueError("Only batch size 1 supported")

    device = next(model.parameters()).device
    full = torch.cat([prefix_tokens_2d.to(device), continuation_tokens_1d.unsqueeze(0).to(device)], dim=1)
    prefix_len = int(prefix_tokens_2d.shape[1])
    if continuation_tokens_1d.numel() == 0:
        return {"mean_nll": 0.0, "ppl": 1.0}

    with torch.no_grad():
        logits = model(full)  # [1, L, V]
        # predict token t from position t-1
        logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)  # [1, L-1, V]
        target = full[:, 1:]  # [1, L-1]
        # Continuation starts at position prefix_len, which is predicted by index prefix_len-1
        cont_logprobs = logprobs[0, prefix_len - 1 :, :].gather(1, target[0, prefix_len - 1 :].unsqueeze(1)).squeeze(1)
        mean_nll = float((-cont_logprobs).mean().item())
        ppl = float(torch.exp(-cont_logprobs.mean()).item())
    return {"mean_nll": mean_nll, "ppl": ppl}


def detect_degenerate_continuation(
    continuation_tokens_1d: torch.Tensor,
    *,
    model=None,
    prefix_tokens_2d: torch.Tensor | None = None,
    thresholds: dict | None = None,
):
    """
    Detect degenerate continuations using stricter, AND-style rules. Perplexity is supportive only.

    Inputs:
      - continuation_tokens_1d: 1D Tensor of token ids (the generated continuation)
      - model, prefix_tokens_2d: if provided, computes continuation perplexity conditioned on prefix
      - thresholds: override defaults

    Returns: { is_degenerate: bool, reasons: [..], metrics: {...} }
    """
    if thresholds is None:
        thresholds = {
            "min_len_for_rules": 20,
            "extreme_max_run_len": 12,      # single-token loops
            "low_unique_ratio": 0.15,       # very low diversity
            "high_top_k_ratio": 0.90,       # dominated by few tokens
            "high_trigram_repeat": 0.50,    # strong local repetition
            "mid_trigram_repeat": 0.40,     # moderate repetition
            "support_top_k_ratio": 0.80,    # support for mid_trigram rule
            "ppl_support": 1.15,            # only supportive when paired with repetition
        }

    metrics = _compute_repetition_metrics(continuation_tokens_1d)

    ppl_metrics = None
    L = metrics["length"]
    if model is not None and prefix_tokens_2d is not None and L >= thresholds["min_len_for_rules"]:
        try:
            ppl_metrics = _compute_continuation_ppl(model, prefix_tokens_2d, continuation_tokens_1d)
            metrics.update(ppl_metrics)
        except Exception:
            pass

    # Decision rules (stricter): require strong repetition patterns, not PPL alone
    reasons = []
    is_degen = False
    if L >= thresholds["min_len_for_rules"]:
        # Rule 1: Extreme single-token runs
        if metrics["max_run_len"] >= thresholds["extreme_max_run_len"]:
            is_degen = True
            reasons.append("rule:max_run_len_extreme")

        # Rule 2: High trigram repetition + (high top-k OR very low diversity)
        if not is_degen and metrics.get("ngram_repeat_frac_3", 0.0) >= thresholds["high_trigram_repeat"]:
            if metrics["top_k_ratio"] >= thresholds["high_top_k_ratio"] or metrics["unique_ratio"] <= thresholds["low_unique_ratio"]:
                is_degen = True
                reasons.append("rule:high_trigram+support")

        # Rule 3: Moderate trigram repetition + high top-k + low PPL (supportive)
        if (
            not is_degen
            and metrics.get("ngram_repeat_frac_3", 0.0) >= thresholds["mid_trigram_repeat"]
            and metrics["top_k_ratio"] >= thresholds["support_top_k_ratio"]
            and ppl_metrics is not None
            and metrics.get("ppl", 9e9) <= thresholds["ppl_support"]
        ):
            is_degen = True
            reasons.append("rule:mid_trigram+topk+ppl_support")

    return {
        "is_degenerate": is_degen,
        "reasons": reasons,
        "metrics": metrics,
    }


# Example: evaluate baseline and two steering variants
# try:
#     base_cont = out_BL[0, yn:]
#     base_res = detect_degenerate_continuation(base_cont, model=model, prefix_tokens_2d=out_BL[:, :yn])
#     print("\n[Degeneracy] Baseline:", base_res)

#     paren_cont = steering_results['(']['steered'][0]['steered_text']
#     paren_res = detect_degenerate_continuation(paren_cont, model=model, prefix_tokens_2d=inter_toks_BL)
#     print("[Degeneracy] '(' steered:", paren_res)

#     paren_cont = steering_results['(']['steered'][2]['steered_text']
#     paren_res = detect_degenerate_continuation(paren_cont, model=model, prefix_tokens_2d=inter_toks_BL)
#     print("[Degeneracy] '(' steered:", paren_res)

#     two_cont = steering_results['2']['steered'][2]['steered_text']
#     two_res = detect_degenerate_continuation(two_cont, model=model, prefix_tokens_2d=inter_toks_BL)
#     print("[Degeneracy] '2' steered:", two_res)
# except Exception as _e:
#     # Keep notebook running even if any shape/device assumption fails
#     print("[Degeneracy] Skipped evaluation:", str(_e))


# %%

# Labeler: classify cluster outcomes as Plan / Improv / Can't say / Not planning

def _contains_subsequence(haystack: torch.Tensor, needle: torch.Tensor) -> bool:
    if haystack.ndim != 1:
        haystack = haystack.reshape(-1)
    if needle.ndim != 1:
        needle = needle.reshape(-1)
    H = int(haystack.numel())
    N = int(needle.numel())
    if N == 0 or H < N:
        return False
    hay = haystack.detach().cpu().tolist()
    ned = needle.detach().cpu().tolist()
    for i in range(0, H - N + 1):
        if hay[i : i + N] == ned:
            return True
    return False


def _entry_to_text_and_tokens(val, model):
    # Returns (text_str, tokens_1d_tensor)
    if isinstance(val, torch.Tensor):
        toks = val
        try:
            txt = model.to_string(toks)
        except Exception:
            txt = ""
        return txt, toks
    else:
        txt = val if isinstance(val, str) else ""
        if txt:
            try:
                toks = model.to_tokens(txt)[0]
            except Exception:
                toks = torch.tensor([], device=next(model.parameters()).device)
        else:
            toks = torch.tensor([], device=next(model.parameters()).device)
        return txt, toks


def _ym_present_in_output(ym_str: str, out_text: str, out_tokens: torch.Tensor, model) -> bool:
    # String check (robust to spacing)
    if isinstance(out_text, str) and out_text:
        if ym_str in out_text:
            return True
        if (" " + ym_str) in out_text:
            return True

    # Token subsequence check for both direct and space-prefixed forms
    try:
        ym_tok = model.to_tokens(ym_str)[0]
    except Exception:
        ym_tok = torch.tensor([], device=out_tokens.device)
    try:
        ym_tok_sp = model.to_tokens(" " + ym_str)[0]
    except Exception:
        ym_tok_sp = torch.tensor([], device=out_tokens.device)

    if out_tokens.numel() > 0:
        if ym_tok.numel() > 0 and _contains_subsequence(out_tokens, ym_tok):
            return True
        if ym_tok_sp.numel() > 0 and _contains_subsequence(out_tokens, ym_tok_sp):
            return True
    return False


def _ym_present_in_prompt(ym_str: str, prefix_tokens_2d: torch.Tensor, model) -> bool:
    # String containment on the prompt
    try:
        prompt_text = model.to_string(prefix_tokens_2d[0, :])
    except Exception:
        prompt_text = ""
    if prompt_text:
        if ym_str in prompt_text or (" " + ym_str) in prompt_text:
            return True

    # Token subsequence containment on the prompt
    try:
        ym_tok = model.to_tokens(ym_str)[0]
    except Exception:
        ym_tok = torch.tensor([], device=prefix_tokens_2d.device)
    try:
        ym_tok_sp = model.to_tokens(" " + ym_str)[0]
    except Exception:
        ym_tok_sp = torch.tensor([], device=prefix_tokens_2d.device)

    prompt_tokens = prefix_tokens_2d[0]
    if ym_tok.numel() > 0 and _contains_subsequence(prompt_tokens, ym_tok):
        return True
    if ym_tok_sp.numel() > 0 and _contains_subsequence(prompt_tokens, ym_tok_sp):
        return True
    return False


def label_steering_clusters(
    steering_results: dict,
    *,
    model,
    prefix_tokens_2d: torch.Tensor,
    thresholds: dict | None = None,
):
    """
    Label each cluster across coefficients with rules:
      - Plan: next token changed AND target YM absent AND not degenerate
      - Improv: next token changed AND not degenerate (YM may still appear)
      - Can't say: next token changed AND degenerate
      - Not planning: otherwise

    Returns dict[label] = {final_label, chosen, evals}
    where chosen contains the first coeff satisfying the strongest label.
    """
    results = {}
    for ym_str, info in steering_results.items():
        ym_in_prompt = _ym_present_in_prompt(ym_str, prefix_tokens_2d, model)
        evals = []
        for item in info.get("steered", []):
            coeff = item.get("coeff")
            val = item.get("steered_text")

            # Determine if next token changed based on presence of output
            changed = False
            if isinstance(val, torch.Tensor):
                changed = val.numel() > 0
            elif isinstance(val, str):
                changed = len(val) > 0
            else:
                changed = False

            text, toks = _entry_to_text_and_tokens(val, model)
            ym_present = _ym_present_in_output(ym_str, text, toks, model) if changed else False

            # Degeneracy check if there is any continuation
            degen = False
            if changed:
                try:
                    degen_res = detect_degenerate_continuation(
                        toks, model=model, prefix_tokens_2d=prefix_tokens_2d, thresholds=thresholds
                    )
                    degen = bool(degen_res.get("is_degenerate", False))
                except Exception:
                    degen = False

            evals.append(
                {
                    "coeff": coeff,
                    "changed": changed,
                    "ym_present": ym_present,
                    "degenerate": degen,
                }
            )

        # Apply enumerated logic
        chosen = None
        final_label = "Not planning"

        # Candidates that pass YM removal
        ym_pass_evs = [ev for ev in evals if ev["changed"] and (not ev["ym_present"])]

        # Case 4: some pass YM, YM not in prompt, and non-degenerate -> Planning
        for ev in ym_pass_evs:
            if (not ym_in_prompt) and (not ev["degenerate"]):
                chosen = ev
                final_label = "Plan"
                break

        # Case 2/3: some pass YM but all such are degenerate OR YM is in prompt -> Can't say
        if final_label == "Not planning" and len(ym_pass_evs) > 0:
            # If YM is in prompt, immediately Can't say
            if ym_in_prompt:
                chosen = ym_pass_evs[0]
                final_label = "Can't say"
            else:
                # YM not in prompt, but all passing are degenerate
                all_deg = all(ev["degenerate"] for ev in ym_pass_evs)
                if all_deg:
                    chosen = ym_pass_evs[0]
                    final_label = "Can't say"

        # Case 1 and default: either no coeff changed, or none pass YM -> Not planning

        results[ym_str] = {"final_label": final_label, "chosen": chosen, "evals": evals, "ym_in_prompt": ym_in_prompt}

    return results


# Example labeling on token-based steering results
try:
    labels = label_steering_clusters(
        steering_results,
        model=model,
        prefix_tokens_2d=inter_toks_BL,
    )
    print("\n[Cluster Labels]")
    for k, v in labels.items():
        print(f" - {k}: {v['final_label']} (chosen={v['chosen']})")
    # Highlight Can't say cases
    # print("\n[Can't say cases]")
    # any_cs = False
    # for k, v in labels.items():
    #     if v["final_label"] == "Can't say":
    #         any_cs = True
    #         print(f" - {k}: chosen={v['chosen']} ym_in_prompt={v['ym_in_prompt']}")
    # if not any_cs:
    #     print(" (none)")
except Exception as _e:
    print("[Labeling] Skipped:", str(_e))


# %%
