import torch
from typing import Dict


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
    if isinstance(out_text, str) and out_text:
        if ym_str in out_text or (" " + ym_str) in out_text:
            return True
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
    try:
        prompt_text = model.to_string(prefix_tokens_2d[0, :])
    except Exception:
        prompt_text = ""
    if prompt_text and (ym_str in prompt_text or (" " + ym_str) in prompt_text):
        return True
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


def _compute_repetition_metrics(tokens_1d: torch.Tensor, n_list=(2, 3, 4), top_k: int = 3):
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
    max_run = 1
    current_run = 1
    for i in range(1, total_len):
        if tokens_1d[i].item() == tokens_1d[i - 1].item():
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 1
    unique_ratio = float(torch.unique(tokens_1d).numel()) / float(total_len)
    _, counts = torch.unique(tokens_1d, return_counts=True)
    top_counts, _ = torch.topk(counts.float(), k=min(top_k, counts.numel()))
    top_k_ratio = float(top_counts.sum().item() / total_len)
    ngram_repeat_fracs = {}
    for n in n_list:
        if total_len < n:
            ngram_repeat_fracs[f"ngram_repeat_frac_{n}"] = 0.0
            continue
        toks_cpu = tokens_1d.detach().cpu().tolist()
        counts_map: Dict = {}
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
        logits = model(full)
        logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        target = full[:, 1:]
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
    if thresholds is None:
        thresholds = {
            "min_len_for_rules": 20,
            "extreme_max_run_len": 12,
            "low_unique_ratio": 0.15,
            "high_top_k_ratio": 0.90,
            "high_trigram_repeat": 0.50,
            "mid_trigram_repeat": 0.40,
            "support_top_k_ratio": 0.80,
            "ppl_support": 1.15,
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
    reasons = []
    is_degen = False
    if L >= thresholds["min_len_for_rules"]:
        if metrics["max_run_len"] >= thresholds["extreme_max_run_len"]:
            is_degen = True
            reasons.append("rule:max_run_len_extreme")
        if not is_degen and metrics.get("ngram_repeat_frac_3", 0.0) >= thresholds["high_trigram_repeat"]:
            if metrics["top_k_ratio"] >= thresholds["high_top_k_ratio"] or metrics["unique_ratio"] <= thresholds["low_unique_ratio"]:
                is_degen = True
                reasons.append("rule:high_trigram+support")
        if (
            not is_degen
            and metrics.get("ngram_repeat_frac_3", 0.0) >= thresholds["mid_trigram_repeat"]
            and metrics["top_k_ratio"] >= thresholds["support_top_k_ratio"]
            and ppl_metrics is not None
            and metrics.get("ppl", 9e9) <= thresholds["ppl_support"]
        ):
            is_degen = True
            reasons.append("rule:mid_trigram+topk+ppl_support")
    return {"is_degenerate": is_degen, "reasons": reasons, "metrics": metrics}


def label_steering_clusters(
    steering_results: dict,
    *,
    model,
    prefix_tokens_2d: torch.Tensor,
    thresholds: dict | None = None,
):
    results = {}
    for ym_str, info in steering_results.items():
        ym_in_prompt = _ym_present_in_prompt(ym_str, prefix_tokens_2d, model)
        evals = []
        for item in info.get("steered", []):
            coeff = item.get("coeff")
            val = item.get("steered_text")
            changed = False
            if isinstance(val, torch.Tensor):
                changed = val.numel() > 0
            elif isinstance(val, str):
                changed = len(val) > 0
            text, toks = _entry_to_text_and_tokens(val, model)
            ym_present = _ym_present_in_output(ym_str, text, toks, model) if changed else False
            degen = False
            if changed:
                try:
                    degen_res = detect_degenerate_continuation(
                        toks, model=model, prefix_tokens_2d=prefix_tokens_2d, thresholds=thresholds
                    )
                    degen = bool(degen_res.get("is_degenerate", False))
                except Exception:
                    degen = False
            evals.append({"coeff": coeff, "changed": changed, "ym_present": ym_present, "degenerate": degen})
        chosen = None
        final_label = "Not planning"
        ym_pass_evs = [ev for ev in evals if ev["changed"] and (not ev["ym_present"]) ]
        for ev in ym_pass_evs:
            if (not ym_in_prompt) and (not ev["degenerate"]):
                chosen = ev
                final_label = "Plan"
                break
        if final_label == "Not planning" and len(ym_pass_evs) > 0:
            if ym_in_prompt:
                chosen = ym_pass_evs[0]
                final_label = "Can't say"
            else:
                all_deg = all(ev["degenerate"] for ev in ym_pass_evs)
                if all_deg:
                    chosen = ym_pass_evs[0]
                    final_label = "Can't say"
        results[ym_str] = {"final_label": final_label, "chosen": chosen, "evals": evals, "ym_in_prompt": ym_in_prompt}
    return results


