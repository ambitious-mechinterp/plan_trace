"""
Circuit analysis tools for measuring planning effects across different prediction positions.

Shape Suffix Definition: 
- B: batch size 
- L: Num of Input Tokens
"""

import json
import torch
import torch.nn.functional as F
from pathlib import Path
from functools import partial
from dataclasses import dataclass
from typing import Dict, Set, Tuple, List, Iterable, Optional
from .logit_lens import gather_unique_tokens, build_saved_pair_dict_fastest
from .steering import steering_hook


@dataclass
class Config:
    """Configuration for circuit analysis pipeline."""
    # I/O
    data_path: Path = Path("../data/external/first_100_passing_examples.json")
    hits_root: Path = Path("../models/mbpp_task2_2b_mlp")

    # generation
    stop_token_id: int = 1917
    max_new_tokens: int = 150

    # steering
    coeff: float = -200.0

    # SAE
    sae_release: str = "gemma-scope-2b-pt-mlp-canonical"
    layers: Optional[List[int]] = None

    # runtime
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class CircuitAnalyzer:
    """
    Analyze planning circuits for a single (prompt_idx, token_pred_idx) pair.
    
    This class orchestrates the full analysis pipeline:
    1. Loads circuit entries from saved hits
    2. Builds logit lens clusters of latents by predicted tokens  
    3. Measures steering effects of each cluster on baseline prediction
    4. Identifies "planning positions" where steering has significant effects
    """

    def __init__(self, model, saes: List, cfg: Config):
        """
        Initialize analyzer with model, SAEs, and configuration.
        
        Args:
            model: The language model
            saes: List of SAE objects
            cfg: Configuration object
        """
        self.model, self.saes, self.cfg = model, saes, cfg
        if cfg.layers is None:
            cfg.layers = list(range(model.cfg.n_layers))
        with open(cfg.data_path, "r") as f:
            self.dataset = json.load(f)

    def run(
        self,
        prompt_idx: int,
        token_pred_idx: int,
        *,
        keys: Optional[Iterable[str]] = None,
        thresh: float = 0.0,
    ) -> Dict[str, object]:
        """
        Compute circuit positions C, future positions F_all, and planning positions F_prime.

        Args:
            prompt_idx: Index into the dataset
            token_pred_idx: Index of the prediction position to analyze
            keys: List of saved‑pair dict keys to evaluate. None → all keys, [] → skip evaluation
            thresh: Threshold for steering effect magnitude to be considered significant
            
        Returns:
            Dict containing:
            - C: Set of (layer, token) positions in the circuit
            - F_all: Set of (layer, token) positions with future planning latents  
            - F_prime: Set of (layer, token) positions with significant steering effects
            - position_effect: Dict mapping (layer, token) -> steering effect magnitude
        """
        # 1. Load circuit entries (C) & baseline prompt
        hits_path = self.cfg.hits_root / f"prompt_{prompt_idx}/pred_{token_pred_idx}/hits.pt"
        trial = torch.load(hits_path, weights_only=True)
        C = {(l, t) for l, t, *_ in trial["entries"]}
        clean_prompt = trial["prompt"].lstrip("<bos>")

        # 2. Build saved‑pair dict for logit lens clustering
        uniq_ids = gather_unique_tokens(self.model, clean_prompt, stop_tok=self.cfg.stop_token_id)
        prompt_ids = self.model.to_tokens(clean_prompt)[0].tolist()
        filtered = [tok for tok in uniq_ids if tok not in prompt_ids]

        saved_pair_dict = build_saved_pair_dict_fastest(
            self.model, self.saes, trial["entries"], filtered
        )
        
        # Collect all positions that have hits for any predicted token
        hit_positions = set()
        for key in saved_pair_dict:
            for layer_i, _, token_pos_list in saved_pair_dict[key]:
                for token_pos in token_pos_list:
                    hit_positions.add((layer_i, token_pos))

        F_all = hit_positions
        
        # Early‑exit if user provided an empty key list
        if keys is not None and len(list(keys)) == 0:
            return {
                "prompt_idx": prompt_idx,
                "token_pred_idx": token_pred_idx,
                "keys": [],
                "C": C,
                "F_all": F_all,
                "F_prime": set(),
                "position_effect": {},
            }

        # Choose keys
        keys = list(saved_pair_dict.keys()) if keys is None else list(keys)

        # 3. Map (layer, token_pos) → latents for selected keys
        pair_to_latents: Dict[Tuple[int, int], List[int]] = {}
        for k in keys:
            for layer_i, latent_i, tok_pos_list in saved_pair_dict[k]:
                for tok_pos in tok_pos_list:
                    pair_to_latents.setdefault((layer_i, tok_pos), []).append(latent_i)

        # 4. Compute steering effects on demand
        position_effect: Dict[Tuple[int, int], float] = {}
        F_prime: Set[Tuple[int, int]] = set()
        if F_all:
            position_effect = self._measure_effects(pair_to_latents, clean_prompt)
            F_prime = {pos for pos, eff in position_effect.items() if abs(eff) > thresh}

        return {
            "prompt_idx": prompt_idx,
            "token_pred_idx": token_pred_idx,
            "keys": keys,
            "C": C,
            "F_all": F_all,
            "F_prime": F_prime,
            "position_effect": position_effect,
        }

    def _measure_effects(
        self, 
        pair_to_latents: Dict[Tuple[int, int], List[int]], 
        clean_prompt: str
    ) -> Dict[Tuple[int, int], float]:
        """
        Measure steering effects of latent clusters on the baseline prediction.
        
        For each (layer, token_pos) with associated latents, apply steering
        and measure the change in probability for the baseline predicted token.
        
        Args:
            pair_to_latents: Mapping from (layer, token_pos) to list of latent indices
            clean_prompt: The input prompt string
            
        Returns:
            Dict mapping (layer, token_pos) -> effect magnitude (prob change)
        """
        toks_prefix = self.model.to_tokens(clean_prompt)
        self.model.reset_hooks(including_permanent=True)
        
        # Get baseline prediction
        with torch.no_grad():
            logits = self.model(toks_prefix)[:, -1]
        baseline_id = logits.argmax(-1).item()
        baseline_prob = F.softmax(logits, dim=-1)[0, baseline_id]

        pos_eff: Dict[Tuple[int, int], float] = {}
        
        for (layer, tok_pos), latents in pair_to_latents.items():
            sae = self.saes[layer]
            # Add steering hooks for all latents at this position
            for latent_idx in latents:
                self.model.add_hook(
                    sae.cfg.hook_name,
                    partial(
                        steering_hook, 
                        sae=sae, 
                        latent_idx=latent_idx, 
                        coeff=self.cfg.coeff, 
                        steering_token_index=tok_pos
                    ),
                )
            
            # Measure steered probability
            with torch.no_grad():
                logits = self.model(toks_prefix)[:, -1]
            prob = F.softmax(logits, dim=-1)[0, baseline_id]
            pos_eff[(layer, tok_pos)] = (prob - baseline_prob).item()
            self.model.reset_hooks(including_permanent=True)
            
        return pos_eff


def analyze_batch(
    analyzer: CircuitAnalyzer,
    prompt_idx: int,
    mapping: Dict[int, List[str]],  # token_pred_idx → key list (may be empty)
    *,
    thresh: float = 0.0,
) -> Dict[str, Set[Tuple[int, int]]]:
    """
    Run analyzer over several predictions and return unified sets.
    
    This function analyzes multiple prediction positions for the same prompt
    and aggregates the results to understand overall planning patterns.
    
    Args:
        analyzer: CircuitAnalyzer instance
        prompt_idx: Index into the dataset
        mapping: Dict mapping token_pred_idx -> list of keys to analyze
        thresh: Threshold for significant steering effects
        
    Returns:
        Dict containing:
        - union_C: All circuit positions across predictions
        - union_F_all: All future planning positions across predictions  
        - union_F_prime: All significant planning positions across predictions
        - circuit_minus_future: Circuit positions that aren't future planning positions
    """
    union_C: Set[Tuple[int, int]] = set()
    union_F: Set[Tuple[int, int]] = set()
    union_Fp: Set[Tuple[int, int]] = set()

    for token_pred_idx, key_list in mapping.items():
        res = analyzer.run(prompt_idx, token_pred_idx, keys=key_list, thresh=thresh)
        union_C |= res["C"]
        union_F |= res["F_all"]
        union_Fp |= res["F_prime"]

    return {
        "union_C": union_C,
        "union_F_all": union_F,
        "union_F_prime": union_Fp,
        "circuit_minus_future": union_C - union_F,
    } 