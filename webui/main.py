"""
Goal: make an html or something for circuit display along with evidence of planning. 

# ui inputs 

- output_dir (base or instruct)
- prompt_id 
- yn_ind (current token being predicted)


# files to get 

- clusters: /work/pi_jensen_umass_edu/jnainani_umass_edu/plan_trace/outputs/{output_dir}/prompt_{prompt_id}/token_{yn_ind}/clusters.json
- planning_analysis: /work/pi_jensen_umass_edu/jnainani_umass_edu/plan_trace/outputs/{output_dir}/prompt_{prompt_id}/token_{yn_ind}/planning_analysis.json
- steering_results: /work/pi_jensen_umass_edu/jnainani_umass_edu/plan_trace/outputs/{output_dir}/prompt_{prompt_id}/token_{yn_ind}/steering_results.json

## file structure

### clusters:
{
"<ym_1>": [
    [
      15, <- layer_ind 
      101, <- latent_ind
      [     <- token ids active 
        103
      ]
    ],
    [
      15, <- layer_ind
      1871, <- latent_ind
      [ <- token ids active 
        272,
        274,
        269,
        273,
        275,
        278,
        288
      ]
    ], ....
    ], 
"<ym_2>": [ ... ], ...
}

### planning analysis:
{
  "<ym_1>": "Plan",
  "<ym_2>": "Not planning",
  ...
}

### steering results: 
{
    "<ym_1>": <-- future token
    {
        "base_text": ...,
        "steered": [
            {
            "coeff": coeff,
            "is_tokens": true,
            "steered_text": [tokens...],
            "decoded_text": ...., 
            },
            {
            "coeff": coeff,...
            },
            ...
            
    }, 
    "<ym_2>": ...
}


# main interface 

x axis has the string tokens, y axis has the layer indices. x axis can be quite big (> 200 tokens) so maybe better to allow horizontal scrolling than to break 

# clusters 

at a given token_pos, layer, we want to make a cluster of latents for each ym_i. hovering over each cluster should show a pop up, where a slim top row is allowing selecting the latent index for that (ym, layer, token). the rest of the pop up is showing the embed for the selected latent from neuronpedia 

# neuron/latent pop up

<iframe src=https://www.neuronpedia.org/gemma-2-2b/{layer_ind}-gemmascope-mlp-16k/{latent_ind}?embed=true&embedexplanation=true&embedplots=true&embedtest=false" title="Neuronpedia" style="height: 300px; width: 540px;"></iframe>

"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from starlette.staticfiles import StaticFiles


# Resolve important paths
WEBUI_DIR = Path(__file__).resolve().parent
REPO_ROOT = WEBUI_DIR.parent
STATIC_DIR = WEBUI_DIR / "static"
OUTPUTS_DIR = REPO_ROOT / "outputs"
TOKEN_MAP_PATH = REPO_ROOT / "outputs" / "prompt_tokenized_map.json"


def _safe_read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _build_data_response(
    output_dir: str,
    prompt_id: int,
    yn_ind: int,
) -> Dict[str, Any]:
    base_dir = OUTPUTS_DIR / output_dir / f"prompt_{prompt_id}" / f"token_{yn_ind}"

    clusters_path = base_dir / "clusters.json"
    planning_path = base_dir / "planning_analysis.json"
    steering_path = base_dir / "steering_results.json"
    metadata_path = base_dir / "metadata.json"

    clusters = _safe_read_json(clusters_path)
    planning = _safe_read_json(planning_path)
    steering = _safe_read_json(steering_path)
    metadata = _safe_read_json(metadata_path)
    token_map = _safe_read_json(TOKEN_MAP_PATH)

    # Compute helpful metadata and an index structure for quick lookup
    meta: Dict[str, Any] = {
        "hasClusters": clusters is not None,
        "hasPlanning": planning is not None,
        "hasSteering": steering is not None,
        "hasMetadata": metadata is not None,
        "hasTokenized": False,
        "yms": [],
        "layers": [],
        "tokenIds": [],
    }

    index: Dict[str, Dict[str, List[int]]] = {}
    unique_layers: set[int] = set()
    unique_token_ids: set[int] = set()

    if isinstance(clusters, dict):
        yms = list(clusters.keys())
        meta["yms"] = yms
        for ym_key, entries in clusters.items():
            # entries: List[[layer_ind, latent_ind, [token_ids...]], ...]
            ym_map: Dict[str, List[int]] = {}
            if isinstance(entries, list):
                for entry in entries:
                    if (
                        isinstance(entry, list)
                        and len(entry) == 3
                        and isinstance(entry[0], int)
                        and isinstance(entry[1], int)
                        and isinstance(entry[2], list)
                    ):
                        layer_ind = entry[0]
                        latent_ind = entry[1]
                        token_ids = [t for t in entry[2] if isinstance(t, int)]
                        unique_layers.add(layer_ind)
                        for t in token_ids:
                            unique_token_ids.add(t)
                            key = f"{layer_ind}|{t}"
                            ym_map.setdefault(key, []).append(latent_ind)
            index[ym_key] = ym_map

        meta["layers"] = sorted(unique_layers)
        # Ensure we render a continuous token range up to the current token index
        max_in_clusters = max(unique_token_ids) if unique_token_ids else -1
        max_token = max(yn_ind, max_in_clusters)
        meta["tokenIds"] = list(range(0, max_token + 1))

    # Resolve tokenized strings for this prompt and token index (trim BOS by slicing from index 1)
    tokens_input: List[str] = []
    tokens_baseline: List[str] = []
    try:
        prompt_key = str(prompt_id)
        token_key = str(yn_ind)
        if isinstance(token_map, dict) and prompt_key in token_map:
            token_results = token_map[prompt_key].get("token_results") or {}
            entry = token_results.get(token_key)
            if isinstance(entry, dict):
                inp = entry.get("input_prefix_token_strings") or []
                base = entry.get("baseline_token_strings") or []
                if isinstance(inp, list):
                    tokens_input = [str(x) for x in inp[1:]]  # keep one BOS
                if isinstance(base, list):
                    tokens_baseline = [str(x) for x in base[1:]]  # drop BOS entirely
    except Exception:
        # Ignore tokenization errors; keep empty
        pass

    # If we know how many tokens are on the x-axis, align input tokens length
    if meta.get("tokenIds"):
        try:
            tokens_input = tokens_input[: len(meta["tokenIds"])]
        except Exception:
            pass

    if tokens_input or tokens_baseline:
        meta["hasTokenized"] = True

    return {
        "ok": True,
        "paths": {
            "clusters": str(clusters_path),
            "planning": str(planning_path),
            "steering": str(steering_path),
            "metadata": str(metadata_path),
            "token_map": str(TOKEN_MAP_PATH),
        },
        "clusters": clusters,
        "planning": planning,
        "steering": steering,
        "metadata": metadata,
        "tokens": {
            "input": tokens_input,
            "baseline": tokens_baseline,
        },
        "meta": meta,
        "index": index,
    }


app = FastAPI(title="Plan Trace UI")


@app.get("/api/data")
def get_data(
    output_dir: str = Query("instruct", description="Output directory: 'base' or 'instruct'"),
    prompt_id: int = Query(..., description="Prompt ID, e.g., 15"),
    yn_ind: int = Query(..., description="Token index (current token), e.g., 293"),
) -> JSONResponse:
    # Validate output_dir
    if output_dir not in ["base", "instruct"]:
        raise HTTPException(status_code=400, detail="output_dir must be 'base' or 'instruct'")
    
    resp = _build_data_response(output_dir=output_dir, prompt_id=prompt_id, yn_ind=yn_ind)
    if not (
        resp["meta"].get("hasClusters")
        or resp["meta"].get("hasPlanning")
        or resp["meta"].get("hasSteering")
        or resp["meta"].get("hasMetadata")
        or resp["meta"].get("hasTokenized")
    ):
        raise HTTPException(
            status_code=404, 
            detail=f"No data files found for output_dir='{output_dir}', prompt_id={prompt_id}, yn_ind={yn_ind}"
        )
    return JSONResponse(content=resp)


# Serve static frontend and index
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not built yet: missing index.html")
    return FileResponse(str(index_path))


if __name__ == "__main__":
    # Run: python webui/main.py
    import sys
    import uvicorn

    # Ensure repo root is importable so 'webui.main:app' works under the reloader
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    uvicorn.run(
        "webui.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(REPO_ROOT)],
    )