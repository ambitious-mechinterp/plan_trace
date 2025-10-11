"""
Goal: make an html or something for circuit display along with evidence of planning. 

# ui inputs 

- prompt_id 
- yn_ind (current token being predicted)


# files to get 

- clusters: /work/pi_jensen_umass_edu/jnainani_umass_edu/plan_trace/outputs/topkfile/prompt_{prompt_id}/token_{yn_ind}/clusters.json
- planning_analysis: /work/pi_jensen_umass_edu/jnainani_umass_edu/plan_trace/outputs/topkfile/prompt_{prompt_id}/token_{yn_ind}/planning_analysis.json
- steering_results: /work/pi_jensen_umass_edu/jnainani_umass_edu/plan_trace/outputs/topkfile/prompt_{prompt_id}/token_{yn_ind}/steering_results.json

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
OUTPUTS_DIR = REPO_ROOT / "outputs" / "topkfile"


def _safe_read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _build_data_response(
    prompt_id: int,
    yn_ind: int,
) -> Dict[str, Any]:
    base_dir = OUTPUTS_DIR / f"prompt_{prompt_id}" / f"token_{yn_ind}"

    clusters_path = base_dir / "clusters.json"
    planning_path = base_dir / "planning_analysis.json"
    steering_path = base_dir / "steering_results.json"

    clusters = _safe_read_json(clusters_path)
    planning = _safe_read_json(planning_path)
    steering = _safe_read_json(steering_path)

    # Compute helpful metadata and an index structure for quick lookup
    meta: Dict[str, Any] = {
        "hasClusters": clusters is not None,
        "hasPlanning": planning is not None,
        "hasSteering": steering is not None,
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
        meta["tokenIds"] = sorted(unique_token_ids)

    return {
        "ok": True,
        "paths": {
            "clusters": str(clusters_path),
            "planning": str(planning_path),
            "steering": str(steering_path),
        },
        "clusters": clusters,
        "planning": planning,
        "steering": steering,
        "meta": meta,
        "index": index,
    }


app = FastAPI(title="Plan Trace UI")


@app.get("/api/data")
def get_data(
    prompt_id: int = Query(..., description="Prompt ID, e.g., 15"),
    yn_ind: int = Query(..., description="Token index (current token), e.g., 293"),
) -> JSONResponse:
    resp = _build_data_response(prompt_id=prompt_id, yn_ind=yn_ind)
    if not (resp["meta"]["hasClusters"] or resp["meta"]["hasPlanning"] or resp["meta"]["hasSteering"]):
        raise HTTPException(status_code=404, detail="No data files found for the given prompt_id and yn_ind")
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