# Plan Trace UI - GitHub Pages Static Build

This folder contains a standalone static version of the UI that can be hosted on GitHub Pages.

How it works
- No backend. The JS code fetches JSON directly from the `data/` directory.
- Expected data layout (copy from your repo outputs):
  - `data/topkfile/prompt_<PROMPT_ID>/token_<YN_IND>/clusters.json`
  - `data/topkfile/prompt_<PROMPT_ID>/token_<YN_IND>/planning_analysis.json`
  - `data/topkfile/prompt_<PROMPT_ID>/token_<YN_IND>/steering_results.json`
  - `data/topkfile/prompt_<PROMPT_ID>/token_<YN_IND>/metadata.json`
  - `data/prompt_tokenized_map.json`

Usage
1. Copy the needed files from your repo into `ghpages/data/` following the structure above.
2. Push the `ghpages/` directory to a GitHub repository configured with Pages (root or `docs/`).
3. If deploying from a subpath (e.g., `https://<user>.github.io/<repo>/`), the relative paths are already local (`app.js`, `styles.css`, `data/...`).

Notes
- Large JSONs are fine but GitHub has file size limits; split or compress externally if needed.
- The UI mirrors the FastAPI behavior: it builds the same composite response on the client.
- Neuronpedia content is embedded via iframe and works on GitHub Pages.


