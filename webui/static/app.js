/* globals window, document, fetch */

const state = {
  promptId: null,
  ynInd: null,
  data: null,
  selected: { layer: null, tokenId: null, ym: null, latent: null },
};

// YM color palette (deterministic mapping)
const YM_PALETTE = [
  '#66c2ff', '#6bff9c', '#ff6b6b', '#ffd166', '#b28dff',
  '#8dd3c7', '#80b1d3', '#fdb462', '#b3de69', '#fccde5',
  '#bc80bd', '#ccebc5', '#fb8072', '#bebada', '#ffffb3'
];

function colorForYm(ymKey) {
  let h = 0;
  for (let i = 0; i < ymKey.length; i++) h = (h * 31 + ymKey.charCodeAt(i)) >>> 0;
  return YM_PALETTE[h % YM_PALETTE.length];
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  Object.entries(attrs).forEach(([k, v]) => {
    if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v;
    else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  });
  for (const child of children) node.append(child);
  return node;
}

function setStatus(msg, isError = false) {
  const status = document.getElementById('status');
  status.textContent = msg || '';
  status.className = 'status' + (isError ? ' error' : '');
}

async function fetchData(promptId, ynInd) {
  const url = `/api/data?prompt_id=${encodeURIComponent(promptId)}&yn_ind=${encodeURIComponent(ynInd)}`;
  const res = await fetch(url);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Request failed: ${res.status}`);
  }
  return res.json();
}

function renderAxes(meta) {
  const xAxis = document.getElementById('x-axis-inner');
  xAxis.innerHTML = '';
  const tokenIds = meta.tokenIds || [];
  tokenIds.forEach((t) => {
    xAxis.append(el('span', { class: 'token', text: String(t) }));
  });

  const yAxis = document.getElementById('y-axis-inner');
  yAxis.innerHTML = '';
  (meta.layers || []).forEach((layer) => {
    yAxis.append(el('div', { class: 'layer', text: String(layer) }));
  });
}

// Legend now shows YM-based colors
function renderLegend(yms) {
  const legend = document.getElementById('legend');
  legend.innerHTML = '';
  if (!Array.isArray(yms) || !yms.length) return;
  yms.forEach((ymKey) => {
    const dot = el('span', { class: 'dot' });
    dot.style.background = colorForYm(ymKey);
    legend.append(el('span', {}, [dot, el('span', { text: ymKey })]));
  });
}

function openPopover({ layer, tokenId, ym, latents }) {
  state.selected.layer = layer;
  state.selected.tokenId = tokenId;
  state.selected.ym = ym;
  const pop = document.getElementById('popover');
  const title = document.getElementById('popover-title');
  const ymSelect = document.getElementById('ym-select');
  const latentSelect = document.getElementById('latent-select');

  title.textContent = `Layer ${layer} · token ${tokenId}`;

  // Populate ym select
  ymSelect.innerHTML = '';
  (state.data.meta.yms || []).forEach((ymKey) => {
    const opt = el('option', { value: ymKey, text: ymKey });
    if (ymKey === ym) opt.selected = true;
    ymSelect.append(opt);
  });

  function setIframe(layerIdx, latentIdx) {
    const container = document.getElementById('neuronpedia');
    container.innerHTML = '';
    if (layerIdx == null || latentIdx == null) {
      container.append(el('div', { class: 'muted', text: 'No latent selected' }));
      return;
    }
    const src = `https://www.neuronpedia.org/gemma-2-2b/${layerIdx}-gemmascope-mlp-16k/${latentIdx}?embed=true&embedexplanation=true&embedplots=true&embedtest=false`;
    const iframe = el('iframe', { src, title: 'Neuronpedia', style: 'height: 300px; width: 100%; border:0;' });
    container.append(iframe);
  }

  function populateLatentSelect(ymKey) {
    const key = `${layer}|${tokenId}`;
    const latentsList = (state.data.index[ymKey] && state.data.index[ymKey][key]) || [];
    latentSelect.innerHTML = '';
    if (!latentsList.length) {
      latentSelect.append(el('option', { value: '', text: 'No latents' }));
      setIframe(null, null);
      return;
    }
    latentsList.forEach((latent) => latentSelect.append(el('option', { value: String(latent), text: String(latent) })));
    const first = latentsList[0];
    state.selected.latent = first;
    latentSelect.value = String(first);
    setIframe(layer, first);
  }

  ymSelect.addEventListener('change', () => {
    state.selected.ym = ymSelect.value;
    populateLatentSelect(state.selected.ym);
  });

  latentSelect.addEventListener('change', () => {
    const val = latentSelect.value ? Number(latentSelect.value) : null;
    state.selected.latent = val;
    setIframe(layer, val);
  });

  populateLatentSelect(ym);
  pop.classList.remove('hidden');
}

function closePopover() {
  document.getElementById('popover').classList.add('hidden');
}

function renderGrid(data) {
  const grid = document.getElementById('grid');
  grid.innerHTML = '';

  const tokenIds = data.meta.tokenIds || [];
  const layers = data.meta.layers || [];

  // For each layer, create a row of cells across tokenIds
  layers.forEach((layer) => {
    const row = el('div', { class: 'cell-row' });
    tokenIds.forEach((t) => {
      const cell = el('div', { class: 'cell', 'data-layer': String(layer), 'data-token': String(t) });
      // For each ym, if there are latents for this layer|token, render a small dot
      let hasAny = false;
      (data.meta.yms || []).forEach((ymKey) => {
        const key = `${layer}|${t}`;
        const latents = (data.index[ymKey] && data.index[ymKey][key]) || [];
        if (latents.length) {
          hasAny = true;
          const dot = el('div', { class: 'cluster', title: `${ymKey} • ${latents.length} latents` });
          dot.style.background = colorForYm(ymKey);
          dot.addEventListener('click', () => openPopover({ layer, tokenId: t, ym: ymKey, latents }));
          cell.append(dot);
        }
      });
      if (!hasAny) cell.append(el('div', { class: 'cluster empty', title: 'No latents' }));
      row.append(cell);
    });
    grid.append(row);
  });
}

function renderPlanning(planning) {
  const container = document.getElementById('planning');
  container.innerHTML = '';
  if (!planning) {
    container.append(el('div', { class: 'muted', text: 'No planning data' }));
    return;
  }
  container.append(el('h3', { text: 'Planning labels' }));
  Object.entries(planning).forEach(([ym, label]) => {
    container.append(el('div', { class: 'ym' }, [
      el('strong', { text: ym }),
      el('span', { class: 'muted', text: ` – ${String(label)}` }),
    ]));
  });
}

function renderSteering(steering) {
  const container = document.getElementById('steering');
  container.innerHTML = '';
  if (!steering) {
    container.append(el('div', { class: 'muted', text: 'No steering results' }));
    return;
  }
  container.append(el('h3', { text: 'Steering results' }));
  Object.entries(steering).forEach(([ym, payload]) => {
    const box = el('div', { class: 'ym' });
    box.append(el('strong', { text: ym }));
    if (payload && Array.isArray(payload.steered)) {
      payload.steered.forEach((entry) => {
        const coeff = entry.coeff;
        const decoded = entry.decoded_text;
        box.append(el('div', { class: 'steer' }, [
          el('div', { class: 'coeff', text: `coeff: ${String(coeff)}` }),
          el('div', { text: decoded || '' }),
        ]));
      });
    }
    container.append(box);
  });
}

function hydrate(data) {
  state.data = data;
  renderAxes(data.meta || {});
  renderLegend((data.meta && data.meta.yms) || []);
  renderGrid(data);
  renderPlanning(data.planning || null);
  renderSteering(data.steering || null);
  setupScrollSync();
}

function initFormFromQuery() {
  const params = new URLSearchParams(window.location.search);
  const promptId = params.get('prompt_id');
  const ynInd = params.get('yn_ind');
  if (promptId) document.getElementById('prompt-id').value = Number(promptId);
  if (ynInd) document.getElementById('yn-ind').value = Number(ynInd);
}

async function onSubmit(e) {
  e.preventDefault();
  const promptId = Number(document.getElementById('prompt-id').value);
  const ynInd = Number(document.getElementById('yn-ind').value);
  if (Number.isNaN(promptId) || Number.isNaN(ynInd)) {
    setStatus('Invalid inputs', true);
    return;
  }
  state.promptId = promptId;
  state.ynInd = ynInd;
  setStatus('Loading...');
  try {
    const data = await fetchData(promptId, ynInd);
    hydrate(data);
    setStatus('Loaded');
    const url = new URL(window.location.href);
    url.searchParams.set('prompt_id', String(promptId));
    url.searchParams.set('yn_ind', String(ynInd));
    window.history.replaceState({}, '', url.toString());
  } catch (err) {
    console.error(err);
    setStatus('Failed to load data', true);
  }
}

function main() {
  document.getElementById('query-form').addEventListener('submit', onSubmit);
  document.getElementById('popover-close').addEventListener('click', closePopover);
  initFormFromQuery();
  // Auto-submit if both values present
  const pid = document.getElementById('prompt-id').value;
  const ti = document.getElementById('yn-ind').value;
  if (pid && ti) {
    document.getElementById('query-form').dispatchEvent(new Event('submit'));
  }
}

function setupScrollSync() {
  const grid = document.getElementById('grid');
  const xAxis = document.getElementById('x-axis-inner');
  const yAxis = document.getElementById('y-axis-inner');
  if (!grid || !xAxis || !yAxis) return;
  function sync() {
    xAxis.style.transform = `translateX(${-grid.scrollLeft}px)`;
    yAxis.style.transform = `translateY(${-grid.scrollTop}px)`;
  }
  grid.removeEventListener('scroll', sync);
  grid.addEventListener('scroll', sync, { passive: true });
  // Initialize once
  sync();
}

window.addEventListener('DOMContentLoaded', main);


