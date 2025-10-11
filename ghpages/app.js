/* globals window, document, fetch */

const state = { data: null, selected: { layer: null, tokenId: null, ym: null, latent: null } };

const YM_PALETTE = ['#66c2ff','#6bff9c','#ff6b6b','#ffd166','#b28dff','#8dd3c7','#80b1d3','#fdb462','#b3de69','#fccde5','#bc80bd','#ccebc5','#fb8072','#bebada','#ffffb3'];
function colorForYm(ymKey){ let h=0; for(let i=0;i<ymKey.length;i++) h=(h*31+ymKey.charCodeAt(i))>>>0; return YM_PALETTE[h%YM_PALETTE.length]; }

function el(tag, attrs={}, children=[]) { const node=document.createElement(tag); Object.entries(attrs).forEach(([k,v])=>{ if(k==='class') node.className=v; else if(k==='text') node.textContent=v; else if(k.startsWith('on')&&typeof v==='function') node.addEventListener(k.slice(2), v); else node.setAttribute(k,v); }); for(const c of children) node.append(c); return node; }

function setStatus(msg, isError=false){ const s=document.getElementById('status'); s.textContent=msg||''; s.className='status'+(isError?' error':''); }

async function readJson(path){ const res=await fetch(path); if(!res.ok) throw new Error(`Failed to fetch ${path}: ${res.status}`); return res.json(); }

// Build data similar to FastAPI endpoint from raw files placed under ghpages/data/
async function buildData(promptId, ynInd){
  const baseDir = `data/topkfile/prompt_${promptId}/token_${ynInd}`;
  const clusters = await safe(readJson(`${baseDir}/clusters.json`));
  const planning = await safe(readJson(`${baseDir}/planning_analysis.json`));
  const steering = await safe(readJson(`${baseDir}/steering_results.json`));
  const metadata = await safe(readJson(`${baseDir}/metadata.json`));
  const tokenMap = await safe(readJson(`data/prompt_tokenized_map.json`));

  const meta = { hasClusters: !!clusters, hasPlanning: !!planning, hasSteering: !!steering, hasMetadata: !!metadata, hasTokenized: false, yms: [], layers: [], tokenIds: [] };

  const index = {}; const uniqueLayers=new Set(); const uniqueTokenIds=new Set();
  if (clusters && typeof clusters==='object'){
    const yms=Object.keys(clusters); meta.yms=yms;
    for(const ym of yms){
      const entries = clusters[ym]||[]; const ymMap={};
      for(const entry of entries){
        if(Array.isArray(entry)&&entry.length===3&&Number.isInteger(entry[0])&&Number.isInteger(entry[1])&&Array.isArray(entry[2])){
          const layerInd=entry[0]; const latentInd=entry[1]; const tokenIds=entry[2].filter((t)=>Number.isInteger(t));
          uniqueLayers.add(layerInd);
          for(const t of tokenIds){ uniqueTokenIds.add(t); const key=`${layerInd}|${t}`; (ymMap[key]||(ymMap[key]=[])).push(latentInd); }
        }
      }
      index[ym]=ymMap;
    }
    meta.layers=[...uniqueLayers].sort((a,b)=>a-b);
    const maxInClusters = uniqueTokenIds.size? Math.max(...uniqueTokenIds): -1;
    const maxToken = Math.max(ynInd, maxInClusters);
    meta.tokenIds = Array.from({length: maxToken+1}, (_,i)=>i);
  }

  let tokens_input=[], tokens_baseline=[];
  try{
    const pKey=String(promptId); const tKey=String(ynInd);
    if(tokenMap && tokenMap[pKey] && tokenMap[pKey].token_results){
      const entry = tokenMap[pKey].token_results[tKey];
      if(entry){
        const inp = Array.isArray(entry.input_prefix_token_strings)? entry.input_prefix_token_strings: [];
        const base = Array.isArray(entry.baseline_token_strings)? entry.baseline_token_strings: [];
        tokens_input = inp.slice(1).map(String); // keep one BOS overall
        tokens_baseline = base.slice(1).map(String); // drop BOS entirely
      }
    }
  }catch(_e){}
  if(meta.tokenIds && meta.tokenIds.length) tokens_input = tokens_input.slice(0, meta.tokenIds.length);
  if(tokens_input.length || tokens_baseline.length) meta.hasTokenized=true;

  return { ok:true, paths:{ baseDir }, clusters, planning, steering, metadata, tokens:{ input: tokens_input, baseline: tokens_baseline }, meta, index };
}

async function safe(p){ try{ return await p; } catch(_e){ return null; } }

function renderAxes(meta){ const x=document.getElementById('x-axis-inner'); x.innerHTML=''; (meta.tokenIds||[]).forEach((t)=>x.append(el('span',{class:'token',text:String(t)}))); const y=document.getElementById('y-axis-inner'); y.innerHTML=''; (meta.layers||[]).forEach((l)=>y.append(el('div',{class:'layer',text:String(l)}))); }

function renderLegend(yms){ const legend=document.getElementById('legend'); legend.innerHTML=''; if(!Array.isArray(yms)||!yms.length) return; yms.forEach((ym)=>{ const dot=el('span',{class:'dot'}); dot.style.background=colorForYm(ym); legend.append(el('span',{},[dot, el('span',{text:ym})])); }); }

function openPopover({layer, tokenId, ym}){
  state.selected.layer=layer; state.selected.tokenId=tokenId; state.selected.ym=ym;
  const title=document.getElementById('viewer-title'); const ymSelect=document.getElementById('ym-select'); const latentSelect=document.getElementById('latent-select');
  title.textContent=`Layer ${layer} · token ${tokenId}`;
  ymSelect.innerHTML=''; (state.data.meta.yms||[]).forEach((k)=>{ const opt=el('option',{value:k,text:k}); if(k===ym) opt.selected=true; ymSelect.append(opt); });
  function setIframe(layerIdx, latentIdx){ const container=document.getElementById('neuronpedia'); container.innerHTML=''; if(layerIdx==null||latentIdx==null){ container.append(el('div',{class:'muted',text:'No latent selected'})); return; } const src=`https://www.neuronpedia.org/gemma-2-2b/${layerIdx}-gemmascope-mlp-16k/${latentIdx}?embed=true&embedexplanation=true&embedplots=true&embedtest=false`; const iframe=el('iframe',{src, title:'Neuronpedia', style:'height: 500px; width: 100%; border:0;'}); container.append(iframe); }
  function populateLatentSelect(ymKey){ const key=`${layer}|${tokenId}`; const list=(state.data.index[ymKey] && state.data.index[ymKey][key])||[]; latentSelect.innerHTML=''; if(!list.length){ latentSelect.append(el('option',{value:'',text:'No latents'})); setIframe(null,null); return; } list.forEach((latent)=>latentSelect.append(el('option',{value:String(latent), text:String(latent)}))); const first=list[0]; state.selected.latent=first; latentSelect.value=String(first); setIframe(layer, first); }
  ymSelect.onchange=()=>{ state.selected.ym=ymSelect.value; populateLatentSelect(state.selected.ym); };
  latentSelect.onchange=()=>{ const val=latentSelect.value? Number(latentSelect.value): null; state.selected.latent=val; setIframe(layer, val); };
  populateLatentSelect(ym);
}

function renderGrid(data){ const grid=document.getElementById('grid'); grid.innerHTML=''; const tokenIds=data.meta.tokenIds||[]; const layers=data.meta.layers||[]; layers.forEach((layer)=>{ const row=el('div',{class:'cell-row'}); tokenIds.forEach((t)=>{ const cell=el('div',{class:'cell','data-layer':String(layer),'data-token':String(t)}); let hasAny=false; (data.meta.yms||[]).forEach((ymKey)=>{ const key=`${layer}|${t}`; const latents=(data.index[ymKey] && data.index[ymKey][key])||[]; if(latents.length){ hasAny=true; const dot=el('div',{class:'cluster', title:`${ymKey} • ${latents.length} latents`}); dot.style.background=colorForYm(ymKey); dot.addEventListener('click',()=>{ openPopover({layer, tokenId:t, ym:ymKey}); highlightInputToken(t); }); cell.append(dot);} }); if(!hasAny) cell.append(el('div',{class:'cluster empty', title:'No latents'})); row.append(cell); }); grid.append(row); }); }

function renderPlanning(planning){ const c=document.getElementById('planning'); c.innerHTML=''; if(!planning){ c.append(el('div',{class:'muted',text:'No planning data'})); return; } c.append(el('h3',{text:'Planning labels'})); Object.entries(planning).forEach(([ym,label])=>{ c.append(el('div',{class:'ym'},[ el('strong',{text:ym}), el('span',{class:'muted', text:` – ${String(label)}`}) ])); }); }

function renderSteering(steering){ const c=document.getElementById('steering'); c.innerHTML=''; if(!steering){ c.append(el('div',{class:'muted',text:'No steering results'})); return; } c.append(el('h3',{text:'Steering results'})); Object.entries(steering).forEach(([ym,payload])=>{ const box=el('div',{class:'ym'}); box.append(el('strong',{text:ym})); if(payload && Array.isArray(payload.steered)){ payload.steered.forEach((entry)=>{ const coeff=entry.coeff; const decoded=entry.decoded_text; if(!decoded||decoded.length===0) return; box.append(el('div',{class:'steer'},[ el('div',{class:'coeff', text:`coeff: ${String(coeff)}`}), el('div',{text:decoded}) ])); }); } if(box.children.length>1) c.append(box); }); }

function renderIO(fullData){ const strip=document.getElementById('input-strip'); const baselineStrip=document.getElementById('baseline-strip'); if(!strip||!baselineStrip) return; strip.innerHTML=''; baselineStrip.innerHTML=''; const inputTokens=(fullData && fullData.tokens && Array.isArray(fullData.tokens.input))? fullData.tokens.input: null; if(inputTokens && inputTokens.length){ inputTokens.forEach((tok,idx)=>{ const tokenEl=el('span',{class:'input-token', text:String(tok)}); tokenEl.dataset.tokenId=String(idx); strip.append(tokenEl); }); } else { const tokenIds=(fullData && fullData.meta && fullData.meta.tokenIds)||[]; tokenIds.forEach((tid)=>{ const tokenEl=el('span',{class:'input-token', text:String(tid)}); tokenEl.dataset.tokenId=String(tid); strip.append(tokenEl); }); }
  const baselineTokens=(fullData && fullData.tokens && Array.isArray(fullData.tokens.baseline))? fullData.tokens.baseline: []; baselineTokens.forEach((tok)=> baselineStrip.append(el('span',{class:'baseline-token', text:String(tok)})));
}

function hydrate(data){ state.data=data; renderAxes(data.meta||{}); renderLegend((data.meta&&data.meta.yms)||[]); renderGrid(data); renderPlanning(data.planning||null); renderSteering(data.steering||null); renderIO(data); setupScrollSync(); }

function setupScrollSync(){ const grid=document.getElementById('grid'); const xAxis=document.getElementById('x-axis-inner'); const yAxis=document.getElementById('y-axis-inner'); if(!grid||!xAxis||!yAxis) return; function sync(){ xAxis.style.transform=`translateX(${-grid.scrollLeft}px)`; yAxis.style.transform=`translateY(${-grid.scrollTop}px)`; } grid.removeEventListener('scroll', sync); grid.addEventListener('scroll', sync, { passive:true }); sync(); }

async function onSubmit(e){ e.preventDefault(); const promptId=Number(document.getElementById('prompt-id').value); const ynInd=Number(document.getElementById('yn-ind').value); if(Number.isNaN(promptId)||Number.isNaN(ynInd)){ setStatus('Invalid inputs', true); return; } setStatus('Loading...'); try{ const data=await buildData(promptId, ynInd); if(!(data.meta.hasClusters||data.meta.hasPlanning||data.meta.hasSteering||data.meta.hasMetadata||data.meta.hasTokenized)){ throw new Error('No data files found'); } hydrate(data); setStatus('Loaded'); const url=new URL(window.location.href); url.searchParams.set('prompt_id', String(promptId)); url.searchParams.set('yn_ind', String(ynInd)); window.history.replaceState({},'',url.toString()); } catch(err){ console.error(err); setStatus('Failed to load data', true); } }

function initFormFromQuery(){ const params=new URLSearchParams(window.location.search); const promptId=params.get('prompt_id'); const ynInd=params.get('yn_ind'); if(promptId) document.getElementById('prompt-id').value=Number(promptId); if(ynInd) document.getElementById('yn-ind').value=Number(ynInd); }

function highlightInputToken(tokenId){ const strip=document.getElementById('input-strip'); if(!strip) return; strip.querySelectorAll('.input-token').forEach((elx)=> elx.classList.remove('active')); const elx=strip.querySelector(`.input-token[data-token-id="${String(tokenId)}"]`); if(elx) elx.classList.add('active'); }

function main(){ document.getElementById('query-form').addEventListener('submit', onSubmit); initFormFromQuery(); const pid=document.getElementById('prompt-id').value; const ti=document.getElementById('yn-ind').value; if(pid && ti){ document.getElementById('query-form').dispatchEvent(new Event('submit')); } }

window.addEventListener('DOMContentLoaded', main);


