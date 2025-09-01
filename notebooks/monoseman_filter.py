# %%
"""
for a set of latents in the same layer, 

1. calc the logit lens tokens in the same way as the logit lens clustering function
2. find cosine sim between the top k logit lens tokens for each latent
3. drop latents with cosine sim < 0.5

example case: token 180, task 24, ym "2"

"""

# %%

import json 
import os 
import sys 
import torch 

sys.path.append("../")

from plan_trace.pipeline import run_single_token_analysis
from plan_trace.utils import load_model, load_pretrained_saes, cleanup_cuda
from plan_trace.circuit_discovery import discover_circuit
from plan_trace.logit_lens import find_logit_lens_clusters, gather_unique_tokens
from plan_trace.steering import run_steering_sweep

# %%
cluster_path = "../outputs/prompt_-1/token_180/clusters.json"
with open(cluster_path, "r") as f:
    cluster_dict = json.load(f)
# %%
cluster_dict["2"]
# %%
from tqdm import tqdm 
from collections import defaultdict
from typing import Dict


layer_to_latents: Dict[int, set[int]] = defaultdict(set)
for l, lat, toks in cluster_dict["2"]:
    layer_to_latents[l].add(lat)

# %%
model_name = "gemma-2-2b-it"
device = "cuda"
model = load_model(model_name, device=device, use_custom_cache=True, dtype=torch.bfloat16)
layers = list(range(model.cfg.n_layers))
saes = load_pretrained_saes(
    layers=layers, 
    release="gemma-scope-2b-pt-mlp-canonical", 
    width="16k", 
    device=device, 
    canon=True
)

# %%
layer_i = 24 
with torch.no_grad():
    W_U = model.W_U.float().to(device)  # [D_model, V]
    W_dec = saes[layer_i].W_dec.to(device)  # [S, D_model]
    dirs = W_dec[list(layer_to_latents[layer_i])]            # [batch_size, D_model]
    logits = dirs @ W_U            # [batch_size, V]
    topk_scores, topk_idx = torch.topk(logits, 15, dim=1)  # [batch_size, tok_k_pos_logits]
    print(topk_idx.shape)



# %%

def cohesion(topk_ids, EMB):
    """Calculate cohesion score for a list of token IDs using cosine similarity."""
    if len(topk_ids) < 2:
        return 0.0
    vecs = EMB[topk_ids]                              # [K, d_model] 
    μ = vecs.mean(dim=0, keepdim=True)               # [1, d_model] - centroid
    sim = torch.nn.functional.cosine_similarity(vecs, μ, dim=1)  # [K] - similarity to centroid
    return sim.mean().item()                         # Average similarity (1 = perfectly cohesive)

#%%

for ind, lat in enumerate(topk_idx):
    print(ind)
    print(cohesion(lat, EMB = model.W_E ))
    print("-"*100)

# %%
model.to_tokens("2")[:, -1]

# %%
from typing import List, Tuple
# Config for testing scoring/thresholding
tok_k_pos_logits = 15
batch_size = 4096
score_threshold = 0.5       # set None to disable thresholding
count_weight = 0.03         # weight for match count
min_match_count = 1         # minimum number of unique-token matches in top-k
cohesion_mode = "spherical" # one of: "spherical", "pairwise", "centroid"
saved_pair_dict: Dict[str, List[Tuple[int, int, List[int]]]] = defaultdict(list)
unique_token_tensor = torch.tensor([model.to_tokens("2")[:, -1].tolist()], device=device)
with torch.no_grad():
    W_U = model.W_U.float().to(device)  # [D_model, V]

    for layer_i, latents in tqdm(layer_to_latents.items()):
        latents = sorted(latents)
        W_dec = saes[layer_i].W_dec.to(device)  # [S, D_model]
        print("-"*100)
        print(layer_i)
        for start in range(0, len(latents), batch_size):
            batch = latents[start : start + batch_size]
            dirs = W_dec[batch]            # [batch_size, D_model]

            logits = dirs @ W_U            # [batch_size, V]
            topk_scores, topk_idx = torch.topk(logits, tok_k_pos_logits, dim=1)  # [batch_size, tok_k_pos_logits]

            # GPU intersection: find which topk tokens match unique_token_ids
            # broadcast: [batch_size, tok_k_pos_logits] vs [1, num_unique_ids]
            topk_idx_exp = topk_idx.unsqueeze(-1)                   # [batch, tok_k_pos_logits, 1]
            unique_tok_exp = unique_token_tensor.view(1, 1, -1)     # [1, 1, num_unique_ids]

            matches = (topk_idx_exp == unique_tok_exp).any(-1)      # [batch, tok_k_pos_logits] -> True/False
            matching_token_ids = topk_idx[matches]                  # flatten matches

            # Compute per-latent cohesion over top-k token embeddings
            token_embs = model.W_E.float().to(device)[topk_idx]      # [batch_size, tok_k_pos_logits, d_model]
            if cohesion_mode == "spherical":
                # Normalize embeddings onto the unit sphere and use resultant vector length (R)
                unit = torch.nn.functional.normalize(token_embs, dim=-1)
                resultant = unit.sum(dim=1)                          # [batch_size, d_model]
                cohesion_per_latent = resultant.norm(dim=-1) / tok_k_pos_logits   # R in [0,1]
            elif cohesion_mode == "pairwise":
                # Mean pairwise cosine across the K tokens (off-diagonal mean)
                unit = torch.nn.functional.normalize(token_embs, dim=-1)
                # cosine matrix per batch: [batch, K, K]
                cos_mat = unit @ unit.transpose(1, 2)
                # subtract diagonal ones and average off-diagonals
                K = tok_k_pos_logits
                off_diag_sum = cos_mat.sum(dim=(1,2)) - K
                cohesion_per_latent = off_diag_sum / (K * (K - 1))
            else:  # "centroid" (legacy): cosine to mean vector
                centroid = token_embs.mean(dim=1, keepdim=True)       # [batch_size, 1, d_model]
                cos_per_token = torch.nn.functional.cosine_similarity(token_embs, centroid, dim=-1)  # [batch_size, tok_k_pos_logits]
                cohesion_per_latent = cos_per_token.mean(dim=1)       # [batch_size]
            # if debug:
            print(batch)
            for toks in topk_idx:
                for tok in toks:
                    print(model.to_string([tok]))
                print("*"*50)
            print(cohesion_per_latent)
            # Count how many of the top-k are in the unique set (per latent)
            match_counts = matches.sum(dim=1)                         # [batch_size]
            # if debug:
            print(match_counts)
            # Final score = cohesion + count_weight * count
            final_scores = cohesion_per_latent + count_weight * match_counts.float()
            #   if debug:
            print(final_scores)
            if matching_token_ids.numel() > 0:
                matched_latents, matched_topk = torch.nonzero(matches, as_tuple=True)
                for latent_batch_idx, topk_pos in zip(matched_latents.tolist(), matched_topk.tolist()):
                    # Thresholding per latent: require min matches and score threshold (if set)
                    if score_threshold is not None:
                        if match_counts[latent_batch_idx].item() < min_match_count:
                            continue
                        if final_scores[latent_batch_idx].item() < score_threshold:
                            continue
                    latent_idx_in_batch = batch[latent_batch_idx]
                    matched_token_id = topk_idx[latent_batch_idx, topk_pos].item()
                    label_str = model.to_string([matched_token_id]).strip()

                    if not label_str:
                        continue

                    matching_toks = [
                        toks for l, lat, toks in cluster_dict["2"]
                        if l == layer_i and lat == latent_idx_in_batch
                    ]
                    saved_pair_dict[label_str].append(
                        (layer_i, latent_idx_in_batch, matching_toks)
                    )



# %%
print(cluster_dict["2"])
# %%
[[20, 5990, [116, 178]], [20, 16136, [150]], [22, 2456, [147, 93]], [22, 15849, [116, 165]], [24, 2765, [116]], [24, 13682, [116, 49, 176, 99, 165, 24, 119]], [23, 6768, [91, 116, 34, 146]], [23, 6832, [179, 175, 93, 34, 178]], [21, 2020, [116, 165, 49, 172]], [17, 8604, [172]], [7, 14962, [56, 69]], [1, 7459, [33, 92]], [1, 11380, [153, 159]], [1, 12443, [23, 111, 149, 146, 18, 6, 89, 90, 29, 156, 76]], [0, 41, [47, 32, 28, 37, 30, 94, 29, 109]], [0, 601, [99, 22, 110, 94, 48, 151, 72, 126, 47, 35, 141, 61, 16, 150, 105, 34, 104, 179, 169, 10, 26, 20, 175, 18, 109, 119, 148, 28, 178, 83, 140, 114, 127, 156, 133, 118]], [0, 1024, [178, 92, 89, 5, 6, 146, 90, 4, 179, 162, 175, 149, 87, 80, 8]], [0, 5199, [42]], [0, 5889, [167, 154]], [0, 8959, [24]], [0, 8984, [33]], [0, 10409, [22, 93, 110, 21]], [2, 4478, [34]], [2, 14333, [23, 94, 96, 22, 92, 122]], [3, 9918, [24, 20, 111, 159, 16, 166, 19, 14, 72]], [6, 7540, [23, 96]], [6, 7576, [179, 22, 21]], [6, 11679, [98, 22, 108, 96, 20, 21, 111, 142, 16, 17, 37, 93, 139, 122, 48]], [5, 686, [1]], [5, 4698, [178, 169]]]

----------------------------------------------------------------------------------------------------
20
[5990, 16136]
,
 in
 de
 for
1
によっては
 dis
2
 (
 to
it
a
 of
in
 from
**************************************************
 оригіналу
2
 Six
 ivelany
 two
IMPORTED
六
Földrajzportál
 nhàng
two
geist
二维
mann
бурга
дву
**************************************************
tensor([0.6660, 0.3581], device='cuda:0')
tensor([1, 1], device='cuda:0')
tensor([0.6960, 0.3881], device='cuda:0')
----------------------------------------------------------------------------------------------------
22
[2456, 15849]
 





,
<eos>
 (
.
  
-
1
'
...
2
 -
 I
**************************************************
8
八
 eight
 oito
2
1
 eighth
eigh
 nine
 Eighteen
 八
 eighties
9
zkopf
 Eight
**************************************************
tensor([0.7865, 0.5018], device='cuda:0')
tensor([1, 1], device='cuda:0')
tensor([0.8165, 0.5318], device='cuda:0')
----------------------------------------------------------------------------------------------------
24
[2765, 13682]


 
0
p
</strong>
1
op
2
-
3
5



<strong>
</h2>
</em>
**************************************************
 


,



.
 (
<eos>
  
1
 I
…
...
2
-
/
**************************************************
tensor([0.7224, 0.7790], device='cuda:0')
tensor([1, 1], device='cuda:0')
tensor([0.7524, 0.8090], device='cuda:0')
----------------------------------------------------------------------------------------------------
23
[6768, 6832]
.
-
,
 
1
2
f
3
si
k
me
c
w
<eos>
4
**************************************************
<bos>
.
-
  
2
_
!
3
1


j



P
...
(
**************************************************
tensor([0.7396, 0.7145], device='cuda:0')
tensor([1, 1], device='cuda:0')
tensor([0.7696, 0.7445], device='cuda:0')
----------------------------------------------------------------------------------------------------
21
[2020]
 
<eos>


1
2



,
<strong>
 (
 O
</strong>
 B
<unused61>
3
<unused63>
**************************************************
tensor([0.7214], device='cuda:0')
tensor([1], device='cuda:0')
tensor([0.7514], device='cuda:0')
----------------------------------------------------------------------------------------------------
17
[8604]
st
3
2
ST
XDECREF
Active
 r
0
 I
 g
 EventArgs
 c
ap
 judiciaire
5
**************************************************
tensor([0.4597], device='cuda:0')
tensor([1], device='cuda:0')
tensor([0.4897], device='cuda:0')
----------------------------------------------------------------------------------------------------
7
[14962]
zustellen
2
 ^{-
 MatDialog
5
gestellt
jScrollPane
1
nivers
layoutParams
urto
udesta
ntö
ming
urals
**************************************************
tensor([0.3620], device='cuda:0')
tensor([1], device='cuda:0')
tensor([0.3920], device='cuda:0')
----------------------------------------------------------------------------------------------------
1
[7459, 11380, 12443]


 
<eos>
 v



2
/
...
 my
…
id
 i
 go
 I
-
**************************************************
DockStyle
 awakeFromNib
 مرئيه
2
 StringTokenizer
1
 tartalomajánló
 "..\..\..\
principalTable
 noqa
 viewDidLoad
 createStore
 bezeichneter
StructEnd
0
**************************************************
 kasarigan
EndProject
 resourceCulture
1
GEBURTSDATUM
sta



z
2
es
<eos>
 es
  
 din
й
**************************************************
tensor([0.6955, 0.4171, 0.3491], device='cuda:0')
tensor([1, 1, 1], device='cuda:0')
tensor([0.7255, 0.4471, 0.3791], device='cuda:0')
----------------------------------------------------------------------------------------------------
0
[41, 601, 1024, 5199, 5889, 8959, 8984, 10409]
The
1
 }}$}
<i>
 latter
2
5
<strong>
 invi
 relationship
')]
 ')
Traits
3
mybatisplus
**************************************************


2
1
 uiteindelijk
.
 tampoco
toyage
stalgia
 verità
 nonetheless
 in
vlád
less
 and
indest
**************************************************
'





1
A
2
(
4
3
{
v
5
S
i
s
**************************************************
 Diſ
 Reſ
 ―――――
 Monfieur
 Anſ
 Perſ
 Inſ
2
 Houſe
{\
保
(\
 NDEBUG
न
 experien
**************************************************
2
0
1
4
5
$
帯
3
 time
 minutes
hora
6
 hour
8
horas
**************************************************
<bos>
2
 various
 the
GEBURTS
UnifiedTopology
 Logement
 cal
3
0
 pri
 ac
 resourceCulture
 they
inheritDoc
**************************************************
1
2
3
 greateſt
8
 espagnole
 Chriftian
7
9
5
قایناقلار
 MainAxisSize
 Efq
4
 Encuentra
**************************************************
меч
1
oredCriteria
 Hiller
ukunft
2
gjenge
 mør
arande
 Gilli
inck
犷
 figliu
MethodType
kå
**************************************************
tensor([0.3308, 0.3034, 0.7539, 0.5347, 0.5507, 0.2890, 0.3418, 0.4163],
       device='cuda:0')
tensor([1, 1, 1, 1, 1, 1, 1, 1], device='cuda:0')
tensor([0.3608, 0.3334, 0.7839, 0.5647, 0.5807, 0.3190, 0.3718, 0.4463],
       device='cuda:0')
----------------------------------------------------------------------------------------------------
2
[4478, 14333]
4
5
3
1
2
6
7
0
8
subsection
9
일에
 imparare
UNRELATED
 geb
**************************************************
1
atorics
,
gms
 =
-
大
飽
多
 GMA



併
2
(
#
**************************************************
tensor([0.5808, 0.4155], device='cuda:0')
tensor([1, 1], device='cuda:0')
tensor([0.6108, 0.4455], device='cuda:0')
----------------------------------------------------------------------------------------------------
3
[9918]
'





’
{
.
‘
1
(
"
 
 '
2
//
4
**************************************************
tensor([0.7459], device='cuda:0')
tensor([1], device='cuda:0')
tensor([0.7759], device='cuda:0')
----------------------------------------------------------------------------------------------------
6
[7540, 7576, 11679]
2
 II
1
ਤੇ
oporosis
Bands
GeneratedMessage
baselines
 Two
 ska
 conflicts
II
flikt
径
jScrollPane
**************************************************



 


1




  
 ‘
 “
 '
.
 "
2
 cell
ibley
                               
**************************************************
<bos>
<eos>



-
 
At
2
'
Since
 deres
at
log
.
if
Below
**************************************************
tensor([0.3327, 0.6451, 0.5413], device='cuda:0')
tensor([1, 1, 1], device='cuda:0')
tensor([0.3627, 0.6751, 0.5713], device='cuda:0')
----------------------------------------------------------------------------------------------------
5
100%|██████████| 13/13 [00:00<00:00, 60.71it/s][686, 4698]
g
 Club
,
G
 ro
0
3
9
2
Club
sp
 club
 Che
ro
6
**************************************************
?
4
.
g
5
2
!
}{
 em
 ma
%
)
-
 for
en
**************************************************
tensor([0.5758, 0.6674], device='cuda:0')
tensor([1, 1], device='cuda:0')
tensor([0.6058, 0.6974], device='cuda:0')