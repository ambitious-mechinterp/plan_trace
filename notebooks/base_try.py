# %%
import requests
import json 
import os 
import sys 
import math
import html
import numpy as np
import torch 
from torch import Tensor
from typing import Dict, Sequence, Tuple, Union
from IPython.display import HTML, display

sys.path.append("../")

from plan_trace.utils import load_model, load_pretrained_saes, cleanup_cuda
from plan_trace.hooks import run_with_saes, register_sae_hooks
from sae_lens import HookedSAETransformer, SAE

# %%

model_name = "gemma-2-2b"
device = "cuda"
model = load_model(model_name, device=device, use_custom_cache=True, dtype=torch.bfloat16)
# layers = list(range(model.cfg.n_layers))
# saes = load_pretrained_saes(
#     layers=layers, 
#     release="gemma-scope-2b-pt-mlp-canonical", 
#     width="16k", 
#     device=device, 
#     canon=True
# )
# %%
model_it = load_model("gemma-2-2b-it", device=device, use_custom_cache=True, dtype=torch.bfloat16)


# %%
input_text = "You are an expert Python programmer, and here is your task: Write a python function to find the largest number that can be formed with the given list of digits. No docstring. Your code should pass these tests:\n\nassert find_Max_Num([1,2,3]) == 321\nassert find_Max_Num([4,5,6,1]) == 6541\nassert find_Max_Num([1,2,3,9]) == 9321\nWrite your code below starting with \"```python\" and ending with \"```\".\n```python\n"
resp = model.generate(input_text, max_new_tokens=20, temperature=0.0)
print(resp)
# %%
resp = model.generate(input_text, max_new_tokens=20, temperature=0.0)
print(resp)

# %%

input_prefix_token_strings = [
          "You",
          " are",
          " an",
          " expert",
          " Python",
          " programmer",
          ",",
          " and",
          " here",
          " is",
          " your",
          " task",
          ":",
          " Write",
          " a",
          " python",
          " function",
          " to",
          " find",
          " the",
          " largest",
          " number",
          " that",
          " can",
          " be",
          " formed",
          " with",
          " the",
          " given",
          " list",
          " of",
          " digits",
          ".",
          " Your",
          " code",
          " should",
          " pass",
          " these",
          " tests",
          ":",
          "\n\n",
          "assert",
          " find",
          "_",
          "Max",
          "_",
          "Num",
          "([",
          "1",
          ",",
          "2",
          ",",
          "3",
          "])",
          " ==",
          " ",
          "3",
          "2",
          "1",
          "\n",
          "assert",
          " find",
          "_",
          "Max",
          "_",
          "Num",
          "([",
          "4",
          ",",
          "5",
          ",",
          "6",
          ",",
          "1",
          "])",
          " ==",
          " ",
          "6",
          "5",
          "4",
          "1",
          "\n",
          "assert",
          " find",
          "_",
          "Max",
          "_",
          "Num",
          "([",
          "1",
          ",",
          "2",
          ",",
          "3",
          ",",
          "9",
          "])",
          " ==",
          " ",
          "9",
          "3",
          "2",
          "1",
          "\n",
          "Write",
          " your",
          " code",
          " below",
          " starting",
          " with",
          " \"",
          "```",
          "python",
          "\"",
          " and",
          " ending",
          " with",
          " \"",
          "```",
          "\".",
          "\n",
          "```",
          "python",
          "\n",
          "def",
          " find",
          "_",
          "Max",
          "_",
          "Num",
          "(",
          "digits",
          "):",
          "\n",
          "    ",
          "\"\"\"",
          "\n",
          "    ",
          "Find",
          "s",
          " the",
          " largest",
          " number",
          " that",
          " can",
          " be",
          " formed",
          " with",
          " the",
          " given",
          " list",
          " of",
          " digits",
          ".",
          "\n\n",
          "    ",
          "Args",
          ":",
          "\n",
          "        ",
          "digits",
          ":",
          " A",
          " list",
          " of",
          " digits",
          ".",
          "\n\n",
          "    ",
          "Returns",
          ":",
          "\n",
          "        ",
          "The",
          " largest",
          " number",
          " that",
          " can",
          " be",
          " formed",
          " with",
          " the",
          " given",
          " digits",
          ".",
          "\n",
          "    ",
          "\"\"\"",
          "\n",
          "    "
        ]

docstring_input = "".join(input_prefix_token_strings)
print(docstring_input)
# %%
resp_with_docstring = model.generate(docstring_input, max_new_tokens=20, temperature=0.0)
print(resp_with_docstring)
# %%
resp_with_docstring = model_it.generate(docstring_input, max_new_tokens=20, temperature=0.0)
print(resp_with_docstring)
# %%
