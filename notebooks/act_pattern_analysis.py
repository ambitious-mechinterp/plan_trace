# %%

import requests

url = "https://www.neuronpedia.org/api/activation/get"

payload = {
    "modelId": "gemma-2-2b",
    "source": "4-gemmascope-mlp-16k",
    "index": "9586"
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print(response.status_code)
print(response.json())  # or response.text if not JSON

# %%
out_dict = response.json()
print(out_dict[0]['tokens'])

# %%

"".join(out_dict[0]['tokens']).replace("\u2581", " ")
# %%
out_dict[0]['values']
# %%
ym = ","

for ind, val in enumerate(out_dict[0]['values']):
    if val > 0 and ind > 3:
        ppt = "".join(out_dict[0]['tokens'][ind-3:ind+3]).replace("\u2581", " ")
        if ym in ppt:
            print(ppt)


# %%