# for a given prompt, 
# we want to iterate through each forward pass
# discover circuit with >60% perf recovery
# cluster based on decoding directions logit lens 
# test steering effect of clusters
# test effect of each latent / position in cluster

model = ""
saes = ""

prompt = ""



# (layer, latent, tok, effect)
circuit = discover_circuit() 

# {ym : (layer, latent, tok) 
logit_lens_cluster = get_ll_clusters()

steered_clusters = test_steering()

planning_positions = test_steering()
