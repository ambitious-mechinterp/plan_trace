# Achievements

- finite differences step by step debugging
- the call to run_with_saes was not recording the sae activations, so delta was always 0
- tested a zero ablation based edge ranking that tests one upstream to all downstream 

# Next Steps

- Check if we should do perturb (alpha addition) or zero ablation for finite differences (both are confirmed working but whats more principled?)
- Check if jvp is working now (it should be faster than iterating over upstream latents) 
- In the circuit performance recovery, we test both top abs components and top negative components, should do the same for edges
- do all up and downstream layer combinations, zero ablation results showed significance of this many edges from layer 0 where to layer 8 or 12 etc. Should be a simple for loop. 
- [MOST IMPORTANT] run whatever method for all combinations for the (prompt, inter token) thats loaded in the notebook and find the edges from the planning features. Summarize the circuit with the edges. 
    - this involves finding the in and out edges of the planning features
    - looking at the latents and noting down their functional roles
    - also looking at other high ranking latents and their edge connections
    - dump all latent roles in a tldraw with edges connecting 
    - provide a semantic summary of the circuit

# Open Questions

- whats the citation for finite differences? Intuitively its what I was doing already and makes a lot of sense but i've never seen it in a paper/post.
- I disagree with the idea of perturbing the activation of an upstream latent by adding an alpha. I believe it should zero ablate it as saes are sparsely trained. What do you think? Its what i have been doing and it has shown interpretable results albeit i havent done thorough investigations.