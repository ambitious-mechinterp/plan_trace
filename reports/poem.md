Running pipeline analysis on generated tokens...
============================================================
Analyzing token positions 18 to 25
Tokens to analyze:
  Position 18: 'A'
  Position 19: ' tasty'
  Position 20: ' treat'
  Position 21: ','
  Position 22: ' a'
  Position 23: ' crunchy'
  Position 24: ' habit'
  Position 25: '.'

Starting analysis...
============================================================

--- Analyzing token position 18: 'A' ---
Baseline continuation: A tasty treat, a crunchy habit....
Found minimum K for absolute effects: 10001 (metric: 0.3164, target: 0.1629)
Found circuit with 10001 entries
Found 5 tokens not in prompt: ['.', ' crunchy', ' tasty', ' habit', ' treat']
100%|██████████| 26/26 [00:05<00:00,  4.78it/s]
Found 3 clusters: ['.', 'treat', 'tasty']
  🎯 Planning evidence found for: ['.', 'tasty']

--- Analyzing token position 19: ' tasty' ---
Baseline continuation:  tasty treat, a crunchy habit....
Found minimum K for negative effects: 10001 (metric: 0.9258, target: 0.0554)
Found minimum K for absolute effects: 20001 (metric: 0.0752, target: 0.0554)
Found circuit with 10001 entries
Found 5 tokens not in prompt: ['.', ' crunchy', ' tasty', ' habit', ' treat']
100%|██████████| 26/26 [00:05<00:00,  4.80it/s]
Found 3 clusters: ['.', 'tasty', 'habit']
  🎯 Planning evidence found for: ['.', 'tasty']

--- Analyzing token position 20: ' treat' ---
Baseline continuation:  treat, a crunchy habit....
Found minimum K for negative effects: 10001 (metric: 0.9844, target: 0.3680)
Found minimum K for absolute effects: 10001 (metric: 0.6250, target: 0.3680)
Found circuit with 10001 entries
Found 4 tokens not in prompt: ['.', ' crunchy', ' habit', ' treat']
100%|██████████| 26/26 [00:05<00:00,  4.79it/s]
Found 2 clusters: ['.', 'treat']
  🎯 Planning evidence found for: ['.', 'treat']

--- Analyzing token position 21: ',' ---
Baseline continuation: , a crunchy habit....
Found minimum K for negative effects: 1 (metric: 0.5898, target: 0.3000)
Found minimum K for absolute effects: 1 (metric: 0.5898, target: 0.3000)
Found circuit with 1 entries
Found 3 tokens not in prompt: ['.', ' crunchy', ' habit']
100%|██████████| 1/1 [00:00<00:00,  4.03it/s]
Found 0 clusters: []
  ❌ No planning evidence found

--- Analyzing token position 22: ' a' ---
Baseline continuation:  a crunchy habit....
Found minimum K for negative effects: 1 (metric: 0.3379, target: 0.1711)
Found minimum K for absolute effects: 1 (metric: 0.3379, target: 0.1711)
Found circuit with 1 entries
Found 3 tokens not in prompt: ['.', ' crunchy', ' habit']
100%|██████████| 1/1 [00:00<00:00,  4.30it/s]
Found 0 clusters: []
  ❌ No planning evidence found

--- Analyzing token position 23: ' crunchy' ---
Baseline continuation:  crunchy habit....
Found minimum K for negative effects: 10001 (metric: 0.9180, target: 0.0902)
Found circuit with 10001 entries
Found 3 tokens not in prompt: ['.', ' crunchy', ' habit']
100%|██████████| 26/26 [00:05<00:00,  4.80it/s]
Found 2 clusters: ['.', 'habit']
  🎯 Planning evidence found for: ['.']

--- Analyzing token position 24: ' habit' ---
Baseline continuation:  habit....
Found minimum K for negative effects: 10001 (metric: 0.9883, target: 0.3305)
Found minimum K for absolute effects: 20001 (metric: 0.3594, target: 0.3305)
Found circuit with 10001 entries
Found 2 tokens not in prompt: ['.', ' habit']
100%|██████████| 26/26 [00:05<00:00,  4.73it/s]
Found 2 clusters: ['.', 'habit']
  🎯 Planning evidence found for: ['.']

--- Analyzing token position 25: '.' ---
Baseline continuation: ....
Found minimum K for negative effects: 10001 (metric: 0.9648, target: 0.5531)
Found minimum K for absolute effects: 10001 (metric: 0.9727, target: 0.5531)
Found circuit with 10001 entries
Found 1 tokens not in prompt: ['.']
100%|██████████| 26/26 [00:05<00:00,  4.76it/s]
Found 1 clusters: ['.']
  🎯 Planning evidence found for: ['.']
