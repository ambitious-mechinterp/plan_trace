# goals

1 input and output display
- display the input sequence till yn below the grid 
- also display the base generation after the input

2 tokenizing: this probably needs a separate script 
- we will have to look into plan_trace/pipeline.py on how it iterates over tokens. 
- Save the string token inputs and base generation for each ym in the metadata.json that already exists. 

3 highlighting 
- in the input display (input tokenzied strings), when we select a cluster from the grid and a pop up appears, that token id should be highlighted 
- we might need to make sure that the input is never hidden behind the pop up. maybe we can keep a fixed empty place for it so the input never flows there? whatever works is fine 