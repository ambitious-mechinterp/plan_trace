import json

if __name__ == "__main__":
    with open("../outputs/tokenizer.json") as f:
        tokenizer_map = json.load(f)
    vocab = tokenizer_map["model"]["vocab"]
    token_map = {int(v): k for k, v in vocab.items()}
    with open("../outputs/token_map.json", "w") as f:
        json.dump(token_map, f, indent=2)