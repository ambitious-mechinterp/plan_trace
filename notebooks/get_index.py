import json

if __name__ == "__main__":
    with open("data/first_100_passing_examples.json", "r") as f:
        data = json.load(f)
    with open("data/external/first_100_passing_examples_without_docstrings.json", "r") as f:
        ext_data = json.load(f)
    for i in range(len(ext_data)):
        if ext_data[i]["task_id"] == 64:
            print(i, "This is 64")
        if ext_data[i]["task_id"] == 80:
            print(i, "This is 80")