import json

if __name__ == "__main__":
    # pass_file = "data/external/first_100_passing_examples_without_docstrings_base_model_og_prompt.json"
    pass_file = "data/external/first_100_passing_examples_without_docstrings_base_model_og_prompt_V2.json"
    # fail_file = "data/external/first_100_failing_examples_without_docstrings_base_model_og_prompt.json"
    fail_file = "data/external/first_100_failing_examples_without_docstrings_base_model_og_prompt_V2.json"
    with open(pass_file, "r") as f:
        pass_data = json.load(f)
    with open(fail_file, "r") as f:
        fail_data = json.load(f)
    # pass_taskids = [64, 66, 232, 89, 309] - v1
    # fail_taskids = [3, 80, 139, 397, 404] - v1
    pass_taskids = [394, 397, 390, 292, 282, 62] ## v2
    fail_taskids = [80, 96, 251, 285, 85, 139] ## -v2; consider 85 and 139 as cases where the base model gives a different answer instead of failing.
    print("For the passing data:")
    for i, entry in enumerate(pass_data):
        if entry["task_id"] in pass_taskids:
            print(f"Task id {entry["task_id"]} is at index: {i}")
    print("For the failing data:")
    for i, entry in enumerate(fail_data):
        if entry["task_id"] in fail_taskids:
            print(f"Task id {entry["task_id"]} is at index {i}")

"""
V2 response:

For the passing data:
Task id 62 is at index: 6
Task id 282 is at index: 48
Task id 292 is at index: 50
Task id 390 is at index: 53
Task id 394 is at index: 54
Task id 397 is at index: 56
For the failing data:
Task id 80 is at index 8
Task id 85 is at index 10
Task id 96 is at index 11
Task id 139 is at index 15
Task id 251 is at index 27
Task id 285 is at index 34

"""