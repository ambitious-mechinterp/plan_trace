"""
    After running the low coeff grid minimization experiments, 
    this script does categorization of the cases where the instruct model suceeds and base model fails into A1, A2 and A3 based on the steered generations of plans.
"""

import argparse
import ast
import glob
import json
import os
from collections import Counter
from zss import simple_distance, Node

DEFAULT_OUTPUT_ROOT = "outputs"
DEFAULT_SCALE_FOLDER = "/work/pi_jensen_umass_edu/abhishekmish_umass_edu/plan_trace/outputs/all_scale"


def dump_json(filename, my_dict):
    with open(filename, "w") as f:
        json.dump(my_dict, f, indent=2)


def load_json(file):
    with open(file, "r") as f:
        return json.load(f)
    
class Normalizer(ast.NodeTransformer):
    def __init__(self):
        self.var_map = {}
        self.func_map = {}
        self.class_map = {}

    def _rename(self, name, mapping):
        if name not in mapping:
            mapping[name] = f"id_{len(mapping)}"
        return mapping[name]

    def visit_Name(self, node):
        node.id = self._rename(node.id, self.var_map)
        return node

    def visit_arg(self, node):
        node.arg = self._rename(node.arg, self.var_map)
        return node

    def visit_FunctionDef(self, node):
        node.name = self._rename(node.name, self.func_map)
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node):
        node.name = self._rename(node.name, self.class_map)
        self.generic_visit(node)
        return node


def normalize_ast(code):
    try:
        tree = ast.parse(code)
        norm = Normalizer().visit(tree)
        ast.fix_missing_locations(norm)
        return norm
    except Exception:
        return None

def to_zss(node):
    if not isinstance(node, ast.AST):
        return None
    children = []
    for f in node._fields:
        value = getattr(node, f)
        if isinstance(value, list):
            for child in value:
                if isinstance(child, ast.AST):
                    children.append(to_zss(child))
        elif isinstance(value, ast.AST):
            children.append(to_zss(value))
    return Node(node.__class__.__name__, children=children)

def _steered_answer_goes_closer_to_complement(current_generation, steered_generation, complement_generation, gamma = 0.5, lamda = 10):
    try:
        current_code = normalize_ast(current_generation)
        steered_code = normalize_ast(steered_generation)
        complement_code = normalize_ast(complement_generation)
        d_orig = simple_distance(to_zss(current_code), to_zss(complement_code))
        d_new = simple_distance(to_zss(steered_code), to_zss(complement_code))
        if isinstance(d_orig, float) and isinstance(d_new, float):
            d_ratio = (d_new + 1) / (d_orig + 1)
            return d_ratio <= gamma and d_new <= lamda, d_ratio, d_new
    except:
        pass
    return False, 1, 30

# def _create_input_prompt_prefix(task_prompt, test_list, mode = "base"):
#     prompt = (
#             "You are an expert Python programmer, and here is your task: "
#             f"{task_prompt} Your code should pass these tests:\n\n"
#             + "\n".join(test_list) + "\nWrite your code below starting with \"```python\" and ending with \"```\".\n```python\n"
#         )
#     if mode == "instruct":
#         prompt = (
#             "You are an expert Python programmer, and here is your task: "
#             f"{task_prompt} Your code should pass these tests:\n\n"
#             + "\n".join(test_list) + "\nWrite your code, without docstrings, below starting with \"```python\" and ending with \"```\".\n```python\n"
#         )
    
#     return prompt

def _extract_code_part(input_prefix_text, suffix_text):
    try:
        full_text = input_prefix_text + suffix_text
        return "def" + full_text.split("```python\n")[-1].split("def")[1]
    except:
        # print("Could not extract code from ", suffix_text)
        return None

def _obtain_min(objects, scores):
    objects, scores = zip(*sorted(zip(objects, scores), key=lambda x: x[1]))
    objects = list(objects)
    scores = list(scores)
    return objects[0], scores[0]

def _curate_entry(base, steer, complement, idx, tag = "base to instruct"):
    return {
        "model_code": base,
        "steered_code": steer,
        "complement_code": complement,
        "entry_idx": idx,
        "tag": tag,
        "label": False
    }

def _get_all_valid_steers(file, y_ms):
    with open(file, "r") as f:
        data = json.load(f)
    meta_file = file.replace("earliest_position.json", "metadata.json")
    with open(meta_file, "r") as f:
        meta_data = json.load(f)
        # print(meta_data.keys())
        input_prefix = meta_data["input_prefix_text"]
    all_valid_steers = []
    for cand, val in data.items():
        if cand in y_ms:
            for steer in val["steered"]:
                if "decoded_text" in steer:
                    decoded_text = steer["decoded_text"]
                    extracted_code = _extract_code_part(input_prefix, decoded_text)
                    all_valid_steers.append(extracted_code)
    
    return all_valid_steers

def _get_zss_sim(code1, code2):
    node1 = normalize_ast(code1)
    node2 = normalize_ast(code2)
    try:
        distance = simple_distance(to_zss(node1), to_zss(node2))
        return distance
    except:
        # print("got exception for ", code1, code2)
        return 30

def _get_closest_entry(all_steering_files: list[str], complement_code: str, yms: list[str]):
    all_steers = [_get_all_valid_steers(steering_file, yms) for steering_file in all_steering_files]
    all_steers = [x for sub in all_steers for x in sub]
    all_steer_zss = [_get_zss_sim(code, complement_code) for code in all_steers]
    if len(all_steers) > 0:
        closest_steer, _ = _obtain_min(all_steers, all_steer_zss)
        return closest_steer
    else:
        return None    

def _find_closest_entries(base_steering_files: list[str], instruct_steering_files: list[str], ym_bases: list[str], ym_instructs: list[str]):
    all_base_steers = [_get_all_valid_steers(steering_file, ym_bases) for steering_file in base_steering_files]
    all_base_steers = [x for sub in all_base_steers for x in sub]
    all_instruct_steers = [_get_all_valid_steers(steering_file, ym_instructs) for steering_file in instruct_steering_files]
    all_instruct_steers = [x for sub in all_instruct_steers for x in sub]
    min_sim = 30
    closest_base_steer = None
    closest_instruct_steer = None
    for base_steer in all_base_steers:
        for instruct_steer in all_instruct_steers:
            zss_sim = _get_zss_sim(base_steer, instruct_steer)
            if zss_sim <= min_sim:
                min_sim = zss_sim
                closest_base_steer = base_steer
                closest_instruct_steer = instruct_steer 
    
    return closest_base_steer, closest_instruct_steer
    
    
def _match_sweep(e, tag, idx):
    return e["tag"] == tag and e["entry_idx"] == idx
    
def main(data, scale_folder):
    sweep_entries = []
    for idx, entry in enumerate(data):
        actual_base_code, actual_instruct_code = entry["model_output"], entry["instruct_code"]
        try:    
            ym_bases_e, ym_instructs_e = entry["base_plans_e"], entry["instruct_plans_e"]
        except:
            continue
        if ym_bases_e is not None and ym_instructs_e is not None and len(ym_bases_e) > 0 and len(ym_instructs_e) > 0:
            base_token_files_e = glob.glob(f"{scale_folder}/base/prompt_{idx}/token_*/earliest_position.json")
            instruct_token_files_e = glob.glob(f"{scale_folder}/instruct/prompt_{idx}/token_*/earliest_position.json")
            closest_base_steer_e = _get_closest_entry(base_token_files_e, actual_instruct_code, ym_bases_e)
            closest_instruct_steer_e = _get_closest_entry(instruct_token_files_e, actual_base_code, ym_instructs_e)
            entry_base_e = _curate_entry(actual_base_code, closest_base_steer_e, actual_instruct_code, idx, tag = "base to instruct (e)")
            sweep_entries.append(entry_base_e)
            entry_instruct_e = _curate_entry(actual_instruct_code, closest_instruct_steer_e, actual_base_code, idx, tag = "instruct to base (e)")
            sweep_entries.append(entry_instruct_e)
    
    for _, entry in enumerate(sweep_entries):
        calculated_label, d_ratio, d_new = _steered_answer_goes_closer_to_complement(entry["model_code"], entry["steered_code"], entry["complement_code"])
        entry["label"] = bool(calculated_label)

    def _make_entry(e, tag, idx):
        sweeped = next(x for x in sweep_entries if _match_sweep(x, tag, idx))
        e[tag] = {
            "code": sweeped["steered_code"],
            "label": sweeped["label"]
        }
    
    for idx, entry in enumerate(data):        
        try:
            ym_bases_e, ym_instructs_e = entry["base_plans_e"], entry["instruct_plans_e"]
        except:
            continue
        if ym_bases_e is not None and ym_instructs_e is not None and len(ym_bases_e) > 0 and len(ym_instructs_e) > 0:
            _make_entry(entry, "base to instruct (e)", idx)
            _make_entry(entry, "instruct to base (e)", idx)
    
    for idx, entry in enumerate(data):
        if entry["base_pass"] == False and entry["instruct_pass"] == True and entry["base_plans_e"] is not None and len(entry["base_plans_e"]) > 0 and entry["instruct_plans_e"] is not None and len(entry["instruct_plans_e"]) > 0:
            tag = "A3"
            if entry["base to instruct (e)"]["label"] == True:
                tag = "A1"
            elif entry["instruct to base (e)"]["label"] == True:
                tag = "A2"
            else:
                base_token_files_e = glob.glob(f"{scale_folder}/base/prompt_{idx}/token_*/earliest_position.json")
                instruct_token_files_e = glob.glob(f"{scale_folder}/instruct/prompt_{idx}/token_*/earliest_position.json")
                closest_base_steer, closest_instruct_steer = _find_closest_entries(base_token_files_e, instruct_token_files_e, entry["base_plans_e"], entry["instruct_plans_e"])
                goes_closer_smallest, _, _ = _steered_answer_goes_closer_to_complement(entry["instruct_code"], closest_instruct_steer, closest_base_steer)
                # print("Goes closer smallest:", goes_closer_smallest, "for entry idx:", idx, "with closest_base_steer:", closest_base_steer, "and closest_instruct_steer:", closest_instruct_steer)
                if goes_closer_smallest:
                    tag = "A4" # new category where atleast once base and instruct steered generations come close to each other, while still producing coherent code.
                    entry["closest_proxy_pair"] = {
                        "base_to_instruct_steer": closest_base_steer,
                        "instruct_to_base_steer": closest_instruct_steer
                    }
            entry["A_tag (e)"] = tag

    a1, a2, a3, a4 = 0, 0, 0, 0
    for entry in data:
        try:
            if entry['A_tag (e)'] == 'A1':
                a1 += 1
            elif entry['A_tag (e)'] == "A2":
                a2 += 1
            elif entry['A_tag (e)'] == "A3":
                a3 += 1
            elif entry['A_tag (e)'] == "A4":
                a4 += 1
        except:
            continue

    # print(f"a1: {a1}, a2: {a2}, a3: {a3}, a4: {a4}")
    print("Final counts for A1, A2, A3, A4:", Counter([entry.get('A_tag (e)', 'None') for entry in data if 'A_tag (e)' in entry]))
    return data    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify earliest-position planning and save 4x4 grid plot."
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scale-folder", default=DEFAULT_SCALE_FOLDER)
    parser.add_argument("--phase-1-data-path", default=None, required=True)
    parser.add_argument("--save-data-path", default=None)
    args = parser.parse_args()

    if args.phase_1_data_path is None:
        raise ValueError("Please provide analysis folder path using --analysis-folder argument.")

    resolved_saved_data_path = args.save_data_path or os.path.join(args.output_root, "base_vs_instruct_analysis", "report", "processed_phase2.json")
    os.makedirs(os.path.dirname(resolved_saved_data_path), exist_ok=True)

    phase1_data = load_json(args.phase_1_data_path)
    processed_data = main(phase1_data, args.scale_folder)
    dump_json(resolved_saved_data_path, processed_data)

