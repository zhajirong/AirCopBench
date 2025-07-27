import os
import yaml

SCENARIO_DIR = "scenarios"

def summarize_yaml_scenes(directory):
    summaries = []
    for filename in os.listdir(directory):
        if filename.endswith(".yaml"):
            path = os.path.join(directory, filename)
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                name = data.get("scene_name", filename)
                desc = data.get("description", "(无描述)")
                summaries.append((filename, name, desc.strip()))
    return summaries

if __name__ == "__main__":
    scenes = summarize_yaml_scenes(SCENARIO_DIR)
    print("所有场景概览：\n")
    for fname, name, desc in scenes:
        print(f"{fname} | 名称: {name}\n  简介: {desc}\n")
