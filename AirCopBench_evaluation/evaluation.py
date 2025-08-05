import os
import glob
import json
import re
import base64
import random
from datetime import datetime
from typing import Dict, List
from openai import AzureOpenAI
from PIL import Image

endpoint = "YOUR_AZURE"
model_name = "gpt-4o"
deployment = "gpt-4o"
subscription_key = "YOUR_KEY"
api_version = "YOUR_VERSION"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

class GPTInference:
    """
    Inference engine based on Azure GPT-4o, adapted for benchmark format
    """
    def __init__(
        self,
        model_path: str = "Azure_GPT-4o",
    ):
        self.model_path = model_path
        self.client = client
        self.deployment = deployment
        
        # Generation parameters
        self.gen_kwargs = dict(
            max_tokens=100,
            temperature=0.0,
            top_p=1.0,
        )
        
        print(f"GPT-4o client initialized successfully")

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            raise

    def build_messages(
        self,
        uav_paths: Dict[str, str],
        question: str,
        options: Dict[str, str],
    ):
        """Construct message format"""
        opts = "\n".join([f"{k}. {v}" for k, v in options.items()])
        prompt = f"{question}\n{opts}\nAnswer with the option's letter from the given choices directly."
        
        # Message format - images first, then text
        message = []
        for uav, path in uav_paths.items():
            full_path = '/your/path/to/' + path
            message.append(dict(type='image', value=full_path))
        message.append(dict(type='text', value=prompt))
        
        return message

    def prepare_inputs(self, message):
        """Prepare inputs for API call"""
        images = [x['value'] for x in message if x['type'] == 'image']
        texts = [x['value'] for x in message if x['type'] == 'text']
        
        # Construct content format for OpenAI API
        content = []
        
        # Add text
        if texts:
            content.append({
                "type": "text",
                "text": texts[0]  # Take first text as prompt
            })
        
        # Add images
        for img_path in images:
            if not os.path.exists(img_path):
                print(f"Warning: Image file does not exist: {img_path}")
                continue
                
            try:
                # Verify image can be opened
                with Image.open(img_path) as img:
                    pass  # Just validate image
                
                base64_image = self.encode_image(img_path)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    }
                })
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue
        
        return content

    def extract_answer(self, res: str) -> str:
        """Extract answer from GPT response"""
        # Handle thinking mode output if present
        think_end = res.rfind('<think>')
        if think_end != -1:
            think_end += len('</think>')
            res = res[think_end:].strip()
        
        txt = res.strip().upper()
        
        # Multiple regex patterns
        patterns = [
            r"ANSWER:\s*([ABCD])",
            r"THE ANSWER IS\s*([ABCD])", 
            r"\(([ABCD])\)",
            r"^([ABCD]):",
            r"OPTION\s*([ABCD])",
            r"([ABCD])\s*\.?\s*$"
        ]
        
        for pat in patterns:
            m = re.search(pat, txt, re.MULTILINE)
            if m:
                return m.group(1)
        
        # Fallback: iterate through characters
        for c in txt:
            if c in ["A", "B", "C", "D"]:
                return c
        
        # Return random if no answer found
        return random.choice(["A", "B", "C", "D"])

    def infer_one(
        self,
        uav_paths: Dict[str, str],
        question: str,
        options: Dict[str, str],
    ) -> str:
        """Infer for a single question"""
        try:
            # Build message
            message = self.build_messages(uav_paths, question, options)
            
            # Prepare input
            content = self.prepare_inputs(message)
            
            if not content:
                print("No valid content prepared, returning random answer")
                return random.choice(["A", "B", "C", "D"])
            
            # Call API
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                **self.gen_kwargs
            )
            
            # Get response text
            response_text = response.choices[0].message.content
            print(f"----->model response: {response_text}")
            
            # Extract answer
            answer = self.extract_answer(response_text)
            return answer
            
        except Exception as e:
            print(f"Error during inference: {e}")
            print(f"Falling back to random answer")
            return random.choice(["A", "B", "C", "D"])

# Configuration section
engine = GPTInference()

# Mapping of primary categories to their secondary subcategories
CATEGORY_MAP = {
    "Scene Understanding": [
        "Scene Description",
        "Scene Comparison",
        "Observing Posture"
    ],
    "Object Understanding": [
        "Object Recognition",
        "Object Counting",
        "Object Grounding",
        "Object Matching"
    ],
    "Perception Assessment": [
        "Quality Assessment",
        "Usability Assessment",
        "Causal Assessment"
    ],
    "Collaborative Decision": [
        "When to Collaborate",
        "What to Collaborate",
        "Who to Collaborate",
        "Why to Collaborate"
    ],
}

DATASETS = {
    "Real2_VQA": {
        "questions": "/your/path/to/test/Real2_VQA_test.json",
        "images":   "/your/path/to/Real_2_UAVs/Samples",
    },
    "Sim5_VQA": {
        "questions": "/your/path/to/test/Sim5_VQA_test.json",
        "images":   "/your/path/to/Sim_5_UAVs/Samples",
    },
    "Sim6_VQA": {
        "questions": "/your/path/to/test/Sim6_VQA_test.json",
        "images":   "/your/path/to/Sim_6_UAVs/Samples",
    },
    "Sim3_VQA": {
        "questions": "/your/path/to/test/Sim3_VQA_test.json",
        "images":   "/your/path/to/Sim_3_UAVs/Samples",
    }
}

RESULTS_DIR = "/your/path/to/results"


def parse_subcat(qtype: str) -> str:
    m = re.match(r"\d+\.\d+\s+(.+?)\s*(?:\(|$)", qtype)
    return m.group(1) if m else qtype

# Checkpoint management
def load_checkpoint(path: str) -> List[Dict]:
    print(f"Loading checkpoint: {path}")
    return json.load(open(path, 'r', encoding='utf-8'))

def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# Metric calculation
def compute_accuracy(results: List[Dict]) -> float:
    correct = sum(1 for r in results if r["predicted_answer"] == r["correct_answer"])
    return correct / len(results) if results else 0.0

def aggregate_metrics(all_responses: List[Dict], datasets_done: List[str]):
    # Overall accuracy per dataset
    ds_metrics: Dict[str, float] = {}
    # Cumulative results for secondary subcategories
    sub_metrics: Dict[str, List[Dict]] = {}
    # Cumulative results for (dataset, subcategory)
    ds_sub_metrics: Dict[tuple, List[Dict]] = {}

    for ds in datasets_done:
        rs = [r for r in all_responses if r["dataset"] == ds]
        ds_metrics[ds] = compute_accuracy(rs)
        for r in rs:
            sub = parse_subcat(r["question_type"])
            sub_metrics.setdefault(sub, []).append(r)
            ds_sub_metrics.setdefault((ds, sub), []).append(r)

    # Calculate accuracy for secondary subcategories
    sub_acc = {s: compute_accuracy(v) for s, v in sub_metrics.items()}
    # Calculate accuracy for secondary subcategories within each dataset
    ds_sub_acc = {f"{ds}----{sub}": compute_accuracy(v) for (ds, sub), v in ds_sub_metrics.items()}

    # Calculate primary categories
    cat_metrics: Dict[str, float] = {}
    ds_cat_metrics: Dict[str, float] = {}
    for cat, subs in CATEGORY_MAP.items():
        # Overall primary category
        cat_rs = [r for r in all_responses if parse_subcat(r["question_type"]) in subs]
        cat_metrics[cat] = compute_accuracy(cat_rs)
        # Primary category per dataset
        for ds in datasets_done:
            ds_cat_rs = [
                r for r in all_responses
                if r["dataset"] == ds and parse_subcat(r["question_type"]) in subs
            ]
            ds_cat_metrics[f"{ds}----{cat}"] = compute_accuracy(ds_cat_rs)

    return ds_metrics, sub_acc, cat_metrics, ds_sub_acc, ds_cat_metrics

# Main process
def main():
    print(f"model_path: {engine.model_path}")
    model_name = os.path.basename(engine.model_path.rstrip("/"))

    # Find existing checkpoints
    pattern = os.path.join(RESULTS_DIR, f"{model_name}_*_responses.json")
    candidates = glob.glob(pattern)
    if candidates:
        resp_path = max(candidates, key=os.path.getmtime)
        all_responses = load_checkpoint(resp_path)
        done_ds = sorted({r["dataset"] for r in all_responses})
        print(f"Completed datasets: {done_ds}")
    else:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        resp_path = os.path.join(RESULTS_DIR, f"{model_name}_{now}_responses.json")
        all_responses = []
        done_ds = []

    # Process only unfinished datasets
    remaining_datasets = [ds_name for ds_name in DATASETS.keys() if ds_name not in done_ds]
    
    if not remaining_datasets:
        print("All datasets completed testing, generating final metrics...")
    else:
        print(f"Datasets to test: {remaining_datasets}")

    for ds_name in remaining_datasets:
        cfg = DATASETS[ds_name]
        print(f"Starting dataset testing: {ds_name}")

        raw = json.load(open(cfg["questions"], 'r', encoding='utf-8'))
        qs = raw.get("results", raw)

        for item in qs:
            qid = item["question_id"]
            if any(r["dataset"] == ds_name and r["question_id"] == qid for r in all_responses):
                continue
            print(f"starting next question")
            print(f"Processing {ds_name} -- {qid}")
            pred = engine.infer_one(item["uav_paths"], item["question"], item["options"])
            print(f"---->predicted_answer: {pred}")
            all_responses.append({
                "dataset": ds_name,
                "question_id": qid,
                "question_type": item["question_type"],
                "correct_answer": item["correct_answer"],
                "predicted_answer": pred,
            })
            save_json(all_responses, resp_path)

        done_ds.append(ds_name)
        print(f"Dataset {ds_name} testing completed")

        # Save intermediate results after each dataset
        ds_metrics, sub_acc, cat_acc, ds_sub_acc, ds_cat_acc = aggregate_metrics(all_responses, done_ds)
        current_overall = compute_accuracy(all_responses)
        
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_path = os.path.join(RESULTS_DIR, f"{model_name}_{now}_metrics.json")
        
        partial_metrics = {
            "datasets_done": done_ds,
            "per_dataset": ds_metrics,
            "per_subcategory": sub_acc,
            "per_category": cat_acc,
            "per_dataset_subcategory": ds_sub_acc,
            "per_dataset_category": ds_cat_acc,
            "overall_accuracy": current_overall
        }
        
        save_json(partial_metrics, metrics_path)
        print(f"Intermediate metrics saved to: {metrics_path}")
        print(f"Current overall accuracy (based on {len(done_ds)} datasets): {current_overall}")

    # Generate final metrics
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_metrics_path = os.path.join(RESULTS_DIR, f"{model_name}_{now}_final_metrics.json")
    
    ds_metrics, sub_acc, cat_acc, ds_sub_acc, ds_cat_acc = aggregate_metrics(all_responses, done_ds)
    
    # Calculate final overall accuracy (including all datasets)
    final_overall = compute_accuracy(all_responses)
    print(f"Final overall accuracy (based on all {len(done_ds)} datasets): {final_overall}")
    
    final_metrics = {
        "datasets_done": done_ds,
        "per_dataset": ds_metrics,
        "per_subcategory": sub_acc,
        "per_category": cat_acc,
        "per_dataset_subcategory": ds_sub_acc,
        "per_dataset_category": ds_cat_acc,
        "overall_accuracy": final_overall
    }
    
    save_json(final_metrics, final_metrics_path)
    print(f"Final metrics saved to: {final_metrics_path}")

    print(f"===============================\nEvaluation completed: {engine.model_path}\n===============================")

if __name__ == "__main__":
    main()
