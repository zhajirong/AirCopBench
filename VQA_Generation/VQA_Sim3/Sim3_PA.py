import base64
from PIL import Image
import io
import os
import json
import glob
import re  # For better annotation parsing and filename extraction
import difflib  # For SequenceMatcher
import random
import time
from collections import defaultdict
import openai  # Import OpenAI library
from openai import AzureOpenAI

"""
Perception Assessment Script - Tasks 3.1, 3.2, 3.3
"""

# Set API key and base URL (using official OpenAI API)
API_KEY = 'your_api_key'
BASE_URL = 'https://api.openai.com/v1'
client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)


def encode_image(image_path):
    """Encode image to base64 string"""
    with Image.open(image_path) as img:
        buffered = io.BytesIO()
        img.save(buffered, format=img.format if img.format else "JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


def normalize_filename_for_annotation(filename, scene_name=None):
    """Normalize filename to match annotation file format"""
    # Handle new filename format like "UAV1_frame_001.jpg"
    match = re.search(r'UAV(\d+)_frame_(\d+)', filename)
    if match:
        uav_num = match.group(1)
        frame_num = match.group(2)
        
        # Create possible annotation filenames based on scene and frame
        # The all_samples.json format is: /data/upload/9/8f2a9605-scene_004-UAV1_frame_005.jpg
        possible_names = []
        
        if scene_name:
            # Remove 'scene_' prefix if it already exists
            clean_scene_name = scene_name.replace('scene_', '')
            # Try to match the exact scene and frame combination
            possible_names.extend([
                f"scene_{clean_scene_name}-UAV{uav_num}_frame_{frame_num}.jpg",
                f"scene_{clean_scene_name}-UAV{uav_num}_frame_{frame_num}.png",
                f"UAV{uav_num}_frame_{frame_num}.jpg",
                f"UAV{uav_num}_frame_{frame_num}.png",
                f"frame_{frame_num}.jpg",
                f"frame_{frame_num}.png",
            ])
        else:
            # Fallback without scene information
            possible_names.extend([
                f"UAV{uav_num}_frame_{frame_num}.jpg",
                f"UAV{uav_num}_frame_{frame_num}.png",
                f"frame_{frame_num}.jpg",
                f"frame_{frame_num}.png",
            ])
        
        return possible_names
    
    # Fallback for old format
    # Remove UAV suffix and extension
    base_name = re.sub(r'-UAV\d+\.(png|jpg)$', '', filename)
    
    # Split by '-' to get parts
    parts = base_name.split('-')
    
    if len(parts) >= 3:
        # Format: "3-40m-1623936157944367872"
        sequence_num = parts[0]  # "3"
        timestamp = parts[2]     # "1623936157944367872"
        
        # Create possible annotation filenames
        # The annotation file might have different hash prefixes
        possible_names = [
            f"{timestamp}.png",  # Direct timestamp match
            f"{sequence_num}-{timestamp}.png",  # With sequence number
        ]
        
        return possible_names
    
    return [filename]


def load_annotations(annotation_file):
    """Load annotation file"""
    try:
        with open(annotation_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)

        annotation_map = {}
        for annotation in annotations:
            if 'img1' in annotation:
                full_filename = os.path.basename(annotation['img1'])
                # Handle format like "8f2a9605-scene_004-UAV1_frame_005.jpg"
                # Extract scene, UAV, and frame information
                scene_match = re.search(r'scene_(\d+)-UAV(\d+)_frame_(\d+)', full_filename)
                if scene_match:
                    scene_num = scene_match.group(1)
                    uav_num = scene_match.group(2)
                    frame_num = scene_match.group(3)
                    
                    # Create keys for different matching strategies
                    # Key 1: scene_frame format (e.g., "scene_004_frame_005")
                    scene_frame_key = f"scene_{scene_num.zfill(3)}_frame_{frame_num}"
                    # Key 2: UAV_frame format (e.g., "UAV1_frame_005")
                    uav_frame_key = f"UAV{uav_num}_frame_{frame_num}"
                    # Key 3: scene_UAV_frame format (e.g., "scene_004_UAV1_frame_005")
                    full_key = f"scene_{scene_num.zfill(3)}_UAV{uav_num}_frame_{frame_num}"
                    
                    # Store annotation with multiple keys for flexible matching
                    # Only store if the key doesn't already exist to avoid overwriting
                    if scene_frame_key not in annotation_map:
                        annotation_map[scene_frame_key] = annotation
                    if uav_frame_key not in annotation_map:
                        annotation_map[uav_frame_key] = annotation
                    if full_key not in annotation_map:
                        annotation_map[full_key] = annotation
                    
                    # Also store with original filename for direct matching
                    annotation_map[full_filename] = annotation
                    
                    print(f"Loaded annotation for {full_filename} -> keys: {scene_frame_key}, {uav_frame_key}, {full_key}")
                else:
                    # Handle format like "5064a99b-1623936157944367872.png"
                    parts = full_filename.split('-', 1)
                    if len(parts) == 2 and len(parts[0]) == 8 and parts[0].isalnum():
                        # Extract timestamp part
                        timestamp = parts[1].replace('.png', '')
                        annotation_map[timestamp] = annotation
                        # Also store with .png extension for compatibility
                        annotation_map[f"{timestamp}.png"] = annotation
                    else:
                        # Fallback for other formats
                        filename = full_filename
                        annotation_map[filename] = annotation

        print(f"Loaded {len(annotation_map)} annotation entries")
        return annotation_map
    except Exception as e:
        print(f"Error loading annotations: {str(e)}")
        return {}








def get_image_groups(base_dir):
    """Get corresponding image groups from 3 UAV folders across all scenes"""
    image_map = defaultdict(list)

    # Find all scene directories
    scene_dirs = glob.glob(os.path.join(base_dir, "scene_*"))
    scene_dirs.sort()  # Sort to ensure consistent processing order
    
    print(f"Found {len(scene_dirs)} scene directories: {[os.path.basename(d) for d in scene_dirs]}")

    for scene_dir in scene_dirs:
        scene_name = os.path.basename(scene_dir)
        
        # Look for UAV directories in each scene
        for uav_num in range(1, 4):
            uav_dir = os.path.join(scene_dir, f"UAV{uav_num}")
            if not os.path.exists(uav_dir):
                print(f"Warning: UAV{uav_num} directory not found in {scene_name}")
                continue
                
            # Look for both .jpg and .png files
            images = glob.glob(os.path.join(uav_dir, "*.jpg")) + glob.glob(os.path.join(uav_dir, "*.png"))
            for img_path in images:
                filename = os.path.basename(img_path)
                # Extract frame number from filename like "UAV1_frame_001.jpg"
                match = re.search(r'UAV(\d+)_frame_(\d+)', filename)
                if match:
                    uav_num_from_filename = int(match.group(1))
                    frame_num = match.group(2)
                    # Create group key using scene and frame number
                    group_key = f"{scene_name}_frame_{frame_num}"
                    
                    image_map[group_key].append({
                        'uav': uav_num_from_filename,
                        'path': img_path,
                        'filename': filename,
                        'scene': scene_name,
                        'frame': frame_num
                    })

    # Filter complete groups with exactly 3 UAVs
    image_groups = []
    for key, group in sorted(image_map.items()):
        if len(group) == 3:
            group.sort(key=lambda x: x['uav'])  # Sort by UAV number
            image_groups.append({
                'sequence_frame': key,
                'group': group
            })

    return image_groups








def extract_annotation_info(annotation):
    """Extract useful information from annotation"""
    if not annotation:
        return ""

    info_parts = []

    if 'PerceptionIssues' in annotation and annotation['PerceptionIssues']:
        issues = set()
        for issue in annotation['PerceptionIssues']:
            if 'rectanglelabels' in issue:
                issues.update(issue['rectanglelabels'])
        if issues:
            info_parts.append(f"Perception issues: {', '.join(issues)}")

    if 'Degradation' in annotation:
        if isinstance(annotation['Degradation'], dict) and 'choices' in annotation['Degradation']:
            degradation = ', '.join(annotation['Degradation']['choices'])
        else:
            degradation = str(annotation['Degradation'])
        info_parts.append(f"Perception degradation: {degradation}")

    if 'Quality' in annotation:
        quality = annotation['Quality']
        info_parts.append(f"Image quality: {quality}")

    # Handle both "Usability" and "Usibility" (typo in JSON)
    if 'Usability' in annotation:
        usability = annotation['Usability']
        info_parts.append(f"Image usability: {usability}")
    elif 'Usibility' in annotation:
        usability = annotation['Usibility']
        info_parts.append(f"Image usability: {usability}")
    else:
        info_parts.append("Image usability: Not specified")

    if 'Collaboration_when' in annotation:
        collaboration_when = annotation['Collaboration_when']
        info_parts.append(f"Collaboration time: {collaboration_when}")

    if 'Collaboration_what' in annotation:
        if isinstance(annotation['Collaboration_what'], dict) and 'choices' in annotation['Collaboration_what']:
            collaboration_what = ', '.join(annotation['Collaboration_what']['choices'])
        else:
            collaboration_what = str(annotation['Collaboration_what'])
        info_parts.append(f"Collaboration content: {collaboration_what}")

    if 'Collaboration_who' in annotation:
        info_parts.append(f"Collaboration partner: {annotation['Collaboration_who']}")

    if 'Collaboration_why' in annotation:
        if isinstance(annotation['Collaboration_why'], dict) and 'choices' in annotation['Collaboration_why']:
            collaboration_reasons = ', '.join(annotation['Collaboration_why']['choices'])
        elif isinstance(annotation['Collaboration_why'], str):
            collaboration_reasons = annotation['Collaboration_why']
        else:
            collaboration_reasons = str(annotation['Collaboration_why'])
        info_parts.append(f"Collaboration reasons: {collaboration_reasons}")

    # Add object information
    if 'Object_type' in annotation:
        object_type = annotation['Object_type']
        info_parts.append(f"Object type: {object_type}")

    if 'Object_count' in annotation:
        object_count = annotation['Object_count']
        info_parts.append(f"Object count: {object_count}")

    return '; '.join(info_parts)


def compute_quality_statistics(all_results):
    """Compute statistics on image quality across all pairs"""
    quality_scores = []
    for result in all_results['results']:
        for info in [result['combined_info1'], result['combined_info2']]:
            match = re.search(r'Image quality: .*?(\d+)/5', info)
            if match:
                quality_scores.append(int(match.group(1)))

    if quality_scores:
        stats = {
            'average_quality': sum(quality_scores) / len(quality_scores),
            'min_quality': min(quality_scores),
            'max_quality': max(quality_scores),
            'count_low_quality': len([s for s in quality_scores if s <= 3])
        }
        return stats
    return {}


def validate_annotation_coverage(image_groups, annotation_map):
    """Validate annotation coverage for all image groups"""
    total_images = 0
    covered_images = 0
    
    print("\n=== Annotation Coverage Validation ===")
    
    for group_data in image_groups:
        group = group_data['group']
        group_scene = group[0]['scene'] if group else "unknown"
        group_frame = group[0]['frame'] if group else "unknown"
        
        for item in group:
            uav_num = item['uav']
            current_filename = item['filename']
            total_images += 1
            
            # Try to find annotation
            scene_frame_key = f"{group_scene}_frame_{group_frame}"
            uav_frame_key = f"UAV{uav_num}_frame_{group_frame}"
            full_key = f"{group_scene}_UAV{uav_num}_frame_{group_frame}"
            
            found = False
            if full_key in annotation_map or scene_frame_key in annotation_map or uav_frame_key in annotation_map:
                found = True
            else:
                # Try pattern matching
                for key in annotation_map.keys():
                    if f"{group_scene}-UAV{uav_num}_frame_{group_frame}" in key:
                        found = True
                        break
            
            if found:
                covered_images += 1
            else:
                print(f"  Missing annotation: {current_filename} (scene: {group_scene}, frame: {group_frame}, UAV: {uav_num})")
    
    coverage_rate = (covered_images / total_images) * 100 if total_images > 0 else 0
    print(f"Annotation coverage: {covered_images}/{total_images} ({coverage_rate:.1f}%)")
    return coverage_rate


def save_to_json(data, filename="VQA_Sim3_PA.json"):
    """Save data to JSON file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Failed to save file: {str(e)}")
        return False


def parse_quality_or_usability(value_str):
    """Parse quality or usability from descriptive string using regex"""
    if not value_str:
        return None, None

    # Regex to match level (word) and score (number/5)
    match = re.search(r'(\w+(?:\s\w+)?)\s*\((\d+)/5\)', value_str)
    if match:
        level = match.group(1).strip()
        score = int(match.group(2))
        return level, score

    # Fallback to just level or score
    level_match = re.search(
        r'(Excellent|Good|Fair|Poor|Very Poor|Unusable|Highly Usable|Usable|Partially Usable|Barely Usable|Not Usable)',
        value_str, re.IGNORECASE)
    score_match = re.search(r'(\d+)/5', value_str)

    level = level_match.group(1) if level_match else None
    score = int(score_match.group(1)) if score_match else None

    return level, score


def generate_rule_based_quality_q(annotation, uav_id, q_id):
    """
    [Rule-Based] Generate image quality assessment questions based on JSON annotation data.
    Improved parsing with regex for descriptive strings.
    """
    if not annotation or 'Quality' not in annotation:
        return None

    quality_value = annotation['Quality']
    if isinstance(quality_value, dict) and 'choices' in quality_value:
        quality_str = ', '.join(quality_value['choices'])
    else:
        quality_str = str(quality_value)

    level, score = parse_quality_or_usability(quality_str)
    if level is None and score is None:
        return None

    if score is not None:
        # Numeric score (1-5)
        correct = str(score)
        options = {correct}
        while len(options) < 4:
            distractor = random.randint(1, 5)
            if str(distractor) != correct:
                options.add(str(distractor))

        shuffled_options = random.sample(list(options), len(options))
        option_dict = {chr(65 + i): opt for i, opt in enumerate(shuffled_options)}
        correct_letter = [k for k, v in option_dict.items() if v == correct][0]

        return {
            "question_id": f"Sim3_QA_{uav_id}_{q_id}",
            "question_type": f"3.1 Quality Assessment ({uav_id})",
            "question": f"What is the perception quality assessment score (1-5) for target detection in the image captured by {uav_id}?",
            "options": option_dict,
            "correct_answer": correct_letter,
            "source": "Rule-Based from JSON"
        }
    elif level:
        # Text level
        possible_options = ["Excellent", "Good", "Fair", "Poor", "Very Poor"]
        normalized_level = level.capitalize()
        if normalized_level not in possible_options:
            return None

        options = {normalized_level}
        distractors = [opt for opt in possible_options if opt != normalized_level]
        while len(options) < min(4, len(possible_options)):
            options.add(random.choice(distractors))

        shuffled_options = random.sample(list(options), len(options))
        option_dict = {chr(65 + i): opt for i, opt in enumerate(shuffled_options)}
        correct_letter = [k for k, v in option_dict.items() if v == normalized_level][0]

        return {
            "question_id": f"Sim3_QA_{uav_id}_{q_id}",
            "question_type": f"3.1 Quality Assessment ({uav_id})",
            "question": f"What is the perception quality assessment level for target detection in the image captured by {uav_id}?",
            "options": option_dict,
            "correct_answer": correct_letter,
            "source": "Rule-Based from JSON"
        }


def generate_rule_based_usability_q(annotation, uav_id, q_id):
    """
    [Rule-Based] Generate image usability assessment questions based on JSON annotation data.
    Improved with fallback if missing.
    """
    # Handle both "Usability" and "Usibility" (typo in JSON)
    if 'Usability' not in annotation and 'Usibility' not in annotation:
        # Fallback if missing - assume based on quality or default
        if 'Quality' in annotation:
            quality_level, _ = parse_quality_or_usability(annotation['Quality'])
            if quality_level:
                # Map quality to usability roughly
                mapping = {
                    "Excellent": "Yes, highly usable",
                    "Good": "Yes, usable",
                    "Fair": "Yes, partially usable",
                    "Poor": "Yes, barely usable",
                    "Very Poor": "No, not usable"
                }
                usability_value = mapping.get(quality_level, "Yes, partially usable")
            else:
                return None
        else:
            return None
    else:
        # Get usability value from either field
        if 'Usability' in annotation:
            usability_value = annotation['Usability']
        else:
            usability_value = annotation['Usibility']
            
        if isinstance(usability_value, dict) and 'choices' in usability_value:
            usability_str = ', '.join(usability_value['choices'])
        else:
            usability_str = str(usability_value)

        # Handle "no" and "yes" values from JSON annotations
        if usability_str.lower() == "no":
            usability_value = "No, not usable"
        elif usability_str.lower() == "yes":
            # If usability is "yes", determine level based on quality
            if 'Quality' in annotation:
                quality_level, _ = parse_quality_or_usability(annotation['Quality'])
                if quality_level:
                    # Map quality to usability when usability is "yes"
                    mapping = {
                        "Excellent": "Yes, highly usable",
                        "Good": "Yes, usable",
                        "Fair": "Yes, partially usable",
                        "Poor": "Yes, barely usable",
                        "Very Poor": "No, not usable"
                    }
                    usability_value = mapping.get(quality_level, "Yes, usable")
                else:
                    usability_value = "Yes, usable"  # Default for "yes" without quality info
            else:
                usability_value = "Yes, usable"  # Default for "yes" without quality info
        else:
            # Try to parse as level using existing parser
            level, _ = parse_quality_or_usability(usability_str)
            usability_value = level if level else usability_str

    possible_options = ["Yes, highly usable", "Yes, usable", "Yes, partially usable", "Yes, barely usable", "No, not usable"]

    if usability_value not in possible_options:
        return None

    options = {usability_value}
    distractors = [opt for opt in possible_options if opt != usability_value]
    while len(options) < min(4, len(possible_options)):
        options.add(random.choice(distractors))

    shuffled_options = random.sample(list(options), len(options))
    option_dict = {chr(65 + i): opt for i, opt in enumerate(shuffled_options)}
    correct_letter = [k for k, v in option_dict.items() if v == usability_value][0]

    return {
        "question_id": f"Sim3_UA_{uav_id}_{q_id}",
        "question_type": f"3.2 Usability Assessment ({uav_id})",
        "question": f"Is the image captured by {uav_id} usable for detecting drones, vehicles, pedestrians, and bicycles?",
        "options": option_dict,
        "correct_answer": correct_letter,
        "source": "Rule-Based from JSON"
    }


def call_chatgpt_api(messages, retries=3):
    """Universal ChatGPT API call function with retry and option diversity check from Sim5_CD.py"""
    for attempt in range(retries):
        try:
            # Adapt messages for OpenAI format
            adapted_messages = []
            for msg in messages:
                if msg['role'] == 'system':
                    adapted_messages.append({"role": "system", "content": msg['content']})
                elif msg['role'] == 'user':
                    content_list = []
                    for item in msg['content']:
                        if 'image' in item:
                            content_list.append({
                                "type": "image_url",
                                "image_url": {"url": item['image']}
                            })
                        elif 'text' in item:
                            content_list.append({
                                "type": "text",
                                "text": item['text']
                            })
                    adapted_messages.append({"role": "user", "content": content_list})
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=adapted_messages
            )
            content = response.choices[0].message.content
            # Try to parse JSON
            try:
                if isinstance(content, str):
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    if json_start != -1 and json_end != 0:
                        json_str = content[json_start:json_end]
                        result = json.loads(json_str)
                        if "options" in result:
                            is_diverse, issue = check_option_diversity(result["options"])
                            if not is_diverse:
                                return {"error": f"Option diversity check failed: {issue}"}
                        return result
                    else:
                        return {"error": "Unable to find valid JSON format", "content": content}
                else:
                    return {"error": "Response content format is incorrect", "content_type": str(type(content)), "content": content}
            except json.JSONDecodeError:
                return {"error": "JSON parsing failed", "raw_content": content}
        except Exception as e:
            if attempt < retries - 1:
                print(f"Retrying API call ({attempt + 1}/{retries}) due to exception: {str(e)}")
                time.sleep(1)
                continue
            return {"error": f"API call failed after {retries} attempts: {str(e)}"}


def check_option_diversity(options):
    """Check if options are sufficiently different from split scripts"""
    for i, opt1 in enumerate(options.values()):
        for j, opt2 in enumerate(options.values()):
            if i < j:
                similarity = difflib.SequenceMatcher(None, opt1, opt2).ratio()
                if similarity > 0.85:
                    return False, f"Options {i + 1} and {j + 1} too similar: {opt1} vs {opt2}"
    return True, ""


def evaluate_question_quality(result):
    """Evaluate the quality of generated questions"""
    if result is None:
        return {"quality": "ERROR", "issues": ["Result is None"]}
    if "error" in result:
        return {"quality": "ERROR", "issues": [result["error"]]}

    issues = []
    quality_score = 0

    # Check required fields
    required_fields = ["question_id", "question_type", "question", "options", "correct_answer"]
    for field in required_fields:
        if field not in result:
            issues.append(f"Missing required field: {field}")
        else:
            quality_score += 1

    # Check options format
    if "options" in result:
        options = result["options"]
        if not isinstance(options, dict):
            issues.append("Options must be a dictionary")
        elif len(options) != 4:
            issues.append("Must have exactly 4 options")
        elif not all(key in options for key in ["A", "B", "C", "D"]):
            issues.append("Options must have keys A, B, C, D")
        else:
            quality_score += 1

    # Check correct answer
    if "correct_answer" in result and "options" in result:
        correct = result["correct_answer"]
        options = result["options"]
        if correct not in options:
            issues.append("Correct answer must be one of the option keys (A, B, C, D)")
        else:
            quality_score += 1

    # Check question clarity
    if "question" in result:
        question = result["question"]
        if len(question) < 10:
            issues.append("Question too short")
        elif len(question) > 200:
            issues.append("Question too long")
        else:
            quality_score += 1

    # Cap quality_score at 4
    quality_score = min(quality_score, 4)

    # Determine quality level
    if quality_score == 4:
        quality = "EXCELLENT"
    elif quality_score == 3:
        quality = "GOOD"
    elif quality_score == 2:
        quality = "FAIR"
    else:
        quality = "POOR"

    return {
        "quality": quality,
        "score": quality_score,
        "issues": issues
    }


def try_generate_qa(func, *args, max_attempts=3, **kwargs):
    """Try generating QA with retries if it fails"""
    for attempt in range(max_attempts):
        result = func(*args, **kwargs)
        quality = evaluate_question_quality(result)
        if "error" not in result:
            return result, quality
        print(f"Attempt {attempt + 1} failed: {result.get('error')}. Retrying...")
        time.sleep(1)
    return {"error": f"Failed after {max_attempts} attempts"}, {"quality": "ERROR", "issues": ["Max attempts reached"]}


def generate_few_shot_quality_assessment_q(img_path, uav_id, q_id):
    """Generate quality assessment questions with few-shot examples (optimized for multi-UAV)"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' ability to assess image quality for perception tasks in multi-UAV views, with focus on drone, vehicle, pedestrian, and bicycle detection.

CRITICAL RULES:
1. ALWAYS respond in English only
2. Follow a structured thinking process: analyze → identify quality factors → formulate question → create options → verify correctness
3. Questions must be based on actual visual content or provided description
4. Each question should have exactly 4 options (A, B, C, D) with only one correct answer
5. Options should be plausible, distinct in meaning, and avoid minor rephrasing
6. Output must be valid JSON format only
7. Focus on quality factors that affect detection of drones, vehicles, pedestrians, and bicycles

THINKING PROCESS:
1. First, describe the quality factors (clarity, noise, color balance, etc.) in the image or description
2. Identify the focus based on generation index
3. Formulate a clear, specific question about image quality for target detection
4. Create 4 distinct options where only one is correct
5. Verify the question is unambiguous and answerable"""

    quality_types = [
        "overall image clarity for target detection",
        "presence of noise or distortion affecting drone/vehicle/pedestrian/bicycle visibility"
    ]

    few_shot_examples = """
EXAMPLES OF GOOD QUALITY ASSESSMENT QUESTIONS:

Example 1:
{
    "question_id": "Sim3_QA_UAV1_1001",
    "question_type": "3.1 Quality Assessment (UAV1)",
    "question": "How would you rate the image clarity for detecting drones and vehicles in this scene?",
    "options": {
        "A": "Excellent with sharp details on all targets",
        "B": "Good with minor blur on some objects",
        "C": "Fair with noticeable distortion affecting detection",
        "D": "Poor with significant artifacts obscuring targets"
    },
    "correct_answer": "A",
    "image_description": "The image shows excellent clarity with sharp details on drones, vehicles, pedestrians, and bicycles."
}

Example 2:
{
    "question_id": "Sim3_QA_UAV2_1002",
    "question_type": "3.1 Quality Assessment (UAV2)",
    "question": "What is the level of noise affecting the visibility of pedestrians and bicycles?",
    "options": {
        "A": "Minimal noise with clear target visibility",
        "B": "Moderate noise affecting some target areas",
        "C": "High noise throughout the image",
        "D": "No noticeable distortion on targets"
    },
    "correct_answer": "A",
    "image_description": "The image has minimal noise, providing clear visuals of all target objects including drones, vehicles, pedestrians, and bicycles."
}
"""

    user_prompt = f"""First, provide a brief description (50-100 words) of the quality factors affecting target detection (drones, vehicles, pedestrians, bicycles) in this image from {uav_id}.
Then, create a multiple-choice question about quality assessment based on this description.

REQUIREMENTS:
- Question should test understanding of image quality for target detection tasks
- Focus on: {quality_types[random.randint(0, 1)]}
- Ensure question is specific, unambiguous, and DISTINCT from examples in structure, focus, and vocabulary
- Create 4 plausible options with distinct meanings
- Use clear, professional English
- Include the image description in the output JSON as 'image_description'
- IMPORTANT: Use the EXACT question_id format provided in the JSON template
- Focus on quality factors that impact detection of drones, vehicles, pedestrians, and bicycles

{few_shot_examples}

JSON format:
{{
    "question_id": "Sim3_QA_{uav_id}_{q_id}",
    "question_type": "3.1 Quality Assessment ({uav_id})",
    "question": "...",
    "options": {{ "A": "...", "B": "...", "C": "...", "D": "..." }},
    "correct_answer": "...",
    "image_description": "..."
}}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"image": f"data:image/jpeg;base64,{encode_image(img_path)}"},
                {"text": user_prompt}
            ]
        }
    ]

    result = call_chatgpt_api(messages)
    if "error" not in result:
        result["source"] = "Model-Based (Few-Shot)"
        result["question_id"] = f"Sim3_QA_{uav_id}_{q_id}"
    return result


def generate_few_shot_usability_assessment_q(img_path, uav_id, q_id):
    """Generate usability assessment questions with few-shot examples (optimized for multi-UAV)"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' ability to assess image usability for perception tasks in multi-UAV views, with focus on drone, vehicle, pedestrian, and bicycle detection and tracking.

CRITICAL RULES:
1. ALWAYS respond in English only
2. Follow a structured thinking process: analyze → identify usability factors → formulate question → create options → verify correctness
3. Questions must be based on actual visual content or provided description
4. Each question should have exactly 4 options (A, B, C, D) with only one correct answer
5. Options should be plausible, distinct in meaning, and avoid minor rephrasing
6. Output must be valid JSON format only
7. Focus on usability factors that affect detection and tracking of drones, vehicles, pedestrians, and bicycles

THINKING PROCESS:
1. First, describe the usability factors (suitability for target detection, tracking, etc.) in the image or description
2. Identify the focus based on generation index
3. Formulate a clear, specific question about image usability for target tasks
4. Create 4 distinct options where only one is correct
5. Verify the question is unambiguous and answerable"""

    usability_types = [
        "usability for drone/vehicle/pedestrian/bicycle detection",
        "usability for tracking moving drones, vehicles, pedestrians, and bicycles"
    ]

    few_shot_examples = """
EXAMPLES OF GOOD USABILITY ASSESSMENT QUESTIONS:

Example 1:
{
    "question_id": "Sim3_UA_UAV1_1001",
    "question_type": "3.2 Usability Assessment (UAV1)",
    "question": "Is this image usable for detecting drones and vehicles?",
    "options": {
        "A": "Highly usable with clear target boundaries",
        "B": "Moderately usable with some target occlusions",
        "C": "Low usability due to heavy blurring of targets",
        "D": "Not usable due to severe perception degradation"
    },
    "correct_answer": "A",
    "image_description": "The image provides high usability for target detection with clear boundaries and good contrast for drones, vehicles, pedestrians, and bicycles."
}

Example 2:
{
    "question_id": "Sim3_UA_UAV2_1002",
    "question_type": "3.2 Usability Assessment (UAV2)",
    "question": "Is this image usable for tracking moving drones and vehicles?",
    "options": {
        "A": "Highly suitable with distinct motion cues for targets",
        "B": "Moderately suitable with some target overlapping",
        "C": "Low suitability due to severe motion blur on targets",
        "D": "Unsuitable due to low image quality for target tracking"
    },
    "correct_answer": "A",
    "image_description": "The scene shows moving drones and vehicles with distinct paths, highly suitable for tracking."
}
"""

    user_prompt = f"""First, provide a brief description (50-100 words) of the usability factors for target detection and tracking (drones, vehicles, pedestrians, bicycles) in this image from {uav_id}.
Then, create a multiple-choice question about usability assessment based on this description.

REQUIREMENTS:
- Question should test understanding of image usability for target detection and tracking tasks
- Focus on: {usability_types[random.randint(0, 1)]}
- Ensure question is specific, unambiguous, and DISTINCT from examples in structure, focus, and vocabulary
- Create 4 plausible options with distinct meanings
- Use clear, professional English
- Include the image description in the output JSON as 'image_description'
- IMPORTANT: Use the EXACT question_id format provided in the JSON template
- Focus on usability factors that impact detection and tracking of drones, vehicles, pedestrians, and bicycles

{few_shot_examples}

JSON format:
{{
    "question_id": "Sim3_UA_{uav_id}_{q_id}",
    "question_type": "3.2 Usability Assessment ({uav_id})",
    "question": "...",
    "options": {{ "A": "...", "B": "...", "C": "...", "D": "..." }},
    "correct_answer": "...",
    "image_description": "..."
}}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"image": f"data:image/jpeg;base64,{encode_image(img_path)}"},
                {"text": user_prompt}
            ]
        }
    ]

    result = call_chatgpt_api(messages)
    if "error" not in result:
        result["source"] = "Model-Based (Few-Shot)"
        result["question_id"] = f"Sim3_UA_{uav_id}_{q_id}"
    return result


def generate_few_shot_causal_assessment_q(img_path, uav_id, q_id):
    """Generate causal assessment questions with few-shot examples (optimized for multi-UAV)"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' ability to analyze causes of perception quality issues in multi-UAV views, with focus on drone, vehicle, pedestrian, and bicycle detection.

CRITICAL RULES:
1. ALWAYS respond in English only
2. Follow a structured thinking process: analyze image → identify potential causes → formulate question → create options → verify correctness
3. Questions must be based on actual visual content or provided description
4. Each question should have exactly 4 options (A, B, C, D) with only one correct answer
5. Options should be plausible, distinct in meaning, and avoid minor rephrasing
6. Output must be valid JSON format only
7. Focus on causes that affect detection of drones, vehicles, pedestrians, and bicycles

THINKING PROCESS:
1. First, analyze the image and identify potential causes of perception quality issues (e.g., blur, occlusion, lighting) affecting target detection
2. Then, formulate a question about the primary cause
3. Create 4 distinct options, with only one option being correct
4. Verify the question is unambiguous and answerable"""

    causal_types = [
        "primary cause of reduced target visibility",
        "main factor affecting drone/vehicle/pedestrian/bicycle detection"
    ]

    few_shot_examples = """
EXAMPLES OF GOOD CAUSAL ASSESSMENT QUESTIONS:

Example 1:
{
    "question_id": "Sim3_CA_UAV1_1001",
    "question_type": "3.3 Causal Assessment (UAV1)",
    "question": "What is the primary cause of reduced visibility for drones and vehicles in this image?",
    "options": {
        "A": "Heavy noise due to sensor failure",
        "B": "Overexposure due to bright sunlight",
        "C": "Motion blur from fast-moving targets",
        "D": "Occlusion by foreground trees"
    },
    "correct_answer": "C",
    "image_description": "The image shows blurred drones and vehicles due to motion, reducing target visibility."
}

Example 2:
{
    "question_id": "Sim3_CA_UAV2_1002",
    "question_type": "3.3 Causal Assessment (UAV2)",
    "question": "What main factor might affect detection of pedestrians and bicycles in this scene?",
    "options": {
        "A": "Low contrast due to overcast weather",
        "B": "High noise from image compression",
        "C": "Partial occlusion by buildings",
        "D": "Partial image loss due to sensor failure"
    },
    "correct_answer": "C",
    "image_description": "Pedestrians and bicycles are partially occluded by buildings, affecting target detection."
}

Example 3:
{
    "question_id": "Sim3_CA_UAV2_1003",
    "question_type": "3.3 Causal Assessment (UAV2)",
    "question": "What main factor might affect drone detection in this scene?",
    "options": {
        "A": "Severe noise from sensor failure",
        "B": "Low contrast due to lighting issues",
        "C": "Partial occlusion by other targets",
        "D": "Data loss due to incomplete image capture"
    },
    "correct_answer": "A",
    "image_description": "The image quality is degraded by severe noise caused by sensor failure, making drone detection challenging."
}

Example 4:
{
    "question_id": "Sim3_CA_UAV2_1004",
    "question_type": "3.3 Causal Assessment (UAV2)",
    "question": "What is the primary issue affecting the completeness of target observation in UAV2?",
    "options": {
        "A": "Severe image noise from sensor failure",
        "B": "Incomplete observation due to sensor malfunction",
        "C": "Targets being too far for clear detection",
        "D": "Partial image loss due to network interruption"
    },
    "correct_answer": "B",
    "image_description": "The image from UAV2 is missing crucial parts due to sensor malfunction, leading to incomplete target observations."
}

"""

    user_prompt = f"""First, analyze the image from {uav_id} and provide a brief description (50-100 words) of potential causes of perception quality issues affecting target detection (drones, vehicles, pedestrians, bicycles).
Then, create a multiple-choice question about causal assessment based on this analysis.

REQUIREMENTS:
- Question should test understanding of causes affecting target detection quality
- Focus on: {causal_types[random.randint(0, 1)]}
- Ensure question is specific, unambiguous, and DISTINCT from examples
- Create 4 plausible options with distinct meanings, with only one option being correct
- Use clear, professional English
- Include the analysis in 'image_description'
- IMPORTANT: Use the EXACT question_id format provided in the JSON template
- Focus on causes that impact detection of drones, vehicles, pedestrians, and bicycles

{few_shot_examples}

JSON format:
{{
    "question_id": "Sim3_CA_{uav_id}_{q_id}",
    "question_type": "3.3 Causal Assessment ({uav_id})",
    "question": "...",
    "options": {{ "A": "...", "B": "...", "C": "...", "D": "..." }},
    "correct_answer": "...",
    "image_description": "..."
}}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"image": f"data:image/jpeg;base64,{encode_image(img_path)}"},
                {"text": user_prompt}
            ]
        }
    ]

    result = call_chatgpt_api(messages)
    if "error" not in result:
        result["source"] = "Model-Based (Few-Shot)"
        result["question_id"] = f"Sim3_CA_{uav_id}_{q_id}"
    return result


def main():
    # Set image directory paths
    base_dir = "Samples/images"
    
    # Check if base directory exists
    if not os.path.exists(base_dir):
        print(f"Error: Base directory not found: {base_dir}")
        return

    # Set annotation file paths
    annotation_file = "Annotations/all_samples.json"

    if not os.path.exists(annotation_file):
        print(f"Error: Annotation file not found: {annotation_file}")
        return

    print("Loading data...")
    annotation_map = load_annotations(annotation_file)
    image_groups = get_image_groups(base_dir)

    print(f"Loaded {len(annotation_map)} annotation entries")
    print(f"Found {len(image_groups)} image groups to process")
    
    # Validate annotation coverage
    coverage_rate = validate_annotation_coverage(image_groups, annotation_map)
    if coverage_rate < 80:
        print(f"WARNING: Low annotation coverage ({coverage_rate:.1f}%). Some images may not have corresponding annotations.")
    else:
        print(f"Good annotation coverage ({coverage_rate:.1f}%).")

    all_results = {
        "dataset": "Sim_3_UAVs",  # Updated dataset name for 3 UAVs
        "total_groups": len(image_groups),
        "results": []
    }

    quality_stats = {"EXCELLENT": 0, "GOOD": 0, "FAIR": 0, "POOR": 0, "ERROR": 0}
    group_results = []

    # Initialize counters for sequential numbering per UAV (3 UAVs)
    counters = {f"UAV{i}": {"quality": 1, "usability": 1, "causal": 1} for i in range(1, 4)}

    # Process each image group
    for g_idx, group_data in enumerate(image_groups):
        print(f"\n--- Processing group {g_idx + 1}/{len(image_groups)}: {group_data['sequence_frame']} ---")

        group = group_data['group']
        group_paths = {item['uav']: item['path'] for item in group}
        group_filenames = {item['uav']: item['filename'] for item in group}
        group_scene = group[0]['scene'] if group else "unknown"
        group_frame = group[0]['frame'] if group else "unknown"

        group_questions = {
            "sequence_frame": group_data['sequence_frame'],
            "scene": group_scene,
            "frame": group_frame,
            "uav_paths": {f"UAV{k}": v for k, v in group_paths.items()},
            "uav_filenames": {f"UAV{k}": v for k, v in group_filenames.items()},
            "questions_per_uav": {}
        }

        for uav_num in range(1, 4):
            uav_id = f"UAV{uav_num}"
            current_path = group_paths[uav_num]
            current_filename = group_filenames[uav_num]

            print(f"  Processing {uav_id}")

            # Extract annotation information
            # For 3-UAV setup, we'll use scene and frame info to find annotations
            annotation = {}
            
            # Try multiple matching strategies to find annotation
            scene_frame_key = f"{group_scene}_frame_{group_frame}"
            uav_frame_key = f"UAV{uav_num}_frame_{group_frame}"
            full_key = f"{group_scene}_UAV{uav_num}_frame_{group_frame}"
            
            # Try to find annotation using different keys - prioritize the most specific match
            annotation = None
            matched_key = None
            
            # Priority 1: Full key (most specific - scene + UAV + frame)
            if full_key in annotation_map:
                annotation = annotation_map[full_key]
                matched_key = full_key
                print(f"    Found annotation using full_key: {full_key}")
            # Priority 2: Scene frame key (scene + frame)
            elif scene_frame_key in annotation_map:
                annotation = annotation_map[scene_frame_key]
                matched_key = scene_frame_key
                print(f"    Found annotation using scene_frame_key: {scene_frame_key}")
            # Priority 3: UAV frame key (UAV + frame) - least specific, may match wrong scene
            elif uav_frame_key in annotation_map:
                annotation = annotation_map[uav_frame_key]
                matched_key = uav_frame_key
                print(f"    Found annotation using uav_frame_key: {uav_frame_key}")
            else:
                # Fallback to original filename matching
                possible_filenames = normalize_filename_for_annotation(current_filename, group_scene)
                for filename in possible_filenames:
                    if filename in annotation_map:
                        annotation = annotation_map[filename]
                        matched_key = filename
                        print(f"    Found annotation using filename: {filename}")
                        break
                
                # Additional fallback: try to construct the original filename from all_samples.json format
                if annotation is None:
                    # Try to match the original filename format from all_samples.json
                    # Format: "8f2a9605-scene_004-UAV1_frame_005.jpg"
                    original_filename = f"8f2a9605-{group_scene}-UAV{uav_num}_frame_{group_frame}.jpg"
                    if original_filename in annotation_map:
                        annotation = annotation_map[original_filename]
                        matched_key = original_filename
                        print(f"    Found annotation using original filename: {original_filename}")
                    
                    # Try with different hash prefixes
                    if annotation is None:
                        for key in annotation_map.keys():
                            if f"{group_scene}-UAV{uav_num}_frame_{group_frame}" in key:
                                annotation = annotation_map[key]
                                matched_key = key
                                print(f"    Found annotation using pattern matching: {key}")
                                break

            # Extract annotation info
            annotation_info = extract_annotation_info(annotation)

            # Create simple info string for new format
            match = re.search(r'UAV(\d+)_frame_(\d+)', current_filename)
            if match:
                uav_num_from_filename = int(match.group(1))
                frame_num = match.group(2)
                json_info = f"Frame {frame_num} from {uav_id} in {group_scene}"
            else:
                json_info = f"Unknown filename format: {current_filename}"

            # Combine annotation information
            combined_info = f"{annotation_info}; {json_info}".strip('; ')

            # Print annotation information for debugging
            if annotation_info or json_info:
                print(f"    {uav_id} annotation info: {combined_info}")
            
            # Debug: Print detailed annotation information if found
            if annotation and matched_key:
                print(f"    {uav_id} matched annotation key: {matched_key}")
                if 'Quality' in annotation:
                    print(f"    {uav_id} Quality: {annotation['Quality']}")
                if 'Usability' in annotation:
                    print(f"    {uav_id} Usability: {annotation['Usability']}")
                elif 'Usibility' in annotation:
                    print(f"    {uav_id} Usibility: {annotation['Usibility']}")
                if 'Object_type' in annotation:
                    print(f"    {uav_id} Object type: {annotation['Object_type']}")
                if 'Object_count' in annotation:
                    print(f"    {uav_id} Object count: {annotation['Object_count']}")
            else:
                print(f"    {uav_id} WARNING: No annotation found!")
                print(f"    {uav_id} Tried keys: {scene_frame_key}, {uav_frame_key}, {full_key}")
                print(f"    {uav_id} Available keys in annotation_map: {list(annotation_map.keys())[:10]}...")

            group_questions["questions_per_uav"][uav_id] = {
                "annotation_info": annotation_info,
                "json_info": json_info,
                "combined_info": combined_info,
                "questions": []
            }

            uav_questions = group_questions["questions_per_uav"][uav_id]["questions"]
            pair_quality_stats = {"EXCELLENT": 0, "GOOD": 0, "FAIR": 0, "POOR": 0, "ERROR": 0}

            print("Generating rule-based questions...")

            # Task 3.1: Quality Assessment (based on JSON annotations)
            if annotation:
                quality_q = generate_rule_based_quality_q(annotation, uav_id, counters[uav_id]["quality"])
                quality_quality = evaluate_question_quality(quality_q)
                if quality_q:
                    print(f"  Generated quality question for {uav_id}")
                    print(f"Quality: {quality_quality['quality']} (Score: {quality_quality['score']})")
                    if quality_quality['issues']:
                        print(f"Issues: {quality_quality['issues']}")
                    uav_questions.append(quality_q)
                    pair_quality_stats[quality_quality['quality']] += 1
                    counters[uav_id]["quality"] += 1
                else:
                    # Fallback to few-shot model if rule fails
                    quality_q, quality_quality = try_generate_qa(generate_few_shot_quality_assessment_q,
                                                                 current_path, uav_id, counters[uav_id]["quality"])
                    if "error" not in quality_q:
                        print(f"  Generated fallback few-shot quality for {uav_id}")
                        print(f"Quality: {quality_quality['quality']} (Score: {quality_quality['score']})")
                        if quality_quality['issues']:
                            print(f"Issues: {quality_quality['issues']}")
                        uav_questions.append(quality_q)
                        pair_quality_stats[quality_quality['quality']] += 1
                        counters[uav_id]["quality"] += 1
                    else:
                        print(
                            f"  Failed to generate fallback Quality Q for {uav_id}: {quality_q.get('error', 'Unknown error')}")
                        pair_quality_stats["ERROR"] += 1

                usability_q = generate_rule_based_usability_q(annotation, uav_id, counters[uav_id]["usability"])
                quality_usability = evaluate_question_quality(usability_q)
                if usability_q:
                    print(f"  Generated usability question for {uav_id}")
                    print(f"Quality: {quality_usability['quality']} (Score: {quality_usability['score']})")
                    if quality_usability['issues']:
                        print(f"Issues: {quality_usability['issues']}")
                    uav_questions.append(usability_q)
                    pair_quality_stats[quality_usability['quality']] += 1
                    counters[uav_id]["usability"] += 1
                else:
                    # Fallback to few-shot model if rule fails
                    usability_q, quality_usability = try_generate_qa(generate_few_shot_usability_assessment_q,
                                                                     current_path, uav_id, counters[uav_id]["usability"])
                    if "error" not in usability_q:
                        print(f"  Generated fallback few-shot usability for {uav_id}")
                        print(f"Quality: {quality_usability['quality']} (Score: {quality_usability['score']})")
                        if quality_usability['issues']:
                            print(f"Issues: {quality_usability['issues']}")
                        uav_questions.append(usability_q)
                        pair_quality_stats[quality_usability['quality']] += 1
                        counters[uav_id]["usability"] += 1
                    else:
                        print(
                            f"  Failed to generate fallback Usability Q for {uav_id}: {usability_q.get('error', 'Unknown error')}")
                        pair_quality_stats["ERROR"] += 1

            print("Generating model-based questions...")

            # Task 3.3: Causal Assessment (for poor quality images)
            def generate_causal_if_low_quality(annotation, img_path, uav_id):
                if 'Quality' in annotation:
                    _, score = parse_quality_or_usability(annotation['Quality'])
                    if score is not None and score <= 5:  # Lowered to <=3 to include Fair
                        causal_q, quality_causal = try_generate_qa(generate_few_shot_causal_assessment_q, img_path, uav_id,
                                                                   counters[uav_id]["causal"])
                        if "error" not in causal_q:
                            print(f"  Generated causal assessment question for {uav_id} (quality score: {score})")
                            print(f"Quality: {quality_causal['quality']} (Score: {quality_causal['score']})")
                            if quality_causal['issues']:
                                print(f"Issues: {quality_causal['issues']}")
                            uav_questions.append(causal_q)
                            pair_quality_stats[quality_causal['quality']] += 1
                            counters[uav_id]["causal"] += 1
                        else:
                            print(
                                f"  Failed to generate Causal Assessment Q for {uav_id}: {causal_q.get('error', 'Unknown error')}")
                            pair_quality_stats["ERROR"] += 1

            if annotation:
                generate_causal_if_low_quality(annotation, current_path, uav_id)

            # Aggregate quality stats
            for k, v in pair_quality_stats.items():
                quality_stats[k] += v

        all_results["results"].append(group_questions)

        # Save intermediate results every 5 groups
        if (g_idx + 1) % 5 == 0:
            save_to_json(all_results, "VQA_Sim3_PA.json")
            print(f"Saved intermediate results after processing {g_idx + 1} groups")

    # Compute and add quality statistics
    all_results['quality_statistics'] = quality_stats

    # Save final results
    if save_to_json(all_results, "VQA_Sim3_PA.json"):
        print(f"\nAll results saved to VQA_Sim3_PA.json")
        print(f"Successfully processed {len(all_results['results'])} image groups")

        # Count total questions generated
        total_questions = sum(len(uav_data['questions']) for result in all_results['results'] for uav_data in result['questions_per_uav'].values())
        print(f"Total questions generated: {total_questions}")
    else:
        print("Failed to save final results")

    # Overall quality summary
    print(f"\n=== Overall Quality Summary ===")
    print(f"Quality distribution: {quality_stats}")

if __name__ == "__main__":
    main()
