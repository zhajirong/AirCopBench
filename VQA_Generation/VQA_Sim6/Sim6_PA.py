import base64
from PIL import Image
import io
import os
import json
import glob
import re  # For better annotation parsing and filename extraction
import difflib  # For SequenceMatcher
import openai  # Import OpenAI library
from openai import AzureOpenAI
import random
import time
from collections import defaultdict

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


def normalize_filename(filename):
    """Normalize filename by stripping prefix and handling time zone"""
    # Remove prefix if present (8 alnum chars + '-')
    parts = filename.split('-', 1)
    if len(parts) == 2 and len(parts[0]) == 8 and parts[0].isalnum():
        filename = parts[1]
    # Remove timezone '+0800' or appended '0800' in time
    # Find time pattern like -YYYY-MM-DD-HH-MM-SS+TZ or -YYYY-MM-DD-HH-MM-SSTZ
    match = re.search(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})(\+?\d{4})?', filename)
    if match:
        base_time = match.group(1)
        filename = filename.replace(match.group(0), base_time)
    return filename


def load_annotations(annotation_file):
    """Load annotation file"""
    try:
        with open(annotation_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)

        annotation_map = {}
        for annotation in annotations:
            if 'img1' in annotation:
                full_path = annotation['img1']
                filename = normalize_filename(os.path.basename(full_path))
                annotation_map[filename] = annotation

        return annotation_map
    except Exception as e:
        print(f"Error loading annotations: {str(e)}")
        return {}


def get_image_groups(uav_dirs):
    """Get corresponding image groups from 6 UAV folders"""
    image_map = defaultdict(list)

    for uav_dir in uav_dirs:
        images = glob.glob(os.path.join(uav_dir, "*.jpg"))
        for img_path in images:
            filename = os.path.basename(img_path)
            match = re.search(r'_UAV_(\d+)_', filename)
            if match:
                uav_num = int(match.group(1))
                group_key = filename.replace(f"_UAV_{uav_num}_", "_")
                image_map[group_key].append({
                    'uav': uav_num,
                    'path': img_path,
                    'filename': filename
                })

    # Filter complete groups with exactly 6 UAVs
    image_groups = []
    for key, group in sorted(image_map.items()):
        if len(group) == 6:
            group.sort(key=lambda x: x['uav'])  # Sort by UAV number
            image_groups.append({
                'sequence_frame': key,
                'group': group
            })

    return image_groups


def parse_json_annotation(json_file_path):
    """Parse JSON annotation file, extract object information (assuming similar structure to original XML)"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tracks = []
        if 'tracks' in data:
            for track in data['tracks']:
                track_info = {
                    'id': track.get('id'),
                    'label': track.get('label'),
                    'boxes': []
                }
                if 'boxes' in track:
                    for box in track['boxes']:
                        box_info = {
                            'frame': int(box.get('frame', 0)),
                            'occluded': int(box.get('occluded', 0)),
                            'outside': int(box.get('outside', 0)),
                            'xtl': float(box.get('xtl', 0)),
                            'ytl': float(box.get('ytl', 0)),
                            'xbr': float(box.get('xbr', 0)),
                            'ybr': float(box.get('ybr', 0))
                        }
                        track_info['boxes'].append(box_info)
                tracks.append(track_info)

        return tracks
    except Exception as e:
        print(f"Error parsing JSON file {json_file_path}: {str(e)}")
        return []


def get_json_annotation_for_frame(json_file_path, frame_number):
    """Get JSON annotation information for specified frame"""
    tracks = parse_json_annotation(json_file_path)

    frame_objects = []
    for track in tracks:
        for box in track['boxes']:
            if box['frame'] == frame_number and box['occluded'] == 0 and box['outside'] == 0:
                frame_objects.append({
                    'label': track['label'],
                    'track_id': track['id'],
                    'bbox': {
                        'xtl': box['xtl'],
                        'ytl': box['ytl'],
                        'xbr': box['xbr'],
                        'ybr': box['ybr']
                    }
                })

    return frame_objects


def get_json_annotation_for_scene(json_file_path):
    """Get JSON annotation information for the entire scene"""
    tracks = parse_json_annotation(json_file_path)

    scene_objects = []
    object_types = defaultdict(int)
    total_objects = 0
    visible_objects = 0
    occluded_objects = 0

    for track in tracks:
        label = track['label']
        object_types[label] += 1
        total_objects += 1

        has_visible = any(box['occluded'] == 0 and box['outside'] == 0 for box in track['boxes'])
        if has_visible:
            visible_objects += 1
        else:
            occluded_objects += 1

        for box in track['boxes']:
            if box['occluded'] == 0 and box['outside'] == 0:
                scene_objects.append({
                    'label': label,
                    'track_id': track['id'],
                    'occluded': box['occluded'],
                    'outside': box['outside'],
                    'bbox': {
                        'xtl': box['xtl'],
                        'ytl': box['ytl'],
                        'xbr': box['xbr'],
                        'ybr': box['ybr']
                    }
                })
                break

    return {
        'objects': scene_objects,
        'object_types': dict(object_types),
        'total_objects': total_objects,
        'visible_objects': visible_objects,
        'occluded_objects': occluded_objects
    }


def find_json_file_for_image(image_filename, json_base_dir):
    """Find corresponding JSON file for image filename"""
    parts = image_filename.split('-')
    if len(parts) >= 2:
        sequence_number = parts[0]

        match = re.search(r'UAV_(\d+)', image_filename)
        if match:
            uav_num = match.group(1)
            json_filename = f"{sequence_number}-{uav_num}.json"
            json_path = os.path.join(json_base_dir, json_filename)
            if os.path.exists(json_path):
                return json_path

    return None


def extract_frame_number_from_filename(image_filename):
    """Extract frame number from image filename"""
    parts = image_filename.rsplit('_', 1)
    if len(parts) == 2:
        frame_str = parts[1].replace('.jpg', '')
        if frame_str.isdigit():
            return int(frame_str)
    return 0


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

    if 'Usability' in annotation:
        usability = annotation['Usability']
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


def save_to_json(data, filename="VQA_MDMT_PA.json"):
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
            "question_id": f"MDMT_QA_{uav_id}_{q_id}",
            "question_type": f"3.1 Quality Assessment ({uav_id})",
            "question": f"What is the perception quality assessment score (1-5) for the image captured by {uav_id}?",
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
            "question_id": f"MDMT_QA_{uav_id}_{q_id}",
            "question_type": f"3.1 Quality Assessment ({uav_id})",
            "question": f"What is the perception quality assessment level for the image captured by {uav_id}?",
            "options": option_dict,
            "correct_answer": correct_letter,
            "source": "Rule-Based from JSON"
        }


def generate_rule_based_usability_q(annotation, uav_id, q_id):
    """
    [Rule-Based] Generate image usability assessment questions based on JSON annotation data.
    Improved with fallback if missing.
    """
    if 'Usability' not in annotation:
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
        usability_value = annotation['Usability']
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
        "question_id": f"MDMT_UA_{uav_id}_{q_id}",
        "question_type": f"3.2 Usability Assessment ({uav_id})",
        "question": f"Is the image captured by {uav_id} usable for target perception tasks?",
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
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' ability to assess image quality for perception tasks in multi-UAV views.

CRITICAL RULES:
1. ALWAYS respond in English only
2. Follow a structured thinking process: analyze → identify quality factors → formulate question → create options → verify correctness
3. Questions must be based on actual visual content or provided description
4. Each question should have exactly 4 options (A, B, C, D) with only one correct answer
5. Options should be plausible, distinct in meaning, and avoid minor rephrasing
6. Output must be valid JSON format only

THINKING PROCESS:
1. First, describe the quality factors (clarity, noise, color balance, etc.) in the image or description
2. Identify the focus based on generation index
3. Formulate a clear, specific question about image quality
4. Create 4 distinct options where only one is correct
5. Verify the question is unambiguous and answerable"""

    quality_types = [
        "overall image clarity",
        "presence of noise or distortion"
    ]

    few_shot_examples = """
EXAMPLES OF GOOD QUALITY ASSESSMENT QUESTIONS:

Example 1:
{
    "question_id": "MDMT_QA_UAV1_1001",
    "question_type": "3.1 Quality Assessment (UAV1)",
    "question": "How would you rate the overall image clarity in this scene?",
    "options": {
        "A": "Excellent with sharp details",
        "B": "Good with minor blur",
        "C": "Fair with noticeable distortion",
        "D": "Poor with significant artifacts"
    },
    "correct_answer": "A",
    "image_description": "The image shows excellent clarity with sharp details on all objects."
}

Example 2:
{
    "question_id": "MDMT_QA_UAV2_1002",
    "question_type": "3.1 Quality Assessment (UAV2)",
    "question": "What is the level of noise or distortion in this image?",
    "options": {
        "A": "Minimal noise with clear visuals",
        "B": "Moderate noise affecting some areas",
        "C": "High noise throughout the image",
        "D": "No noticeable distortion"
    },
    "correct_answer": "A",
    "image_description": "The image has minimal noise, providing clear visuals of the scene."
}
"""

    user_prompt = f"""First, provide a brief description (50-100 words) of the quality factors in this image from {uav_id}.
Then, create a multiple-choice question about quality assessment based on this description.

REQUIREMENTS:
- Question should test understanding of image quality for perception tasks
- Focus on: {quality_types[random.randint(0, 1)]}
- Ensure question is specific, unambiguous, and DISTINCT from examples in structure, focus, and vocabulary
- Create 4 plausible options with distinct meanings
- Use clear, professional English
- Include the image description in the output JSON as 'image_description'
- IMPORTANT: Use the EXACT question_id format provided in the JSON template

{few_shot_examples}

JSON format:
{{
    "question_id": "MDMT_QA_{uav_id}_{q_id}",
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
        result["question_id"] = f"MDMT_QA_{uav_id}_{q_id}"
    return result


def generate_few_shot_usability_assessment_q(img_path, uav_id, q_id):
    """Generate usability assessment questions with few-shot examples (optimized for multi-UAV)"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' ability to assess image usability for perception tasks in multi-UAV views.

CRITICAL RULES:
1. ALWAYS respond in English only
2. Follow a structured thinking process: analyze → identify usability factors → formulate question → create options → verify correctness
3. Questions must be based on actual visual content or provided description
4. Each question should have exactly 4 options (A, B, C, D) with only one correct answer
5. Options should be plausible, distinct in meaning, and avoid minor rephrasing
6. Output must be valid JSON format only

THINKING PROCESS:
1. First, describe the usability factors (suitability for object detection, tracking, etc.) in the image or description
2. Identify the focus based on generation index
3. Formulate a clear, specific question about image usability
4. Create 4 distinct options where only one is correct
5. Verify the question is unambiguous and answerable"""

    usability_types = [
        "usability for object detection",
        "usability for tracking moving objects"
    ]

    few_shot_examples = """
EXAMPLES OF GOOD USABILITY ASSESSMENT QUESTIONS:

Example 1:
{
    "question_id": "MDMT_UA_UAV1_1001",
    "question_type": "3.2 Usability Assessment (UAV1)",
    "question": "Is this image usable for object detection tasks?",
    "options": {
        "A": "Highly usable with clear object boundaries",
        "B": "Moderately usable with some occlusions",
        "C": "Low usability due to heavy blurring",
        "D": "Not usable due to severe perception degradation"
    },
    "correct_answer": "A",
    "image_description": "The image provides high usability for object detection with clear boundaries and good contrast."
}

Example 2:
{
    "question_id": "MDMT_UA_UAV2_1002",
    "question_type": "3.2 Usability Assessment (UAV2)",
    "question": "Is this image usable for tracking moving objects?",
    "options": {
        "A": "Highly suitable with distinct motion cues",
        "B": "Moderately suitable with some overlapping",
        "C": "Low suitability due to severe motion blur",
        "D": "Unsuitable due to low image quality "
    },
    "correct_answer": "A",
    "image_description": "The scene shows moving vehicles with distinct paths, highly suitable for tracking."
}
"""

    user_prompt = f"""First, provide a brief description (50-100 words) of the usability factors in this image from {uav_id}.
Then, create a multiple-choice question about usability assessment based on this description.

REQUIREMENTS:
- Question should test understanding of image usability for perception tasks
- Focus on: {usability_types[random.randint(0, 1)]}
- Ensure question is specific, unambiguous, and DISTINCT from examples in structure, focus, and vocabulary
- Create 4 plausible options with distinct meanings
- Use clear, professional English
- Include the image description in the output JSON as 'image_description'
- IMPORTANT: Use the EXACT question_id format provided in the JSON template

{few_shot_examples}

JSON format:
{{
    "question_id": "MDMT_UA_{uav_id}_{q_id}",
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
        result["question_id"] = f"MDMT_UA_{uav_id}_{q_id}"
    return result


def generate_few_shot_causal_assessment_q(img_path, uav_id, q_id):
    """Generate causal assessment questions with few-shot examples (optimized for multi-UAV)"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' ability to analyze causes of perception quality issues in multi-UAV views.

CRITICAL RULES:
1. ALWAYS respond in English only
2. Follow a structured thinking process: analyze image → identify potential causes → formulate question → create options → verify correctness
3. Questions must be based on actual visual content or provided description
4. Each question should have exactly 4 options (A, B, C, D) with only one correct answer
5. Options should be plausible, distinct in meaning, and avoid minor rephrasing
6. Output must be valid JSON format only

THINKING PROCESS:
1. First, analyze the image and identify potential causes of perception quality issues (e.g., blur, occlusion, lighting)
2. Then, formulate a question about the primary cause
3. Create 4 distinct options, with only one option being correct
4. Verify the question is unambiguous and answerable"""

    causal_types = [
        "primary cause of reduced visibility",
        "main factor affecting object detection"
    ]

    few_shot_examples = """
EXAMPLES OF GOOD CAUSAL ASSESSMENT QUESTIONS:

Example 1:
{
    "question_id": "MDMT_CA_UAV1_1001",
    "question_type": "3.3 Causal Assessment (UAV1)",
    "question": "What is the primary cause of reduced visibility in this image?",
    "options": {
        "A": "Heavy noise due to sensor failure",
        "B": "Overexposure due to bright sunlight",
        "C": "Motion blur from fast-moving vehicles",
        "D": "Occlusion by foreground trees"
    },
    "correct_answer": "C",
    "image_description": "The image shows blurred vehicles due to motion, reducing visibility."
}

Example 2:
{
    "question_id": "MDMT_CA_UAV2_1002",
    "question_type": "3.3 Causal Assessment (UAV2)",
    "question": "What main factor might affect object detection in this scene?",
    "options": {
        "A": "Low contrast due to overcast weather",
        "B": "High noise from image compression",
        "C": "Partial occlusion by buildings",
        "D": "Partial image loss due to sensor failure"
    },
    "correct_answer": "C",
    "image_description": "Objects are partially occluded by buildings, affecting detection."
}

Example 3:
{
    "question_id": "MDMT_CA_UAV2_1003",
    "question_type": "3.3 Causal Assessment (UAV2)",
    "question": "What main factor might affect object detection in this scene?",
    "options": {
        "A": "Severe noise from sensor failure",
        "B": "Low contrast due to lighting issues",
        "C": "Partial occlusion by other vehicles",
        "D": "Data loss due to incomplete image capture"
    },
    "correct_answer": "A",
    "image_description": "The image quality is degraded by severe noise caused by sensor failure, making object detection challenging."
}

Example 4:
{
    "question_id": "MDMT_CA_UAV2_1004",
    "question_type": "3.3 Causal Assessment (UAV2)",
    "question": "What is the primary issue affecting the completeness of the scene in UAV2?",
    "options": {
        "A": "Severe image noise from sensor failure",
        "B": "Incomplete observation due to sensor malfunction",
        "C": "Objects being too far for clear detection",
        "D": "Partial image loss due to network interruption"
    },
    "correct_answer": "B",
    "image_description": "The image from UAV2 is missing crucial parts due to sensor malfunction, leading to incomplete observations."
}

"""

    user_prompt = f"""First, analyze the image from {uav_id} and provide a brief description (50-100 words) of potential causes of perception quality issues.
Then, create a multiple-choice question about causal assessment based on this analysis.

REQUIREMENTS:
- Question should test understanding of causes affecting perception quality
- Focus on: {causal_types[random.randint(0, 1)]}
- Ensure question is specific, unambiguous, and DISTINCT from examples
- Create 4 plausible options with distinct meanings, with only one option being correct
- Use clear, professional English
- Include the analysis in 'image_description'
- IMPORTANT: Use the EXACT question_id format provided in the JSON template

{few_shot_examples}

JSON format:
{{
    "question_id": "MDMT_CA_{uav_id}_{q_id}",
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
        result["question_id"] = f"MDMT_CA_{uav_id}_{q_id}"
    return result


def main():
    # Set image directory paths
    base_dir = "/Users/starryyu/Documents/tinghuasummer/Sim_6_UAVs/Samples_testpa"
    uav_dirs = [os.path.join(base_dir, f"UAV{i}") for i in range(1, 7)]

    # Set annotation file paths
    annotation_file = "/Users/starryyu/Documents/tinghuasummer/Sim_6_UAVs/Annotations/all_samples.json"

    # Set JSON annotation file paths (replaces XML)
    json_base_dir = "/Users/starryyu/Documents/tinghuasummer/Sim_6_UAVs/Annotations/original_json"

    # Check if directories exist
    for uav_dir in uav_dirs:
        if not os.path.exists(uav_dir):
            print(f"Error: UAV directory not found: {uav_dir}")
            return

    if not os.path.exists(annotation_file):
        print(f"Error: Annotation file not found: {annotation_file}")
        return

    if not os.path.exists(json_base_dir):
        print(f"Warning: JSON directory not found: {json_base_dir}")
        print("Continuing without JSON annotations...")

    print("Loading data...")
    annotation_map = load_annotations(annotation_file)
    image_groups = get_image_groups(uav_dirs)

    print(f"Loaded {len(annotation_map)} annotation entries")
    print(f"Found {len(image_groups)} image groups to process")

    all_results = {
        "dataset": "Sim_6_UAVs",
        "total_groups": len(image_groups),
        "results": []
    }

    quality_stats = {"EXCELLENT": 0, "GOOD": 0, "FAIR": 0, "POOR": 0, "ERROR": 0}
    group_results = []

    # Initialize counters for sequential numbering
    counters = {f"UAV{i}": {"quality": 1, "usability": 1, "causal": 1} for i in range(1, 7)}

    # Process each image group
    for g_idx, group_data in enumerate(image_groups):
        print(f"\n--- Processing group {g_idx + 1}/{len(image_groups)}: {group_data['sequence_frame']} ---")

        group = group_data['group']
        group_paths = {item['uav']: item['path'] for item in group}
        group_filenames = {item['uav']: item['filename'] for item in group}

        group_questions = {
            "sequence_frame": group_data['sequence_frame'],
            "uav_paths": {f"UAV{k}": v for k, v in group_paths.items()},
            "uav_filenames": {f"UAV{k}": v for k, v in group_filenames.items()},
            "questions_per_uav": {}
        }

        for uav_num in range(1, 7):
            uav_id = f"UAV{uav_num}"
            current_path = group_paths[uav_num]
            current_filename = group_filenames[uav_num]

            print(f"  Processing {uav_id}")

            # Extract annotation information
            possible_filenames = [normalize_filename(current_filename)]
            annotation = {}
            for filename in possible_filenames:
                if filename in annotation_map:
                    annotation = annotation_map[filename]
                    break

            # Extract annotation info
            annotation_info = extract_annotation_info(annotation)

            # Extract JSON annotation information - Now use frame-level, fallback to scene if 0
            json_info = ""
            frame_objects = []

            frame_number = extract_frame_number_from_filename(current_filename)

            # For current UAV
            json_file = find_json_file_for_image(current_filename, json_base_dir)
            if json_file:
                frame_objects = get_json_annotation_for_frame(json_file, frame_number)
                if not frame_objects:
                    scene_info = get_json_annotation_for_scene(json_file)
                    frame_objects = scene_info['objects']  # Fallback to scene visible
                json_info = f"Frame objects: {len(frame_objects)} visible objects"

            # Combine annotation information
            combined_info = f"{annotation_info}; {json_info}".strip('; ')

            # Print annotation information for debugging
            if annotation_info or json_info:
                print(f"    {uav_id} annotation info: {combined_info}")

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
            save_to_json(all_results, "VQA_MDMT_PA.json")
            print(f"Saved intermediate results after processing {g_idx + 1} groups")

    # Compute and add quality statistics
    all_results['quality_statistics'] = quality_stats

    # Save final results
    if save_to_json(all_results, "VQA_MDMT_PA.json"):
        print(f"\nAll results saved to VQA_MDMT_PA.json")
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
