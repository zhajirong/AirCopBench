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
from collections import defaultdict  # Added import for defaultdict

"""
Collaborative Decision Script - Tasks 4.1, 4.2, 4.3, 4.4
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


def normalize_filename_for_annotation(filename):
    """Normalize filename to match annotation file format (no suffixes in new dataset)"""
    # Replace '+' with '' to match annotation time format
    normalized = filename.replace('+', '')
    return [normalized]


def load_annotations(annotation_file):
    """Load annotation file"""
    try:
        with open(annotation_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)

        annotation_map = {}
        for annotation in annotations:
            if 'img1' in annotation:
                full_filename = os.path.basename(annotation['img1'])
                # Strip hash prefix if present (e.g., 3c5ed8a3-)
                parts = full_filename.split('-', 1)
                if len(parts) == 2 and len(parts[0]) == 8 and parts[0].isalnum():
                    filename = parts[1]
                else:
                    filename = full_filename
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


def save_to_json(data, filename="VQA_MDMT_CD.json"):
    """Save data to JSON file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Failed to save file: {str(e)}")
        return False


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


def generate_rule_based_collaboration_when_q(annotation, uav_id, counter=1):
    """Generate rule-based collaboration timing questions with few-shot examples (optimized for multi-UAV)"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' understanding of when collaboration between multiple UAVs (up to 6) is necessary.

CRITICAL RULES:
1. ALWAYS respond in English only
2. Follow a structured thinking process: analyze annotation → determine collaboration need → formulate question → create options → verify correctness
3. Questions must be based on annotation data
4. Each question should have exactly 4 options (A, B, C, D) with only one correct answer
5. Options should be plausible, distinct in meaning, and avoid minor rephrasing
6. Output must be valid JSON format only

THINKING PROCESS:
1. Analyze the annotation to determine if collaboration is needed
2. Formulate a clear question about the need for collaboration
3. Create 4 distinct options where only one is correct
4. Verify the question is unambiguous and answerable"""

    collaboration_types = [
        "need for collaboration due to incomplete information from multiple views",
        "need for collaboration due to environmental factors across UAVs"
    ]

    few_shot_examples = """
EXAMPLES OF GOOD COLLABORATION WHEN QUESTIONS:

Example 1:
{
    "question_id": "MDMT_when2col_UAV1_1001",
    "question_type": "4.1 When to Collaborate (UAV1)",
    "question": "Should UAV1 collaborate with other UAVs to obtain supplementary information due to incomplete observation data in the multi-UAV setup?",
    "options": {
        "A": "Yes, due to partial occlusion of key objects across views",
        "B": "No, all views are fully visible",
        "C": "Yes, due to poor visibility in multiple UAV views",
        "D": "No, all objects are clearly visible in all UAVs"
    },
    "correct_answer": "A",
    "annotation_info": "Annotation indicates partial occlusion requiring collaboration across UAVs."
}

Example 2:
{
    "question_id": "MDMT_when2col_UAV2_1002",
    "question_type": "4.1 When to Collaborate (UAV2)",
    "question": "Should UAV2 collaborate with other UAVs to address environmental challenges for better perception across multiple UAV views?",
    "options": {
        "A": "Yes, due to poor lighting conditions in multiple views",
        "B": "No, lighting is adequate across all UAVs",
        "C": "Yes, due to low image resolution in some views",
        "D": "No, the environment is clear for all UAVs"
    },
    "correct_answer": "A",
    "annotation_info": "Annotation indicates poor lighting conditions necessitating collaboration across multiple UAVs."
}
"""

    if not annotation or 'Collaboration_when' not in annotation:
        return None

    collaboration_when_value = annotation['Collaboration_when']

    correct_answer = "Yes" if "1" in collaboration_when_value else "No"

    focus = collaboration_types[random.randint(0, 1)]
    if focus == "need for collaboration due to incomplete information from multiple views":
        options = {
            "A": "Yes, due to partial occlusion of key objects across views",
            "B": "No, all views are fully visible",
            "C": "Yes, due to poor visibility in multiple UAV views",
            "D": "No, all objects are clearly captured in all UAVs"
        }
        correct_option = "A" if correct_answer == "Yes" else "B"
    else:
        options = {
            "A": "Yes, due to poor lighting conditions in multiple views",
            "B": "No, lighting is adequate across all UAVs",
            "C": "Yes, due to low image resolution in some views",
            "D": "No, the environment is clear for all UAVs"
        }
        correct_option = "A" if correct_answer == "Yes" else "B"

    result = {
        "question_id": f"MDMT_when2col_{uav_id}_{counter}",
        "question_type": f"4.1 When to Collaborate ({uav_id})",
        "question": f"Should {uav_id} collaborate with other UAVs to address {focus}?",
        "options": options,
        "correct_answer": correct_option,
        "source": "Rule-Based (Few-Shot)",
        "annotation_info": f"Annotation indicates: {collaboration_when_value}"
    }

    return result


def generate_few_shot_collaboration_what_q(current_path, other_paths, uav_id, counter=1):
    """Generate model-based collaboration content questions with few-shot examples (optimized for multi-UAV)"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' understanding of what specific information should be shared between multiple UAVs (up to 6).

CRITICAL RULES:
1. ALWAYS respond in English only
2. Follow a structured thinking process: analyze images → identify information gaps → formulate question → create options → verify correctness
3. Questions must be based on actual visual content or provided descriptions
4. Each question should have exactly 4 options (A, B, C, D) with at least one correct answer
5. Options should be plausible, distinct in meaning, and avoid minor rephrasing
6. Output must be valid JSON format only

THINKING PROCESS:
1. Analyze all images to identify information gaps or complementary data across multiple UAV views
2. Identify the focus based on generation index
3. Formulate a clear question about what specific information to collaborate on
4. Create 4 distinct options, with at least one being correct
5. Verify the question is unambiguous and answerable"""

    collaboration_types = [
        "specific object information (e.g., vehicle/pedestrian/bicycle details across views)",
        "scene context information (e.g., traffic flow from multiple UAVs)"
    ]

    few_shot_examples = """
EXAMPLES OF GOOD COLLABORATION WHAT QUESTIONS:

Example 1:
{
    "question_id": "MDMT_what2col_UAV1_1001",
    "question_type": "4.2 What to Collaborate (UAV1)",
    "question": "What specific object information should UAV1 share with other UAVs to improve multi-view perception?",
    "options": {
        "A": "Details of the red car's position and movement across views",
        "B": "Details of the road's lane markings from all UAVs",
        "C": "Details of the traffic light status in multiple views",
        "D": "Details of the surrounding buildings from different angles"
    },
    "correct_answer": "A",
    "image_description": "UAV1 shows a red car moving on a multi-lane road, partially occluded in other UAV views."
}

Example 2:
{
    "question_id": "MDMT_what2col_UAV2_1002",
    "question_type": "4.2 What to Collaborate (UAV2)",
    "question": "What scene context information should UAV2 share with other UAVs to enhance multi-UAV understanding?",
    "options": {
        "A": "Traffic flow patterns across the intersection from multiple views",
        "B": "Weather conditions affecting visibility in all UAVs",
        "C": "Types of vehicles on the road from different angles",
        "D": "Layout of pedestrian crossings in multi-view setup"
    },
    "correct_answer": "A",
    "image_description": "UAV2 provides a broader view of traffic flow at an intersection, complementing other UAV views."
}
"""

    user_prompt = f"""First, provide a brief description (50-100 words) of the information gaps or complementary data between the image from {uav_id} (first image) and the images from other UAVs (subsequent images).
Then, create a multiple-choice question about what specific information {uav_id} should collaborate on, based on this description.

REQUIREMENTS:
- Question should test understanding of what data to share across multiple UAVs
- Focus on: {collaboration_types[random.randint(0, 1)]}
- Ensure question is specific, unambiguous, and DISTINCT from examples
- Create 4 plausible options with distinct meanings
- Use clear, professional English
- Include the image description in the output JSON as 'image_description'
- IMPORTANT: Use the EXACT question_id format provided in the JSON template

{few_shot_examples}

JSON format:
{{
    "question_id": "MDMT_what2col_{uav_id}_{counter}",
    "question_type": "4.2 What to Collaborate ({uav_id})",
    "question": "...",
    "options": {{ "A": "...", "B": "...", "C": "...", "D": "..." }},
    "correct_answer": "...",
    "image_description": "..."
}}"""

    content_list = [
        {"image": f"data:image/jpeg;base64,{encode_image(current_path)}"}  # First: current UAV
    ] + [
        {"image": f"data:image/jpeg;base64,{encode_image(p)}"} for p in other_paths  # Others
    ] + [
        {"text": user_prompt}
    ]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content_list}
    ]

    result = call_chatgpt_api(messages)
    if "error" not in result:
        result["source"] = "Model-Based (Few-Shot)"
        result["question_id"] = f"MDMT_what2col_{uav_id}_{counter}"
    return result


def generate_rule_based_collaboration_who_q_with_annotation(annotation, uav_id, counter=1):
    """Generate rule-based collaboration partner questions with few-shot examples (optimized for multi-UAV)"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' understanding of which UAV(s) should be the collaboration partner in a multi-UAV setup (up to 6 UAVs).

CRITICAL RULES:
1. ALWAYS respond in English only
2. Follow a structured thinking process: analyze annotation → determine collaboration partner → formulate question → create options → verify correctness
3. Questions must be based on annotation data
4. Each question should have exactly 4 options (A, B, C, D) with only one correct answer
5. Options should be plausible, distinct in meaning, and avoid minor rephrasing
6. Output must be valid JSON format only

THINKING PROCESS:
1. Analyze the annotation to identify the collaboration partner
2. Formulate a clear question about the collaboration partner
3. Create 4 distinct options where only one is correct
4. Verify the question is unambiguous and answerable"""

    collaboration_types = [
        "collaboration partner for complementary perspective in multi-UAV setup",
        "collaboration partner for specific object data across UAVs"
    ]

    few_shot_examples = """
EXAMPLES OF GOOD COLLABORATION WHO QUESTIONS:

Example 1:
{
    "question_id": "MDMT_who2col_UAV1_1001",
    "question_type": "4.3 Who to Collaborate (UAV1)",
    "question": "Which UAV should UAV1 collaborate with to gain a complementary perspective in the multi-UAV setup?",
    "options": {
        "A": "UAV2",
        "B": "None (no need for collaboration)",
        "C": "A ground-based sensor",
        "D": "UAV3"
    },
    "correct_answer": "A",
    "annotation_info": "Annotation indicates UAV2 as the collaboration partner."
}

Example 2:
{
    "question_id": "MDMT_who2col_UAV2_1002",
    "question_type": "4.3 Who to Collaborate (UAV2)",
    "question": "Which UAV should UAV2 collaborate with to obtain specific object data in the multi-UAV setup?",
    "options": {
        "A": "UAV1",
        "B": "None (no need for collaboration)",
        "C": "UAV4",
        "D": "A satellite system"
    },
    "correct_answer": "A",
    "annotation_info": "Annotation indicates UAV1 as the collaboration partner."
}
"""

    if not annotation or 'Collaboration_who' not in annotation:
        return None

    partner = annotation['Collaboration_who']
    if isinstance(partner, str):
        partner = partner.strip().upper()

    if partner in ["NONE", "NO"]:
        correct_answer = "None"
    elif "UAV" in partner:
        correct_answer = partner.replace('_', '')
    else:
        return None

    # Generate plausible wrong options
    all_uavs = [f"UAV{i}" for i in range(1, 7) if f"UAV{i}" != uav_id and f"UAV{i}" != correct_answer]
    wrong_uav1 = random.choice(all_uavs)
    all_uavs.remove(wrong_uav1)
    wrong_uav2 = random.choice(all_uavs) if all_uavs else "A ground-based sensor"

    options = {
        "A": correct_answer,
        "B": "None (no need for collaboration)" if correct_answer != "None" else wrong_uav1,
        "C": wrong_uav1,
        "D": wrong_uav2
    }
    correct_option = "A"

    focus = collaboration_types[random.randint(0, 1)]
    result = {
        "question_id": f"MDMT_who2col_{uav_id}_{counter}",
        "question_type": f"4.3 Who to Collaborate ({uav_id})",
        "question": f"Which UAV should {uav_id} collaborate with for {focus}?",
        "options": options,
        "correct_answer": correct_option,
        "source": "Rule-Based (Few-Shot)",
        "annotation_info": f"Annotation indicates: {partner}"
    }

    return result


def generate_hybrid_collaboration_why_q(current_path, other_paths, uav_id, annotation, counter=1):
    """Generate hybrid collaboration reason questions with few-shot examples (optimized for multi-UAV)"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' understanding of why collaboration between multiple UAVs (up to 6) is necessary.

CRITICAL RULES:
1. ALWAYS respond in English only
2. Follow a structured thinking process: analyze images and annotation → identify reasons for collaboration → formulate question → create options → verify correctness
3. Questions must integrate annotation data and visual content
4. Each question should have exactly 4 options (A, B, C, D) with only one correct answer
5. Options should be plausible, distinct in meaning, and avoid minor rephrasing
6. Output must be valid JSON format only

THINKING PROCESS:
1. Analyze annotation and images to identify reasons for collaboration across multiple UAV views
2. Formulate a clear question about the reason for collaboration
3. Create 4 distinct options, with only one being correct
4. Verify the question is unambiguous and answerable"""

    reason_types = [
        "reason due to visibility limitations in multi-UAV views",
        "reason due to information gaps across UAVs"
    ]

    few_shot_examples = """
EXAMPLES OF GOOD COLLABORATION WHY QUESTIONS:

Example 1:
{
    "question_id": "MDMT_why2col_UAV1_1001",
    "question_type": "4.4 Why to Collaborate (UAV1)",
    "question": "What is the main reason UAV1 should collaborate with other UAVs?",
    "options": {
        "A": "To overcome partial occlusion of the objects across multiple views",
        "B": "To adjust for lighting conditions in multi-view setup",
        "C": "To capture a wider area from different UAV angles",
        "D": "To reduce computational load"
    },
    "correct_answer": "A",
    "image_description": "UAV1 shows a scene with objects partially occluded; other UAVs provide clearer views."
}

Example 2:
{
    "question_id": "MDMT_why2col_UAV2_1002",
    "question_type": "4.4 Why to Collaborate (UAV2)",
    "question": "What is the primary reason UAV2 should collaborate with other UAVs?",
    "options": {
        "A": "To obtain more clear information about specific objects across views",
        "B": "To compensate for low image quality in some views",
        "C": "To supplement missing information due to limited field of view (FoV) in multi-UAV setup",
        "D": "To synchronize time stamps"
    },
    "correct_answer": "A",
    "image_description": "UAV2 lacks clear pedestrian movement data; other UAVs provide complementary details."
}
"""

    reasons = []
    if 'Collaboration_why' in annotation:
        why = annotation['Collaboration_why']
        if isinstance(why, dict) and 'choices' in why:
            reasons = why['choices']
        elif isinstance(why, list):
            reasons = why
        elif isinstance(why, str):
            reasons = [why]
    reasons_text = ", ".join(reasons) if reasons else "No specific reasons provided."

    focus = reason_types[random.randint(0, 1)]
    user_prompt = f"""First, provide a brief description (50-100 words) of the reasons for collaboration based on the image from {uav_id} (first image) and images from other UAVs (subsequent images), and annotation: '{reasons_text}'.
Then, create a multiple-choice question about why {uav_id} should collaborate, integrating annotation and image analysis.

REQUIREMENTS:
- Question should test understanding of collaboration reasons across multiple UAVs
- Focus on: {focus}
- Ensure question is specific, unambiguous, and DISTINCT from examples
- Create 4 plausible options with distinct meanings, with only one being correct
- Use clear, professional English
- Include the description in 'image_description'
- IMPORTANT: Use the EXACT question_id format provided in the JSON template

{few_shot_examples}

JSON format:
{{
    "question_id": "MDMT_why2col_{uav_id}_{counter}",
    "question_type": "4.4 Why to Collaborate ({uav_id})",
    "question": "...",
    "options": {{ "A": "...", "B": "...", "C": "...", "D": "..." }},
    "correct_answer": "...",
    "image_description": "..."
}}"""

    content_list = [
        {"image": f"data:image/jpeg;base64,{encode_image(current_path)}"}  # First: current UAV
    ] + [
        {"image": f"data:image/jpeg;base64,{encode_image(p)}"} for p in other_paths  # Others
    ] + [
        {"text": user_prompt}
    ]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content_list}
    ]

    result = call_chatgpt_api(messages)
    if "error" not in result:
        result["source"] = "Rule-Based & Model-Based (Few-Shot)"
        result["annotation_info"] = reasons_text
        result["question_id"] = f"MDMT_why2col_{uav_id}_{counter}"
    return result


def generate_model_based_collaboration_when_q(current_path, other_paths, uav_id, counter=1):
    """Fallback model-based for collaboration when if rule-based fails (optimized for multi-UAV)"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' understanding of when collaboration between multiple UAVs (up to 6) is necessary.

CRITICAL RULES:
1. ALWAYS respond in English only
2. Follow a structured thinking process: analyze images → determine collaboration need → formulate question → create options → verify correctness
3. Questions must be based on actual visual content
4. Each question should have exactly 4 options (A, B, C, D) with only one correct answer
5. Options should be plausible, distinct in meaning, and avoid minor rephrasing
6. Output must be valid JSON format only

THINKING PROCESS:
1. Analyze the images to determine if collaboration is needed across multiple UAV views
2. Formulate a clear question about the need for collaboration
3. Create 4 distinct options where only one is correct
4. Verify the question is unambiguous and answerable"""

    collaboration_types = [
        "need for collaboration due to incomplete information from multiple views",
        "need for collaboration due to environmental factors across UAVs"
    ]

    few_shot_examples = """
EXAMPLES OF GOOD COLLABORATION WHEN QUESTIONS:

Example 1:
{
    "question_id": "MDMT_when2col_UAV1_1001",
    "question_type": "4.1 When to Collaborate (UAV1)",
    "question": "Should UAV1 collaborate with other UAVs to obtain supplementary information due to incomplete observation data in the multi-UAV setup?",
    "options": {
        "A": "Yes, due to partial occlusion of key objects across views",
        "B": "No, all views are fully visible",
        "C": "Yes, due to poor visibility in multiple UAV views",
        "D": "No, all objects are clearly captured in all UAVs"
    },
    "correct_answer": "A",
    "annotation_info": "Annotation indicates partial occlusion requiring collaboration across UAVs."
}

Example 2:
{
    "question_id": "MDMT_when2col_UAV2_1002",
    "question_type": "4.1 When to Collaborate (UAV2)",
    "question": "Should UAV2 collaborate with other UAVs to address environmental challenges for better perception in the multi-UAV setup?",
    "options": {
        "A": "Yes, due to poor lighting conditions in multiple views",
        "B": "No, lighting is adequate across all UAVs",
        "C": "Yes, due to low image resolution in some views",
        "D": "No, the environment is clear for all UAVs"
    },
    "correct_answer": "A",
    "annotation_info": "Annotation indicates poor lighting conditions necessitating collaboration across multiple UAVs."
}
"""

    user_prompt = f"""First, provide a brief description (50-100 words) of the need for collaboration in the image from {uav_id} (first image) and images from other UAVs (subsequent images).
Then, create a multiple-choice question about when {uav_id} should collaborate based on this description.

REQUIREMENTS:
- Question should test understanding of when to share data across multiple UAVs
- Focus on: {collaboration_types[random.randint(0, 1)]}
- Ensure question is specific, unambiguous, and DISTINCT from examples
- Create 4 plausible options with distinct meanings
- Use clear, professional English
- Random seed for diversity: {random.randint(1, 1000)}
- Include the image description in the output JSON as 'image_description'

{few_shot_examples}

JSON format:
{{
    "question_id": "MDMT_when2col_{uav_id}_{counter}",
    "question_type": "4.1 When to Collaborate ({uav_id})",
    "question": "...",
    "options": {{ "A": "...", "B": "...", "C": "...", "D": "..." }},
    "correct_answer": "...",
    "image_description": "..."
}}"""

    content_list = [
        {"image": f"data:image/jpeg;base64,{encode_image(current_path)}"}  # First: current UAV
    ] + [
        {"image": f"data:image/jpeg;base64,{encode_image(p)}"} for p in other_paths  # Others
    ] + [
        {"text": user_prompt}
    ]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content_list}
    ]

    result = call_chatgpt_api(messages)
    if "error" not in result:
        result["source"] = "Model-Based (Few-Shot Fallback)"
        result["question_id"] = f"MDMT_when2col_{uav_id}_{counter}"
    return result


def generate_model_based_collaboration_why_q(current_path, other_paths, uav_id, counter=1):
    """Fallback model-based for collaboration why if hybrid fails or no annotation (optimized for multi-UAV)"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' understanding of why collaboration between multiple UAVs (up to 6) is necessary.

CRITICAL RULES:
1. ALWAYS respond in English only
2. Follow a structured thinking process: analyze images → identify reasons for collaboration → formulate question → create options → verify correctness
3. Questions must be based on actual visual content
4. Each question should have exactly 4 options (A, B, C, D) with only one correct answer
5. Options should be plausible, distinct in meaning, and avoid minor rephrasing
6. Output must be valid JSON format only

THINKING PROCESS:
1. Analyze images to identify reasons for collaboration across multiple UAV views
2. Formulate a clear question about the reason for collaboration
3. Create 4 distinct options where only one is correct
4. Verify the question is unambiguous and answerable"""

    reason_types = [
        "reason due to visibility limitations in multi-UAV views",
        "reason due to information gaps across UAVs"
    ]

    few_shot_examples = """
EXAMPLES OF GOOD COLLABORATION WHY QUESTIONS:

Example 1:
{
    "question_id": "MDMT_why2col_UAV1_1001",
    "question_type": "4.4 Why to Collaborate (UAV1)",
    "question": "What is the main reason UAV1 should collaborate with other UAVs?",
    "options": {
        "A": "To overcome partial occlusion of the objects across multiple views",
        "B": "To adjust for lighting conditions in multi-view setup",
        "C": "To capture a wider area from different UAV angles",
        "D": "To reduce computational load"
    },
    "correct_answer": "A",
    "image_description": "UAV1 shows a scene with objects partially occluded; other UAVs provide clearer views."
}

Example 2:
{
    "question_id": "MDMT_why2col_UAV2_1002",
    "question_type": "4.4 Why to Collaborate (UAV2)",
    "question": "What is the primary reason UAV2 should collaborate with other UAVs?",
    "options": {
        "A": "To obtain more clear information about specific objects across views",
        "B": "To compensate for low image quality in some views",
        "C": "To supplement missing information due to limited field of view (FoV) in multi-UAV setup",
        "D": "To synchronize time stamps"
    },
    "correct_answer": "A",
    "image_description": "UAV2 lacks clear pedestrian movement data; other UAVs provide complementary details."
}
"""

    focus = reason_types[random.randint(0, 1)]
    user_prompt = f"""First, provide a brief description (50-100 words) of the reasons for collaboration based on the image from {uav_id} (first image) and images from other UAVs (subsequent images).
Then, create a multiple-choice question about why {uav_id} should collaborate based on this description.

REQUIREMENTS:
- Question should test understanding of collaboration reasons across multiple UAVs
- Focus on: {focus}
- Ensure question is specific, unambiguous, and DISTINCT from examples
- Create 4 plausible options with distinct meanings, with only one being correct
- Use clear, professional English
- Random seed for diversity: {random.randint(1, 1000)}
- Include the description in 'image_description'

{few_shot_examples}

JSON format:
{{
    "question_id": "MDMT_why2col_{uav_id}_{counter}",
    "question_type": "4.4 Why to Collaborate ({uav_id})",
    "question": "...",
    "options": {{ "A": "...", "B": "...", "C": "...", "D": "..." }},
    "correct_answer": "...",
    "image_description": "..."
}}"""

    content_list = [
        {"image": f"data:image/jpeg;base64,{encode_image(current_path)}"}  # First: current UAV
    ] + [
        {"image": f"data:image/jpeg;base64,{encode_image(p)}"} for p in other_paths  # Others
    ] + [
        {"text": user_prompt}
    ]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content_list}
    ]

    result = call_chatgpt_api(messages)
    if "error" not in result:
        result["source"] = "Model-Based (Few-Shot Fallback)"
        result["question_id"] = f"MDMT_why2col_{uav_id}_{counter}"
    return result


def generate_model_based_collaboration_who_q(current_path, other_paths, uav_id, counter=1):
    """Fallback model-based for collaboration who if rule-based fails (optimized for multi-UAV)"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' understanding of which UAV(s) should be the collaboration partner in a multi-UAV setup (up to 6 UAVs).

CRITICAL RULES:
1. ALWAYS respond in English only
2. Follow a structured thinking process: analyze images → determine collaboration partner → formulate question → create options → verify correctness
3. Questions must be based on actual visual content
4. Each question should have exactly 4 options (A, B, C, D) with only one correct answer
5. Options should be plausible, distinct in meaning, and avoid minor rephrasing
6. Output must be valid JSON format only

THINKING PROCESS:
1. Analyze the images to identify the best collaboration partner across multiple UAV views
2. Formulate a clear question about the collaboration partner
3. Create 4 distinct options where only one is correct
4. Verify the question is unambiguous and answerable"""

    collaboration_types = [
        "collaboration partner for complementary perspective in multi-UAV setup",
        "collaboration partner for specific object data across UAVs"
    ]

    few_shot_examples = """
EXAMPLES OF GOOD COLLABORATION WHO QUESTIONS:

Example 1:
{
    "question_id": "MDMT_who2col_UAV1_1001",
    "question_type": "4.3 Who to Collaborate (UAV1)",
    "question": "Which UAV should UAV1 collaborate with to gain a complementary perspective in the multi-UAV setup?",
    "options": {
        "A": "UAV2",
        "B": "None (no need for collaboration)",
        "C": "A ground-based sensor",
        "D": "UAV3"
    },
    "correct_answer": "A",
    "annotation_info": "Annotation indicates UAV2 as the collaboration partner."
}

Example 2:
{
    "question_id": "MDMT_who2col_UAV2_1002",
    "question_type": "4.3 Who to Collaborate (UAV2)",
    "question": "Which UAV should UAV2 collaborate with to obtain specific object data in the multi-UAV setup?",
    "options": {
        "A": "UAV1",
        "B": "None (no need for collaboration)",
        "C": "UAV4",
        "D": "A satellite system"
    },
    "correct_answer": "A",
    "annotation_info": "Annotation indicates UAV1 as the collaboration partner."
}
"""

    user_prompt = f"""First, provide a brief description (50-100 words) of the best collaboration partner in the image from {uav_id} (first image) and images from other UAVs (subsequent images).
Then, create a multiple-choice question about who {uav_id} should collaborate with based on this description.

REQUIREMENTS:
- Question should test understanding of collaboration partners across multiple UAVs
- Focus on: {collaboration_types[random.randint(0, 1)]}
- Ensure question is specific, unambiguous, and DISTINCT from examples
- Create 4 plausible options with distinct meanings
- Use clear, professional English
- Random seed for diversity: {random.randint(1, 1000)}
- Include the image description in the output JSON as 'image_description'

{few_shot_examples}

JSON format:
{{
    "question_id": "MDMT_who2col_{uav_id}_{counter}",
    "question_type": "4.3 Who to Collaborate ({uav_id})",
    "question": "...",
    "options": {{ "A": "...", "B": "...", "C": "...", "D": "..." }},
    "correct_answer": "...",
    "image_description": "..."
}}"""

    content_list = [
        {"image": f"data:image/jpeg;base64,{encode_image(current_path)}"}  # First: current UAV
    ] + [
        {"image": f"data:image/jpeg;base64,{encode_image(p)}"} for p in other_paths  # Others
    ] + [
        {"text": user_prompt}
    ]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content_list}
    ]

    result = call_chatgpt_api(messages)
    if "error" not in result:
        result["source"] = "Model-Based (Few-Shot Fallback)"
        result["question_id"] = f"MDMT_who2col_{uav_id}_{counter}"
    return result


def main():
    # Set image directory paths
    base_dir = "/Users/starryyu/Documents/tinghuasummer/Sim_6_UAVs/Samples_testco"
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

    # Initialize counters for sequential numbering per UAV
    counters = {f"UAV{i}": {"when": 1, "what": 1, "who": 1, "why": 1} for i in range(1, 7)}

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
            other_paths = [group_paths[n] for n in range(1, 7) if n != uav_num]
            current_filename = group_filenames[uav_num]

            print(f"  Processing {uav_id}")

            # Extract annotation information
            possible_filenames = normalize_filename_for_annotation(current_filename)
            annotation = {}
            for filename in possible_filenames:
                if filename in annotation_map:
                    annotation = annotation_map[filename]
                    break

            # Extract annotation info
            annotation_info = extract_annotation_info(annotation)

            # Extract JSON annotation information
            json_info = ""
            frame_objects = []

            frame_number = extract_frame_number_from_filename(current_filename)

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

            pair_quality_stats = {"EXCELLENT": 0, "GOOD": 0, "FAIR": 0, "POOR": 0, "ERROR": 0}
            uav_questions = []

            group_questions["questions_per_uav"][uav_id] = {
                "annotation_info": annotation_info,
                "json_info": json_info,
                "combined_info": combined_info,
                "questions": uav_questions
            }

            print("  Generating rule-based questions...")

            # Task 4.1: Collaboration Decision (When)
            collaboration_when_q = generate_rule_based_collaboration_when_q(annotation, uav_id, counter=counters[uav_id]["when"])
            if collaboration_when_q:
                quality_when = evaluate_question_quality(collaboration_when_q)
                print(f"    Generated collaboration when question for {uav_id}")
                print(f"Quality: {quality_when['quality']} (Score: {quality_when['score']})")
                if quality_when['issues']:
                    print(f"Issues: {quality_when['issues']}")
                uav_questions.append(collaboration_when_q)
                pair_quality_stats[quality_when['quality']] += 1
                counters[uav_id]["when"] += 1
            else:
                print(f"    Failed to generate Collaboration When Q for {uav_id}: No annotation data")
                # Fallback to model-based if rule-based fails
                collaboration_when_q, quality_when = try_generate_qa(generate_model_based_collaboration_when_q, current_path, other_paths,
                                                                    uav_id, counter=counters[uav_id]["when"])
                if "error" not in collaboration_when_q:
                    print(f"    Generated fallback model-based collaboration when for {uav_id}")
                    print(f"Quality: {quality_when['quality']} (Score: {quality_when['score']})")
                    if quality_when['issues']:
                        print(f"Issues: {quality_when['issues']}")
                    uav_questions.append(collaboration_when_q)
                    pair_quality_stats[quality_when['quality']] += 1
                    counters[uav_id]["when"] += 1
                else:
                    print(f"    Failed to generate fallback Collaboration When Q for {uav_id}: {collaboration_when_q.get('error', 'Unknown error')}")
                    pair_quality_stats["ERROR"] += 1

            print("  Generating model-based questions...")

            # Task 4.2: Collaboration Decision (What)
            collaboration_what_q, quality_what = try_generate_qa(generate_few_shot_collaboration_what_q, current_path, other_paths, uav_id, counter=counters[uav_id]["what"])
            if "error" not in collaboration_what_q:
                print(f"    Generated collaboration what question for {uav_id}")
                print(f"Quality: {quality_what['quality']} (Score: {quality_what['score']})")
                if quality_what['issues']:
                    print(f"Issues: {quality_what['issues']}")
                uav_questions.append(collaboration_what_q)
                pair_quality_stats[quality_what['quality']] += 1
                counters[uav_id]["what"] += 1
            else:
                print(f"    Failed to generate Collaboration What Q for {uav_id}: {collaboration_what_q.get('error', 'Unknown error')}")
                pair_quality_stats["ERROR"] += 1

            # Task 4.3: Collaboration Decision (Who)
            collaboration_who_q = generate_rule_based_collaboration_who_q_with_annotation(annotation, uav_id, counter=counters[uav_id]["who"])
            if collaboration_who_q:
                quality_who = evaluate_question_quality(collaboration_who_q)
                print(f"    Generated collaboration who question for {uav_id}")
                print(f"Quality: {quality_who['quality']} (Score: {quality_who['score']})")
                if quality_who['issues']:
                    print(f"Issues: {quality_who['issues']}")
                uav_questions.append(collaboration_who_q)
                pair_quality_stats[quality_who['quality']] += 1
                counters[uav_id]["who"] += 1
            else:
                print(f"    Failed to generate rule-based Collaboration Who Q for {uav_id}: No annotation data")
                # Fallback to model-based if rule-based fails
                collaboration_who_q, quality_who = try_generate_qa(generate_model_based_collaboration_who_q, current_path, other_paths,
                                                                   uav_id, counter=counters[uav_id]["who"])
                if "error" not in collaboration_who_q:
                    print(f"    Generated fallback model-based collaboration who for {uav_id}")
                    print(f"Quality: {quality_who['quality']} (Score: {quality_who['score']})")
                    if quality_who['issues']:
                        print(f"Issues: {quality_who['issues']}")
                    uav_questions.append(collaboration_who_q)
                    pair_quality_stats[quality_who['quality']] += 1
                    counters[uav_id]["who"] += 1
                else:
                    print(f"    Failed to generate fallback Collaboration Who Q for {uav_id}: {collaboration_who_q.get('error', 'Unknown error')}")
                    pair_quality_stats["ERROR"] += 1

            # Task 4.4: Collaboration Decision (Why)
            if annotation and 'Collaboration_why' in annotation:
                print(f"    Using Collaboration_why annotation for {uav_id}: {annotation['Collaboration_why']}")
                collaboration_why_q, quality_why = try_generate_qa(generate_hybrid_collaboration_why_q,
                                                                   current_path, other_paths, uav_id, annotation, counter=counters[uav_id]["why"])
            else:
                collaboration_why_q, quality_why = try_generate_qa(generate_model_based_collaboration_why_q, current_path, other_paths,
                                                                   uav_id, counter=counters[uav_id]["why"])

            if "error" not in collaboration_why_q:
                print(f"    Generated collaboration why question for {uav_id}")
                print(f"Quality: {quality_why['quality']} (Score: {quality_why['score']})")
                if quality_why['issues']:
                    print(f"Issues: {quality_why['issues']}")
                uav_questions.append(collaboration_why_q)
                pair_quality_stats[quality_why['quality']] += 1
                counters[uav_id]["why"] += 1
            else:
                print(f"    Failed to generate Collaboration Why Q for {uav_id}: {collaboration_why_q.get('error', 'Unknown error')}")
                pair_quality_stats["ERROR"] += 1

            # Aggregate quality stats
            for k, v in pair_quality_stats.items():
                quality_stats[k] += v

        all_results["results"].append(group_questions)

        # Save intermediate results every 5 groups
        if (g_idx + 1) % 5 == 0:
            save_to_json(all_results, "VQA_MDMT_CD.json")
            print(f"Saved intermediate results after processing {g_idx + 1} groups")

    # Compute and add quality statistics
    all_results['quality_statistics'] = quality_stats

    # Save final results
    if save_to_json(all_results, "VQA_MDMT_CD.json"):
        print(f"\nAll results saved to VQA_MDMT_CD.json")
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
