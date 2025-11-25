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

"""
Object Understanding Script - Tasks 2.1, 2.2, 2.3, 2.4
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
    # Replace timezone format
    normalized = filename.replace("20+0800", "200800")
    # Remove the frame number part _numbers.jpg
    normalized = re.sub(r'_\d+\.(jpg|png)$', '', normalized)
    return [normalized]


def load_annotations(annotation_file):
    """Load annotation file"""
    try:
        with open(annotation_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)

        annotation_map = {}
        for annotation in annotations:
            if 'img1' in annotation:
                filename = os.path.basename(annotation['img1'])
                match = re.match(r'[0-9a-f]{8}-(.*)', filename, re.IGNORECASE)
                if match:
                    full_after = match.group(1)
                else:
                    full_after = filename
                # Remove the frame number part _numbers.jpg
                key = re.sub(r'_\d+\.(jpg|png)$', '', full_after)
                annotation_map[key] = annotation

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

    if 'Object_type' in annotation:
        if isinstance(annotation['Object_type'], dict) and 'choices' in annotation['Object_type']:
            object_type = ', '.join(annotation['Object_type']['choices'])
        else:
            object_type = str(annotation['Object_type'])
        info_parts.append(f"Object type: {object_type}")

    if 'Object_count' in annotation:
        object_count = annotation['Object_count']
        info_parts.append(f"Object count: {object_count}")

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


def save_to_json(data, filename="VQA_MDMT_OU.json"):
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


def generate_few_shot_object_recognition_q(img_path, uav_id, counter=1, combined_info=""):
    """Generate object recognition questions with few-shot examples (optimized for multi-UAV)"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' ability to identify objects in a scene from multiple UAV views.

CRITICAL RULES:
1. ALWAYS respond in English only
2. Follow a structured thinking process: analyze → identify objects → formulate question → create options → verify correctness
3. Questions must be based on actual visual content or provided description
4. Each question should have exactly 4 options (A, B, C, D) with only one correct answer
5. Options should be plausible, distinct in meaning, and avoid minor rephrasing
6. Output must be valid JSON format only
7. Use the provided annotation and JSON info to ensure accuracy in object identification

THINKING PROCESS:
1. First, describe the key objects in the image or description
2. Identify the focus based on generation index
3. Formulate a clear, specific question about object recognition
4. Create 4 distinct options where only one is correct
5. Verify the question is unambiguous and answerable"""

    object_types = [
        "type of a prominent vehicle",
        "type of a prominent non-vehicle object (e.g., pedestrian, cyclist)"
    ]

    few_shot_examples = """
EXAMPLES OF GOOD OBJECT RECOGNITION QUESTIONS:

Example 1:
{
    "question_id": "MDMT_Model_ObjRec_UAV1_1001",
    "question_type": "2.1 Object Recognition (UAV1)",
    "question": "What type of vehicle is most prominently visible in this scene?",
    "options": {
        "A": "A white van",
        "B": "A beige sedan",
        "C": "A red car",
        "D": "A black truck"
    },
    "correct_answer": "A",
    "image_description": "The scene shows a white van driving on a multi-lane road with other vehicles nearby."
}

Example 2:
{
    "question_id": "MDMT_Model_ObjRec_UAV2_1002",
    "question_type": "2.1 Object Recognition (UAV2)",
    "question": "How many types of objects are there in the image??",
    "options": {
        "A": "Three, pedestrian, vehicle, and bicycle",
        "B": "Two, vehicle and pedestrian",
        "C": "One, vehicle",
        "D": "One, pedestrian"
    },
    "correct_answer": "B",
    "image_description": "The scene depicts an intersection with a pedestrian crossing the road and vehicles waiting."
}
"""

    user_prompt = f"""Annotation and JSON info: {combined_info}

First, provide a brief description (50-100 words) of the key objects in this image from {uav_id}.
Then, create a multiple-choice question about object recognition based on this description and the annotation info.

REQUIREMENTS:
- Question should test identification of a specific object type
- Focus on: {object_types[random.randint(0, 1)]}
- Ensure question is specific, unambiguous, and DISTINCT from examples in structure, focus, and vocabulary
- Create 4 plausible options with distinct meanings
- Use clear, professional English
- IMPORTANT: Use the EXACT question_id format provided in the JSON template
- Include the image description in the output JSON as 'image_description'
- Ensure the correct answer aligns with the annotation info if provided

{few_shot_examples}

JSON format:
{{
    "question_id": "MDMT_OR_{uav_id}_{counter}",
    "question_type": "2.1 Object Recognition ({uav_id})",
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
        result["question_id"] = f"MDMT_OR_{uav_id}_{counter}"
    return result


def generate_rule_based_counting_q(annotation, frame_objects, uav_id, counter=1):
    """
    [Rule-Based] Generate object counting questions based on annotation Object_count or JSON annotation data.
    Prioritize Object_count from annotation if available.
    """
    correct_count = 0
    source = "JSON"
    if 'Object_count' in annotation:
        try:
            correct_count = int(annotation['Object_count'])
            source = "Annotation"
        except ValueError:
            correct_count = len(frame_objects)
    else:
        correct_count = len(frame_objects)

    # Generate distractors
    options = {str(correct_count)}
    while len(options) < 4:
        offset = random.randint(-3, 3)
        if offset == 0: continue
        distractor = max(0, correct_count + offset)
        options.add(str(distractor))

    shuffled_options = random.sample(list(options), len(options))
    option_dict = {chr(65 + i): opt for i, opt in enumerate(shuffled_options)}
    correct_letter = [k for k, v in option_dict.items() if v == str(correct_count)][0]

    return {
        "question_id": f"MDMT_OC_{uav_id}_{counter}",
        "question_type": f"2.2 Object Counting ({uav_id})",
        "question": f"Based on the image analysis, how many targets (vehicles, pedestrians, bicycles) can be observed in {uav_id}'s perspective?",
        "options": option_dict,
        "correct_answer": correct_letter,
        "source": f"Rule-Based from {source}"
    }, correct_count


def generate_few_shot_object_grounding_q(img_path, uav_id, counter=1, combined_info=""):
    """Generate object grounding questions with few-shot examples (optimized for multi-UAV)"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' understanding of object spatial positioning in multi-UAV views.

CRITICAL RULES:
1. ALWAYS respond in English only
2. Follow a structured thinking process: analyze → identify object positions → formulate question → create options → verify correctness
3. Questions must be based on actual visual content or provided description
4. Each question should have exactly 4 options (A, B, C, D) with only one correct answer
5. Options should be plausible, distinct in meaning, and avoid minor rephrasing
6. Output must be valid JSON format only
7. Use the provided annotation and JSON info to ensure accuracy in spatial positioning

THINKING PROCESS:
1. First, describe the key objects and their spatial positions in the image or description
2. Identify the focus based on generation index
3. Formulate a clear, specific question about object grounding
4. Create 4 distinct options where only one is correct
5. Verify the question is unambiguous and answerable"""

    grounding_types = [
        "position of a specific object relative to another",
        "spatial relationship within the scene"
    ]

    few_shot_examples = """
EXAMPLES OF GOOD OBJECT GROUNDING QUESTIONS:

Example 1:
{
    "question_id": "MDMT_OG_UAV1_1001",
    "question_type": "2.3 Object Grounding (UAV1)",
    "question": "Where is the white van located relative to the red car in this scene?",
    "options": {
        "A": "The white van is directly in front of the red car",
        "B": "The white van is parked on the opposite side of the road",
        "C": "The white van is next to the red car in the same lane",
        "D": "The white van is in a different lane behind the red car"
    },
    "correct_answer": "A",
    "image_description": "The scene shows a white van driving in front of a red car on a multi-lane road."
}

Example 2:
{
    "question_id": "MDMT_OG_UAV2_1002",
    "question_type": "2.3 Object Grounding (UAV2)",
    "question": "How is the pedestrian positioned within the scene?",
    "options": {
        "A": "The pedestrian is on the sidewalk near the road",
        "B": "The pedestrian is in the center of the intersection",
        "C": "The pedestrian is standing next to a vehicle",
        "D": "The pedestrian is crossing the road"
    },
    "correct_answer": "A",
    "image_description": "The scene depicts an intersection with a pedestrian on the sidewalk near the road."
}
"""

    user_prompt = f"""Annotation and JSON info: {combined_info}

First, provide a brief description (50-100 words) of the key objects and their spatial positions in this image from {uav_id}.
Then, create a multiple-choice question about object grounding based on this description and the annotation info.

REQUIREMENTS:
- Question should test understanding of object spatial positions
- Focus on: {grounding_types[random.randint(0, 1)]}
- Ensure question is specific, unambiguous, and DISTINCT from examples in structure, focus, and vocabulary
- Create 4 plausible options with distinct meanings (avoid minor rephrasing or symmetric phrasing like 'left vs. right' or 'ahead vs. behind')
- Use clear, professional English
- IMPORTANT: Use the EXACT question_id format provided in the JSON template
- Include the image description in the output JSON as 'image_description'
- Ensure the correct answer aligns with the annotation info if provided

{few_shot_examples}

JSON format:
{{
    "question_id": "MDMT_OG_{uav_id}_{counter}",
    "question_type": "2.3 Object Grounding ({uav_id})",
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
        result["question_id"] = f"MDMT_OG_{uav_id}_{counter}"
    return result


def generate_few_shot_object_matching_q(current_path, other_paths, uav_id, counter=1, combined_info=""):
    """Generate object matching questions with few-shot examples (optimized for multi-UAV)"""
    system_prompt = f"""You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' ability to match objects across multiple UAV perspectives.

CRITICAL RULES:
1. ALWAYS respond in English only
2. Follow a structured thinking process: analyze all images → identify matching objects → formulate question → create options → verify correctness
3. Questions must be based on actual visual content or provided descriptions
4. Each question should have exactly 4 options (A, B, C, D) with only one correct answer
5. Options should be plausible, distinct in meaning, and avoid minor rephrasing
6. Output must be valid JSON format only
7. Use the provided annotation and JSON info to ensure accuracy in object matching

A simple question like "What is the white truck in image 1?" with the answer "The white truck in image 2" is USELESS. Avoid this.
Instead, follow this reasoning process to create a high-quality question:

THINKING PROCESS:
1. Identify a Candidate Object: In the first image ({uav_id}), find a distinct object that is also clearly visible in one of the subsequent images (other UAVs). Let's call this the "target object".
2. Analyze the Change: Critically compare the target object's appearance and context between the views. Focus on what has CHANGED. Examples of changes include:
* Perspective: "The truck seen from the side" in {uav_id} is now "seen from the rear" in another view.
* Relative Position: "The car behind the bus" in {uav_id} is now "the car beside a red sedan" in another view.
* Action/State: "The person walking towards the crosswalk" in {uav_id} is now "the person waiting at the crosswalk" in another view.
* Occlusion: "The partially occluded blue car" in {uav_id} is "fully visible" in another view.
3. Formulate the Question: The question should describe the object in {uav_id} using its appearance or relative position. For example: "The silver SUV turning at the intersection in {uav_id}'s view corresponds to which object in another UAV's view?"
4. Create the Answer and Distractors:
* The correct answer must describe the SAME object in the other view, highlighting the CHANGE you analyzed in step 2. (e.g., "The silver SUV now seen from behind, approaching the bridge in UAV3.")
* The distractors should be plausible but incorrect. They could describe other similar objects in the other views, or describe the correct object with a wrong state/location.
5. Final Output: Assemble everything into the required JSON format. Only output the final JSON.

"""

    matching_types = [
        "matching a vehicle across perspectives",
        "matching a non-vehicle object (e.g., pedestrian, cyclist)",
        "matching a static object (e.g., traffic light, sign)"
    ]

    few_shot_examples = """
EXAMPLES OF GOOD OBJECT MATCHING QUESTIONS:

Example 1:
{
    "question_id": "MDMT_OM_UAV1_1001",
    "question_type": "2.4 Object Matching (UAV1)",
    "question": "Which object in another UAV's view corresponds to the red truck seen from the front in UAV1?",
    "options": {
        "A": "The red truck now seen from the side, parked along the curb in UAV2",
        "B": "The blue sedan now seen from the front, moving towards the intersection in UAV3",
        "C": "The white van now seen from behind, parked on the right lane in UAV4",
        "D": "The green bus now seen from the front, stationary in UAV5"
    },
    "correct_answer": "A",
    "image_description": "UAV1 shows a red truck seen from the front; UAV2 shows the same red truck from the side, parked along the curb."
}

Example 2:
{
    "question_id": "MDMT_OM_UAV2_1002",
    "question_type": "2.4 Object Matching (UAV2)",
    "question": "Which object in another UAV's view corresponds to the person standing near the traffic light in UAV2?",
    "options": {
        "A": "The person now sitting on the bench near the traffic light in UAV1",
        "B": "The person seen walking towards the crosswalk in UAV3",
        "C": "The person now standing near a car in UAV4",
        "D": "The person seen sitting at a table in UAV5"
    },
    "correct_answer": "A",
    "image_description": "UAV2 shows a person standing near a traffic light; UAV1 shows the same person sitting on a bench near the traffic light."
}
"""

    user_prompt = f"""Annotation and JSON info: {combined_info}

First, provide a brief description (50-100 words) of the key objects visible in the image from {uav_id} (first image) and images from other UAVs (subsequent images).
Then, create a multiple-choice question about object matching based on this description and the annotation info.

REQUIREMENTS:
- Question should test ability to match objects across perspectives
- Focus on: {matching_types[random.randint(0, 2)]}
- Ensure question is specific, unambiguous, and DISTINCT from examples
- Create 4 plausible options with distinct meanings
- Use clear, professional English
- IMPORTANT: Use the EXACT question_id format provided in the JSON template
- Include the image description in the output JSON as 'image_description'
- Ensure the correct answer aligns with the annotation info if provided

{few_shot_examples}

JSON format:
{{
    "question_id": "MDMT_OM_{uav_id}_{counter}",
    "question_type": "2.4 Object Matching ({uav_id})",
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
        result["question_id"] = f"MDMT_OM_{uav_id}_{counter}"
    return result


def main():
    # Set image directory paths
    base_dir = "/Users/starryyu/Documents/tinghuasummer/Sim_6_UAVs/Samples_testob4"
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
    counters = {f"UAV{i}": {"or": 1, "oc": 1, "og": 1, "om": 1} for i in range(1, 7)}

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

            # Task 2.2: Object Counting - rule-based
            counting_q, correct_count = generate_rule_based_counting_q(annotation, frame_objects, uav_id, counter=str(counters[uav_id]["oc"]))
            quality_counting = evaluate_question_quality(counting_q)
            if counting_q:
                print(f"  Generated counting question for {uav_id}: {correct_count} objects")
                print(f"Quality: {quality_counting['quality']} (Score: {quality_counting['score']})")
                if quality_counting['issues']:
                    print(f"Issues: {quality_counting['issues']}")
                uav_questions.append(counting_q)
                pair_quality_stats[quality_counting['quality']] += 1
                counters[uav_id]["oc"] += 1

            print("Generating model-based questions...")

            # Task 2.1: Object Recognition
            object_rec_q, quality_rec = try_generate_qa(generate_few_shot_object_recognition_q, current_path, uav_id,
                                                        counter=str(counters[uav_id]["or"]), combined_info=combined_info)
            if "error" not in object_rec_q:
                print(f"  Generated object recognition question for {uav_id}")
                print(f"Quality: {quality_rec['quality']} (Score: {quality_rec['score']})")
                if quality_rec['issues']:
                    print(f"Issues: {quality_rec['issues']}")
                uav_questions.append(object_rec_q)
                pair_quality_stats[quality_rec['quality']] += 1
                counters[uav_id]["or"] += 1
            else:
                print(f"  Failed to generate Object Recognition Q for {uav_id}: {object_rec_q.get('error', 'Unknown error')}")
                pair_quality_stats["ERROR"] += 1

            # Task 2.3: Object Grounding
            object_grounding_q, quality_ground = try_generate_qa(generate_few_shot_object_grounding_q, current_path,
                                                                 uav_id, counter=str(counters[uav_id]["og"]), combined_info=combined_info)
            if "error" not in object_grounding_q:
                print(f"  Generated object grounding question for {uav_id}")
                print(f"Quality: {quality_ground['quality']} (Score: {quality_ground['score']})")
                if quality_ground['issues']:
                    print(f"Issues: {quality_ground['issues']}")
                uav_questions.append(object_grounding_q)
                pair_quality_stats[quality_ground['quality']] += 1
                counters[uav_id]["og"] += 1
            else:
                print(f"  Failed to generate Object Grounding Q for {uav_id}: {object_grounding_q.get('error', 'Unknown error')}")
                pair_quality_stats["ERROR"] += 1

            # Task 2.4: Object Matching - per UAV, matching to others
            object_matching_q, quality_match = try_generate_qa(generate_few_shot_object_matching_q, current_path, other_paths,
                                                               uav_id, counter=str(counters[uav_id]["om"]), combined_info=combined_info)
            if "error" not in object_matching_q:
                print(f"  Generated object matching question for {uav_id}")
                print(f"Quality: {quality_match['quality']} (Score: {quality_match['score']})")
                if quality_match['issues']:
                    print(f"Issues: {quality_match['issues']}")
                uav_questions.append(object_matching_q)
                pair_quality_stats[quality_match['quality']] += 1
                counters[uav_id]["om"] += 1
            else:
                print(f"  Failed to generate Object Matching Q for {uav_id}: {object_matching_q.get('error', 'Unknown error')}")
                pair_quality_stats["ERROR"] += 1

            # Aggregate quality stats
            for k, v in pair_quality_stats.items():
                quality_stats[k] += v

        all_results["results"].append(group_questions)

        # Save intermediate results every 5 groups
        if (g_idx + 1) % 5 == 0:
            save_to_json(all_results, "VQA_MDMT_OU.json")
            print(f"Saved intermediate results after processing {g_idx + 1} groups")

    # Compute and add quality statistics
    all_results['quality_statistics'] = quality_stats

    # Save final results
    if save_to_json(all_results, "VQA_MDMT_OU.json"):
        print(f"\nAll results saved to VQA_MDMT_OU.json")
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
