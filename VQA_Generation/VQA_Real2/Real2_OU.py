import openai  # Import OpenAI library
from openai import AzureOpenAI
import base64
from PIL import Image
import io
import os
import json
import glob
import xml.etree.ElementTree as ET
from collections import defaultdict
import random
import time
import re  # Added for better annotation parsing
import difflib  # Added for SequenceMatcher

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
    """Normalize filename to match annotation file format, supporting -loss and -noise suffixes"""
    if "-loss.jpg" in filename or "-noise.jpg" in filename:
        return [filename]

    base_name = filename.replace("-UAV1.jpg", "").replace("-UAV2.jpg", "")

    possible_filenames = [
        filename,  # Original filename
        f"{base_name}-UAV1-loss.jpg",
        f"{base_name}-UAV2-loss.jpg",
        f"{base_name}-UAV1-noise.jpg",
        f"{base_name}-UAV2-noise.jpg"
    ]

    return possible_filenames


def load_annotations(annotation_file):
    """Load annotation file, supporting -loss and -noise suffix image filenames"""
    try:
        with open(annotation_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)

        annotation_map = {}
        for annotation in annotations:
            if 'img1' in annotation:
                filename = annotation['img1']
                annotation_map[filename] = annotation

        return annotation_map
    except Exception as e:
        print(f"Error loading annotations: {str(e)}")
        return {}


def get_image_pairs(uav1_dir, uav2_dir):
    """Get corresponding image pairs from UAV1 and UAV2 folders, supporting -loss and -noise suffixes"""
    uav1_images = glob.glob(os.path.join(uav1_dir, "*.jpg"))
    uav2_images = glob.glob(os.path.join(uav2_dir, "*.jpg"))

    uav1_map = {}
    uav2_map = {}

    for img_path in uav1_images:
        filename = os.path.basename(img_path)
        if "-UAV1-loss.jpg" in filename:
            base_name = filename.replace("-UAV1-loss.jpg", "-loss")
        elif "-UAV1-noise.jpg" in filename:
            base_name = filename.replace("-UAV1-noise.jpg", "-noise")
        else:
            base_name = filename.replace("-UAV1.jpg", "")
        uav1_map[base_name] = img_path

    for img_path in uav2_images:
        filename = os.path.basename(img_path)
        if "-UAV2-loss.jpg" in filename:
            base_name = filename.replace("-UAV2-loss.jpg", "-loss")
        elif "-UAV2-noise.jpg" in filename:
            base_name = filename.replace("-UAV2-noise.jpg", "-noise")
        else:
            base_name = filename.replace("-UAV2.jpg", "")
        uav2_map[base_name] = img_path

    common_keys = set(uav1_map.keys()) & set(uav2_map.keys())

    image_pairs = []
    for key in sorted(common_keys):
        if key.endswith("-loss"):
            base_without_suffix = key.replace("-loss", "")
            uav1_filename = f"{base_without_suffix}-UAV1-loss.jpg"
            uav2_filename = f"{base_without_suffix}-UAV2-loss.jpg"
        elif key.endswith("-noise"):
            base_without_suffix = key.replace("-noise", "")
            uav1_filename = f"{base_without_suffix}-UAV1-noise.jpg"
            uav2_filename = f"{base_without_suffix}-UAV2-noise.jpg"
        else:
            uav1_filename = f"{key}-UAV1.jpg"
            uav2_filename = f"{key}-UAV2.jpg"

        image_pairs.append({
            'sequence_frame': key,
            'uav1_path': uav1_map[key],
            'uav2_path': uav2_map[key],
            'uav1_filename': uav1_filename,
            'uav2_filename': uav2_filename
        })

    return image_pairs


def parse_xml_annotation(xml_file_path):
    """Parse XML annotation file, extract object information"""
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        tracks = []
        for track in root.findall('track'):
            track_info = {
                'id': track.get('id'),
                'label': track.get('label'),
                'boxes': []
            }

            for box in track.findall('box'):
                box_info = {
                    'frame': int(box.get('frame')),
                    'occluded': int(box.get('occluded')),
                    'outside': int(box.get('outside')),
                    'xtl': float(box.get('xtl')),
                    'ytl': float(box.get('ytl')),
                    'xbr': float(box.get('xbr')),
                    'ybr': float(box.get('ybr'))
                }
                track_info['boxes'].append(box_info)

            tracks.append(track_info)

        return tracks
    except Exception as e:
        print(f"Error parsing XML file {xml_file_path}: {str(e)}")
        return []


def get_xml_annotation_for_frame(xml_file_path, frame_number):
    """Get XML annotation information for specified frame"""
    tracks = parse_xml_annotation(xml_file_path)

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


def get_xml_annotation_for_scene(xml_file_path):
    """Get XML annotation information for the entire scene"""
    tracks = parse_xml_annotation(xml_file_path)

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


def find_xml_file_for_image(image_filename, xml_base_dir):
    """Find corresponding XML file for image filename, supporting -loss and -noise suffixes"""
    parts = image_filename.split('-')

    if len(parts) >= 2:
        sequence_number = parts[0]

        if 'UAV1' in xml_base_dir:
            xml_filename = f"{sequence_number}-1.xml"
        elif 'UAV2' in xml_base_dir:
            xml_filename = f"{sequence_number}-2.xml"
        else:
            xml_filename = f"{sequence_number}-1.xml"

        xml_path = os.path.join(xml_base_dir, xml_filename)

        if os.path.exists(xml_path):
            return xml_path

    return None


def extract_frame_number_from_filename(image_filename):
    """Extract frame number from image filename, supporting -loss and -noise suffixes"""
    parts = image_filename.split('-')
    if len(parts) >= 2:
        frame_part = parts[1]
        frame_number = int(''.join(filter(str.isdigit, frame_part)))
        return frame_number

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
        if isinstance(annotation['Quality'], dict) and 'choices' in annotation['Quality']:
            quality = ', '.join(annotation['Quality']['choices'])
        else:
            quality = str(annotation['Quality'])
        info_parts.append(f"Image quality: {quality}")

    if 'Usability' in annotation:
        if isinstance(annotation['Usability'], dict) and 'choices' in annotation['Usability']:
            usability = ', '.join(annotation['Usability']['choices'])
        else:
            usability = str(annotation['Usability'])
        info_parts.append(f"Image usability: {usability}")
    else:
        info_parts.append("Image usability: Not specified")

    if 'Collaboration_when' in annotation:
        if isinstance(annotation['Collaboration_when'], dict) and 'choices' in annotation['Collaboration_when']:
            collaboration_when = ', '.join(annotation['Collaboration_when']['choices'])
        else:
            collaboration_when = str(annotation['Collaboration_when'])
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


def generate_rule_based_counting_q(frame_objects, uav_id, counter=1):
    """
    [Rule-Based] Generate object counting questions based on XML annotation data.
    Now generates even if count=0.
    """
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
        "source": "Rule-Based from XML"
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


def generate_few_shot_object_recognition_q(img_path, uav_id, counter=1):
    """Generate object recognition questions with few-shot examples from Object_Understanding.py"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' ability to identify objects in a scene.

CRITICAL RULES:
1. ALWAYS respond in English only
2. Follow a structured thinking process: analyze → identify objects → formulate question → create options → verify correctness
3. Questions must be based on actual visual content or provided description
4. Each question should have exactly 4 options (A, B, C, D) with only one correct answer
5. Options should be plausible, distinct in meaning, and avoid minor rephrasing
6. Output must be valid JSON format only

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

    user_prompt = f"""First, provide a brief description (50-100 words) of the key objects in this image from {uav_id}.
Then, create a multiple-choice question about object recognition based on this description.

REQUIREMENTS:
- Question should test identification of a specific object type
- Focus on: {object_types[random.randint(0, 1)]}
- Ensure question is specific, unambiguous, and DISTINCT from examples in structure, focus, and vocabulary
- Create 4 plausible options with distinct meanings
- Use clear, professional English
- IMPORTANT: Use the EXACT question_id format provided in the JSON template
- Include the image description in the output JSON as 'image_description'

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


def generate_few_shot_object_grounding_q(img_path, uav_id, counter=1):
    """Generate object grounding questions with few-shot examples from Object_Understanding.py"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' understanding of object spatial positioning.

CRITICAL RULES:
1. ALWAYS respond in English only
2. Follow a structured thinking process: analyze → identify object positions → formulate question → create options → verify correctness
3. Questions must be based on actual visual content or provided description
4. Each question should have exactly 4 options (A, B, C, D) with only one correct answer
5. Options should be plausible, distinct in meaning, and avoid minor rephrasing
6. Output must be valid JSON format only

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

    user_prompt = f"""First, provide a brief description (50-100 words) of the key objects and their spatial positions in this image from {uav_id}.
Then, create a multiple-choice question about object grounding based on this description.

REQUIREMENTS:
- Question should test understanding of object spatial positions
- Focus on: {grounding_types[random.randint(0, 1)]}
- Ensure question is specific, unambiguous, and DISTINCT from examples in structure, focus, and vocabulary
- Create 4 plausible options with distinct meanings (avoid minor rephrasing or symmetric phrasing like 'left vs. right' or 'ahead vs. behind')
- Use clear, professional English
- IMPORTANT: Use the EXACT question_id format provided in the JSON template
- Include the image description in the output JSON as 'image_description'

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


def generate_few_shot_object_matching_q(img_path1, img_path2, counter=1):
    """Generate object matching questions with few-shot examples from Object_Understanding.py"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' ability to match objects across two UAV perspectives.

CRITICAL RULES:
1. ALWAYS respond in English only
2. Follow a structured thinking process: analyze both images → identify matching objects → formulate question → create options → verify correctness
3. Questions must be based on actual visual content or provided descriptions
4. Each question should have exactly 4 options (A, B, C, D) with only one correct answer
5. Options should be plausible, distinct in meaning, and avoid minor rephrasing
6. Output must be valid JSON format only

A simple question like "What is the white truck in image 1?" with the answer "The 
white truck in image 2" is USELESS. Avoid this.
Instead, follow this reasoning process to create a high-quality question:

THINKING PROCESS:
1. Identify a Candidate Object: In the first image (UAV1), find a distinct object 
that is also clearly visible in the second image (UAV2). Let's call this the "target 
object".
2. Analyze the Change: Critically compare the target object's appearance and context 
between the two views. Focus on what has CHANGED. Examples of changes include:
* Perspective: "The truck seen from the side" in image 1 is now "seen from the 
rear" in image 2.
* Relative Position: "The car behind the bus" in image 1 is now "the car beside 
a red sedan" in image 2.
* Action/State: "The person walking towards the crosswalk" in image 1 is now 
"the person waiting at the crosswalk" in image 2.
* Occlusion: "The partially occluded blue car" in image 1 is "fully visible" in 
image 2.
3. Formulate the Question: The question should describe the object in Image 1 using 
its appearance or relative position. For example: "The silver SUV turning at the 
intersection in UAV1's view corresponds to which object in UAV2's view?"
4. Create the Answer and Distractors:
* The correct answer must describe the SAME object in Image 2, highlighting the 
CHANGE you analyzed in step 2. (e.g., "The silver SUV now seen from behind, approaching 
the bridge.")
* The distractors should be plausible but incorrect. They could describe other 
similar objects in Image 2, or describe the correct object with a wrong state/location.
5. Final Output: Assemble everything into the required JSON format. Only output the 
final JSON.

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
    "question_id": "MDMT_OM_1001",
    "question_type": "2.4 Object Matching",
    "question": "Which object in UAV2 corresponds to the red truck seen from the front in UAV1?",
        "options": {
            "A": "The red truck now seen from the side, parked along the curb in UAV2",
            "B": "The blue sedan now seen from the front, moving towards the intersection in UAV2",
            "C": "The white van now seen from behind, parked on the right lane in UAV2",
            "D": "The green bus now seen from the front, stationary in UAV2"
        },
        "correct_answer": "A",
        "image_description": "UAV1 shows a red truck seen from the front; UAV2 shows the same red truck from the side, parked along the curb."
}

Example 2:
{
    "question_id": "MDMT_OM_1002",
    "question_type": "2.4 Object Matching",
    "question": "Which object in UAV2 corresponds to the person standing near the traffic light in UAV1?",
        "options": {
            "A": "The person now sitting on the bench near the traffic light in UAV2",
            "B": "The person seen walking towards the crosswalk in UAV2",
            "C": "The person now standing near a car in UAV2",
            "D": "The person seen sitting at a table in UAV2"
        },
        "correct_answer": "A",
        "image_description": "UAV1 shows a person standing near a traffic light; UAV2 shows the same person sitting on a bench near the traffic light."
}
"""

    user_prompt = f"""First, provide a brief description (50-100 words) of the key objects visible in both images from UAV1 and UAV2.
Then, create a multiple-choice question about object matching based on this description.

REQUIREMENTS:
- Question should test ability to match objects across perspectives
- Focus on: {matching_types[random.randint(0, 2)]}
- Ensure question is specific, unambiguous, and DISTINCT from examples
- Create 4 plausible options with distinct meanings
- Use clear, professional English
- IMPORTANT: Use the EXACT question_id format provided in the JSON template
- Include the image description in the output JSON as 'image_description'

{few_shot_examples}

JSON format:
{{
    "question_id": "MDMT_OM_{counter}",
    "question_type": "2.4 Object Matching",
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
                {"image": f"data:image/jpeg;base64,{encode_image(img_path1)}"},
                {"image": f"data:image/jpeg;base64,{encode_image(img_path2)}"},
                {"text": user_prompt}
            ]
        }
    ]

    result = call_chatgpt_api(messages)
    if "error" not in result:
        result["source"] = "Model-Based (Few-Shot)"
        result["question_id"] = f"MDMT_OM_{counter}"
    return result


def main():
    # Set image directory paths
    # base_dir = "Samples_test"
    base_dir = "All_Samples"
    uav1_dir = os.path.join(base_dir, "UAV1")
    uav2_dir = os.path.join(base_dir, "UAV2")

    # Set annotation file paths
    annotation_file = "Annotations/all_samples.json"

    # Set XML annotation file paths
    xml_base_dir = "Annotations/original_xml"
    uav1_xml_dir = os.path.join(xml_base_dir, "UAV1")
    uav2_xml_dir = os.path.join(xml_base_dir, "UAV2")

    # Check if directories exist
    if not os.path.exists(uav1_dir) or not os.path.exists(uav2_dir):
        print(f"Error: UAV directories not found. UAV1: {uav1_dir}, UAV2: {uav2_dir}")
        return

    # Check if annotation file exists
    if not os.path.exists(annotation_file):
        print(f"Error: Annotation file not found: {annotation_file}")
        return

    # Check if XML directories exist
    if not os.path.exists(uav1_xml_dir) or not os.path.exists(uav2_xml_dir):
        print(f"Warning: XML directories not found. UAV1 XML: {uav1_xml_dir}, UAV2 XML: {uav2_xml_dir}")
        print("Continuing without XML annotations...")

    print("Loading data...")
    annotation_map = load_annotations(annotation_file)
    image_pairs = get_image_pairs(uav1_dir, uav2_dir)

    print(f"Loaded {len(annotation_map)} annotation entries")
    print(f"Found {len(image_pairs)} image pairs to process")

    all_results = {
        "dataset": "Real_2_UAVs",
        "total_pairs": len(image_pairs),
        "results": []
    }

    quality_stats = {"EXCELLENT": 0, "GOOD": 0, "FAIR": 0, "POOR": 0, "ERROR": 0}
    pair_results = []

    # Initialize counters for sequential numbering
    or_uav1_counter = 1
    or_uav2_counter = 1
    oc_uav1_counter = 1
    oc_uav2_counter = 1
    og_uav1_counter = 1
    og_uav2_counter = 1
    om_counter = 1

    # Process each image pair
    for i, pair in enumerate(image_pairs):
        print(f"\n--- Processing pair {i + 1}/{len(image_pairs)}: {pair['sequence_frame']} ---")

        # Extract annotation information for both UAVs
        possible_uav1_filenames = normalize_filename_for_annotation(pair['uav1_filename'])
        possible_uav2_filenames = normalize_filename_for_annotation(pair['uav2_filename'])

        annotation1 = {}
        annotation2 = {}

        for filename in possible_uav1_filenames:
            if filename in annotation_map:
                annotation1 = annotation_map[filename]
                break

        for filename in possible_uav2_filenames:
            if filename in annotation_map:
                annotation2 = annotation_map[filename]
                break

        # Extract annotation info
        annotation_info1 = extract_annotation_info(annotation1)
        annotation_info2 = extract_annotation_info(annotation2)

        # Extract XML annotation information - Now use frame-level, fallback to scene if 0
        xml_info1 = ""
        xml_info2 = ""
        frame_objects1 = []
        frame_objects2 = []

        frame_number = extract_frame_number_from_filename(pair['uav1_filename'])

        # For UAV1
        xml_file1 = find_xml_file_for_image(pair['uav1_filename'], uav1_xml_dir)
        if xml_file1:
            frame_objects1 = get_xml_annotation_for_frame(xml_file1, frame_number)
            if not frame_objects1:
                scene_info = get_xml_annotation_for_scene(xml_file1)
                frame_objects1 = scene_info['objects']  # Fallback to scene visible
            xml_info1 = f"Frame objects: {len(frame_objects1)} visible objects"

        # For UAV2
        xml_file2 = find_xml_file_for_image(pair['uav2_filename'], uav2_xml_dir)
        if xml_file2:
            frame_objects2 = get_xml_annotation_for_frame(xml_file2, frame_number)
            if not frame_objects2:
                scene_info = get_xml_annotation_for_scene(xml_file2)
                frame_objects2 = scene_info['objects']  # Fallback to scene visible
            xml_info2 = f"Frame objects: {len(frame_objects2)} visible objects"

        # Combine annotation information
        combined_info1 = f"{annotation_info1}; {xml_info1}".strip('; ')
        combined_info2 = f"{annotation_info2}; {xml_info2}".strip('; ')

        # Print annotation information for debugging
        if annotation_info1 or xml_info1:
            print(f"  UAV1 annotation info: {combined_info1}")
        if annotation_info2 or xml_info2:
            print(f"  UAV2 annotation info: {combined_info2}")

        pair_quality_stats = {"EXCELLENT": 0, "GOOD": 0, "FAIR": 0, "POOR": 0, "ERROR": 0}
        pair_all_results = []

        pair_questions = {
            "sequence_frame": pair['sequence_frame'],
            "uav1_path": pair['uav1_path'],
            "uav2_path": pair['uav2_path'],
            "uav1_filename": pair['uav1_filename'],
            "uav2_filename": pair['uav2_filename'],
            "annotation_info1": annotation_info1,
            "annotation_info2": annotation_info2,
            "xml_info1": xml_info1,
            "xml_info2": xml_info2,
            "combined_info1": combined_info1,
            "combined_info2": combined_info2,
            "questions": []
        }

        print("Generating rule-based questions...")

        # For UAV1 - use frame objects
        counting_q1 = generate_rule_based_counting_q(frame_objects1, "UAV1", counter=str(oc_uav1_counter))
        quality_counting1 = evaluate_question_quality(counting_q1)
        if counting_q1:
            print(f"  Generated counting question for UAV1: {len(frame_objects1)} objects")
            print(f"Quality: {quality_counting1['quality']} (Score: {quality_counting1['score']})")
            if quality_counting1['issues']:
                print(f"Issues: {quality_counting1['issues']}")
            pair_all_results.append(counting_q1)
            pair_quality_stats[quality_counting1['quality']] += 1
            pair_questions["questions"].append(counting_q1)
            oc_uav1_counter += 1

        # For UAV2 - use frame objects
        counting_q2 = generate_rule_based_counting_q(frame_objects2, "UAV2", counter=str(oc_uav2_counter))
        quality_counting2 = evaluate_question_quality(counting_q2)
        if counting_q2:
            print(f"  Generated counting question for UAV2: {len(frame_objects2)} objects")
            print(f"Quality: {quality_counting2['quality']} (Score: {quality_counting2['score']})")
            if quality_counting2['issues']:
                print(f"Issues: {quality_counting2['issues']}")
            pair_all_results.append(counting_q2)
            pair_quality_stats[quality_counting2['quality']] += 1
            pair_questions["questions"].append(counting_q2)
            oc_uav2_counter += 1

        print("Generating model-based questions...")

        # Task 2.1: Object Recognition
        object_rec_q1, quality_rec1 = try_generate_qa(generate_few_shot_object_recognition_q, pair['uav1_path'], "UAV1",
                                                      counter=str(or_uav1_counter))
        if "error" not in object_rec_q1:
            print("  Generated object recognition question for UAV1")
            print(f"Quality: {quality_rec1['quality']} (Score: {quality_rec1['score']})")
            if quality_rec1['issues']:
                print(f"Issues: {quality_rec1['issues']}")
            pair_all_results.append(object_rec_q1)
            pair_quality_stats[quality_rec1['quality']] += 1
            pair_questions["questions"].append(object_rec_q1)
            or_uav1_counter += 1
        else:
            print(f"  Failed to generate Object Recognition Q for UAV1: {object_rec_q1.get('error', 'Unknown error')}")
            pair_quality_stats["ERROR"] += 1

        object_rec_q2, quality_rec2 = try_generate_qa(generate_few_shot_object_recognition_q, pair['uav2_path'], "UAV2",
                                                      counter=str(or_uav2_counter))
        if "error" not in object_rec_q2:
            print("  Generated object recognition question for UAV2")
            print(f"Quality: {quality_rec2['quality']} (Score: {quality_rec2['score']})")
            if quality_rec2['issues']:
                print(f"Issues: {quality_rec2['issues']}")
            pair_all_results.append(object_rec_q2)
            pair_quality_stats[quality_rec2['quality']] += 1
            pair_questions["questions"].append(object_rec_q2)
            or_uav2_counter += 1
        else:
            print(f"  Failed to generate Object Recognition Q for UAV2: {object_rec_q2.get('error', 'Unknown error')}")
            pair_quality_stats["ERROR"] += 1

        # Task 2.3: Object Grounding
        object_grounding_q1, quality_ground1 = try_generate_qa(generate_few_shot_object_grounding_q, pair['uav1_path'],
                                                               "UAV1", counter=str(og_uav1_counter))
        if "error" not in object_grounding_q1:
            print("  Generated object grounding question for UAV1")
            print(f"Quality: {quality_ground1['quality']} (Score: {quality_ground1['score']})")
            if quality_ground1['issues']:
                print(f"Issues: {quality_ground1['issues']}")
            pair_all_results.append(object_grounding_q1)
            pair_quality_stats[quality_ground1['quality']] += 1
            pair_questions["questions"].append(object_grounding_q1)
            og_uav1_counter += 1
        else:
            print(
                f"  Failed to generate Object Grounding Q for UAV1: {object_grounding_q1.get('error', 'Unknown error')}")
            pair_quality_stats["ERROR"] += 1

        object_grounding_q2, quality_ground2 = try_generate_qa(generate_few_shot_object_grounding_q, pair['uav2_path'],
                                                               "UAV2", counter=str(og_uav2_counter))
        if "error" not in object_grounding_q2:
            print("  Generated object grounding question for UAV2")
            print(f"Quality: {quality_ground2['quality']} (Score: {quality_ground2['score']})")
            if quality_ground2['issues']:
                print(f"Issues: {quality_ground2['issues']}")
            pair_all_results.append(object_grounding_q2)
            pair_quality_stats[quality_ground2['quality']] += 1
            pair_questions["questions"].append(object_grounding_q2)
            og_uav2_counter += 1
        else:
            print(
                f"  Failed to generate Object Grounding Q for UAV2: {object_grounding_q2.get('error', 'Unknown error')}")
            pair_quality_stats["ERROR"] += 1

        # Task 2.4: Object Matching
        object_matching_q, quality_match = try_generate_qa(generate_few_shot_object_matching_q, pair['uav1_path'],
                                                           pair['uav2_path'], counter=str(om_counter))
        if "error" not in object_matching_q:
            print("  Generated object matching question")
            print(f"Quality: {quality_match['quality']} (Score: {quality_match['score']})")
            if quality_match['issues']:
                print(f"Issues: {quality_match['issues']}")
            pair_all_results.append(object_matching_q)
            pair_quality_stats[quality_match['quality']] += 1
            pair_questions["questions"].append(object_matching_q)
            om_counter += 1
        else:
            print(f"  Failed to generate Object Matching Q: {object_matching_q.get('error', 'Unknown error')}")
            pair_quality_stats["ERROR"] += 1

        all_results["results"].append(pair_questions)

        # Aggregate for this pair
        for k, v in pair_quality_stats.items():
            quality_stats[k] += v

        pair_results.append({
            "pair_sequence_frame": pair['sequence_frame'],
            "pair_quality_statistics": pair_quality_stats,
            "pair_results": pair_all_results
        })

        print(f"  Total questions generated for this pair: {len(pair_questions['questions'])}")

        # Save intermediate results every 5 image pairs to prevent data loss
        if (i + 1) % 5 == 0:
            save_to_json(all_results, "VQA_MDMT_OU.json")
            print(f"Saved intermediate results after processing {i + 1} pairs")

    # Compute and add quality statistics
    all_results['quality_statistics'] = quality_stats

    # Save final results
    if save_to_json(all_results, "VQA_MDMT_OU.json"):
        print(f"\nAll results saved to VQA_MDMT_OU.json")
        print(f"Successfully processed {len(all_results['results'])} image pairs")

        # Count total questions generated
        total_questions = sum(len(result['questions']) for result in all_results['results'])
        print(f"Total questions generated: {total_questions}")
    else:
        print("Failed to save final results")

    # Overall quality summary
    print(f"\n=== Overall Quality Summary ===")
    print(f"Quality distribution: {quality_stats}")

    # Detailed quality assessment for all
    print(f"\n=== Detailed Quality Assessment for All Pairs ===")
    for pair_idx, pair_result in enumerate(pair_results):
        print(f"\n--- Pair {pair_idx + 1}: {pair_result['pair_sequence_frame']} ---")
        for i, result in enumerate(pair_result["pair_results"]):
            quality = evaluate_question_quality(result)
            print(f"\nQuestion {i + 1}:")
            print(f"Type: {result.get('question_type', 'N/A')}")
            print(f"Question: {result.get('question', 'N/A')}")
            print(f"Options: {result.get('options', {})}")
            print(f"Correct Answer: {result.get('correct_answer', 'N/A')}")
            print(f"Image Description: {result.get('image_description', 'N/A')}")
            print(f"Quality: {quality['quality']} (Score: {quality['score']})")
            if quality['issues']:
                print(f"Issues: {quality['issues']}")


if __name__ == "__main__":
    main()
