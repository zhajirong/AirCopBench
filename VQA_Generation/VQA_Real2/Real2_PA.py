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
        # "question": f"What is the usability assessment for the image captured by {uav_id} for target perception tasks?",
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
    """Generate quality assessment questions with few-shot examples"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' ability to assess image quality for perception tasks.

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
    """Generate usability assessment questions with few-shot examples from Perception_Assessment.py"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' ability to assess image usability for perception tasks.

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
    """Generate causal assessment questions with few-shot examples from Perception_Assessment.py"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' ability to analyze causes of perception quality issues.

CRITICAL RULES:
1. ALWAYS respond in English only
2. Follow a structured thinking process: analyze image → identify potential causes → formulate question → create options → verify correctness
3. Questions must be based on actual visual content or provided description
4. Each question should have exactly 4 options (A, B, C, D) with at least one correct answer
5. Options should be plausible, distinct in meaning, and avoid minor rephrasing
6. Output must be valid JSON format only

THINKING PROCESS:
1. First, analyze the image and identify potential causes of perception quality issues (e.g., blur, occlusion, lighting)
2. Then, formulate a question about the primary cause
3. Create 4 distinct options, with at least one option being correct
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
- Create 4 plausible options with distinct meanings, with at least one option being correct
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

    # Initialize sequential counters for each question type
    quality_id = 1
    usability_id = 1
    causal_id = 1

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

        # Task 3.1: Quality Assessment (based on JSON annotations)
        if annotation1:
            quality_q1 = generate_rule_based_quality_q(annotation1, "UAV1", quality_id)
            quality_quality1 = evaluate_question_quality(quality_q1)
            if quality_q1:
                print("  Generated quality question for UAV1")
                print(f"Quality: {quality_quality1['quality']} (Score: {quality_quality1['score']})")
                if quality_quality1['issues']:
                    print(f"Issues: {quality_quality1['issues']}")
                pair_all_results.append(quality_q1)
                pair_quality_stats[quality_quality1['quality']] += 1
                pair_questions["questions"].append(quality_q1)
                quality_id += 1
            else:
                # Fallback to few-shot model if rule fails
                quality_q1, quality_quality1 = try_generate_qa(generate_few_shot_quality_assessment_q,
                                                               pair['uav1_path'], "UAV1", quality_id)
                if "error" not in quality_q1:
                    print("  Generated fallback few-shot quality for UAV1")
                    print(f"Quality: {quality_quality1['quality']} (Score: {quality_quality1['score']})")
                    if quality_quality1['issues']:
                        print(f"Issues: {quality_quality1['issues']}")
                    pair_all_results.append(quality_q1)
                    pair_quality_stats[quality_quality1['quality']] += 1
                    pair_questions["questions"].append(quality_q1)
                    quality_id += 1
                else:
                    print(
                        f"  Failed to generate fallback Quality Q for UAV1: {quality_q1.get('error', 'Unknown error')}")
                    pair_quality_stats["ERROR"] += 1

            usability_q1 = generate_rule_based_usability_q(annotation1, "UAV1", usability_id)
            quality_usability1 = evaluate_question_quality(usability_q1)
            if usability_q1:
                print("  Generated usability question for UAV1")
                print(f"Quality: {quality_usability1['quality']} (Score: {quality_usability1['score']})")
                if quality_usability1['issues']:
                    print(f"Issues: {quality_usability1['issues']}")
                pair_all_results.append(usability_q1)
                pair_quality_stats[quality_usability1['quality']] += 1
                pair_questions["questions"].append(usability_q1)
                usability_id += 1
            else:
                # Fallback to few-shot model if rule fails
                usability_q1, quality_usability1 = try_generate_qa(generate_few_shot_usability_assessment_q,
                                                                   pair['uav1_path'], "UAV1", usability_id)
                if "error" not in usability_q1:
                    print("  Generated fallback few-shot usability for UAV1")
                    print(f"Quality: {quality_usability1['quality']} (Score: {quality_usability1['score']})")
                    if quality_usability1['issues']:
                        print(f"Issues: {quality_usability1['issues']}")
                    pair_all_results.append(usability_q1)
                    pair_quality_stats[quality_usability1['quality']] += 1
                    pair_questions["questions"].append(usability_q1)
                    usability_id += 1
                else:
                    print(
                        f"  Failed to generate fallback Usability Q for UAV1: {usability_q1.get('error', 'Unknown error')}")
                    pair_quality_stats["ERROR"] += 1

        if annotation2:
            quality_q2 = generate_rule_based_quality_q(annotation2, "UAV2", quality_id)
            quality_quality2 = evaluate_question_quality(quality_q2)
            if quality_q2:
                print("  Generated quality question for UAV2")
                print(f"Quality: {quality_quality2['quality']} (Score: {quality_quality2['score']})")
                if quality_quality2['issues']:
                    print(f"Issues: {quality_quality2['issues']}")
                pair_all_results.append(quality_q2)
                pair_quality_stats[quality_quality2['quality']] += 1
                pair_questions["questions"].append(quality_q2)
                quality_id += 1
            else:
                # Fallback to few-shot model if rule fails
                quality_q2, quality_quality2 = try_generate_qa(generate_few_shot_quality_assessment_q,
                                                               pair['uav2_path'], "UAV2", quality_id)
                if "error" not in quality_q2:
                    print("  Generated fallback few-shot quality for UAV2")
                    print(f"Quality: {quality_quality2['quality']} (Score: {quality_quality2['score']})")
                    if quality_quality2['issues']:
                        print(f"Issues: {quality_quality2['issues']}")
                    pair_all_results.append(quality_q2)
                    pair_quality_stats[quality_quality2['quality']] += 1
                    pair_questions["questions"].append(quality_q2)
                    quality_id += 1
                else:
                    print(
                        f"  Failed to generate fallback Quality Q for UAV2: {quality_q2.get('error', 'Unknown error')}")
                    pair_quality_stats["ERROR"] += 1

            usability_q2 = generate_rule_based_usability_q(annotation2, "UAV2", usability_id)
            quality_usability2 = evaluate_question_quality(usability_q2)
            if usability_q2:
                print("  Generated usability question for UAV2")
                print(f"Quality: {quality_usability2['quality']} (Score: {quality_usability2['score']})")
                if quality_usability2['issues']:
                    print(f"Issues: {quality_usability2['issues']}")
                pair_all_results.append(usability_q2)
                pair_quality_stats[quality_usability2['quality']] += 1
                pair_questions["questions"].append(usability_q2)
                usability_id += 1
            else:
                # Fallback to few-shot model if rule fails
                usability_q2, quality_usability2 = try_generate_qa(generate_few_shot_usability_assessment_q,
                                                                   pair['uav2_path'], "UAV2", usability_id)
                if "error" not in usability_q2:
                    print("  Generated fallback few-shot usability for UAV2")
                    print(f"Quality: {quality_usability2['quality']} (Score: {quality_usability2['score']})")
                    if quality_usability2['issues']:
                        print(f"Issues: {quality_usability2['issues']}")
                    pair_all_results.append(usability_q2)
                    pair_quality_stats[quality_usability2['quality']] += 1
                    pair_questions["questions"].append(usability_q2)
                    usability_id += 1
                else:
                    print(
                        f"  Failed to generate fallback Usability Q for UAV2: {usability_q2.get('error', 'Unknown error')}")
                    pair_quality_stats["ERROR"] += 1

        print("Generating model-based questions...")

        # Task 3.3: Causal Assessment (for poor quality images) - Now symmetric for both UAVs
        def generate_causal_if_low_quality(annotation, img_path, uav_id):
            nonlocal causal_id
            if 'Quality' in annotation:
                _, score = parse_quality_or_usability(annotation['Quality'])
                if score is not None and score <= 3:  # Lowered to <=3 to include Fair
                    quality_desc = f"Low quality (score: {score})"
                    causal_q, quality_causal = try_generate_qa(generate_few_shot_causal_assessment_q, img_path, uav_id,
                                                               causal_id)
                    if "error" not in causal_q:
                        print(f"  Generated causal assessment question for {uav_id} (quality score: {score})")
                        print(f"Quality: {quality_causal['quality']} (Score: {quality_causal['score']})")
                        if quality_causal['issues']:
                            print(f"Issues: {quality_causal['issues']}")
                        pair_all_results.append(causal_q)
                        pair_quality_stats[quality_causal['quality']] += 1
                        pair_questions["questions"].append(causal_q)
                        causal_id += 1
                    else:
                        print(
                            f"  Failed to generate Causal Assessment Q for {uav_id}: {causal_q.get('error', 'Unknown error')}")
                        pair_quality_stats["ERROR"] += 1

        if annotation1:
            generate_causal_if_low_quality(annotation1, pair['uav1_path'], "UAV1")
        if annotation2:
            generate_causal_if_low_quality(annotation2, pair['uav2_path'], "UAV2")

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
            save_to_json(all_results, "VQA_MDMT_PA.json")
            print(f"Saved intermediate results after processing {i + 1} pairs")

    # Compute and add quality statistics
    all_results['quality_statistics'] = quality_stats

    # Save final results
    if save_to_json(all_results, "VQA_MDMT_PA.json"):
        print(f"\nAll results saved to VQA_MDMT_PA.json")
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
