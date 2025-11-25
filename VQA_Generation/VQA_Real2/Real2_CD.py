# Collaborative_Decision.py (modified)
# Collaborative_Decision.py
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
    """Generate rule-based collaboration timing questions with few-shot examples from Collaborativ_Decision.py"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' understanding of when collaboration between UAVs is necessary.

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
        "need for collaboration due to incomplete information",
        "need for collaboration due to environmental factors"
    ]

    few_shot_examples = """
EXAMPLES OF GOOD COLLABORATION WHEN QUESTIONS:

Example 1:
{
    "question_id": "MDMT_when2col_UAV1_1001",
    "question_type": "4.1 When to Collaborate (UAV1)",
    "question": "Should UAV1 collaborate with another UAV to obtain supplementary information due to incomplete observation data in the scene?",
    "options": {
        "A": "Yes, due to partial occlusion of key objects",
        "B": "No, the scene is fully visible",
        "C": "Yes, due to poor visibility of the objects",
        "D": "No, all objects are clearly captured"
    },
    "correct_answer": "A",
    "annotation_info": "Annotation indicates partial occlusion of objects requiring collaboration."
}

Example 2:
{
    "question_id": "MDMT_when2col_UAV2_1002",
    "question_type": "4.1 When to Collaborate (UAV2)",
    "question": "Should UAV2 collaborate with another UAV to address environmental challenges for better perception in the scene?",
    "options": {
        "A": "Yes, due to poor lighting conditions",
        "B": "No, lighting is adequate",
        "C": "Yes, due to low image resolution",
        "D": "No, the environment is clear"
    },
    "correct_answer": "A",
    "annotation_info": "Annotation indicates poor lighting conditions necessitating collaboration."
}
"""

    if not annotation or 'Collaboration_when' not in annotation:
        return None

    collaboration_when_value = None
    if isinstance(annotation['Collaboration_when'], dict) and 'choices' in annotation['Collaboration_when']:
        collaboration_when_value = annotation['Collaboration_when']['choices'][0]
    elif isinstance(annotation['Collaboration_when'], str):
        collaboration_when_value = annotation['Collaboration_when']

    if not collaboration_when_value:
        return None

    correct_answer = "Yes" if "Yes" in collaboration_when_value or "1" in collaboration_when_value else "No"

    focus = collaboration_types[random.randint(0, 1)]
    if focus == "need for collaboration due to incomplete information":
        options = {
            "A": "Yes, due to partial occlusion of key objects",
            "B": "No, the scene is fully visible",
            "C": "Yes, due to poor visibility of the objects",
            "D": "No, all objects are clearly captured"
        }
        correct_option = "A" if correct_answer == "Yes" else "B"
    else:
        options = {
            "A": "Yes, due to poor lighting conditions",
            "B": "No, lighting is adequate",
            "C": "Yes, due to low image resolution",
            "D": "No, the environment is clear"
        }
        correct_option = "A" if correct_answer == "Yes" else "B"

    result = {
        "question_id": f"MDMT_when2col_{uav_id}_{counter}",
        "question_type": f"4.1 When to Collaborate ({uav_id})",
        "question": f"Should {uav_id} collaborate with another UAV to address {focus}?",
        "options": options,
        "correct_answer": correct_option,
        "source": "Rule-Based (Few-Shot)",
        "annotation_info": f"Annotation indicates: {collaboration_when_value}"
    }

    return result


def generate_few_shot_collaboration_what_q(img_path1, img_path2, uav_id, counter=1):
    """Generate model-based collaboration content questions with few-shot examples from Collaborativ_Decision.py"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' understanding of what specific information should be shared between UAVs.

CRITICAL RULES:
1. ALWAYS respond in English only
2. Follow a structured thinking process: analyze images → identify information gaps → formulate question → create options → verify correctness
3. Questions must be based on actual visual content or provided descriptions
4. Each question should have exactly 4 options (A, B, C, D) with at least one correct answer
5. Options should be plausible, distinct in meaning, and avoid minor rephrasing
6. Output must be valid JSON format only

THINKING PROCESS:
1. Analyze both images to identify information gaps or complementary data
2. Identify the focus based on generation index
3. Formulate a clear question about what specific information to collaborate on
4. Create 4 distinct options, with at least one being correct
5. Verify the question is unambiguous and answerable"""

    collaboration_types = [
        "specific object information (e.g., vehicle/pedestrian/bicycle details)",
        "scene context information (e.g., traffic flow)"
    ]

    few_shot_examples = """
EXAMPLES OF GOOD COLLABORATION WHAT QUESTIONS:

Example 1:
{
    "question_id": "MDMT_what2col_UAV1_1001",
    "question_type": "4.2 What to Collaborate (UAV1)",
    "question": "What specific object information should UAV1 share with UAV2 to improve perception?",
    "options": {
        "A": "Details of the red car's position and movement",
        "B": "Details of the road's lane markings",
        "C": "Details of the traffic light status",
        "D": "Details of the surrounding buildings"
    },
    "correct_answer": "A",
    "image_description": "UAV1 shows a red car moving on a multi-lane road, partially occluded in UAV2."
}

Example 2:
{
    "question_id": "MDMT_what2col_UAV2_1002",
    "question_type": "4.2 What to Collaborate (UAV2)",
    "question": "What scene context information should UAV2 share with UAV1 to enhance understanding?",
    "options": {
        "A": "Traffic flow patterns across the intersection",
        "B": "Weather conditions affecting visibility",
        "C": "Types of vehicles on the road",
        "D": "Layout of pedestrian crossings"
    },
    "correct_answer": "A",
    "image_description": "UAV2 provides a broader view of traffic flow at an intersection, complementing UAV1's view."
}
"""

    user_prompt = f"""First, provide a brief description (50-100 words) of the information gaps or complementary data between the two images from UAV1 and UAV2.
Then, create a multiple-choice question about what specific information {uav_id} should collaborate on, based on this description.

REQUIREMENTS:
- Question should test understanding of what data to share
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
        result["question_id"] = f"MDMT_what2col_{uav_id}_{counter}"
    return result


def generate_rule_based_collaboration_who_q_with_annotation(annotation, uav_id, counter=1):
    """Generate rule-based collaboration partner questions with few-shot examples from Collaborativ_Decision.py"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' understanding of which UAV should be the collaboration partner.

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
        "collaboration partner for complementary perspective",
        "collaboration partner for specific object data"
    ]

    few_shot_examples = """
EXAMPLES OF GOOD COLLABORATION WHO QUESTIONS:

Example 1:
{
    "question_id": "MDMT_who2col_UAV1_1001",
    "question_type": "4.3 Who to Collaborate (UAV1)",
    "question": "Which UAV should UAV1 collaborate with to gain a complementary perspective?",
    "options": {
        "A": "UAV2",
        "B": "None (no need for collaboration)",
        "C": "A ground-based sensor",
        "D": "None (no suitable collaboration partner)"
    },
    "correct_answer": "A",
    "annotation_info": "Annotation indicates UAV2 as the collaboration partner."
}

Example 2:
{
    "question_id": "MDMT_who2col_UAV2_1002",
    "question_type": "4.3 Who to Collaborate (UAV2)",
    "question": "Which UAV should UAV2 collaborate with to obtain specific object data?",
    "options": {
        "A": "UAV1",
        "B": "None (no need for collaboration)",
        "C": "A satellite system",
        "D": "None (no suitable collaboration partner)"
    },
    "correct_answer": "A",
    "annotation_info": "Annotation indicates UAV1 as the collaboration partner."
}
"""

    if not annotation or 'Collaboration_who' not in annotation:
        return None

    partner = annotation['Collaboration_who']
    if isinstance(partner, str):
        partner = partner.strip().lower()

    if partner in ["none", "no"]:
        correct_answer = "None"
    elif "uav1" in partner:
        correct_answer = "UAV1"
    elif "uav2" in partner:
        correct_answer = "UAV2"
    else:
        return None

    options = {
        "A": correct_answer,
        "B": "None (no need for collaboration)" if correct_answer != "None" else "UAV1",
        "C": "A ground-based sensor",
        "D": "None (no suitable collaboration partner)" if correct_answer != "None" else "UAV1"
    }

    focus = collaboration_types[random.randint(0, 1)]
    result = {
        "question_id": f"MDMT_who2col_{uav_id}_{counter}",
        "question_type": f"4.3 Who to Collaborate ({uav_id})",
        "question": f"Which UAV should {uav_id} collaborate with for {focus}?",
        "options": options,
        "correct_answer": "A",
        "source": "Rule-Based (Few-Shot)",
        "annotation_info": f"Annotation indicates: {partner}"
    }

    return result


def generate_hybrid_collaboration_why_q(img_path1, img_path2, uav_id, annotation, counter=1):
    """Generate hybrid collaboration reason questions with few-shot examples from Collaborativ_Decision.py"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' understanding of why collaboration between UAVs is necessary.

CRITICAL RULES:
1. ALWAYS respond in English only
2. Follow a structured thinking process: analyze images and annotation → identify reasons for collaboration → formulate question → create options → verify correctness
3. Questions must integrate annotation data and visual content
4. Each question should have exactly 4 options (A, B, C, D) with only one correct answer
5. Options should be plausible, distinct in meaning, and avoid minor rephrasing
6. Output must be valid JSON format only

THINKING PROCESS:
1. Analyze annotation and images to identify reasons for collaboration
2. Formulate a clear question about the reason for collaboration
3. Create 4 distinct options, with at least one being correct
4. Verify the question is unambiguous and answerable"""

    reason_types = [
        "reason due to visibility limitations",
        "reason due to information gaps"
    ]

    few_shot_examples = """
EXAMPLES OF GOOD COLLABORATION WHY QUESTIONS:

Example 1:
{
    "question_id": "MDMT_why2col_UAV1_1001",
    "question_type": "4.4 Why to Collaborate (UAV1)",
    "question": "Why should UAV1 collaborate with another UAV?",
    "options": {
        "A": "To overcome partial occlusion of the objects",
        "B": "To improve visibility of the objects",
        "C": "To adjust for lighting conditions",
        "D": "To capture a wider area"
    },
    "correct_answer": "A, B",
    "image_description": "UAV1 shows a scene with objects partially occluded; UAV2 provides a clearer view."
}

Example 2:
{
    "question_id": "MDMT_why2col_UAV2_1002",
    "question_type": "4.4 Why to Collaborate (UAV2)",
    "question": "Why should UAV2 collaborate with another UAV?",
    "options": {
        "A": "To obtain more clear information about specific objects",
        "B": "To obtain more detailed information about the objects",
        "C": "To compensate for low image quality",
        "D": "To supplement missing information due to limited field of view (FoV)"
    },
    "correct_answer": "A, B",
    "image_description": "UAV2 lacks clear pedestrian movement data; UAV1 provides complementary details."
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
    user_prompt = f"""First, provide a brief description (50-100 words) of the reasons for collaboration based on the two images from UAV1 and UAV2, and annotation: '{reasons_text}'.
Then, create a multiple-choice question about why {uav_id} should collaborate, integrating annotation and image analysis.

REQUIREMENTS:
- Question should test understanding of collaboration reasons
- Focus on: {focus}
- Ensure question is specific, unambiguous, and DISTINCT from examples
- Create 4 plausible options with distinct meanings, with at least one being correct
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
        result["source"] = "Rule-Based & Model-Based (Few-Shot)"
        result["annotation_info"] = reasons_text
        result["question_id"] = f"MDMT_why2col_{uav_id}_{counter}"
    return result


def generate_model_based_collaboration_when_q(img_path1, img_path2, uav_id, counter=1):
    """Fallback model-based for collaboration when if rule-based fails"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' understanding of when collaboration between UAVs is necessary.

CRITICAL RULES:
1. ALWAYS respond in English only
2. Follow a structured thinking process: analyze images → determine collaboration need → formulate question → create options → verify correctness
3. Questions must be based on actual visual content
4. Each question should have exactly 4 options (A, B, C, D) with only one correct answer
5. Options should be plausible, distinct in meaning, and avoid minor rephrasing
6. Output must be valid JSON format only

THINKING PROCESS:
1. Analyze the images to determine if collaboration is needed
2. Formulate a clear question about the need for collaboration
3. Create 4 distinct options where only one is correct
4. Verify the question is unambiguous and answerable"""

    collaboration_types = [
        "need for collaboration due to incomplete information",
        "need for collaboration due to environmental factors"
    ]

    few_shot_examples = """
EXAMPLES OF GOOD COLLABORATION WHEN QUESTIONS:

Example 1:
{
    "question_id": "MDMT_when2col_UAV1_1001",
    "question_type": "4.1 When to Collaborate (UAV1)",
    "question": "Should UAV1 collaborate with another UAV to obtain supplementary information due to incomplete observation data in the scene?",
    "options": {
        "A": "Yes, due to partial occlusion of key objects",
        "B": "No, the scene is fully visible",
        "C": "Yes, due to poor visibility of the objects",
        "D": "No, all objects are clearly captured"
    },
    "correct_answer": "A",
    "annotation_info": "Annotation indicates partial occlusion of objects requiring collaboration."
}

Example 2:
{
    "question_id": "MDMT_when2col_UAV2_1002",
    "question_type": "4.1 When to Collaborate (UAV2)",
    "question": "Should UAV2 collaborate with another UAV to address environmental challenges for better perception in the scene?",
    "options": {
        "A": "Yes, due to poor lighting conditions",
        "B": "No, lighting is adequate",
        "C": "Yes, due to low image resolution",
        "D": "No, the environment is clear"
    },
    "correct_answer": "A",
    "annotation_info": "Annotation indicates poor lighting conditions necessitating collaboration."
}
"""

    user_prompt = f"""First, provide a brief description (50-100 words) of the need for collaboration in the images from UAV1 and UAV2.
Then, create a multiple-choice question about when {uav_id} should collaborate based on this description.

REQUIREMENTS:
- Question should test understanding of when to share data
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
        result["source"] = "Model-Based (Few-Shot Fallback)"
        result["question_id"] = f"MDMT_when2col_{uav_id}_{counter}"
    return result


def generate_model_based_collaboration_why_q(img_path1, img_path2, uav_id, counter=1):
    """Fallback model-based for collaboration why if hybrid fails or no annotation"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' understanding of why collaboration between UAVs is necessary.

CRITICAL RULES:
1. ALWAYS respond in English only
2. Follow a structured thinking process: analyze images → identify reasons for collaboration → formulate question → create options → verify correctness
3. Questions must be based on actual visual content
4. Each question should have exactly 4 options (A, B, C, D) with at least one correct answer
5. Options should be plausible, distinct in meaning, and avoid minor rephrasing
6. Output must be valid JSON format only

THINKING PROCESS:
1. Analyze images to identify reasons for collaboration
2. Formulate a clear question about the reason for collaboration
3. Create 4 distinct options where only one is correct
4. Verify the question is unambiguous and answerable"""

    reason_types = [
        "reason due to visibility limitations",
        "reason due to information gaps"
    ]

    few_shot_examples = """
EXAMPLES OF GOOD COLLABORATION WHY QUESTIONS:

Example 1:
{
    "question_id": "MDMT_why2col_UAV1_1001",
    "question_type": "4.4 Why to Collaborate (UAV1)",
    "question": "Why should UAV1 collaborate with another UAV?",
    "options": {
        "A": "To overcome partial occlusion of the objects",
        "B": "To improve visibility of the objects",
        "C": "To adjust for lighting conditions",
        "D": "To capture a wider area"
    },
    "correct_answer": "A, B",
    "image_description": "UAV1 shows a scene with objects partially occluded; UAV2 provides a clearer view."
}

Example 2:
{
    "question_id": "MDMT_why2col_UAV2_1002",
    "question_type": "4.4 Why to Collaborate (UAV2)",
    "question": "Why should UAV2 collaborate with another UAV?",
    "options": {
        "A": "To obtain more clear information about specific objects",
        "B": "To obtain more detailed information about the objects",
        "C": "To compensate for low image quality",
        "D": "To supplement missing information due to limited field of view (FoV)"
    },
    "correct_answer": "A, B",
    "image_description": "UAV2 lacks clear pedestrian movement data; UAV1 provides complementary details."
}
"""

    focus = reason_types[random.randint(0, 1)]
    user_prompt = f"""First, provide a brief description (50-100 words) of the reasons for collaboration based on the two images from UAV1 and UAV2.
Then, create a multiple-choice question about why {uav_id} should collaborate based on this description.

REQUIREMENTS:
- Question should test understanding of collaboration reasons
- Focus on: {focus}
- Ensure question is specific, unambiguous, and DISTINCT from examples
- Create 4 plausible options with distinct meanings, with at least one being correct
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
        result["source"] = "Model-Based (Few-Shot Fallback)"
        result["question_id"] = f"MDMT_why2col_{uav_id}_{counter}"
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
    when_uav1_counter = 1
    when_uav2_counter = 1
    what_uav1_counter = 1
    what_uav2_counter = 1
    who_uav1_counter = 1
    who_uav2_counter = 1
    why_uav1_counter = 1
    why_uav2_counter = 1

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

        # Task 4.1: Collaboration Decision (When)
        collaboration_when_q1 = generate_rule_based_collaboration_when_q(annotation1, "UAV1", counter=str(when_uav1_counter))
        if collaboration_when_q1:
            quality_when1 = evaluate_question_quality(collaboration_when_q1)
            print("  Generated collaboration when question for UAV1")
            print(f"Quality: {quality_when1['quality']} (Score: {quality_when1['score']})")
            if quality_when1['issues']:
                print(f"Issues: {quality_when1['issues']}")
            pair_all_results.append(collaboration_when_q1)
            pair_quality_stats[quality_when1['quality']] += 1
            pair_questions["questions"].append(collaboration_when_q1)
            when_uav1_counter += 1
        else:
            print("  Failed to generate Collaboration When Q for UAV1: No annotation data")
            # Fallback to model-based if rule-based fails
            collaboration_when_q1, quality_when1 = try_generate_qa(generate_model_based_collaboration_when_q, pair['uav1_path'], pair['uav2_path'],
                                                                              "UAV1", counter=str(when_uav1_counter))
            if "error" not in collaboration_when_q1:
                print("  Generated fallback model-based collaboration when for UAV1")
                print(f"Quality: {quality_when1['quality']} (Score: {quality_when1['score']})")
                if quality_when1['issues']:
                    print(f"Issues: {quality_when1['issues']}")
                pair_all_results.append(collaboration_when_q1)
                pair_quality_stats[quality_when1['quality']] += 1
                pair_questions["questions"].append(collaboration_when_q1)
                when_uav1_counter += 1
            else:
                print(f"  Failed to generate fallback Collaboration When Q for UAV1: {collaboration_when_q1.get('error', 'Unknown error')}")
                pair_quality_stats["ERROR"] += 1

        collaboration_when_q2 = generate_rule_based_collaboration_when_q(annotation2, "UAV2", counter=str(when_uav2_counter))
        if collaboration_when_q2:
            quality_when2 = evaluate_question_quality(collaboration_when_q2)
            print("  Generated collaboration when question for UAV2")
            print(f"Quality: {quality_when2['quality']} (Score: {quality_when2['score']})")
            if quality_when2['issues']:
                print(f"Issues: {quality_when2['issues']}")
            pair_all_results.append(collaboration_when_q2)
            pair_quality_stats[quality_when2['quality']] += 1
            pair_questions["questions"].append(collaboration_when_q2)
            when_uav2_counter += 1
        else:
            print("  Failed to generate Collaboration When Q for UAV2: No annotation data")
            # Fallback to model-based if rule-based fails
            collaboration_when_q2, quality_when2 = try_generate_qa(generate_model_based_collaboration_when_q, pair['uav2_path'], pair['uav1_path'],
                                                                              "UAV2", counter=str(when_uav2_counter))
            if "error" not in collaboration_when_q2:
                print("  Generated fallback model-based collaboration when for UAV2")
                print(f"Quality: {quality_when2['quality']} (Score: {quality_when2['score']})")
                if quality_when2['issues']:
                    print(f"Issues: {quality_when2['issues']}")
                pair_all_results.append(collaboration_when_q2)
                pair_quality_stats[quality_when2['quality']] += 1
                pair_questions["questions"].append(collaboration_when_q2)
                when_uav2_counter += 1
            else:
                print(f"  Failed to generate fallback Collaboration When Q for UAV2: {collaboration_when_q2.get('error', 'Unknown error')}")
                pair_quality_stats["ERROR"] += 1

        print("Generating model-based questions...")

        # Task 4.2: Collaboration Decision (What)
        collaboration_what_q1, quality_what1 = try_generate_qa(generate_few_shot_collaboration_what_q, pair['uav1_path'], pair['uav2_path'], "UAV1", counter=str(what_uav1_counter))
        if "error" not in collaboration_what_q1:
            print("  Generated collaboration what question for UAV1")
            print(f"Quality: {quality_what1['quality']} (Score: {quality_what1['score']})")
            if quality_what1['issues']:
                print(f"Issues: {quality_what1['issues']}")
            pair_all_results.append(collaboration_what_q1)
            pair_quality_stats[quality_what1['quality']] += 1
            pair_questions["questions"].append(collaboration_what_q1)
            what_uav1_counter += 1
        else:
            print(
                f"  Failed to generate Collaboration What Q for UAV1: {collaboration_what_q1.get('error', 'Unknown error')}")
            pair_quality_stats["ERROR"] += 1

        collaboration_what_q2, quality_what2 = try_generate_qa(generate_few_shot_collaboration_what_q, pair['uav2_path'], pair['uav1_path'], "UAV2", counter=str(what_uav2_counter))
        if "error" not in collaboration_what_q2:
            print("  Generated collaboration what question for UAV2")
            print(f"Quality: {quality_what2['quality']} (Score: {quality_what2['score']})")
            if quality_what2['issues']:
                print(f"Issues: {quality_what2['issues']}")
            pair_all_results.append(collaboration_what_q2)
            pair_quality_stats[quality_what2['quality']] += 1
            pair_questions["questions"].append(collaboration_what_q2)
            what_uav2_counter += 1
        else:
            print(
                f"  Failed to generate Collaboration What Q for UAV2: {collaboration_what_q2.get('error', 'Unknown error')}")
            pair_quality_stats["ERROR"] += 1

        # Task 4.3: Collaboration Decision (Who)
        collaboration_who_q1 = generate_rule_based_collaboration_who_q_with_annotation(annotation1, "UAV1", counter=str(who_uav1_counter))
        if collaboration_who_q1:
            quality_who1 = evaluate_question_quality(collaboration_who_q1)
            print("  Generated collaboration who question for UAV1")
            print(f"Quality: {quality_who1['quality']} (Score: {quality_who1['score']})")
            if quality_who1['issues']:
                print(f"Issues: {quality_who1['issues']}")
            pair_all_results.append(collaboration_who_q1)
            pair_quality_stats[quality_who1['quality']] += 1
            pair_questions["questions"].append(collaboration_who_q1)
            who_uav1_counter += 1
        else:
            print("  Failed to generate Collaboration Who Q for UAV1: No annotation data")
            pair_quality_stats["ERROR"] += 1

        collaboration_who_q2 = generate_rule_based_collaboration_who_q_with_annotation(annotation2, "UAV2", counter=str(who_uav2_counter))
        if collaboration_who_q2:
            quality_who2 = evaluate_question_quality(collaboration_who_q2)
            print("  Generated collaboration who question for UAV2")
            print(f"Quality: {quality_who2['quality']} (Score: {quality_who2['score']})")
            if quality_who2['issues']:
                print(f"Issues: {quality_who2['issues']}")
            pair_all_results.append(collaboration_who_q2)
            pair_quality_stats[quality_who2['quality']] += 1
            pair_questions["questions"].append(collaboration_who_q2)
            who_uav2_counter += 1
        else:
            print("  Failed to generate Collaboration Who Q for UAV2: No annotation data")
            pair_quality_stats["ERROR"] += 1

        # Task 4.4: Collaboration Decision (Why)
        # Use hybrid approach: combine annotation information with model-based generation
        collaboration_why_q1 = None
        collaboration_why_q2 = None

        # For UAV1
        if annotation1 and 'Collaboration_why' in annotation1:
            print(f"  Using Collaboration_why annotation for UAV1: {annotation1['Collaboration_why']}")
            collaboration_why_q1, quality_why1 = try_generate_qa(generate_hybrid_collaboration_why_q,
                pair['uav1_path'], pair['uav2_path'], "UAV1", annotation1, counter=str(why_uav1_counter)
            )
        else:
            # Fallback to model-based generation without annotation
            collaboration_why_q1, quality_why1 = try_generate_qa(generate_model_based_collaboration_why_q, pair['uav1_path'], pair['uav2_path'],
                                                                            "UAV1", counter=str(why_uav1_counter))

        if "error" not in collaboration_why_q1:
            print("  Generated collaboration why question for UAV1")
            print(f"Quality: {quality_why1['quality']} (Score: {quality_why1['score']})")
            if quality_why1['issues']:
                print(f"Issues: {quality_why1['issues']}")
            pair_all_results.append(collaboration_why_q1)
            pair_quality_stats[quality_why1['quality']] += 1
            pair_questions["questions"].append(collaboration_why_q1)
            why_uav1_counter += 1
        else:
            print(
                f"  Failed to generate Collaboration Why Q for UAV1: {collaboration_why_q1.get('error', 'Unknown error')}")
            pair_quality_stats["ERROR"] += 1

        # For UAV2
        if annotation2 and 'Collaboration_why' in annotation2:
            print(f"  Using Collaboration_why annotation for UAV2: {annotation2['Collaboration_why']}")
            collaboration_why_q2, quality_why2 = try_generate_qa(generate_hybrid_collaboration_why_q,
                pair['uav2_path'], pair['uav1_path'], "UAV2", annotation2, counter=str(why_uav2_counter)
            )
        else:
            # Fallback to model-based generation without annotation
            collaboration_why_q2, quality_why2 = try_generate_qa(generate_model_based_collaboration_why_q, pair['uav2_path'], pair['uav1_path'],
                                                                            "UAV2", counter=str(why_uav2_counter))

        if "error" not in collaboration_why_q2:
            print("  Generated collaboration why question for UAV2")
            print(f"Quality: {quality_why2['quality']} (Score: {quality_why2['score']})")
            if quality_why2['issues']:
                print(f"Issues: {quality_why2['issues']}")
            pair_all_results.append(collaboration_why_q2)
            pair_quality_stats[quality_why2['quality']] += 1
            pair_questions["questions"].append(collaboration_why_q2)
            why_uav2_counter += 1
        else:
            print(
                f"  Failed to generate Collaboration Why Q for UAV2: {collaboration_why_q2.get('error', 'Unknown error')}")
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
            save_to_json(all_results, "VQA_MDMT_CD.json")
            print(f"Saved intermediate results after processing {i + 1} pairs")

    # Compute and add quality statistics
    all_results['quality_statistics'] = quality_stats

    # Save final results
    if save_to_json(all_results, "VQA_MDMT_CD.json"):
        print(f"\nAll results saved to VQA_MDMT_CD.json")
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
