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
                # Handle format like "/data/upload/9/8f2a9605-scene_004-UAV1_frame_005.jpg"
                
                # Extract the scene and UAV frame information
                match = re.search(r'scene_(\d+)-UAV(\d+)_frame_(\d+)', full_filename)
                if match:
                    scene_num = match.group(1)
                    uav_num = match.group(2)
                    frame_num = match.group(3)
                    
                    # Create multiple keys for flexible matching
                    keys = [
                        f"scene_{scene_num}-UAV{uav_num}_frame_{frame_num}.jpg",
                        f"scene_{scene_num}-UAV{uav_num}_frame_{frame_num}.png",
                        f"UAV{uav_num}_frame_{frame_num}.jpg",
                        f"UAV{uav_num}_frame_{frame_num}.png",
                        f"frame_{frame_num}.jpg",
                        f"frame_{frame_num}.png",
                        full_filename,  # Original full path
                    ]
                    
                    for key in keys:
                        annotation_map[key] = annotation
                        
                else:
                    # Fallback for other formats
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
                        annotation_map[full_filename] = annotation

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


def extract_frame_number_from_filename(image_filename):
    """Extract frame number from image filename"""
    # Handle new format: UAV1_frame_001.jpg
    match = re.search(r'UAV\d+_frame_(\d+)', image_filename)
    if match:
        frame_str = match.group(1)
        if frame_str.isdigit():
            return int(frame_str)
    
    # Handle old format: fallback
    parts = image_filename.rsplit('_', 1)
    if len(parts) == 2:
        frame_str = parts[1].replace('.jpg', '').replace('.png', '')
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
        object_type = annotation['Object_type']
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


def save_to_json(data, filename="VQA_Sim3_OU.json"):
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


def generate_few_shot_object_recognition_q(img_path, uav_id, counter=1):
    """Generate object recognition questions with few-shot examples (optimized for multi-UAV)"""
    system_prompt = """You are an expert teacher of the "UAV Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' ability to identify targets from UAV perspectives, focusing specifically on drone detection, vehicle recognition, and pedestrian identification.

CRITICAL RULES:
1. ALWAYS respond in English only
2. Focus on UAV-specific target perception: drones, vehicles, pedestrians
3. Questions must emphasize the UAV perspective and aerial view characteristics
4. Each question should have exactly 4 options (A, B, C, D) with only one correct answer
5. Options should be plausible, distinct in meaning, and avoid minor rephrasing
6. Output must be valid JSON format only

THINKING PROCESS:
1. First, describe the key targets visible from the UAV perspective
2. Focus on UAV-specific target types: drones, vehicles, pedestrians
3. Formulate a clear, specific question about target recognition from aerial view
4. Create 4 distinct options where only one is correct
5. Verify the question emphasizes UAV perspective and target perception"""

    object_types = [
        "drone detection from UAV perspective",
        "vehicle recognition from aerial view",
        "pedestrian identification from UAV camera"
    ]

    few_shot_examples = """
EXAMPLES OF GOOD UAV TARGET RECOGNITION QUESTIONS:

Example 1:
{
    "question_id": "Sim3_Model_ObjRec_UAV1_1001",
    "question_type": "2.1 UAV Target Recognition (UAV1)",
    "question": "From the UAV's aerial perspective, what type of target is most prominently visible in this scene?",
    "options": {
        "A": "A white delivery van",
        "B": "A surveillance drone",
        "C": "A pedestrian crossing the road",
        "D": "A stationary traffic light"
    },
    "correct_answer": "A",
    "image_description": "The UAV captures a white delivery van from above, clearly visible on the multi-lane road with other vehicles nearby."
}

Example 2:
{
    "question_id": "Sim3_Model_ObjRec_UAV2_1002",
    "question_type": "2.1 UAV Target Recognition (UAV2)",
    "question": "How many different types of targets can be identified from this UAV's aerial view?",
    "options": {
        "A": "Three types: drone, vehicle, and pedestrian",
        "B": "Two types: vehicle and pedestrian",
        "C": "One type: only vehicles",
        "D": "Four types: drone, vehicle, pedestrian, and traffic sign"
    },
    "correct_answer": "B",
    "image_description": "The UAV's aerial perspective shows vehicles on the road and a pedestrian crossing at the intersection."
}
"""

    user_prompt = f"""First, provide a brief description (50-100 words) of the key targets visible from the UAV perspective in this image from {uav_id}.
Then, create a multiple-choice question about UAV target recognition based on this description.

REQUIREMENTS:
- Question should test identification of UAV-specific targets: drones, vehicles, pedestrians
- Focus on: {object_types[random.randint(0, 2)]}
- Emphasize the UAV's aerial perspective and target perception capabilities
- Ensure question is specific, unambiguous, and DISTINCT from examples in structure, focus, and vocabulary
- Create 4 plausible options with distinct meanings
- Use clear, professional English emphasizing UAV context
- IMPORTANT: Use the EXACT question_id format provided in the JSON template
- Include the image description in the output JSON as 'image_description'

{few_shot_examples}

JSON format:
{{
    "question_id": "Sim3_OR_{uav_id}_{counter}",
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
        result["question_id"] = f"Sim3_OR_{uav_id}_{counter}"
    return result


def generate_rule_based_counting_q(object_count_from_annotation, uav_id, counter=1):
    """
    [Rule-Based] Generate UAV target counting questions based on all_samples.json annotation data.
    Now generates even if count=0.
    """
    # Parse object count from annotation (handle both string and int formats)
    if isinstance(object_count_from_annotation, str):
        try:
            correct_count = int(object_count_from_annotation)
        except ValueError:
            # If parsing fails, default to 0
            correct_count = 0
    else:
        correct_count = int(object_count_from_annotation) if object_count_from_annotation is not None else 0

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
        "question_id": f"Sim3_OC_{uav_id}_{counter}",
        "question_type": f"2.2 UAV Target Counting ({uav_id})",
        "question": f"From the UAV's aerial perspective, how many targets (drones, vehicles, pedestrians) can be detected in {uav_id}'s field of view?",
        "options": option_dict,
        "correct_answer": correct_letter,
        "source": "Rule-Based from all_samples.json"
    }


def generate_few_shot_object_grounding_q(img_path, uav_id, counter=1):
    """Generate UAV target grounding questions with few-shot examples (optimized for multi-UAV)"""
    system_prompt = """You are an expert teacher of the "UAV Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' understanding of target spatial positioning from UAV aerial perspectives.

CRITICAL RULES:
1. ALWAYS respond in English only
2. Focus on UAV-specific target positioning: drones, vehicles, pedestrians from aerial view
3. Questions must emphasize the UAV's spatial perception capabilities
4. Each question should have exactly 4 options (A, B, C, D) with only one correct answer
5. Options should be plausible, distinct in meaning, and avoid minor rephrasing
6. Output must be valid JSON format only

THINKING PROCESS:
1. First, describe the key targets and their spatial positions from the UAV perspective
2. Focus on UAV-specific target types: drones, vehicles, pedestrians
3. Formulate a clear, specific question about target grounding from aerial view
4. Create 4 distinct options where only one is correct
5. Verify the question emphasizes UAV spatial perception"""

    grounding_types = [
        "position of a specific target relative to another from UAV perspective",
        "spatial relationship of targets within the UAV's field of view"
    ]

    few_shot_examples = """
EXAMPLES OF GOOD UAV TARGET GROUNDING QUESTIONS:

Example 1:
{
    "question_id": "Sim3_OG_UAV1_1001",
    "question_type": "2.3 UAV Target Grounding (UAV1)",
    "question": "From the UAV's aerial perspective, where is the white van positioned relative to the red car?",
    "options": {
        "A": "The white van is directly ahead of the red car in the same lane",
        "B": "The white van is parked on the opposite side of the road",
        "C": "The white van is positioned beside the red car in an adjacent lane",
        "D": "The white van is located behind the red car in the traffic flow"
    },
    "correct_answer": "A",
    "image_description": "The UAV captures a white van driving ahead of a red car on a multi-lane road from above."
}

Example 2:
{
    "question_id": "Sim3_OG_UAV2_1002",
    "question_type": "2.3 UAV Target Grounding (UAV2)",
    "question": "How is the pedestrian positioned within the UAV's field of view?",
    "options": {
        "A": "The pedestrian is on the sidewalk near the road intersection",
        "B": "The pedestrian is in the center of the intersection crossing",
        "C": "The pedestrian is standing next to a parked vehicle",
        "D": "The pedestrian is walking along the road shoulder"
    },
    "correct_answer": "A",
    "image_description": "The UAV's aerial view shows a pedestrian positioned on the sidewalk near the road intersection."
}
"""

    user_prompt = f"""First, provide a brief description (50-100 words) of the key targets and their spatial positions from the UAV perspective in this image from {uav_id}.
Then, create a multiple-choice question about UAV target grounding based on this description.

REQUIREMENTS:
- Question should test understanding of target spatial positions from UAV aerial view
- Focus on: {grounding_types[random.randint(0, 1)]}
- Emphasize the UAV's spatial perception and target positioning capabilities
- Ensure question is specific, unambiguous, and DISTINCT from examples in structure, focus, and vocabulary
- Create 4 plausible options with distinct meanings (avoid minor rephrasing or symmetric phrasing like 'left vs. right' or 'ahead vs. behind')
- Use clear, professional English emphasizing UAV context
- IMPORTANT: Use the EXACT question_id format provided in the JSON template
- Include the image description in the output JSON as 'image_description'

{few_shot_examples}

JSON format:
{{
    "question_id": "Sim3_OG_{uav_id}_{counter}",
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
        result["question_id"] = f"Sim3_OG_{uav_id}_{counter}"
    return result


def generate_few_shot_object_matching_q(current_path, other_paths, uav_id, counter=1):
    """Generate UAV target matching questions with few-shot examples (optimized for multi-UAV)"""
    system_prompt = f"""You are an expert teacher of the "UAV Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' ability to match targets across multiple UAV perspectives, focusing on drone detection, vehicle tracking, and pedestrian identification.

CRITICAL RULES:
1. ALWAYS respond in English only
2. Focus on UAV-specific target matching: drones, vehicles, pedestrians across aerial views
3. Questions must emphasize the UAV's multi-perspective target tracking capabilities
4. Each question should have exactly 4 options (A, B, C, D) with only one correct answer
5. Options should be plausible, distinct in meaning, and avoid minor rephrasing
6. Output must be valid JSON format only

A simple question like "What is the white truck in image 1?" with the answer "The white truck in image 2" is USELESS. Avoid this.
Instead, follow this reasoning process to create a high-quality question:

THINKING PROCESS:
1. Identify a Candidate Target: In the first image ({uav_id}), find a distinct target (drone, vehicle, pedestrian) that is also clearly visible in one of the subsequent images (other UAVs). Let's call this the "target object".
2. Analyze the Change: Critically compare the target object's appearance and context between the UAV views. Focus on what has CHANGED. Examples of changes include:
* Perspective: "The vehicle seen from the side" in {uav_id} is now "seen from the rear" in another UAV view.
* Relative Position: "The car behind the bus" in {uav_id} is now "the car beside a red sedan" in another UAV view.
* Action/State: "The pedestrian walking towards the crosswalk" in {uav_id} is now "the pedestrian waiting at the crosswalk" in another UAV view.
* Occlusion: "The partially occluded blue car" in {uav_id} is "fully visible" in another UAV view.
3. Formulate the Question: The question should describe the target in {uav_id} using its appearance or relative position from UAV perspective. For example: "The silver SUV turning at the intersection in {uav_id}'s aerial view corresponds to which target in another UAV's perspective?"
4. Create the Answer and Distractors:
* The correct answer must describe the SAME target in the other UAV view, highlighting the CHANGE you analyzed in step 2. (e.g., "The silver SUV now seen from behind, approaching the bridge in UAV3.")
* The distractors should be plausible but incorrect. They could describe other similar targets in the other UAV views, or describe the correct target with a wrong state/location.
5. Final Output: Assemble everything into the required JSON format. Only output the final JSON.

"""

    matching_types = [
        "matching a drone across UAV perspectives",
        "matching a vehicle across aerial views",
        "matching a pedestrian across UAV perspectives"
    ]

    few_shot_examples = """
EXAMPLES OF GOOD UAV TARGET MATCHING QUESTIONS:

Example 1:
{
    "question_id": "Sim3_OM_UAV1_1001",
    "question_type": "2.4 UAV Target Matching (UAV1)",
    "question": "Which target in another UAV's aerial view corresponds to the red truck seen from the front in UAV1's perspective?",
    "options": {
        "A": "The red truck now seen from the side, parked along the curb in UAV2's aerial view",
        "B": "The blue sedan now seen from the front, moving towards the intersection in UAV3's perspective",
        "C": "The white van now seen from behind, parked on the right lane in UAV2's field of view",
        "D": "The green bus now seen from the front, stationary in UAV3's aerial perspective"
    },
    "correct_answer": "A",
    "image_description": "UAV1's aerial view shows a red truck seen from the front; UAV2's perspective shows the same red truck from the side, parked along the curb."
}

Example 2:
{
    "question_id": "Sim3_OM_UAV2_1002",
    "question_type": "2.4 UAV Target Matching (UAV2)",
    "question": "Which target in another UAV's perspective corresponds to the pedestrian standing near the traffic light in UAV2's aerial view?",
    "options": {
        "A": "The pedestrian now sitting on the bench near the traffic light in UAV1's perspective",
        "B": "The pedestrian seen walking towards the crosswalk in UAV3's aerial view",
        "C": "The pedestrian now standing near a car in UAV1's field of view",
        "D": "The pedestrian seen sitting at a table in UAV3's perspective"
    },
    "correct_answer": "A",
    "image_description": "UAV2's aerial view shows a pedestrian standing near a traffic light; UAV1's perspective shows the same pedestrian sitting on a bench near the traffic light."
}
"""

    user_prompt = f"""First, provide a brief description (50-100 words) of the key targets visible from the UAV perspectives in the image from {uav_id} (first image) and images from other UAVs (subsequent images).
Then, create a multiple-choice question about UAV target matching based on this description.

REQUIREMENTS:
- Question should test ability to match targets across UAV perspectives
- Focus on: {matching_types[random.randint(0, 2)]}
- Emphasize the UAV's multi-perspective target tracking capabilities
- Ensure question is specific, unambiguous, and DISTINCT from examples
- Create 4 plausible options with distinct meanings
- Use clear, professional English emphasizing UAV context
- IMPORTANT: Use the EXACT question_id format provided in the JSON template
- Include the image description in the output JSON as 'image_description'

{few_shot_examples}

JSON format:
{{
    "question_id": "Sim3_OM_{uav_id}_{counter}",
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
        result["question_id"] = f"Sim3_OM_{uav_id}_{counter}"
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

    all_results = {
        "dataset": "Sim_3_UAVs",
        "total_groups": len(image_groups),
        "results": []
    }

    quality_stats = {"EXCELLENT": 0, "GOOD": 0, "FAIR": 0, "POOR": 0, "ERROR": 0}
    group_results = []

    # Initialize counters for sequential numbering
    counters = {f"UAV{i}": {"or": 1, "oc": 1, "og": 1, "om": 1} for i in range(1, 4)}

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

        for uav_num in range(1, 4):
            uav_id = f"UAV{uav_num}"
            current_path = group_paths[uav_num]
            other_paths = [group_paths[n] for n in range(1, 4) if n != uav_num]
            current_filename = group_filenames[uav_num]

            print(f"  Processing {uav_id}")

            # Extract annotation information from all_samples.json
            scene_name = group_data['sequence_frame'].split('_frame_')[0]  # Extract scene name
            possible_filenames = normalize_filename_for_annotation(current_filename, scene_name)
            annotation = {}
            object_count_from_annotation = None
            object_type_from_annotation = None
            
            for filename in possible_filenames:
                if filename in annotation_map:
                    annotation = annotation_map[filename]
                    # Extract object count and type from all_samples.json
                    object_count_from_annotation = annotation.get('Object_count', None)
                    object_type_from_annotation = annotation.get('Object_type', None)
                    break

            # Extract annotation info
            annotation_info = extract_annotation_info(annotation)

            # Create simple info string for new format
            match = re.search(r'UAV(\d+)_frame_(\d+)', current_filename)
            if match:
                uav_num_from_filename = int(match.group(1))
                frame_num = match.group(2)
                json_info = f"Frame {frame_num} from {uav_id} in {scene_name}"
            else:
                json_info = f"Unknown filename format: {current_filename}"

            # Add all_samples.json information to json_info
            if object_count_from_annotation is not None:
                object_type_str = f" ({object_type_from_annotation})" if object_type_from_annotation else ""
                json_info = f"all_samples.json: {object_count_from_annotation} objects{object_type_str}; " + json_info

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

            # Task 2.2: Object Counting - rule-based (using all_samples.json)
            if object_count_from_annotation is not None:
                counting_q = generate_rule_based_counting_q(object_count_from_annotation, uav_id, counter=str(counters[uav_id]["oc"]))
                quality_counting = evaluate_question_quality(counting_q)
                if counting_q:
                    print(f"  Generated counting question for {uav_id}: {object_count_from_annotation} objects (from all_samples.json)")
                    print(f"Quality: {quality_counting['quality']} (Score: {quality_counting['score']})")
                    if quality_counting['issues']:
                        print(f"Issues: {quality_counting['issues']}")
                    uav_questions.append(counting_q)
                    pair_quality_stats[quality_counting['quality']] += 1
                    counters[uav_id]["oc"] += 1
            else:
                print(f"  Skipping counting question for {uav_id}: no Object_count found in all_samples.json")

            print("Generating model-based questions...")

            # Task 2.1: Object Recognition
            object_rec_q, quality_rec = try_generate_qa(generate_few_shot_object_recognition_q, current_path, uav_id,
                                                        counter=str(counters[uav_id]["or"]))
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
                                                                 uav_id, counter=str(counters[uav_id]["og"]))
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
                                                               uav_id, counter=str(counters[uav_id]["om"]))
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
            save_to_json(all_results, "VQA_Sim3_OU.json")
            print(f"Saved intermediate results after processing {g_idx + 1} groups")

    # Compute and add quality statistics
    all_results['quality_statistics'] = quality_stats

    # Save final results
    if save_to_json(all_results, "VQA_Sim3_OU.json"):
        print(f"\nAll results saved to VQA_Sim3_OU.json")
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
