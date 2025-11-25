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
    """Normalize filename to match annotation file format"""
    # Extract the timestamp part from filename like "3-40m-1623936157944367872-UAV1.png"
    # We need to match against annotation format like "5064a99b-1623936157944367872.png"
    
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


def get_image_groups(uav_dirs):
    """Get corresponding image groups from 5 UAV folders"""
    image_map = defaultdict(list)

    for uav_dir in uav_dirs:
        # Look for both .jpg and .png files
        images = glob.glob(os.path.join(uav_dir, "*.jpg")) + glob.glob(os.path.join(uav_dir, "*.png"))
        for img_path in images:
            filename = os.path.basename(img_path)
            # Updated regex to match the actual filename format: -UAV1, -UAV2, etc.
            match = re.search(r'-UAV(\d+)', filename)
            if match:
                uav_num = int(match.group(1))
                # Create group key by removing the UAV part
                group_key = re.sub(r'-UAV\d+', '', filename)
                image_map[group_key].append({
                    'uav': uav_num,
                    'path': img_path,
                    'filename': filename
                })

    # Filter complete groups with exactly 5 UAVs
    image_groups = []
    for key, group in sorted(image_map.items()):
        if len(group) == 5:
            group.sort(key=lambda x: x['uav'])  # Sort by UAV number
            image_groups.append({
                'sequence_frame': key,
                'group': group
            })

    return image_groups





def find_sample_token_by_timestamp(json_base_dir, sequence_number, height, timestamp):
    """Find the sample token that corresponds to the given timestamp"""
    # Convert timestamp to int for comparison
    target_timestamp = int(timestamp)
    
    # Define possible directory paths to search
    possible_dirs = []
    
    # Primary path based on filename
    if height.endswith('m'):
        primary_dir = os.path.join(json_base_dir, f"{sequence_number}-{height}")
        possible_dirs.append(primary_dir)
    else:
        primary_dir = os.path.join(json_base_dir, sequence_number)
        possible_dirs.append(primary_dir)
    
    # Fallback paths: try other height directories for the same sequence
    if height.endswith('m'):
        # Try other common heights for the same sequence
        for alt_height in ['40m', '60m', '80m']:
            if alt_height != height:
                alt_dir = os.path.join(json_base_dir, f"{sequence_number}-{alt_height}")
                if os.path.exists(alt_dir):
                    possible_dirs.append(alt_dir)
    
    # Also try just the sequence number directory
    seq_dir = os.path.join(json_base_dir, sequence_number)
    if seq_dir not in possible_dirs:
        possible_dirs.append(seq_dir)
    
    # Search in all possible directories
    for json_dir in possible_dirs:
        sample_path = os.path.join(json_dir, "sample.json")
        if os.path.exists(sample_path):
            try:
                with open(sample_path, 'r', encoding='utf-8') as f:
                    samples = json.load(f)
                
                # Find the sample with matching timestamp
                for sample in samples:
                    if sample.get('timestamp') == target_timestamp:
                        print(f"Found timestamp {timestamp} in {json_dir}/sample.json")
                        return sample.get('token')
                
            except Exception as e:
                print(f"Error reading sample.json from {json_dir}: {str(e)}")
                continue
    
    print(f"Warning: No sample found for timestamp {timestamp} in any of the searched directories: {possible_dirs}")
    return None


def load_category_mapping(json_base_dir, sequence_number, height):
    """Load category and instance mappings to get object types"""
    # Define possible directory paths to search
    possible_dirs = []
    
    # Primary path based on filename
    if height.endswith('m'):
        primary_dir = os.path.join(json_base_dir, f"{sequence_number}-{height}")
        possible_dirs.append(primary_dir)
    else:
        primary_dir = os.path.join(json_base_dir, sequence_number)
        possible_dirs.append(primary_dir)
    
    # Fallback paths: try other height directories for the same sequence
    if height.endswith('m'):
        # Try other common heights for the same sequence
        for alt_height in ['40m', '60m', '80m']:
            if alt_height != height:
                alt_dir = os.path.join(json_base_dir, f"{sequence_number}-{alt_height}")
                if os.path.exists(alt_dir):
                    possible_dirs.append(alt_dir)
    
    # Also try just the sequence number directory
    seq_dir = os.path.join(json_base_dir, sequence_number)
    if seq_dir not in possible_dirs:
        possible_dirs.append(seq_dir)
    
    category_mapping = {}
    instance_mapping = {}
    
    # Search in all possible directories
    for json_dir in possible_dirs:
        category_path = os.path.join(json_dir, "category.json")
        instance_path = os.path.join(json_dir, "instance.json")
        
        try:
            # Load category mapping
            if os.path.exists(category_path):
                with open(category_path, 'r', encoding='utf-8') as f:
                    categories = json.load(f)
                    for category in categories:
                        category_mapping[category['token']] = category['name']
            
            # Load instance mapping
            if os.path.exists(instance_path):
                with open(instance_path, 'r', encoding='utf-8') as f:
                    instances = json.load(f)
                    for instance in instances:
                        instance_mapping[instance['token']] = instance['category_token']
            
            # If we found both files, we can stop searching
            if category_mapping and instance_mapping:
                print(f"Found category/instance mappings in {json_dir}")
                break
                
        except Exception as e:
            print(f"Error loading category/instance mappings from {json_dir}: {str(e)}")
            continue
    
    return category_mapping, instance_mapping


def get_annotation_by_token(json_base_dir, sequence_number, height, token):
    """Get annotation information for a specific token"""
    # Define possible directory paths to search
    possible_dirs = []
    
    # Primary path based on filename
    if height.endswith('m'):
        primary_dir = os.path.join(json_base_dir, f"{sequence_number}-{height}")
        possible_dirs.append(primary_dir)
    else:
        primary_dir = os.path.join(json_base_dir, sequence_number)
        possible_dirs.append(primary_dir)
    
    # Fallback paths: try other height directories for the same sequence
    if height.endswith('m'):
        # Try other common heights for the same sequence
        for alt_height in ['40m', '60m', '80m']:
            if alt_height != height:
                alt_dir = os.path.join(json_base_dir, f"{sequence_number}-{alt_height}")
                if os.path.exists(alt_dir):
                    possible_dirs.append(alt_dir)
    
    # Also try just the sequence number directory
    seq_dir = os.path.join(json_base_dir, sequence_number)
    if seq_dir not in possible_dirs:
        possible_dirs.append(seq_dir)
    
    # Search in all possible directories
    for json_dir in possible_dirs:
        annotation_path = os.path.join(json_dir, "sample_annotation.json")
        
        if os.path.exists(annotation_path):
            try:
                with open(annotation_path, 'r', encoding='utf-8') as f:
                    annotations = json.load(f)
                
                # Collect all annotations with matching sample_token
                matching_annotations = []
                for annotation in annotations:
                    if annotation.get('sample_token') == token:
                        matching_annotations.append(annotation)
                
                if matching_annotations:
                    print(f"Found annotations for token {token} in {json_dir}")
                    # Return a structure with all matching annotations
                    return {
                        'sample_token': token,
                        'annotations': matching_annotations,
                        'count': len(matching_annotations)
                    }
                
            except Exception as e:
                print(f"Error reading sample_annotation.json from {json_dir}: {str(e)}")
                continue
    
    print(f"Warning: No annotations found for token {token} in any of the searched directories: {possible_dirs}")
    return None


def get_object_type_counts(annotation_data, category_mapping, instance_mapping):
    """Count objects by type from annotation data - only vehicles, pedestrians, bicycles"""
    type_counts = {}
    
    if not annotation_data or 'annotations' not in annotation_data:
        return type_counts
    
    for annotation_item in annotation_data['annotations']:
        instance_token = annotation_item.get('instance_token')
        if instance_token and instance_token in instance_mapping:
            category_token = instance_mapping[instance_token]
            if category_token in category_mapping:
                category_name = category_mapping[category_token]
                
                # Only count vehicles, pedestrians, and bicycles
                if category_name.startswith('vehicle.'):
                    # Extract vehicle type (e.g., "vehicle.car" -> "car")
                    vehicle_type = category_name.split('.')[1] if len(category_name.split('.')) > 1 else 'vehicle'
                    # Map to common categories
                    if vehicle_type in ['car', 'truck', 'bus', 'motorcycle']:
                        object_type = vehicle_type
                    else:
                        object_type = 'car'  # Default for other vehicles
                    type_counts[object_type] = type_counts.get(object_type, 0) + 1
                elif category_name.startswith('human.'):
                    object_type = 'pedestrian'
                    type_counts[object_type] = type_counts.get(object_type, 0) + 1
                elif category_name.startswith('bicycle.'):
                    object_type = 'bicycle'
                    type_counts[object_type] = type_counts.get(object_type, 0) + 1
                # Skip all other categories (e.g., traffic signs, buildings, etc.)
    
    return type_counts





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


def save_to_json(data, filename="VQA_AeroCollab_OU.json"):
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
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' ability to identify objects in a scene from multiple UAV views.

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


def generate_rule_based_counting_q(object_count_from_annotation, uav_id, counter=1):
    """
    [Rule-Based] Generate object counting questions based on all_samples.json annotation data.
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
        "question_id": f"MDMT_OC_{uav_id}_{counter}",
        "question_type": f"2.2 Object Counting ({uav_id})",
        "question": f"Based on the image analysis, how many targets (vehicles, pedestrians, bicycles) can be observed in {uav_id}'s perspective?",
        "options": option_dict,
        "correct_answer": correct_letter,
        "source": "Rule-Based from all_samples.json"
    }


def generate_few_shot_object_grounding_q(img_path, uav_id, counter=1):
    """Generate object grounding questions with few-shot examples (optimized for multi-UAV)"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' understanding of object spatial positioning in multi-UAV views.

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


def generate_few_shot_object_matching_q(current_path, other_paths, uav_id, counter=1):
    """Generate object matching questions with few-shot examples (optimized for multi-UAV)"""
    system_prompt = f"""You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' ability to match objects across multiple UAV perspectives.

CRITICAL RULES:
1. ALWAYS respond in English only
2. Follow a structured thinking process: analyze all images → identify matching objects → formulate question → create options → verify correctness
3. Questions must be based on actual visual content or provided descriptions
4. Each question should have exactly 4 options (A, B, C, D) with only one correct answer
5. Options should be plausible, distinct in meaning, and avoid minor rephrasing
6. Output must be valid JSON format only

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

    user_prompt = f"""First, provide a brief description (50-100 words) of the key objects visible in the image from {uav_id} (first image) and images from other UAVs (subsequent images).
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
    base_dir = "All_samples"
    uav_dirs = [os.path.join(base_dir, f"UAV{i}") for i in range(1, 6)]

    # Set annotation file paths
    annotation_file = "Annotations/all_samples.json"

    # Set JSON annotation file paths (replaces XML)
    json_base_dir = "Annotations/original_json"

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
        "dataset": "Sim_5_UAVs",
        "total_groups": len(image_groups),
        "results": []
    }

    quality_stats = {"EXCELLENT": 0, "GOOD": 0, "FAIR": 0, "POOR": 0, "ERROR": 0}
    group_results = []

    # Initialize counters for sequential numbering
    counters = {f"UAV{i}": {"or": 1, "oc": 1, "og": 1, "om": 1} for i in range(1, 6)}

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

        for uav_num in range(1, 6):
            uav_id = f"UAV{uav_num}"
            current_path = group_paths[uav_num]
            other_paths = [group_paths[n] for n in range(1, 6) if n != uav_num]
            current_filename = group_filenames[uav_num]

            print(f"  Processing {uav_id}")

            # Extract annotation information from all_samples.json
            possible_filenames = normalize_filename_for_annotation(current_filename)
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

            # Extract JSON annotation information using advanced approach (for display purposes only)
            json_info = ""
            frame_objects = []

            # Parse image filename to extract sequence number, height, and timestamp
            parts = current_filename.split('-')
            if len(parts) >= 3:
                sequence_number = parts[0]  # e.g., "3"
                height = parts[1]           # e.g., "40m"
                timestamp = parts[2]        # e.g., "1623936157944367872"
                
                # Load category and instance mappings
                category_mapping, instance_mapping = load_category_mapping(json_base_dir, sequence_number, height)
                
                # Find the sample token for this timestamp
                sample_token = find_sample_token_by_timestamp(json_base_dir, sequence_number, height, timestamp)
                if sample_token:
                    # Get annotation information for this token
                    annotation_data = get_annotation_by_token(json_base_dir, sequence_number, height, sample_token)
                    if annotation_data and 'annotations' in annotation_data:
                        # Extract object information from annotations - only vehicles, pedestrians, bicycles
                        frame_objects = []
                        for annotation_item in annotation_data['annotations']:
                            instance_token = annotation_item.get('instance_token')
                            if instance_token and instance_token in instance_mapping:
                                category_token = instance_mapping[instance_token]
                                if category_token in category_mapping:
                                    category_name = category_mapping[category_token]
                                    
                                    # Only include vehicles, pedestrians, and bicycles
                                    if (category_name.startswith('vehicle.') or 
                                        category_name.startswith('human.') or 
                                        category_name.startswith('bicycle.')):
                                        
                                        # Determine object label
                                        if category_name.startswith('vehicle.'):
                                            vehicle_type = category_name.split('.')[1] if len(category_name.split('.')) > 1 else 'vehicle'
                                            if vehicle_type in ['car', 'truck', 'bus', 'motorcycle']:
                                                label = vehicle_type
                                            else:
                                                label = 'vehicle'
                                        elif category_name.startswith('human.'):
                                            label = 'pedestrian'
                                        elif category_name.startswith('bicycle.'):
                                            label = 'bicycle'
                                        
                                        # Extract basic object information
                                        obj_info = {
                                            'label': label,
                                            'track_id': annotation_item.get('instance_token', ''),
                                            'bbox': {
                                                'translation': annotation_item.get('translation', []),
                                                'size': annotation_item.get('size', []),
                                                'rotation': annotation_item.get('rotation', [])
                                            }
                                        }
                                        frame_objects.append(obj_info)
                        
                        # Get object type counts
                        type_counts = get_object_type_counts(annotation_data, category_mapping, instance_mapping)
                        
                        # Format the json_info with detailed object counts
                        if type_counts:
                            type_info_parts = []
                            for obj_type, count in sorted(type_counts.items()):
                                type_info_parts.append(f"{obj_type}: {count}")
                            type_info = ", ".join(type_info_parts)
                            json_info = f"Frame objects: {len(frame_objects)} objects ({type_info})"
                        else:
                            json_info = f"Frame objects: {len(frame_objects)} visible objects (token: {sample_token})"
                    else:
                        json_info = f"No annotation found for token: {sample_token}"
                else:
                    json_info = f"No sample token found for timestamp: {timestamp}"
            else:
                json_info = f"Could not parse filename: {current_filename}"

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
            save_to_json(all_results, "VQA_AeroCollab_OU.json")
            print(f"Saved intermediate results after processing {g_idx + 1} groups")

    # Compute and add quality statistics
    all_results['quality_statistics'] = quality_stats

    # Save final results
    if save_to_json(all_results, "VQA_AeroCollab_OU.json"):
        print(f"\nAll results saved to VQA_AeroCollab_OU.json")
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
