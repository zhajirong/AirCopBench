# Scene_Understanding.py (modified for 5 UAVs)
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

# Set API key and base URL (using official OpenAI API)
API_KEY = 'your_api_key'
BASE_URL = 'https://api.openai.com/v1'
client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)

def encode_image(image_path):
    """Encode image to base64 string"""
    try:
        with Image.open(image_path) as img:
            buffered = io.BytesIO()
            img.save(buffered, format=img.format if img.format else "JPEG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image {image_path}: {str(e)}")
        return None


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
    """Count objects by type from annotation data"""
    type_counts = {}
    
    if not annotation_data or 'annotations' not in annotation_data:
        return type_counts
    
    for annotation_item in annotation_data['annotations']:
        instance_token = annotation_item.get('instance_token')
        if instance_token and instance_token in instance_mapping:
            category_token = instance_mapping[instance_token]
            if category_token in category_mapping:
                category_name = category_mapping[category_token]
                
                # Simplify category names for better readability
                if category_name.startswith('vehicle.'):
                    # Extract vehicle type (e.g., "vehicle.car" -> "car")
                    vehicle_type = category_name.split('.')[1] if len(category_name.split('.')) > 1 else 'vehicle'
                    # Map to common categories
                    if vehicle_type in ['car', 'truck', 'bus', 'motorcycle']:
                        object_type = vehicle_type
                    else:
                        object_type = 'car'  # Default for other vehicles
                elif category_name.startswith('human.'):
                    object_type = 'person'
                elif category_name.startswith('bicycle.'):
                    object_type = 'bicycle'
                else:
                    object_type = category_name
                
                type_counts[object_type] = type_counts.get(object_type, 0) + 1
    
    return type_counts


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


def get_annotation_for_uav(current_filename, annotation_map, json_base_dir, use_mock=False):
    """Get annotation information for a specific UAV image"""
    annotation_info = ""
    json_info = ""
    combined_info = ""
    
    if not use_mock:
        # Extract annotation information from all_samples.json
        possible_filenames = normalize_filename_for_annotation(current_filename)
        annotation = {}
        for filename in possible_filenames:
            if filename in annotation_map:
                annotation = annotation_map[filename]
                break

        # Extract annotation info
        annotation_info = extract_annotation_info(annotation)

        # Extract JSON annotation information
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
                    # Extract object information from annotations
                    frame_objects = []
                    for annotation_item in annotation_data['annotations']:
                        # Extract basic object information
                        obj_info = {
                            'label': 'vehicle',  # Default label, could be enhanced with category lookup
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

        # Combine annotation information
        combined_info = f"{annotation_info}; {json_info}".strip('; ')
    
    return annotation_info, json_info, combined_info


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


def check_option_diversity(options):
    """Check if options are sufficiently different"""
    for i, opt1 in enumerate(options.values()):
        for j, opt2 in enumerate(options.values()):
            if i < j:
                similarity = difflib.SequenceMatcher(None, opt1, opt2).ratio()
                if similarity > 0.85:  # Adjusted threshold
                    return False, f"Options {i + 1} and {j + 1} too similar: {opt1} vs {opt2}"
    return True, ""


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


def generate_few_shot_scene_description_q(img_path, uav_id, use_mock=False, mock_desc=None, index=0, counter=1):
    """Generate scene description questions with few-shot examples (optimized for multi-UAV)"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' understanding of scene analysis in multi-UAV views.

CRITICAL RULES:
1. ALWAYS respond in English only - never use any other language
2. Follow a structured thinking process: analyze → identify key elements → formulate question → create options → verify correctness
3. Questions must be based on actual visual content or provided description
4. Each question should have exactly 4 options (A, B, C, D) with only one correct answer
5. Options should be plausible, distinct in meaning, and avoid minor rephrasing
6. Output must be valid JSON format only

THINKING PROCESS:
1. First, describe the key objects, relationships, and layout in the image or description
2. Identify the focus based on generation index
3. Formulate a clear, specific question about scene understanding
4. Create 4 distinct options where only one is correct
5. Verify the question is unambiguous and answerable"""

    question_types = [
        "object relationships (e.g., interactions between vehicles and pedestrians)",
        "spatial arrangements (e.g., layout of roads or objects)",
        "scene dynamics (e.g., movement patterns or activities)",
        "environmental features (e.g., urban vs. rural setting)"
    ]

    few_shot_examples = """
EXAMPLES OF GOOD SCENE DESCRIPTION QUESTIONS:

Example 1:
{
    "question_id": "MDMT_SD_UAV1_1001",
    "question_type": "1.1 Scene Description (UAV1)",
    "question": "What is the relationship between the vehicles and the pedestrian crossing in this scene?",
    "options": {
        "A": "Vehicles are stopped at the crossing, allowing pedestrians to pass",
        "B": "Vehicles are driving over the crossing, blocking pedestrians",
        "C": "Vehicles are parked parallel to the crossing",
        "D": "Vehicles are far from the crossing with no interaction"
    },
    "correct_answer": "A",
    "image_description": "The scene shows a busy intersection with vehicles stopped at a pedestrian crossing, allowing pedestrians to pass safely."
}

Example 2:
{
    "question_id": "MDMT_SD_UAV2_1002",
    "question_type": "1.1 Scene Description (UAV2)",
    "question": "How are the buildings arranged relative to the road in this scene?",
    "options": {
        "A": "Buildings are aligned along one side of the road",
        "B": "Buildings surround the road on all sides",
        "C": "Buildings are scattered randomly away from the road",
        "D": "Buildings are absent from the scene"
    },
    "correct_answer": "B",
    "image_description": "The scene depicts a road surrounded by buildings on all sides, forming an urban corridor."
}

Example 3:
{
    "question_id": "MDMT_SD_UAV1_1003",
    "question_type": "1.1 Scene Description (UAV1)",
    "question": "What is the primary movement pattern in this scene?",
    "options": {
        "A": "Vehicles moving in a single direction",
        "B": "Pedestrians crossing the road in multiple directions",
        "C": "Stationary vehicles with no movement",
        "D": "Cyclists riding alongside vehicles"
    },
    "correct_answer": "C",
    "image_description": "The scene shows a quiet road with stationary vehicles parked along the side and no visible pedestrian movement."
}

Example 4:
{
    "question_id": "MDMT_SD_UAV2_1004",
    "question_type": "1.1 Scene Description (UAV2)",
    "question": "What is the dominant environmental feature in this scene?",
    "options": {
        "A": "A dense urban area with tall buildings",
        "B": "A rural area with open fields",
        "C": "A park with scattered trees",
        "D": "A waterfront with docks"
    },
    "correct_answer": "A",
    "image_description": "The scene is a dense urban area with tall buildings lining a multi-lane road."
}
"""

    user_prompt = f"""First, provide a brief description (50-100 words) of the key objects, relationships, and layout in {'this image' if not use_mock else 'the provided scene description'} from {uav_id}.
Then, create a multiple-choice question about scene description based on this description.

REQUIREMENTS:
- Question should test understanding of objects, their relationships, or scene context
- This is generation {index + 1}/4. Focus on: {question_types[index]}
- Ensure question is specific, unambiguous, and DISTINCT from previous generations and examples in structure, focus, and vocabulary
- Create 4 plausible options with distinct meanings (avoid minor rephrasing or symmetric phrasing like 'UAV1 shows more X' vs 'UAV2 shows more X')
- Use clear, professional English
- Include the image description in the output JSON as 'image_description'
- IMPORTANT: Use the EXACT question_id format provided in the JSON template

{few_shot_examples}

Please provide your response in the following JSON format:
{{
    "question_id": "MDMT_SD_{uav_id}_{counter}",
    "question_type": "1.1 Scene Description ({uav_id})",
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
                {"image": f"data:image/jpeg;base64,{encode_image(img_path)}"} if not use_mock else {"text": mock_desc},
                {"text": user_prompt}
            ]
        }
    ]

    result = call_chatgpt_api(messages)
    if "error" not in result:
        result["source"] = "Model-Based (Few-Shot)"
        result["question_id"] = f"MDMT_SD_{uav_id}_{counter}"
    return result


def generate_few_shot_scene_comparison_q(current_path, other_paths, uav_id, use_mock=False, mock_desc1=None, mock_desc2=None,
                                         index=0, counter=1):
    """Generate scene comparison questions with few-shot examples (optimized for multi-UAV)"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' ability to compare and integrate information from multiple UAV perspectives.

CRITICAL RULES:
1. ALWAYS respond in English only - never use any other language
2. Follow a structured thinking process: analyze all images → identify differences/similarities → formulate comparison question → create options → verify correctness
3. Questions must be based on actual visual content or provided descriptions
4. Each question should have exactly 4 options (A, B, C, D) with only one correct answer
5. Options should be plausible, distinct in meaning, and avoid minor rephrasing
6. Output must be valid JSON format only

THINKING PROCESS:
1. First, describe the key differences and similarities in objects, layout, or perspective between the image from {uav_id} (first image) and the other UAV images (subsequent images)
2. Identify the focus based on generation index
3. Formulate a clear question about comparing the perspectives
4. Create 4 distinct options where only one is correct
5. Verify the question is unambiguous and answerable"""

    comparison_types = [
        "visibility differences (e.g., occlusion of objects)",
        "object differences (e.g., number or type of objects)"
    ]

    few_shot_examples = """
EXAMPLES OF GOOD SCENE COMPARISON QUESTIONS:

Example 1:
{
    "question_id": "MDMT_SC_1",
    "question_type": "1.2 Scene Comparison",
    "question": "How does the visibility of the central vehicle differ between the two UAV perspectives?",
    "options": {
        "A": "The vehicle is fully visible in both perspectives",
        "B": "The vehicle is partially occluded in UAV1 but fully visible in UAV2",
        "C": "The vehicle is fully visible in UAV1 but completely hidden in UAV2",
        "D": "The vehicle is partially occluded in both perspectives"
    },
    "correct_answer": "B",
    "image_description": "UAV1 shows a busy intersection with a central vehicle partially blocked by another car; UAV2 shows the same intersection from a higher angle with the vehicle fully visible."
}

Example 2:
{
    "question_id": "MDMT_SC_2",
    "question_type": "1.2 Scene Comparison",
    "question": "Which UAV perspective shows more vehicles on the road?",
    "options": {
        "A": "UAV1 shows more vehicles than UAV2",
        "B": "UAV2 shows more vehicles than UAV1",
        "C": "Both UAV1 and UAV2 show the same number of vehicles",
        "D": "Neither UAV1 nor UAV2 shows any vehicles"
    },
    "correct_answer": "A",
    "image_description": "UAV1 shows a road with 8 vehicles visible in the lanes; UAV2 shows fewer vehicles (3) due to occlusion or limited view."
},

Example 3:
    {
        "question_id": "MDMT_SC_3",
        "question_type": "1.2 Scene Comparison",
        "question": "How do the types of vehicles in UAV1 compare to those in UAV2?",
        "options": {
            "A": "UAV1 shows only cars, while UAV2 shows a mix of cars and trucks",
            "B": "Both UAV1 and UAV2 show only cars",
            "C": "UAV1 shows a mix of cars and trucks, while UAV2 shows only cars",
            "D": "UAV1 shows motorcycles, while UAV2 shows only cars"
        },
        "correct_answer": "A",
        "image_description": "UAV1 shows multiple cars on the road; UAV2 shows a mixture of cars and trucks, with some vehicles partially occluded."
    }

Example 4:
{
    "question_id": "MDMT_SC_4",
    "question_type": "1.2 Scene Comparison",
    "question": "Which perspective provides better information about the pedestrian's location relative to the road?",
    "options": {
        "A": "UAV1 provides clearer spatial context due to its side angle",
        "B": "UAV2 provides clearer spatial context due to its overhead view",
        "C": "Both perspectives provide equally clear information",
        "D": "Neither perspective clearly shows the spatial relationship"
    },
    "correct_answer": "A",
    "image_description": "UAV1 captures a side view of a road with pedestrians on a crossing; UAV2 shows an overhead view with less clear pedestrian positioning."
}

Example 5:
{
    "question_id": "MDMT_SC_5",
    "question_type": "1.2 Scene Comparison",
    "question": "How does the traffic flow appear different between the two UAV perspectives?",
    "options": {
        "A": "UAV1 shows a denser concentration of vehicles",
        "B": "UAV2 provides a wider view of the road layout",
        "C": "UAV1 shows stationary vehicles while UAV2 shows moving ones",
        "D": "Both perspectives show identical traffic patterns"
    },
    "correct_answer": "B",
    "image_description": "UAV1 shows a close-up of a congested road; UAV2 shows a broader view of the same road with clearer lane divisions."
}
"""

    user_prompt = f"""First, provide a brief description (50-100 words) of the key differences and similarities in objects, layout, or perspective between the image from {uav_id} (first image) and the other UAV images (subsequent images).
Then, create a multiple-choice question about scene comparison based on this description.

REQUIREMENTS:
- Question should focus on differences between the perspectives
- This is generation {index + 1}/2. Focus on: {comparison_types[index]}
- Ensure question is specific, unambiguous, and DISTINCT from previous generations and examples in structure, focus, and vocabulary
- Create 4 plausible options with distinct meanings (avoid minor rephrasing or symmetric phrasing like 'UAV1 shows more X' vs 'UAV2 shows more X')
- Use clear, professional English
- Include the image description in the output JSON as 'image_description'
- IMPORTANT: Use the EXACT question_id format provided in the JSON template

{few_shot_examples}

Please provide your response in the following JSON format:
{{
    "question_id": "MDMT_SC_{counter}",
    "question_type": "1.2 Scene Comparison",
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
        result["question_id"] = f"MDMT_SC_{counter}"
    return result


def generate_few_shot_observing_posture_q(current_path, other_paths, uav_id, use_mock=False, mock_desc1=None, mock_desc2=None,
                                          index=0, counter=1):
    """Generate observing posture questions with few-shot examples (optimized for multi-UAV)"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' understanding of observer positioning and perspective analysis in multi-UAV views.

CRITICAL RULES:
1. ALWAYS respond in English only - never use any other language
2. Follow a structured thinking process: analyze all images → identify observer positions → formulate posture question → create options → verify correctness
3. Questions must be based on actual visual content or provided descriptions
4. Each question should have exactly 4 options (A, B, C, D) with only one correct answer
5. Options should be plausible, distinct in meaning, and avoid minor rephrasing
6. Output must be valid JSON format only

THINKING PROCESS:
1. First, describe the observer positions and perspective differences between the image from {uav_id} (first image) and the other UAV images (subsequent images)
2. Identify the focus based on generation index
3. Formulate a clear question about observing posture or perspective relationships
4. Create 4 distinct options where only one is correct
5. Verify the question is unambiguous and answerable"""

    posture_types = [
        "clarity of specific object features (e.g., movement or orientation)",
        "impact of viewing angle on perception (e.g., distance or layout)"
    ]

    few_shot_examples = """
EXAMPLES OF GOOD OBSERVING POSTURE QUESTIONS:

Example 1:
{
    "question_id": "MDMT_OP_1001",
    "question_type": "1.3 Observing Posture",
    "question": "Which UAV perspective provides a better view of the pedestrian's movement direction?",
    "options": {
        "A": "UAV1 provides a clearer view due to its side angle",
        "B": "UAV2 provides a clearer view due to its overhead angle",
        "C": "Both UAVs provide equally clear views",
        "D": "Neither UAV can clearly determine the pedestrian's direction"
    },
    "correct_answer": "A",
    "image_description": "UAV1 shows a side view of a pedestrian crossing a road; UAV2 shows an overhead view with less clear movement direction."
}

Example 2:
{
    "question_id": "MDMT_OP_1002",
    "question_type": "1.3 Observing Posture",
    "question": "How does the viewing angle affect the perception of vehicle distances in this scene?",
    "options": {
        "A": "UAV1's angle makes vehicles appear closer together",
        "B": "UAV2's angle makes vehicles appear further apart",
 "C": "Both angles provide similar distance perception",
        "D": "Neither angle provides reliable distance information"
    },
    "correct_answer": "B",
    "image_description": "UAV1 shows a close-up view of vehicles on a road; UAV2 shows a wider view with clearer distance separation."
}

Example 3:
{
    "question_id": "MDMT_OP_1003",
    "question_type": "1.3 Observing Posture",
    "question": "Which perspective better shows the alignment of the road with surrounding buildings?",
    "options": {
        "A": "UAV1 provides better alignment due to its lower angle",
        "B": "UAV2 provides better alignment due to its higher angle",
        "C": "Both perspectives show equal alignment clarity",
        "D": "Neither perspective clearly shows the alignment"
    },
    "correct_answer": "A",
    "image_description": "UAV1 captures a low-angle view of a road and buildings; UAV2 shows a high-angle view with less clear alignment."
}
"""

    user_prompt = f"""First, provide a brief description (50-100 words) of the observer positions and perspective differences between the image from {uav_id} (first image) and the other UAV images (subsequent images).
Then, create a multiple-choice question about observing posture based on this description.

REQUIREMENTS:
- Question should ask about relative positions and directions of observers in the scene
- This is generation {index + 1}/2. Focus on: {posture_types[index]}
- Ensure question is specific, unambiguous, and DISTINCT from previous generations and examples in structure, focus, and vocabulary
- Create 4 plausible options with distinct meanings (avoid minor rephrasing or symmetric phrasing like 'UAV1 shows more X' vs 'UAV2 shows more X')
- Use clear, professional English
- Include the image description in the output JSON as 'image_description'
- IMPORTANT: Use the EXACT question_id format provided in the JSON template

{few_shot_examples}

Please provide your response in the following JSON format:
{{
    "question_id": "MDMT_OP_{counter}",
    "question_type": "1.3 Observing Posture",
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
        result["question_id"] = f"MDMT_OP_{counter}"
    return result


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
        if correct not in ["A", "B", "C", "D"]:
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


def test_scene_understanding_enhanced(use_mock=False):
    """Enhanced test function with quality evaluation and proper image pairing"""

    # Mock descriptions for simulation
    mock_uav1_desc = "A busy intersection with a red car waiting at a traffic light and a pedestrian crossing the street."
    mock_uav2_desc = "The same intersection from a different angle, showing the rear of the red car and the pedestrian on the crosswalk."

    if not use_mock:
        # Set test image paths
        base_dir = "All_samples"
        uav_dirs = [os.path.join(base_dir, f"UAV{i}") for i in range(1, 6)]

        # Set annotation file paths
        annotation_file = "Annotations/all_samples.json"
        
        # Set JSON annotation file paths
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
        
        # Get image groups
        image_groups = get_image_groups(uav_dirs)

        if not image_groups:
            print("Error: No matching image groups found. Falling back to mock test.")
            use_mock = True
    else:
        image_groups = [{
            'sequence_frame': "mock_traffic_scene",
            'group': [{'uav': i, 'path': None, 'filename': f"mock-UAV{i}.jpg"} for i in range(1, 6)]
        }]
        annotation_map = {}

    quality_stats = {"EXCELLENT": 0, "GOOD": 0, "FAIR": 0, "POOR": 0, "ERROR": 0}
    group_results = []

    # Initialize counters for sequential numbering
    counters = {f"UAV{i}": {"sd": 1, "sc": 1, "op": 1} for i in range(1, 6)}

    for group_idx, group_data in enumerate(image_groups):
        print(f"\n=== Processing Image Group {group_idx + 1}: {group_data['sequence_frame']} ===")
        if not use_mock:
            group = group_data['group']
            group_paths = {item['uav']: item['path'] for item in group}
            group_filenames = {item['uav']: item['filename'] for item in group}

            print(f"Testing with images:")
            for uav_num in range(1, 6):
                print(f"UAV{uav_num}: {os.path.basename(group_paths[uav_num])}")
        else:
            group_paths = {i: None for i in range(1, 6)}
            group_filenames = {i: f"mock-UAV{i}.jpg" for i in range(1, 6)}

        group_quality_stats = {"EXCELLENT": 0, "GOOD": 0, "FAIR": 0, "POOR": 0, "ERROR": 0}
        group_all_results = []



        # Test 1.1 Scene Description
        print("\n=== Testing 1.1 Scene Description ===")
        question_types = [
            "object relationships (e.g., interactions between vehicles and pedestrians)",
            "spatial arrangements (e.g., layout of roads or objects)",
            "scene dynamics (e.g., movement patterns or activities)",
            "environmental features (e.g., urban vs. rural setting)"
        ]

        # 随机选择一种问题类型
        selected_type_index = random.randint(0, len(question_types) - 1)
        selected_type = question_types[selected_type_index]
        print(f"\n--- Scene Description Test (Randomly selected: {selected_type}) ---")

        for uav_num in range(1, 6):
            uav_id = f"UAV{uav_num}"
            current_path = group_paths[uav_num]
            current_filename = group_filenames[uav_num]

            # Get annotation information for this UAV
            annotation_info, json_info, combined_info = get_annotation_for_uav(
                current_filename, annotation_map, json_base_dir, use_mock
            )

            result, quality = try_generate_qa(
                generate_few_shot_scene_description_q,
                current_path, uav_id, use_mock=use_mock, mock_desc=mock_uav1_desc if uav_num == 1 else mock_uav2_desc,
                index=selected_type_index, counter=str(counters[uav_id]["sd"])
            )
            if "error" not in result:
                # Add annotation information to the result
                result["annotation_info"] = annotation_info
                result["json_info"] = json_info
                result["combined_info"] = combined_info
                
                print(f"{uav_id} Question: {result.get('question', 'N/A')}")
                print(f"Options: {result.get('options', {})}")
                print(f"Correct Answer: {result.get('correct_answer', 'N/A')}")
                print(f"Image Description: {result.get('image_description', 'N/A')}")
                print(f"Annotation Info: {combined_info}")
                print(f"Quality: {quality['quality']} (Score: {quality['score']})")
                if quality['issues']:
                    print(f"Issues: {quality['issues']}")
                group_all_results.append(result)
                group_quality_stats[quality['quality']] += 1
            else:
                print(f"{uav_id} Error: {result.get('error')}")
                group_quality_stats["ERROR"] += 1
            counters[uav_id]["sd"] += 1

        # Test 1.2 Scene Comparison
        print("\n=== Testing 1.2 Scene Comparison ===")
        comparison_types = [
            "visibility differences (e.g., occlusion of objects)",
            "object differences (e.g., number or type of objects)"
        ]

        # 随机选择一种问题类型
        selected_comparison_index = random.randint(0, len(comparison_types) - 1)
        selected_comparison_type = comparison_types[selected_comparison_index]
        print(f"\n--- Scene Comparison Test (Randomly selected: {selected_comparison_type}) ---")

        for uav_num in range(1, 6):
            uav_id = f"UAV{uav_num}"
            current_path = group_paths[uav_num]
            current_filename = group_filenames[uav_num]
            other_paths = [group_paths[n] for n in range(1, 6) if n != uav_num]

            # Get annotation information for this UAV
            annotation_info, json_info, combined_info = get_annotation_for_uav(
                current_filename, annotation_map, json_base_dir, use_mock
            )

            result, quality = try_generate_qa(
                generate_few_shot_scene_comparison_q,
                current_path, other_paths, uav_id, use_mock=use_mock,
                mock_desc1=mock_uav1_desc if uav_num == 1 else mock_uav2_desc, mock_desc2=mock_uav2_desc if uav_num == 1 else mock_uav1_desc,
                index=selected_comparison_index, counter=str(counters[uav_id]["sc"])
            )
            if "error" not in result:
                # Add annotation information to the result
                result["annotation_info"] = annotation_info
                result["json_info"] = json_info
                result["combined_info"] = combined_info
                
                print(f"{uav_id} Question: {result.get('question', 'N/A')}")
                print(f"Options: {result.get('options', {})}")
                print(f"Correct Answer: {result.get('correct_answer', 'N/A')}")
                print(f"Image Description: {result.get('image_description', 'N/A')}")
                print(f"Annotation Info: {combined_info}")
                print(f"Quality: {quality['quality']} (Score: {quality['score']})")
                if quality['issues']:
                    print(f"Issues: {quality['issues']}")
                group_all_results.append(result)
                group_quality_stats[quality['quality']] += 1
            else:
                print(f"{uav_id} Error: {result.get('error')}")
                group_quality_stats["ERROR"] += 1
            counters[uav_id]["sc"] += 1

        # Test 1.3 Observing Posture
        print("\n=== Testing 1.3 Observing Posture ===")
        posture_types = [
            "clarity of specific object features (e.g., movement or orientation)",
            "impact of viewing angle on perception (e.g., distance or layout)"
        ]
        for i in range(0, 1):
            print(f"\n--- Observing Posture Test {i + 1} ({posture_types[i]}) ---")
            for uav_num in range(1, 6):
                uav_id = f"UAV{uav_num}"
                current_path = group_paths[uav_num]
                current_filename = group_filenames[uav_num]
                other_paths = [group_paths[n] for n in range(1, 6) if n != uav_num]

                # Get annotation information for this UAV
                annotation_info, json_info, combined_info = get_annotation_for_uav(
                    current_filename, annotation_map, json_base_dir, use_mock
                )

                result, quality = try_generate_qa(
                    generate_few_shot_observing_posture_q,
                    current_path, other_paths, uav_id, use_mock=use_mock,
                    mock_desc1=mock_uav1_desc if uav_num == 1 else mock_uav2_desc,
                    mock_desc2=mock_uav2_desc if uav_num == 1 else mock_uav1_desc, index=i, counter=str(counters[uav_id]["op"])
                )
                if "error" not in result:
                    # Add annotation information to the result
                    result["annotation_info"] = annotation_info
                    result["json_info"] = json_info
                    result["combined_info"] = combined_info
                    
                    print(f"{uav_id} Question: {result.get('question', 'N/A')}")
                    print(f"Options: {result.get('options', {})}")
                    print(f"Correct Answer: {result.get('correct_answer', 'N/A')}")
                    print(f"Image Description: {result.get('image_description', 'N/A')}")
                    print(f"Annotation Info: {combined_info}")
                    print(f"Quality: {quality['quality']} (Score: {quality['score']})")
                    if quality['issues']:
                        print(f"Issues: {quality['issues']}")
                    group_all_results.append(result)
                    group_quality_stats[quality['quality']] += 1
                else:
                    print(f"{uav_id} Error: {result.get('error')}")
                    group_quality_stats["ERROR"] += 1
                counters[uav_id]["op"] += 1

        # Aggregate for this group
        for k, v in group_quality_stats.items():
            quality_stats[k] += v

        group_results.append({
            "sequence_frame": group_data['sequence_frame'],
            "questions": group_all_results
        })

    # Save results
    output_data = {
        "dataset": "Scene_Understanding" + (" (Mock)" if use_mock else ""),
        "total_groups": len(image_groups),
        "results": group_results,
        "quality_statistics": quality_stats
    }

    output_file = "VQA_AeroCollab_SU.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n=== Overall Summary ===")
    print(f"Total questions generated: {sum(len(group['questions']) for group in group_results)}")
    print(f"Overall Quality distribution: {quality_stats}")
    print(f"Results saved to: {output_file}")
    print(f"Annotation information has been successfully integrated for all images.")

    # Detailed quality assessment for all
    print(f"\n=== Detailed Quality Assessment for All Groups ===")
    for group_idx, group_result in enumerate(group_results):
        print(f"\n--- Group {group_idx + 1}: {group_result['sequence_frame']} ---")
        for i, result in enumerate(group_result["questions"]):
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
    test_scene_understanding_enhanced(use_mock=False)
