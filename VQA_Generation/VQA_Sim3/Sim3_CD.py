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
from collections import defaultdict  # Added import for defaultdict
import openai  # Import OpenAI library
from openai import AzureOpenAI 


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
                    if full_filename not in annotation_map:
                        annotation_map[full_filename] = annotation
                else:
                    # Fallback for other formats
                    annotation_map[full_filename] = annotation

        print(f"Loaded {len(annotation_map)} annotation entries")
        print(f"Sample annotation keys: {list(annotation_map.keys())[:5]}")
        return annotation_map
    except Exception as e:
        print(f"Error loading annotations: {str(e)}")
        return {}


def get_image_groups(scene_dirs):
    """Get corresponding image groups from 3 UAV folders across scenes"""
    image_map = defaultdict(list)

    for scene_dir in scene_dirs:
        scene_name = os.path.basename(scene_dir)
        # Look for UAV1, UAV2, UAV3 folders
        for uav_num in range(1, 4):
            uav_dir = os.path.join(scene_dir, f"UAV{uav_num}")
            if os.path.exists(uav_dir):
                # Look for both .jpg and .png files
                images = glob.glob(os.path.join(uav_dir, "*.jpg")) + glob.glob(os.path.join(uav_dir, "*.png"))
                for img_path in images:
                    filename = os.path.basename(img_path)
                    # Extract frame number from filename like "UAV1_frame_001.jpg"
                    frame_match = re.search(r'frame_(\d+)', filename)
                    if frame_match:
                        frame_num = frame_match.group(1)
                        # Create group key using scene and frame
                        group_key = f"{scene_name}_frame_{frame_num}"
                        image_map[group_key].append({
                            'uav': uav_num,
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
    if len(parts) >= 3:
        sequence_number = parts[0]  # e.g., "3"
        height = parts[1]           # e.g., "40m"
        timestamp = parts[2]        # e.g., "1623936157944367872"
        
        # Construct the directory path based on sequence number and height
        if height.endswith('m'):
            json_dir = os.path.join(json_base_dir, f"{sequence_number}-{height}")
        else:
            json_dir = os.path.join(json_base_dir, sequence_number)
        
        # Check if directory exists
        if os.path.exists(json_dir):
            # Look for sample_annotation.json in the directory
            sample_annotation_path = os.path.join(json_dir, "sample_annotation.json")
            if os.path.exists(sample_annotation_path):
                return sample_annotation_path, timestamp
        
        # Fallback: try just the sequence number directory
        fallback_dir = os.path.join(json_base_dir, sequence_number)
        if os.path.exists(fallback_dir):
            sample_annotation_path = os.path.join(fallback_dir, "sample_annotation.json")
            if os.path.exists(sample_annotation_path):
                return sample_annotation_path, timestamp

    return None, None


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
    """Count objects by type from annotation data - prioritize drones as primary targets"""
    type_counts = {}
    
    if not annotation_data or 'annotations' not in annotation_data:
        return type_counts
    
    for annotation_item in annotation_data['annotations']:
        instance_token = annotation_item.get('instance_token')
        if instance_token and instance_token in instance_mapping:
            category_token = instance_mapping[instance_token]
            if category_token in category_mapping:
                category_name = category_mapping[category_token]
                
                # Simplify category names for better readability - prioritize drones
                if category_name.startswith('drone.') or category_name == 'drone':
                    object_type = 'drone'  # Primary target
                elif category_name.startswith('vehicle.'):
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
                    object_type = category_name.split('.')[0] if '.' in category_name else category_name
                
                type_counts[object_type] = type_counts.get(object_type, 0) + 1
    
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

    # Extract quality information
    if 'Quality' in annotation:
        quality = annotation['Quality']
        info_parts.append(f"Image quality: {quality}")

    # Extract usability information
    if 'Usibility' in annotation:
        usability = annotation['Usibility']
        info_parts.append(f"Image usability: {usability}")

    # Extract object information
    if 'Object_type' in annotation:
        object_type = annotation['Object_type']
        info_parts.append(f"Object type: {object_type}")

    if 'Object_count' in annotation:
        object_count = annotation['Object_count']
        info_parts.append(f"Object count: {object_count}")

    # Extract degradation information
    if 'Degradation' in annotation:
        degradation = annotation['Degradation']
        info_parts.append(f"Degradation: {degradation}")

    if 'Other Degradation' in annotation:
        other_degradation = annotation['Other Degradation']
        info_parts.append(f"Other degradation: {other_degradation}")

    # Extract collaboration information
    if 'Collaboration_when' in annotation:
        collaboration_when = annotation['Collaboration_when']
        info_parts.append(f"Collaboration when: {collaboration_when}")

    if 'Collaboration_what' in annotation:
        collaboration_what = annotation['Collaboration_what']
        if isinstance(collaboration_what, list) and len(collaboration_what) > 0:
            # Extract rectangle labels and position information from collaboration_what
            labels = []
            more_info_areas = []
            for item in collaboration_what:
                if 'rectanglelabels' in item:
                    labels.extend(item['rectanglelabels'])
                    # Extract position information for "More information" areas
                    if 'More information' in item['rectanglelabels']:
                        x = item.get('x', 0)
                        y = item.get('y', 0)
                        width = item.get('width', 0)
                        height = item.get('height', 0)
                        more_info_areas.append(f"({x:.1f}, {y:.1f}, {width:.1f}x{height:.1f})")
            if labels:
                info_parts.append(f"Collaboration what: {', '.join(labels)}")
                # Add detailed information about "More information" areas
                if more_info_areas:
                    info_parts.append(f"More information areas: {', '.join(more_info_areas)}")
        else:
            info_parts.append(f"Collaboration what: {collaboration_what}")

    if 'Collaboration_who' in annotation:
        collaboration_who = annotation['Collaboration_who']
        info_parts.append(f"Collaboration who: {collaboration_who}")

    if 'Collaboration_why' in annotation:
        collaboration_why = annotation['Collaboration_why']
        info_parts.append(f"Collaboration why: {collaboration_why}")

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


def save_to_json(data, filename="VQA_Sim3_CD.json"):
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
    """Generate rule-based collaboration timing questions with few-shot examples (optimized for 3-UAV)"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' understanding of when collaboration between multiple UAVs (up to 3) is necessary.

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
    "question_id": "Sim3_when2col_UAV1_1001",
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
    "question_id": "Sim3_when2col_UAV2_1002",
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
        "question_id": f"Sim3_when2col_{uav_id}_{counter}",
        "question_type": f"4.1 When to Collaborate ({uav_id})",
        "question": f"Should {uav_id} collaborate with other UAVs to address {focus}?",
        "options": options,
        "correct_answer": correct_option,
        "source": "Rule-Based (Few-Shot)",
        "annotation_info": f"Annotation indicates: {collaboration_when_value}"
    }

    return result


def generate_few_shot_collaboration_what_q(current_path, other_paths, uav_id, annotation=None, counter=1):
    """Generate model-based collaboration content questions with few-shot examples (optimized for 3-UAV)"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' understanding of what specific object information should be shared between multiple UAVs (up to 3).

CRITICAL RULES:
1. ALWAYS respond in English only
2. Follow a structured thinking process: analyze images → identify specific object information gaps → formulate question → create options → verify correctness
3. Questions must be based on actual visual content or provided descriptions
4. Each question should have exactly 4 options (A, B, C, D) with at least one correct answer
5. Options should be plausible, distinct in meaning, and avoid minor rephrasing
6. Output must be valid JSON format only
7. Focus on specific object descriptions with intuitive location and context details
8. Use intuitive image locations (upper-left corner, center, near landmarks, etc.) instead of numerical positions
9. Prioritize drone detection and tracking as the primary target in all questions

THINKING PROCESS:
1. Analyze all images to identify specific object information gaps in marked regions across multiple UAV views
2. Focus on drone-related object information as the primary target
3. Identify the focus based on generation index
4. Formulate a clear question about what specific object information to share from marked regions
5. Create 4 distinct options, all related to specific object descriptions with intuitive location and context
6. Verify the question is unambiguous and answerable"""

    collaboration_types = [
        "specific object information (e.g., drone/vehicle/pedestrian/bicycle details across views)",
        "scene context information (e.g., traffic flow from multiple UAVs)"
    ]

    few_shot_examples = """
EXAMPLES OF GOOD COLLABORATION WHAT QUESTIONS:

Example 1:
{
    "question_id": "Sim3_what2col_UAV1_1001",
    "question_type": "4.2 What to Collaborate (UAV1)",
    "question": "What specific object information should UAV1 share with other UAVs about the drone in the marked region?",
    "options": {
        "A": "Drone occluded by the tree in the upper-left corner of the image that needs position clarification",
        "B": "Drone flying at low altitude near the bottom edge that requires height verification",
        "C": "Drone moving rapidly from left to right across the center that needs trajectory prediction",
        "D": "Drone with similar color to background near the traffic light that requires contrast enhancement"
    },
    "correct_answer": "A",
    "image_description": "UAV1 shows a drone in the marked region at position (31.5%, 48.1%) with size 6.3%x3.6% that is occluded by a tree in the upper-left corner, requiring detailed position and movement information."
}

Example 2:
{
    "question_id": "Sim3_what2col_UAV2_1002",
    "question_type": "4.2 What to Collaborate (UAV2)",
    "question": "What specific object details should UAV2 share with other UAVs about the drone in the marked region?",
    "options": {
        "A": "Drone under shadows in the center of the image that needs lighting compensation",
        "B": "Drone partially hidden behind power lines near the right edge that requires obstacle mapping",
        "C": "Drone at the far edge of the frame near the horizon that needs range confirmation",
        "D": "Drone with multiple similar objects nearby the building entrance that requires target identification"
    },
    "correct_answer": "A",
    "image_description": "UAV2 captures a drone in the marked region at position (62.4%, 47.3%) with size 6.3%x3.9% that is under shadows in the center of the image, needing precise position and orientation information."
}

Example 3:
{
    "question_id": "Sim3_what2col_UAV3_1003",
    "question_type": "4.2 What to Collaborate (UAV3)",
    "question": "What specific object information should UAV3 share with other UAVs about the drones in the multiple marked regions?",
    "options": {
        "A": "Drone with complex urban background in the upper-right area that is difficult to detect",
        "B": "Drone emerging from behind the tall building on the left side that requires emergence tracking",
        "C": "Drone with reflective surface near the water tower that causes glare and needs glare reduction",
        "D": "Drone in formation with other drones above the bridge that requires formation analysis"
    },
    "correct_answer": "A",
    "image_description": "UAV3 captures drones in two marked regions: position (31.6%, 48.8%) with size 6.7%x2.7% and position (62.2%, 48.1%) with size 6.0%x3.9%, with one drone having complex urban background in the upper-right area that is difficult to detect."
}
"""

    # Extract "More information" areas from annotation if available
    more_info_context = ""
    if annotation and 'Collaboration_what' in annotation:
        collaboration_what = annotation['Collaboration_what']
        if isinstance(collaboration_what, list):
            more_info_areas = []
            for item in collaboration_what:
                if 'rectanglelabels' in item and 'More information' in item['rectanglelabels']:
                    x = item.get('x', 0)
                    y = item.get('y', 0)
                    width = item.get('width', 0)
                    height = item.get('height', 0)
                    more_info_areas.append(f"position ({x:.1f}%, {y:.1f}%) with size {width:.1f}%x{height:.1f}%")
            if more_info_areas:
                more_info_context = f" Note: The annotation indicates specific areas requiring more information at {', '.join(more_info_areas)}."

    user_prompt = f"""First, provide a brief description (50-100 words) of the specific object information gaps in the marked regions between the image from {uav_id} (first image) and images from other UAVs (subsequent images).{more_info_context}
Then, create a multiple-choice question about what specific object information {uav_id} should share about the objects in the marked regions, based on this description.

REQUIREMENTS:
- Question should focus on specific object descriptions with intuitive location and context details
- Focus on: {collaboration_types[random.randint(0, 1)]}
- Prioritize drone detection and tracking as the primary target
- If annotation indicates specific "More information" areas, describe the objects with intuitive location and context (e.g., "drone occluded by tree in upper-left corner", "drone under shadows in center", "drone with complex background in upper-right area", "drone near traffic light", "drone above bridge")
- All options should describe different specific objects with intuitive location and context details
- Use intuitive image locations like: upper-left corner, upper-right area, center, bottom edge, left side, right edge, near [landmark], above [landmark], behind [object]
- Avoid using numerical position values (like percentages or coordinates)
- Ensure question is specific, unambiguous, and DISTINCT from examples
- Create 4 plausible options with distinct meanings, all related to specific object descriptions
- Use clear, professional English
- Include the image description in the output JSON as 'image_description'
- IMPORTANT: Use the EXACT question_id format provided in the JSON template

{few_shot_examples}

JSON format:
{{
    "question_id": "Sim3_what2col_{uav_id}_{counter}",
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
        result["question_id"] = f"Sim3_what2col_{uav_id}_{counter}"
    return result


def generate_rule_based_collaboration_who_q_with_annotation(annotation, uav_id, counter=1):
    """Generate rule-based collaboration partner questions with few-shot examples (optimized for 3-UAV)"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' understanding of which UAV(s) should be the collaboration partner in a multi-UAV setup (up to 3 UAVs).

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
    "question_id": "Sim3_who2col_UAV1_1001",
    "question_type": "4.3 Who to Collaborate (UAV1)",
    "question": "Which UAV should UAV1 collaborate with to gain a complementary perspective in the multi-UAV setup?",
    "options": {
        "A": "UAV2",
        "B": "None (no need for collaboration)",
        "C": "None (no suitable collaboration partner)",
        "D": "UAV3"
    },
    "correct_answer": "A",
    "annotation_info": "Annotation indicates UAV2 as the collaboration partner."
}

Example 2:
{
    "question_id": "Sim3_who2col_UAV2_1002",
    "question_type": "4.3 Who to Collaborate (UAV2)",
    "question": "Which UAV should UAV2 collaborate with to obtain specific object data in the multi-UAV setup?",
    "options": {
        "A": "UAV1",
        "B": "None (no need for collaboration)",
        "C": "UAV3",
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
        partner = partner.strip().upper()

    if partner in ["NONE", "NO"]:
        correct_answer = "None"
    elif "UAV" in partner:
        correct_answer = partner.replace('_', '')
    else:
        return None

    # Generate plausible wrong options
    all_uavs = [f"UAV{i}" for i in range(1, 4) if f"UAV{i}" != uav_id and f"UAV{i}" != correct_answer]
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
        "question_id": f"Sim3_who2col_{uav_id}_{counter}",
        "question_type": f"4.3 Who to Collaborate ({uav_id})",
        "question": f"Which UAV should {uav_id} collaborate with for {focus}?",
        "options": options,
        "correct_answer": correct_option,
        "source": "Rule-Based (Few-Shot)",
        "annotation_info": f"Annotation indicates: {partner}"
    }

    return result


def generate_hybrid_collaboration_why_q(current_path, other_paths, uav_id, annotation, counter=1):
    """Generate hybrid collaboration reason questions with few-shot examples (optimized for 3-UAV)"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' understanding of why collaboration between multiple UAVs (up to 3) is necessary.

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
    "question_id": "Sim3_why2col_UAV1_1001",
    "question_type": "4.4 Why to Collaborate (UAV1)",
    "question": "What is the main reason UAV1 should collaborate with other UAVs?",
    "options": {
        "A": "To overcome partial occlusion of the drones across multiple views",
        "B": "To adjust for lighting conditions in multi-view setup",
        "C": "To capture a wider airspace from different UAV angles",
        "D": "To reduce computational load"
    },
    "correct_answer": "A",
    "image_description": "UAV1 shows a scene with drones partially occluded; other UAVs provide clearer views. The annotation indicates specific areas requiring more information at position (31.5%, 48.1%) with size 6.3%x3.6%."
}

Example 2:
{
    "question_id": "Sim3_why2col_UAV2_1002",
    "question_type": "4.4 Why to Collaborate (UAV2)",
    "question": "What is the primary reason UAV2 should collaborate with other UAVs?",
    "options": {
        "A": "To obtain more clear information about specific drones across views",
        "B": "To compensate for low image quality in some views",
        "C": "To supplement missing information due to limited field of view (FoV) in multi-UAV setup",
        "D": "To synchronize time stamps"
    },
    "correct_answer": "A",
    "image_description": "UAV2 lacks clear drone movement data; other UAVs provide complementary details. The annotation indicates specific areas requiring more information at position (62.4%, 47.3%) with size 6.3%x3.9%."
}

Example 3:
{
    "question_id": "Sim3_why2col_UAV3_1003",
    "question_type": "4.4 Why to Collaborate (UAV3)",
    "question": "What is the primary reason UAV3 should collaborate with other UAVs?",
    "options": {
        "A": "To obtain detailed drone information from multiple regions with limited visibility",
        "B": "To improve overall image quality across all UAVs",
        "C": "To expand the field of view coverage in the airspace",
        "D": "To reduce processing time for drone detection"
    },
    "correct_answer": "A",
    "image_description": "UAV3 captures multiple drone regions with limited visibility. The annotation indicates specific areas requiring more information at position (31.6%, 48.8%) with size 6.7%x2.7% and position (62.2%, 48.1%) with size 6.0%x3.9%."
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
    "question_id": "Sim3_why2col_{uav_id}_{counter}",
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
        result["question_id"] = f"Sim3_why2col_{uav_id}_{counter}"
    return result


def generate_model_based_collaboration_when_q(current_path, other_paths, uav_id, counter=1):
    """Fallback model-based for collaboration when if rule-based fails (optimized for 3-UAV)"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' understanding of when collaboration between multiple UAVs (up to 3) is necessary.

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
    "question_id": "Sim3_when2col_UAV1_1001",
    "question_type": "4.1 When to Collaborate (UAV1)",
    "question": "Should UAV1 collaborate with other UAVs to obtain supplementary information due to incomplete observation data in the multi-UAV setup?",
    "options": {
        "A": "Yes, due to partial occlusion of key drones across views",
        "B": "No, all views are fully visible",
        "C": "Yes, due to poor visibility in multiple UAV views",
        "D": "No, all drones are clearly captured in all UAVs"
    },
    "correct_answer": "A",
    "image_description": "UAV1 shows drones with partial occlusion requiring collaboration. The annotation indicates specific areas requiring more information at position (31.5%, 48.1%) with size 6.3%x3.6%."
}

Example 2:
{
    "question_id": "Sim3_when2col_UAV2_1002",
    "question_type": "4.1 When to Collaborate (UAV2)",
    "question": "Should UAV2 collaborate with other UAVs to address environmental challenges for better perception in the multi-UAV setup?",
    "options": {
        "A": "Yes, due to poor lighting conditions in multiple views",
        "B": "No, lighting is adequate across all UAVs",
        "C": "Yes, due to low image resolution in some views",
        "D": "No, the environment is clear for all UAVs"
    },
    "correct_answer": "A",
    "image_description": "UAV2 shows environmental challenges affecting drone visibility. The annotation indicates specific areas requiring more information at position (62.4%, 47.3%) with size 6.3%x3.9%."
}

Example 3:
{
    "question_id": "Sim3_when2col_UAV3_1003",
    "question_type": "4.1 When to Collaborate (UAV3)",
    "question": "Should UAV3 collaborate with other UAVs to address multiple information gaps in the multi-UAV setup?",
    "options": {
        "A": "Yes, due to multiple drone regions with limited visibility",
        "B": "No, all drone information is clearly visible",
        "C": "Yes, due to single drone region requiring attention",
        "D": "No, no collaboration is needed for this scene"
    },
    "correct_answer": "A",
    "image_description": "UAV3 captures multiple drone regions with limited visibility. The annotation indicates specific areas requiring more information at position (31.6%, 48.8%) with size 6.7%x2.7% and position (62.2%, 48.1%) with size 6.0%x3.9%."
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
    "question_id": "Sim3_when2col_{uav_id}_{counter}",
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
        result["question_id"] = f"Sim3_when2col_{uav_id}_{counter}"
    return result


def generate_model_based_collaboration_why_q(current_path, other_paths, uav_id, counter=1):
    """Fallback model-based for collaboration why if hybrid fails or no annotation (optimized for 3-UAV)"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' understanding of why collaboration between multiple UAVs (up to 3) is necessary.

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
    "question_id": "Sim3_why2col_UAV1_1001",
    "question_type": "4.4 Why to Collaborate (UAV1)",
    "question": "What is the main reason UAV1 should collaborate with other UAVs?",
    "options": {
        "A": "To overcome partial occlusion of the drones across multiple views",
        "B": "To adjust for lighting conditions in multi-view setup",
        "C": "To capture a wider airspace from different UAV angles",
        "D": "To reduce computational load"
    },
    "correct_answer": "A",
    "image_description": "UAV1 shows a scene with drones partially occluded; other UAVs provide clearer views. The annotation indicates specific areas requiring more information at position (31.5%, 48.1%) with size 6.3%x3.6%."
}

Example 2:
{
    "question_id": "Sim3_why2col_UAV2_1002",
    "question_type": "4.4 Why to Collaborate (UAV2)",
    "question": "What is the primary reason UAV2 should collaborate with other UAVs?",
    "options": {
        "A": "To obtain more clear information about specific drones across views",
        "B": "To compensate for low image quality in some views",
        "C": "To supplement missing information due to limited field of view (FoV) in multi-UAV setup",
        "D": "To synchronize time stamps"
    },
    "correct_answer": "A",
    "image_description": "UAV2 lacks clear drone movement data; other UAVs provide complementary details. The annotation indicates specific areas requiring more information at position (62.4%, 47.3%) with size 6.3%x3.9%."
}

Example 3:
{
    "question_id": "Sim3_why2col_UAV3_1003",
    "question_type": "4.4 Why to Collaborate (UAV3)",
    "question": "What is the primary reason UAV3 should collaborate with other UAVs?",
    "options": {
        "A": "To obtain detailed drone information from multiple regions with limited visibility",
        "B": "To improve overall image quality across all UAVs",
        "C": "To expand the field of view coverage in the airspace",
        "D": "To reduce processing time for drone detection"
    },
    "correct_answer": "A",
    "image_description": "UAV3 captures multiple drone regions with limited visibility. The annotation indicates specific areas requiring more information at position (31.6%, 48.8%) with size 6.7%x2.7% and position (62.2%, 48.1%) with size 6.0%x3.9%."
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
    "question_id": "Sim3_why2col_{uav_id}_{counter}",
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
        result["question_id"] = f"Sim3_why2col_{uav_id}_{counter}"
    return result


def generate_model_based_collaboration_who_q(current_path, other_paths, uav_id, counter=1):
    """Fallback model-based for collaboration who if rule-based fails (optimized for 3-UAV)"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' understanding of which UAV(s) should be the collaboration partner in a multi-UAV setup (up to 3 UAVs).

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
    "question_id": "Sim3_who2col_UAV1_1001",
    "question_type": "4.3 Who to Collaborate (UAV1)",
    "question": "Which UAV should UAV1 collaborate with to gain a complementary perspective in the multi-UAV setup?",
    "options": {
        "A": "UAV2",
        "B": "None (no need for collaboration)",
        "C": "None (no suitable collaboration partner)",
        "D": "UAV3"
    },
    "correct_answer": "A",
    "image_description": "UAV1 needs complementary perspective for drone detection. The annotation indicates specific areas requiring more information at position (31.5%, 48.1%) with size 6.3%x3.6%."
}

Example 2:
{
    "question_id": "Sim3_who2col_UAV2_1002",
    "question_type": "4.3 Who to Collaborate (UAV2)",
    "question": "Which UAV should UAV2 collaborate with to obtain specific drone data in the multi-UAV setup?",
    "options": {
        "A": "UAV1",
        "B": "None (no need for collaboration)",
        "C": "UAV3",
        "D": "UAV1, UAV3"
    },
    "correct_answer": "A",
    "image_description": "UAV2 needs specific drone data from other UAVs. The annotation indicates specific areas requiring more information at position (62.4%, 47.3%) with size 6.3%x3.9%."
}

Example 3:
{
    "question_id": "Sim3_who2col_UAV3_1003",
    "question_type": "4.3 Who to Collaborate (UAV3)",
    "question": "Which UAV should UAV3 collaborate with to address multiple information gaps in the multi-UAV setup?",
    "options": {
        "A": "UAV1",
        "B": "UAV2",
        "C": "Both UAV1 and UAV2",
        "D": "None (no collaboration needed)"
    },
    "correct_answer": "C",
    "image_description": "UAV3 has multiple drone regions requiring collaboration. The annotation indicates specific areas requiring more information at position (31.6%, 48.8%) with size 6.7%x2.7% and position (62.2%, 48.1%) with size 6.0%x3.9%."
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
    "question_id": "Sim3_who2col_{uav_id}_{counter}",
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
        result["question_id"] = f"Sim3_who2col_{uav_id}_{counter}"
    return result


def main():
    # Set image directory paths
    base_dir = "Samples/images"
    scene_dirs = [os.path.join(base_dir, f"scene_{i:03d}") for i in range(1, 8)]  # scene_001 to scene_007

    # Set annotation file paths
    annotation_file = "Annotations/all_samples.json"

    # Check if directories exist
    for scene_dir in scene_dirs:
        if not os.path.exists(scene_dir):
            print(f"Warning: Scene directory not found: {scene_dir}")
            continue

    if not os.path.exists(annotation_file):
        print(f"Error: Annotation file not found: {annotation_file}")
        return

    print("Loading data...")
    annotation_map = load_annotations(annotation_file)
    image_groups = get_image_groups(scene_dirs)

    print(f"Loaded {len(annotation_map)} annotation entries")
    print(f"Found {len(image_groups)} image groups to process")

    all_results = {
        "dataset": "Sim_3_UAVs",  # Updated dataset name for 3 UAVs
        "total_groups": len(image_groups),
        "results": []
    }

    quality_stats = {"EXCELLENT": 0, "GOOD": 0, "FAIR": 0, "POOR": 0, "ERROR": 0}
    group_results = []

    # Initialize counters for sequential numbering per UAV (3 UAVs)
    counters = {f"UAV{i}": {"when": 1, "what": 1, "who": 1, "why": 1} for i in range(1, 4)}

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
            other_paths = [group_paths[n] for n in range(1, 4) if n != uav_num]
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
                possible_filenames = normalize_filename_for_annotation(current_filename)
                for filename in possible_filenames:
                    if filename in annotation_map:
                        annotation = annotation_map[filename]
                        matched_key = filename
                        print(f"    Found annotation using filename: {filename}")
                        break
                
                if not annotation:
                    print(f"    No annotation found for {uav_id} frame {group_frame}")
                    print(f"    Tried keys: {scene_frame_key}, {uav_frame_key}, {full_key}")
                    # Show some available keys for debugging
                    available_keys = [k for k in annotation_map.keys() if f'frame_{group_frame}' in k and group_scene in k][:5]
                    print(f"    Available keys for {group_scene} frame {group_frame}: {available_keys}")
            
            # Verify the annotation matches the correct scene and UAV
            if annotation and 'img1' in annotation:
                img1_filename = os.path.basename(annotation['img1'])
                expected_scene_uav = f"scene_{group_scene.split('_')[1]}-UAV{uav_num}"
                if expected_scene_uav not in img1_filename:
                    print(f"    WARNING: Annotation mismatch! Expected {expected_scene_uav} but got {img1_filename}")
                    print(f"    This annotation may be from a different scene or UAV")
                    # Try to find a better match
                    better_match = None
                    for key in annotation_map.keys():
                        if f"{group_scene}_UAV{uav_num}_frame_{group_frame}" in key:
                            better_match = annotation_map[key]
                            break
                    if better_match:
                        annotation = better_match
                        print(f"    Found better matching annotation")
                    else:
                        print(f"    No better match found, using current annotation")

            # Extract annotation info from all_samples.json
            annotation_info = extract_annotation_info(annotation)

            # Extract object information from annotation
            object_info = ""
            if annotation:
                if 'Object_type' in annotation and 'Object_count' in annotation:
                    object_type = annotation['Object_type']
                    object_count = annotation['Object_count']
                    object_info = f"Objects: {object_count} {object_type}"
                elif 'Object_type' in annotation:
                    object_type = annotation['Object_type']
                    object_info = f"Object type: {object_type}"
                elif 'Object_count' in annotation:
                    object_count = annotation['Object_count']
                    object_info = f"Object count: {object_count}"

            # Combine annotation information
            combined_info = f"{annotation_info}; {object_info}".strip('; ')

            # Print annotation information for debugging
            if annotation_info or json_info:
                print(f"    {uav_id} annotation info: {combined_info}")

            pair_quality_stats = {"EXCELLENT": 0, "GOOD": 0, "FAIR": 0, "POOR": 0, "ERROR": 0}
            uav_questions = []

            group_questions["questions_per_uav"][uav_id] = {
                "annotation_info": annotation_info,
                "object_info": object_info,
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
            collaboration_what_q, quality_what = try_generate_qa(generate_few_shot_collaboration_what_q, current_path, other_paths, uav_id, annotation, counter=counters[uav_id]["what"])
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
            save_to_json(all_results, "VQA_Sim3_CD.json")
            print(f"Saved intermediate results after processing {g_idx + 1} groups")

    # Compute and add quality statistics
    all_results['quality_statistics'] = quality_stats

    # Save final results
    if save_to_json(all_results, "VQA_Sim3_CD.json"):
        print(f"\nAll results saved to VQA_Sim3_CD.json")
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
