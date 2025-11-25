# scene_understanding_enhanced.py (modified)
import openai  # Import OpenAI library
from openai import AzureOpenAI
import base64
from PIL import Image
import io
import os
import json
import glob
import random
import time
from difflib import SequenceMatcher

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

    print(
        f"Found {len(image_pairs)} matching image pairs: {', '.join([pair['sequence_frame'] for pair in image_pairs])}")
    return image_pairs


def check_option_diversity(options):
    """Check if options are sufficiently different"""
    for i, opt1 in enumerate(options.values()):
        for j, opt2 in enumerate(options.values()):
            if i < j:
                similarity = SequenceMatcher(None, opt1, opt2).ratio()
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
    """Generate scene description questions with few-shot examples"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' understanding of scene analysis.

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


def generate_few_shot_scene_comparison_q(img_path1, img_path2, use_mock=False, mock_desc1=None, mock_desc2=None,
                                         index=0, counter=1):
    """Generate scene comparison questions with few-shot examples"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' ability to compare and integrate information from multiple perspectives.

CRITICAL RULES:
1. ALWAYS respond in English only - never use any other language
2. Follow a structured thinking process: analyze both images → identify differences/similarities → formulate comparison question → create options → verify correctness
3. Questions must be based on actual visual content or provided descriptions
4. Each question should have exactly 4 options (A, B, C, D) with only one correct answer
5. Options should be plausible, distinct in meaning, and avoid minor rephrasing
6. Output must be valid JSON format only

THINKING PROCESS:
1. First, describe the key differences and similarities in objects, layout, or perspective between both images or descriptions
2. Identify the focus based on generation index
3. Formulate a clear question about comparing the two perspectives
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

    user_prompt = f"""First, provide a brief description (50-100 words) of the key differences and similarities in objects, layout, or perspective between the two {'images' if not use_mock else 'scene descriptions'} from UAV1 and UAV2.
Then, create a multiple-choice question about scene comparison based on this description.

REQUIREMENTS:
- Question should focus on differences between the two perspectives
- This is generation {index +1}/2. Focus on: {comparison_types[index]}
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

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"image": f"data:image/jpeg;base64,{encode_image(img_path1)}"} if not use_mock else {
                    "text": mock_desc1},
                {"image": f"data:image/jpeg;base64,{encode_image(img_path2)}"} if not use_mock else {
                    "text": mock_desc2},
                {"text": user_prompt}
            ]
        }
    ]

    result = call_chatgpt_api(messages)
    if "error" not in result:
        result["source"] = "Model-Based (Few-Shot)"
        result["question_id"] = f"MDMT_SC_{counter}"
    return result


def generate_few_shot_observing_posture_q(img_path1, img_path2, use_mock=False, mock_desc1=None, mock_desc2=None,
                                          index=0, counter=1):
    """Generate observing posture questions with few-shot examples"""
    system_prompt = """You are an expert teacher of the "Multi-view Perception" course. Your role is to create high-quality multiple-choice questions that test students' understanding of observer positioning and perspective analysis.

CRITICAL RULES:
1. ALWAYS respond in English only - never use any other language
2. Follow a structured thinking process: analyze both images → identify observer positions → formulate posture question → create options → verify correctness
3. Questions must be based on actual visual content or provided descriptions
4. Each question should have exactly 4 options (A, B, C, D) with only one correct answer
5. Options should be plausible, distinct in meaning, and avoid minor rephrasing
6. Output must be valid JSON format only

THINKING PROCESS:
1. First, describe the observer positions and perspective differences between both images or descriptions
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

    user_prompt = f"""First, provide a brief description (50-100 words) of the observer positions and perspective differences between the two {'images' if not use_mock else 'scene descriptions'} from UAV1 and UAV2.
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

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"image": f"data:image/jpeg;base64,{encode_image(img_path1)}"} if not use_mock else {
                    "text": mock_desc1},
                {"image": f"data:image/jpeg;base64,{encode_image(img_path2)}"} if not use_mock else {
                    "text": mock_desc2},
                {"text": user_prompt}
            ]
        }
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
        # base_dir = "Samples_test"
        base_dir = "All_samples"
        uav1_dir = os.path.join(base_dir, "UAV1")
        uav2_dir = os.path.join(base_dir, "UAV2")

        # Get image pairs
        image_pairs = get_image_pairs(uav1_dir, uav2_dir)

        if not image_pairs:
            print("Error: No matching image pairs found. Falling back to mock test.")
            use_mock = True
    else:
        image_pairs = [{
            'sequence_frame': "mock_traffic_scene",
            'uav1_path': None,
            'uav2_path': None,
            'uav1_filename': "mock-UAV1.jpg",
            'uav2_filename': "mock-UAV2.jpg"
        }]

    quality_stats = {"EXCELLENT": 0, "GOOD": 0, "FAIR": 0, "POOR": 0, "ERROR": 0}
    pair_results = []

    # Initialize counters for sequential numbering
    sd_uav1_counter = 1
    sd_uav2_counter = 1
    sc_counter = 1
    op_counter = 1

    for pair_idx, pair in enumerate(image_pairs):
        print(f"\n=== Processing Image Pair {pair_idx + 1}: {pair['sequence_frame']} ===")
        if not use_mock:
            test_uav1_path = pair['uav1_path']
            test_uav2_path = pair['uav2_path']
            sequence_frame = pair['sequence_frame']

            print(f"Testing with images:")
            print(f"UAV1: {os.path.basename(test_uav1_path)}")
            print(f"UAV2: {os.path.basename(test_uav2_path)}")
        else:
            test_uav1_path = None
            test_uav2_path = None
            sequence_frame = pair['sequence_frame']

        pair_quality_stats = {"EXCELLENT": 0, "GOOD": 0, "FAIR": 0, "POOR": 0, "ERROR": 0}
        pair_all_results = []

        # Test 1.1 Scene Description
        print("\n=== Testing 1.1 Scene Description ===")
        question_types = [
            "spatial arrangements (e.g., layout of roads or objects)",
            "environmental features (e.g., urban vs. rural setting)"
        ]
        
        # 随机选择一种问题类型
        selected_type_index = random.randint(0, len(question_types) - 1)
        selected_type = question_types[selected_type_index]
        print(f"\n--- Scene Description Test (Randomly selected: {selected_type}) ---")

        # UAV1 Scene Description
        result1, quality1 = try_generate_qa(
            generate_few_shot_scene_description_q,
            test_uav1_path, "UAV1", use_mock=use_mock, mock_desc=mock_uav1_desc, index=selected_type_index, counter=str(sd_uav1_counter)
        )
        if "error" not in result1:
            print(f"UAV1 Question: {result1.get('question', 'N/A')}")
            print(f"Options: {result1.get('options', {})}")
            print(f"Correct Answer: {result1.get('correct_answer', 'N/A')}")
            print(f"Image Description: {result1.get('image_description', 'N/A')}")
            print(f"Quality: {quality1['quality']} (Score: {quality1['score']})")
            if quality1['issues']:
                print(f"Issues: {quality1['issues']}")
            pair_all_results.append(result1)
            pair_quality_stats[quality1['quality']] += 1
        else:
            print(f"UAV1 Error: {result1.get('error')}")
            pair_quality_stats["ERROR"] += 1
        sd_uav1_counter += 1

        # UAV2 Scene Description
        result2, quality2 = try_generate_qa(
            generate_few_shot_scene_description_q,
            test_uav2_path, "UAV2", use_mock=use_mock, mock_desc=mock_uav2_desc, index=selected_type_index, counter=str(sd_uav2_counter)
        )
        if "error" not in result2:
            print(f"UAV2 Question: {result2.get('question', 'N/A')}")
            print(f"Options: {result2.get('options', {})}")
            print(f"Correct Answer: {result2.get('correct_answer', 'N/A')}")
            print(f"Image Description: {result2.get('image_description', 'N/A')}")
            print(f"Quality: {quality2['quality']} (Score: {quality2['score']})")
            if quality2['issues']:
                print(f"Issues: {quality2['issues']}")
            pair_all_results.append(result2)
            pair_quality_stats[quality2['quality']] += 1
        else:
            print(f"UAV2 Error: {result2.get('error')}")
            pair_quality_stats["ERROR"] += 1
        sd_uav2_counter += 1

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
        
        result, quality = try_generate_qa(
            generate_few_shot_scene_comparison_q,
            test_uav1_path, test_uav2_path, use_mock=use_mock,
            mock_desc1=mock_uav1_desc, mock_desc2=mock_uav2_desc, index=selected_comparison_index, counter=str(sc_counter)
        )
        if "error" not in result:
            print(f"Comparison Question: {result.get('question', 'N/A')}")
            print(f"Options: {result.get('options', {})}")
            print(f"Correct Answer: {result.get('correct_answer', 'N/A')}")
            print(f"Image Description: {result.get('image_description', 'N/A')}")
            print(f"Quality: {quality['quality']} (Score: {quality['score']})")
            if quality['issues']:
                print(f"Issues: {quality['issues']}")
            pair_all_results.append(result)
            pair_quality_stats[quality['quality']] += 1
        else:
            print(f"Comparison Error: {result.get('error')}")
            pair_quality_stats["ERROR"] += 1
        sc_counter += 1

        # Test 1.3 Observing Posture
        print("\n=== Testing 1.3 Observing Posture ===")
        posture_types = [
            "clarity of specific object features (e.g., movement or orientation)",
            "impact of viewing angle on perception (e.g., distance or layout)"
        ]
        for i in range(0,1):
            print(f"\n--- Observing Posture Test {i + 1} ({posture_types[i]}) ---")
            result, quality = try_generate_qa(
                generate_few_shot_observing_posture_q,
                test_uav1_path, test_uav2_path, use_mock=use_mock,
                mock_desc1=mock_uav1_desc, mock_desc2=mock_uav2_desc, index=i, counter=str(op_counter)
            )
            if "error" not in result:
                print(f"Posture Question {i + 1}: {result.get('question', 'N/A')}")
                print(f"Options: {result.get('options', {})}")
                print(f"Correct Answer: {result.get('correct_answer', 'N/A')}")
                print(f"Image Description: {result.get('image_description', 'N/A')}")
                print(f"Quality: {quality['quality']} (Score: {quality['score']})")
                if quality['issues']:
                    print(f"Issues: {quality['issues']}")
                pair_all_results.append(result)
                pair_quality_stats[quality['quality']] += 1
            else:
                print(f"Posture Error: {result.get('error')}")
                pair_quality_stats["ERROR"] += 1
            op_counter += 1

        # Aggregate for this pair
        for k, v in pair_quality_stats.items():
            quality_stats[k] += v

        pair_results.append({
            "sequence_frame": sequence_frame,
            "uav1_path": test_uav1_path if not use_mock else mock_uav1_desc,
            "uav2_path": test_uav2_path if not use_mock else mock_uav2_desc,
            "uav1_filename": pair['uav1_filename'],
            "uav2_filename": pair['uav2_filename'],
            "questions": pair_all_results
        })

    # Save results
    output_data = {
        "dataset": "Scene_Understanding" + (" (Mock)" if use_mock else ""),
        "total_pairs": len(image_pairs),
        "results": pair_results,
        "quality_statistics": quality_stats
    }

    output_file = "VQA_MDMT_SU.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n=== Overall Summary ===")
    print(f"Total questions generated: {sum(len(pair['questions']) for pair in pair_results)}")
    print(f"Overall Quality distribution: {quality_stats}")
    print(f"Results saved to: {output_file}")

    # Detailed quality assessment for all
    print(f"\n=== Detailed Quality Assessment for All Pairs ===")
    for pair_idx, pair_result in enumerate(pair_results):
        print(f"\n--- Pair {pair_idx + 1}: {pair_result['sequence_frame']} ---")
        for i, result in enumerate(pair_result["questions"]):
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
