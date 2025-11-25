import os
import sys
import importlib
import json


DATASETS_ROOT = os.path.join(os.path.dirname(__file__), 'datasets')
# eg：datasets/VQA_Sim3/、datasets/VQA_Sim5/...
# =====================

DATASET_CONFIG = {
    'Sim3': {
        'dir': 'VQA_Sim3',
        'uav_num': 3,
        'tasks': {
            'CD': ['when', 'what', 'who', 'why'],
            'OU': ['or', 'oc', 'og', 'om'],
            'PA': ['quality', 'usability', 'causal'],
            'SU': ['scene_description', 'scene_comparison', 'observing_posture']
        }
    },
    'Sim5': {
        'dir': 'VQA_Sim5',
        'uav_num': 5,
        'tasks': {
            'CD': ['when', 'what', 'who', 'why'],
            'OU': ['or', 'oc', 'og', 'om'],
            'PA': ['quality', 'usability', 'causal'],
            'SU': ['scene_description', 'scene_comparison', 'observing_posture']
        }
    },
    'Sim6': {
        'dir': 'VQA_Sim6',
        'uav_num': 6,
        'tasks': {
            'CD': ['when', 'what', 'who', 'why'],
            'OU': ['or', 'oc', 'og', 'om'],
            'PA': ['quality', 'usability', 'causal'],
            'SU': ['scene_description', 'scene_comparison', 'observing_posture']
        }
    },
    'Real2': {
        'dir': 'VQA_Real2',
        'uav_num': 2,
        'tasks': {
            'CD': ['when', 'what', 'who', 'why'],
            'OU': ['or', 'oc', 'og', 'om'],
            'PA': ['quality', 'usability', 'causal'],
            'SU': ['scene_description', 'scene_comparison', 'observing_posture']
        }
    }
}

TASK_FILE_MAP = {
    'CD': 'CD',
    'OU': 'OU',
    'PA': 'PA',
    'SU': 'SU'
}

SUBTASK_FUNC_MAP = {
    # CD
    'when': 'generate_rule_based_collaboration_when_q',
    'what': 'generate_few_shot_collaboration_what_q',
    'who': 'generate_rule_based_collaboration_who_q_with_annotation',
    'why': 'generate_hybrid_collaboration_why_q',
    # OU
    'or': 'generate_few_shot_object_recognition_q',
    'oc': 'generate_rule_based_counting_q',
    'og': 'generate_few_shot_object_grounding_q',
    'om': 'generate_few_shot_object_matching_q',
    # PA
    'quality': 'generate_rule_based_quality_q',
    'usability': 'generate_rule_based_usability_q',
    'causal': 'generate_few_shot_causal_assessment_q',
    # SU
    'scene_description': 'generate_few_shot_scene_description_q',
    'scene_comparison': 'generate_few_shot_scene_comparison_q',
    'observing_posture': 'generate_few_shot_observing_posture_q',
}


def user_select():
    """
    Prompts the user to select a dataset, task, and subtask.
    """
    # Dataset selection
    while True:
        print('\nAvailable Datasets:')
        for ds in DATASET_CONFIG:
            print(f'  - {ds}')
        dataset = input('Please enter the dataset name: ').strip()
        if dataset in DATASET_CONFIG:
            break
        print('Invalid dataset. Please try again.')
        
    # Task selection
    while True:
        print(f'\nAvailable Tasks for {dataset}:')
        for t in DATASET_CONFIG[dataset]['tasks']:
            print(f'  - {t}')
        task = input('Please enter the task name: ').strip()
        if task in DATASET_CONFIG[dataset]['tasks']:
            break
        print('Invalid task. Please try again.')
        
    # Subtask selection
    while True:
        print(f'\nAvailable Subtasks for {dataset} - {task}:')
        for st in DATASET_CONFIG[dataset]['tasks'][task]:
            print(f'  - {st}')
        subtask = input('Please enter the subtask name: ').strip()
        if subtask in DATASET_CONFIG[dataset]['tasks'][task]:
            break
        print('Invalid subtask. Please try again.')
        
    return dataset, task, subtask


def main():
    """
    Main function to ensure dataset root directory exists,
    get user selection, dynamically import the corresponding module,
    and attempt to run the main process or the selected function.
    """
    # Check and create the dataset root directory
    if not os.path.exists(DATASETS_ROOT):
        os.makedirs(DATASETS_ROOT)
        print(f"Created dataset root directory: {DATASETS_ROOT}")
        print("Please place your dataset folders (e.g., VQA_Sim3, VQA_Sim5, etc.) into this directory!")
        sys.exit(0)

    dataset, task, subtask = user_select()
    dataset_dir = os.path.join(DATASETS_ROOT, DATASET_CONFIG[dataset]['dir'])
    # uav_num = DATASET_CONFIG[dataset]['uav_num'] # This variable is not used in the original logic flow
    
    # Construct module information
    script_file = os.path.join(dataset_dir, f"{dataset}_{TASK_FILE_MAP[task]}.py")
    module_name = f"datasets.{DATASET_CONFIG[dataset]['dir']}.{dataset}_{TASK_FILE_MAP[task]}"
    func_name = SUBTASK_FUNC_MAP[subtask]

    # Dynamically import the corresponding script module
    try:
        module = importlib.import_module(module_name)
    except Exception as e:
        print(f"Failed to import module: {module_name}")
        print(f"Error: {e}")
        sys.exit(1)

    # Check if the function exists
    if not hasattr(module, func_name):
        print(f"Function {func_name} not found in module {module_name}")
        sys.exit(1)

    # Run the original 'main' process if available
    if hasattr(module, 'main'):
        print(f"Running {module_name}.main() directly. This should automatically complete all question generation (including all subtasks).")
        module.main()
        return
    else:
        print(f"No 'main' function found. Only single question generation function {func_name} can be called.")
        print("Please manually add the calling logic based on the specific function parameters.")
        sys.exit(1)
        
if __name__ == "__main__":
    main() 
