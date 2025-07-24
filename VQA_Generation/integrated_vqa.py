import os
import sys
import importlib
import json

# =====================
# 数据集统一根目录配置
# =====================
# 请将所有数据集（如Sim3、Sim5、Sim6、Real2）文件夹放在 datasets/ 目录下
DATASETS_ROOT = os.path.join(os.path.dirname(__file__), 'datasets')
# 例如：datasets/VQA_Sim3/、datasets/VQA_Sim5/...
# =====================

# 数据集、任务、子任务映射
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

# 任务到脚本文件名映射
TASK_FILE_MAP = {
    'CD': 'CD',
    'OU': 'OU',
    'PA': 'PA',
    'SU': 'SU'
}

# 子任务到函数名映射（部分函数名有差异，需适配）
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
    # 数据集选择
    while True:
        print('\n可选数据集:')
        for ds in DATASET_CONFIG:
            print(f'  - {ds}')
        dataset = input('请输入数据集名称: ').strip()
        if dataset in DATASET_CONFIG:
            break
        print('无效数据集，请重新输入。')
    # 任务选择
    while True:
        print(f'\n可选任务:')
        for t in DATASET_CONFIG[dataset]['tasks']:
            print(f'  - {t}')
        task = input('请输入任务名: ').strip()
        if task in DATASET_CONFIG[dataset]['tasks']:
            break
        print('无效任务，请重新输入。')
    # 子任务选择
    while True:
        print(f'\n可选子任务:')
        for st in DATASET_CONFIG[dataset]['tasks'][task]:
            print(f'  - {st}')
        subtask = input('请输入子任务名: ').strip()
        if subtask in DATASET_CONFIG[dataset]['tasks'][task]:
            break
        print('无效子任务，请重新输入。')
    return dataset, task, subtask


def main():
    # 检查datasets文件夹是否存在，不存在则创建
    if not os.path.exists(DATASETS_ROOT):
        os.makedirs(DATASETS_ROOT)
        print(f"已创建数据集根目录: {DATASETS_ROOT}")
        print("请将各数据集文件夹（如VQA_Sim3、VQA_Sim5等）放入该目录下！")
        sys.exit(0)

    dataset, task, subtask = user_select()
    dataset_dir = os.path.join(DATASETS_ROOT, DATASET_CONFIG[dataset]['dir'])
    uav_num = DATASET_CONFIG[dataset]['uav_num']
    script_file = os.path.join(dataset_dir, f"{dataset}_{TASK_FILE_MAP[task]}.py")
    module_name = f"datasets.{DATASET_CONFIG[dataset]['dir']}.{dataset}_{TASK_FILE_MAP[task]}"
    func_name = SUBTASK_FUNC_MAP[subtask]

    # 动态导入对应脚本模块
    try:
        module = importlib.import_module(module_name)
    except Exception as e:
        print(f"导入模块失败: {module_name}", e)
        sys.exit(1)

    # 检查函数是否存在
    if not hasattr(module, func_name):
        print(f"{module_name} 中未找到函数 {func_name}")
        sys.exit(1)

    # 运行原main流程，自动适配参数
    if hasattr(module, 'main'):
        print(f"直接运行 {module_name}.main()，将自动完成全部问题生成（包含所有子任务）")
        module.main()
        return
    else:
        print(f"未找到main函数，仅可调用单个问题生成函数 {func_name}")
        print("请根据具体函数参数手动补充调用逻辑")
        sys.exit(1)

if __name__ == "__main__":
    main() 