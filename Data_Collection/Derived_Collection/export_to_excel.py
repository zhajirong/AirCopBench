import json
import pandas as pd

# 读取JSON文件
with open('224_partial_correct.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

rows = []
for item in data['results']:
    question_id = item.get('question_id', '')
    question_type = item.get('question_type', '')
    question = item.get('question', '')
    uav1_path = item.get('uav1_path', '')
    uav2_path = item.get('uav2_path', '')
    options = item.get('options', {})
    options_str = '\n'.join([f"{k}: {v}" for k, v in options.items()])
    correct_answer = item.get('correct_answer', '')
    # 最后两栏留空
    rows.append([
        question_id,
        question_type,
        question,
        f"UAV1: {uav1_path}\nUAV2: {uav2_path}",
        options_str,
        correct_answer,
        '',  # 答案是否错误
        ''   # 错误类型
    ])

columns = [
    '题目ID',
    '题目类型',
    '题目信息',
    '图片信息（UAV1/UAV2）',
    '题目选项（A/B/C/D）',
    '正确答案',
    '答案是否错误',
    '错误类型'
]

df = pd.DataFrame(rows, columns=columns)
df.to_excel('224_partial_correct_export.xlsx', index=False)
print('导出完成：224_partial_correct_export.xlsx') 