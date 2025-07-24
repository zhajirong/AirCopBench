# AirCopBench 多无人机视觉问答（VQA）问题生成平台
# AirCopBench: Multi-UAV Visual Question Answering (VQA) Generation Platform

---

## 项目简介 | Project Introduction

本项目旨在为多无人机协同场景下的视觉问答（VQA）任务，批量生成高质量、多样化的问题数据。项目集成了 4 个数据集（Sim3、Sim5、Sim6、Real2）和 4 大任务（协同决策CD、目标理解OU、感知评估PA、场景理解SU），共 16 个子任务，支持一键选择数据集/任务/子任务自动生成问题。

This project aims to generate high-quality and diverse VQA (Visual Question Answering) questions for multi-UAV collaborative scenarios. It integrates 4 datasets (Sim3, Sim5, Sim6, Real2) and 4 main tasks (Collaborative Decision, Object Understanding, Perception Assessment, Scene Understanding), with a total of 16 subtasks. Users can generate questions by interactively selecting dataset/task/subtask.

- **支持多模态大模型（OpenAI GPT-4o）API自动调用**  
  Supports multi-modal LLM (OpenAI GPT-4o) API auto-calling
- **所有问题生成逻辑已集成到一个脚本，交互式选择，极易上手**  
  All logic integrated in one script, easy interactive selection
- **支持自定义数据集路径，统一管理，便于迁移和扩展**  
  Unified dataset management, easy migration/expansion

---

## 目录结构 | Directory Structure

```
AirCopBench/
│
├── VQA_Generation/
│   ├── integrated_vqa.py         # 集成主脚本 | Integrated main script
│   ├── datasets/                 # 统一数据集根目录 | Datasets root directory
│   ├── VQA_Sim3/                 # Sim3数据集各任务脚本 | Sim3 scripts
│   │   ├── Sim3_CD.py
│   │   ├── Sim3_OU.py
│   │   ├── Sim3_PA.py
│   │   └── Sim3_SU.py
│   ├── VQA_Sim5/                 # Sim5数据集各任务脚本 | Sim5 scripts
│   │   ├── Sim5_CD.py
│   │   ├── Sim5_OU.py
│   │   ├── Sim5_PA.py
│   │   └── Sim5_SU.py
│   ├── VQA_Sim6/                 # Sim6数据集各任务脚本 | Sim6 scripts
│   │   ├── Sim6_CD.py
│   │   ├── Sim6_OU.py
│   │   ├── Sim6_PA.py
│   │   └── Sim6_SU.py
│   └── VQA_Real2/                # Real2数据集各任务脚本 | Real2 scripts
│       ├── Real2_CD.py
│       ├── Real2_OU.py
│       ├── Real2_PA.py
│       └── Real2_SU.py
│
├── requirements.txt              # 一键安装依赖库 | All dependencies
└── README.md                     # 项目说明文档 | This file
```

---

## 依赖环境与一键安装 | Dependencies & Quick Install

推荐使用 Python 3.8 及以上版本。
Recommended: Python 3.8 or above.

**一键安装所有依赖库 | Install all dependencies:**

```bash
cd VQA_Generation
pip install -r requirements.txt
```

如遇到权限问题可加 `--user`：
If you encounter permission issues, add `--user`:
```bash
pip install --user -r requirements.txt
```

---

## OpenAI API Key 配置 | OpenAI API Key Setup

本项目依赖 OpenAI 多模态 API（如 gpt-4o）。你需要准备 OpenAI API Key。
This project requires an OpenAI API Key (for GPT-4o or similar models).

**推荐方式：设置环境变量 | Recommended: set as environment variable**

```bash
export OPENAI_API_KEY=your_api_key
```

如需临时设置（仅本次终端有效）：
For temporary use (current shell only):
```bash
OPENAI_API_KEY=your_api_key python integrated_vqa.py
```

**或直接在各脚本头部 `API_KEY = 'sk-xxxxxx'` 处填写（不推荐，安全性较低）。**
Or fill in the API key directly in the script (not recommended for security).

---

## 数据集准备 | Dataset Preparation

1. **数据集统一放置 | Place all datasets in one folder:**
   - 请将你的所有数据集文件夹（如 `VQA_Sim3`、`VQA_Sim5`、`VQA_Sim6`、`VQA_Real2`）移动到 `VQA_Generation/datasets/` 目录下。
   - Move all dataset folders (e.g. `VQA_Sim3`, `VQA_Sim5`, `VQA_Sim6`, `VQA_Real2`) into `VQA_Generation/datasets/`.
   - 目录结构示例 | Example:
     ```
     VQA_Generation/datasets/
         VQA_Sim3/
         VQA_Sim5/
         VQA_Sim6/
         VQA_Real2/
     ```

2. **数据集内容要求 | Dataset content:**
   - 每个数据集文件夹下应包含原有的图片、注释、标注等子目录和文件，保持与原脚本一致。
   - Each dataset folder should contain images, annotations, and other files as required by the original scripts.

---

## 快速开始 | Quick Start

### 1. 进入项目目录 | Enter project directory

```bash
cd VQA_Generation
```

### 2. 安装依赖 | Install dependencies

```bash
pip install -r requirements.txt
```

### 3. 配置 OpenAI API Key | Set OpenAI API Key

```bash
export OPENAI_API_KEY=your_api_key
```

### 4. 运行集成脚本 | Run the integrated script

```bash
python integrated_vqa.py
```

### 5. 交互式选择 | Interactive selection

- 脚本会依次提示你选择：
  - **数据集**（Sim3、Sim5、Sim6、Real2）| Dataset (Sim3, Sim5, Sim6, Real2)
  - **任务**（CD、OU、PA、SU）| Task (CD, OU, PA, SU)
  - **子任务**（如 when/what/quality/scene_description 等）| Subtask (e.g. when, what, quality, scene_description, etc.)
- 每一步都会列出可选项，输入后自动校验，确保选择有效。
- Each step lists available options and validates your input.

### 6. 自动生成问题 | Auto-generate questions

- 选择完成后，脚本会自动调用对应的原始问题生成函数，批量生成问题并保存为 JSON 文件。
- After selection, the script will call the corresponding function and save results as JSON.

---

## 如何单独运行某个脚本 | How to Run a Single Script

你也可以直接运行任意一个原始任务脚本（如 Sim3_CD.py、Real2_OU.py 等），无需通过集成脚本。
You can also run any original task script directly (e.g. Sim3_CD.py, Real2_OU.py), without using the integrated script.

**所有脚本的输入路径（图片、注释、标注等）均已统一为以 `datasets` 为根目录的相对路径。只需将数据集放入 `VQA_Generation/datasets/` 下，无需手动修改路径，直接运行脚本即可自动找到数据和标注。**
All scripts now use relative paths based on the `datasets` root. Just place your datasets under `VQA_Generation/datasets/`, and you can run any script directly without editing paths.

**操作步骤 | Steps:**

1. 进入对应目录 | Enter the corresponding directory:
   ```bash
   cd VQA_Generation/VQA_Sim3
   # 或 | or
   cd VQA_Generation/VQA_Real2
   ```
2. 运行脚本 | Run the script:
   ```bash
   python Sim3_CD.py
   # 或 | or
   python Real2_OU.py
   ```
3. 结果会自动保存在当前目录下的 JSON 文件中。
   The results will be saved as a JSON file in the current directory.

**注意事项 | Notes:**
- 需确保 API Key 已配置。
- Make sure your API Key is set.
- 数据集路径、注释路径等参数如需修改，请在脚本头部调整。
- If you need to change dataset/annotation paths, edit the script accordingly.

---

## 任务与子任务说明 | Task & Subtask List

- **CD（协同决策 | Collaborative Decision）**：when、what、who、why
- **OU（目标理解 | Object Understanding）**：or（识别 | recognition）、oc（计数 | counting）、og（定位 | grounding）、om（匹配 | matching）
- **PA（感知评估 | Perception Assessment）**：quality、usability、causal
- **SU（场景理解 | Scene Understanding）**：scene_description、scene_comparison、observing_posture

---

## 常见问题 | FAQ

1. **Q: 数据集目录没找到怎么办？**  
   A: 首次运行会自动创建 `datasets/` 文件夹，请将数据集文件夹移动进去。
   Q: What if the dataset directory is missing?  
   A: The script will create `datasets/` if missing. Please move your dataset folders there.

2. **Q: API Key如何安全管理？**  
   A: 推荐用环境变量 `OPENAI_API_KEY`，或在脚本中手动填写。
   Q: How to manage API Key securely?  
   A: Use the environment variable `OPENAI_API_KEY` or fill in the script (not recommended).

3. **Q: 依赖库缺失怎么办？**  
   A: 运行 `pip install -r requirements.txt` 安装依赖。
   Q: What if dependencies are missing?  
   A: Run `pip install -r requirements.txt` to install all dependencies.

4. **Q: 生成结果在哪里？**  
   A: 结果通常保存在当前目录下的 JSON 文件，或原脚本指定的路径。
   Q: Where are the results saved?  
   A: Usually in a JSON file in the current directory, or as specified in the script.

5. **Q: 如何批量运行所有子任务？**  
   A: 目前集成脚本支持单次选择一个子任务，若需批量可多次运行或自行扩展脚本。
   Q: How to run all subtasks in batch?  
   A: The integrated script supports one subtask per run. For batch, run multiple times or extend the script.

---

## 高级用法与扩展 | Advanced Usage & Extension

- 支持自定义数据集、任务、子任务映射，修改 `integrated_vqa.py` 顶部的 `DATASET_CONFIG` 即可。
- You can customize dataset/task/subtask mapping by editing `DATASET_CONFIG` in `integrated_vqa.py`.
- 可根据需要扩展批量运行、结果汇总、日志记录等功能。
- You may extend for batch processing, result aggregation, logging, etc.

---

## 联系与致谢 | Contact & Acknowledgement

如有问题或建议，欢迎联系项目维护者或提交 issue。
For questions or suggestions, please contact the maintainer or submit an issue.

---

**祝你使用愉快，欢迎为本项目贡献代码！**
**Enjoy using AirCopBench! Contributions are welcome!**
