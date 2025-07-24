# AirCopBench 多无人机视觉问答（VQA）问题生成平台

## 项目简介

本项目旨在为多无人机协同场景下的视觉问答（VQA）任务，批量生成高质量、多样化的问题数据。项目集成了 4 个数据集（Sim3、Sim5、Sim6、Real2）和 4 大任务（协同决策CD、目标理解OU、感知评估PA、场景理解SU），共 16 个子任务，支持一键选择数据集/任务/子任务自动生成问题。

- **支持多模态大模型（OpenAI GPT-4o）API自动调用**
- **所有问题生成逻辑已集成到一个脚本，交互式选择，极易上手**
- **支持自定义数据集路径，统一管理，便于迁移和扩展**

---

## 目录结构

```
AirCopBench/
│
├── VQA_Generation/
│   ├── integrated_vqa.py         # 集成主脚本（推荐入口）
│   ├── datasets/                 # 统一数据集根目录（需手动放置数据集文件夹）
│   ├── VQA_Sim3/                 # Sim3数据集各任务脚本
│   │   ├── Sim3_CD.py
│   │   ├── Sim3_OU.py
│   │   ├── Sim3_PA.py
│   │   └── Sim3_SU.py
│   ├── VQA_Sim5/                 # Sim5数据集各任务脚本
│   │   ├── Sim5_CD.py
│   │   ├── Sim5_OU.py
│   │   ├── Sim5_PA.py
│   │   └── Sim5_SU.py
│   ├── VQA_Sim6/                 # Sim6数据集各任务脚本
│   │   ├── Sim6_CD.py
│   │   ├── Sim6_OU.py
│   │   ├── Sim6_PA.py
│   │   └── Sim6_SU.py
│   └── VQA_Real2/                # Real2数据集各任务脚本
│       ├── Real2_CD.py
│       ├── Real2_OU.py
│       ├── Real2_PA.py
│       └── Real2_SU.py
│
├── requirements.txt              # 一键安装依赖库
└── README.md                     # 项目说明文档（本文件）
```

---

## 依赖环境与一键安装

推荐使用 Python 3.8 及以上版本。

**一键安装所有依赖库：**

```bash
cd VQA_Generation
pip install -r requirements.txt
```

如遇到权限问题可加 `--user`，如：
```bash
pip install --user -r requirements.txt
```

---

## OpenAI API Key 配置

本项目依赖 OpenAI 多模态 API（如 gpt-4o）。你需要准备 OpenAI API Key。

**推荐方式：设置环境变量**

```bash
export OPENAI_API_KEY=你的API密钥
```

如需临时设置（仅本次终端有效）：
```bash
OPENAI_API_KEY=你的API密钥 python integrated_vqa.py
```

**或直接在各脚本头部 `API_KEY = 'sk-xxxxxx'` 处填写（不推荐，安全性较低）。**

---

## 数据集准备

1. **数据集统一放置：**
   - 请将你的所有数据集文件夹（如 `VQA_Sim3`、`VQA_Sim5`、`VQA_Sim6`、`VQA_Real2`）移动到 `VQA_Generation/datasets/` 目录下。
   - 目录结构示例：
     ```
     VQA_Generation/datasets/
         VQA_Sim3/
         VQA_Sim5/
         VQA_Sim6/
         VQA_Real2/
     ```

2. **数据集内容要求：**
   - 每个数据集文件夹下应包含原有的图片、注释、标注等子目录和文件，保持与原脚本一致。
   - 例如：`VQA_Sim3/Sim3_CD.py` 期望的数据结构、图片命名、注释文件等需与原始一致。

---

## 快速开始

### 1. 进入项目目录

```bash
cd VQA_Generation
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 OpenAI API Key

```bash
export OPENAI_API_KEY=你的API密钥
```

### 4. 运行集成脚本

```bash
python integrated_vqa.py
```

### 5. 交互式选择

- 脚本会依次提示你选择：
  - **数据集**（Sim3、Sim5、Sim6、Real2）
  - **任务**（CD 协同决策、OU 目标理解、PA 感知评估、SU 场景理解）
  - **子任务**（如 when/what/quality/scene_description 等）
- 每一步都会列出可选项，输入后自动校验，确保选择有效。

### 6. 自动生成问题

- 选择完成后，脚本会自动调用对应的原始问题生成函数，批量生成问题并保存为 JSON 文件。
- 结果输出路径与原脚本一致，或在终端有详细提示。

---

## 常用命令速查

- **一键安装依赖：**
  ```bash
  pip install -r requirements.txt
  ```
- **设置 API Key（推荐）：**
  ```bash
  export OPENAI_API_KEY=你的API密钥
  ```
- **运行主脚本：**
  ```bash
  python integrated_vqa.py
  ```
- **临时设置 API Key 并运行：**
  ```bash
  OPENAI_API_KEY=你的API密钥 python integrated_vqa.py
  ```
- **查看数据集目录结构：**
  ```bash
  tree datasets/
  ```

---

## 任务与子任务说明

- **CD（协同决策）**：when、what、who、why
- **OU（目标理解）**：or（识别）、oc（计数）、og（定位）、om（匹配）
- **PA（感知评估）**：quality、usability、causal
- **SU（场景理解）**：scene_description、scene_comparison、observing_posture

---

## 常见问题

1. **Q: 数据集目录没找到怎么办？**  
   A: 首次运行会自动创建 `datasets/` 文件夹，请将数据集文件夹移动进去。

2. **Q: API Key如何安全管理？**  
   A: 推荐用环境变量 `OPENAI_API_KEY`，或在脚本中手动填写。

3. **Q: 依赖库缺失怎么办？**  
   A: 运行 `pip install -r requirements.txt` 安装依赖。

4. **Q: 生成结果在哪里？**  
   A: 结果通常保存在当前目录下的 JSON 文件，或原脚本指定的路径。

5. **Q: 如何批量运行所有子任务？**  
   A: 目前集成脚本支持单次选择一个子任务，若需批量可多次运行或自行扩展脚本。

---

## 高级用法与扩展

- 支持自定义数据集、任务、子任务映射，修改 `integrated_vqa.py` 顶部的 `DATASET_CONFIG` 即可。
- 可根据需要扩展批量运行、结果汇总、日志记录等功能。

---

## 联系与致谢

如有问题或建议，欢迎联系项目维护者或提交 issue。

---

**祝你使用愉快，欢迎为本项目贡献代码！**
