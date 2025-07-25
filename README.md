# AirCopBench: Benchmarking Vision-Language Models on Multi-UAV Visual Question Answering

The benchmark is designed to evaluate whether vision-language models (VLMs) can process multi-UAV collaborative visual data for question answering, covering perception, reasoning, and decision-making in complex scenarios.

- **Paper**: _Coming soon_
- **Project**: _Coming soon_
- **Dataset**: _Coming soon_

---

## News

- ✅ All datasets and code released
- ✅ Unified question generation pipeline for 4 datasets & 16 tasks
- ✅ One-click integration script for interactive VQA generation

---

## Features

- **Multi-dataset, multi-task**: Supports Sim3, Sim5, Sim6, Real2 datasets, covering 4 major tasks (CD, OU, PA, SU) and 16 subtasks
- **Unified API**: All scripts use OpenAI GPT-4o API for question generation
- **Plug-and-play**: All input paths are relative to `datasets/`, no manual path editing required
- **Flexible**: Both integrated and single-script running supported

---

## Task & Subtask List

- **CD (Collaborative Decision)**
  - `when`: When to collaborate (temporal decision)
  - `what`: What to collaborate (content/goal selection)
  - `who`: Who to collaborate (agent selection)
  - `why`: Why to collaborate (reasoning for collaboration)
- **OU (Object Understanding)**
  - `or`: Object recognition (identify objects in images)
  - `oc`: Object counting (count number of objects)
  - `og`: Object grounding (locate objects in images)
  - `om`: Object matching (match objects across views)
- **PA (Perception Assessment)**
  - `quality`: Quality assessment (evaluate image/data quality)
  - `usability`: Usability assessment (assess usefulness for tasks)
  - `causal`: Causal assessment (reason about cause-effect)
- **SU (Scene Understanding)**
  - `scene_description`: Scene description (describe the scene)
  - `scene_comparison`: Scene comparison (compare different scenes)
  - `observing_posture`: Observing posture (analyze camera/UAV posture)

---

## Dataset Preparation

1. Download or prepare your datasets and place them in:
   ```
   VQA_Generation/datasets/
       VQA_Sim3/
       VQA_Sim5/
       VQA_Sim6/
       VQA_Real2/
   ```
2. Each dataset folder should contain images, annotations, and other files as required by the original scripts.

---

## Quick Start

### 1. Install dependencies

```bash
cd VQA_Generation
pip install -r requirements.txt
```

### 2. Configure OpenAI API Key

```bash
export OPENAI_API_KEY=your_api_key
```

### 3. Run the integrated script

```bash
python integrated_vqa.py
```
- Follow the prompts to select dataset, task, and subtask.
- Results will be saved as JSON files in the current directory.

---

## Running a Single Script

You can also run any original task script directly (e.g. Sim3_CD.py, Real2_OU.py):

```bash
cd VQA_Generation/VQA_Sim3
python Sim3_CD.py
```
- All scripts use relative paths based on `datasets/`. No need to edit paths.
- Results are saved as JSON files in the script directory.

---

## Directory Structure

```
AirCopBench/
├── VQA_Generation/
│   ├── integrated_vqa.py
│   ├── datasets/
│   ├── VQA_Sim3/
│   │   ├── Sim3_CD.py
│   │   ├── Sim3_OU.py
│   │   ├── Sim3_PA.py
│   │   └── Sim3_SU.py
│   ├── VQA_Sim5/
│   │   ├── Sim5_CD.py
│   │   ├── Sim5_OU.py
│   │   ├── Sim5_PA.py
│   │   └── Sim5_SU.py
│   ├── VQA_Sim6/
│   │   ├── Sim6_CD.py
│   │   ├── Sim6_OU.py
│   │   ├── Sim6_PA.py
│   │   └── Sim6_SU.py
│   └── VQA_Real2/
│       ├── Real2_CD.py
│       ├── Real2_OU.py
│       ├── Real2_PA.py
│       └── Real2_SU.py
├── requirements.txt
└── README.md
```

---

## FAQ

- **Q: Where do I put my datasets?**
  - A: Place all dataset folders under `VQA_Generation/datasets/`.
- **Q: How do I set the API key?**
  - A: Use `export OPENAI_API_KEY=your_api_key` before running scripts.
- **Q: Where are the results saved?**
  - A: In the current directory as JSON files.
- **Q: Can I run a single script?**
  - A: Yes, all scripts use unified relative paths.

---

## Acknowledgements

Thanks to all contributors and the open-source community for inspiration and support.

---

## Citation

If you use this project in your research, please cite:

```
@misc{aircopbench2024,
  title={AirCopBench: Benchmarking Vision-Language Models on Multi-UAV Visual Question Answering},
  author={             },
  year={2025},
  url={https://github.com/zhajirong/AirCopBench}
}
```
