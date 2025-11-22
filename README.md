# AirCopBench: A Benchmark for Multi-drone Collaborative Embodied Perception and Reasoning

The benchmark is designed to evaluate whether vision-language models (VLMs) can process multi-UAV collaborative visual data for question answering, covering perception, reasoning, and decision-making in complex scenarios.

- **Paper**: https://arxiv.org/pdf/2511.11025
- **Project**: https://embodiedcity.github.io/AirCopBench/
- **Dataset**: https://drive.google.com/drive/folders/1MeCM2_MA5A-1XsIgvSZZacCWk-sistgp

---

## News
- 🎉 Accepted by AAAI 2026!
- ✅ All datasets and code released
- ✅ Unified question generation pipeline for 4 datasets & 16 tasks
- ✅ One-click integration script for interactive VQA generation
- ✅ **NEW**: Data collection pipeline for EmbodiedCity simulator
- ✅ **NEW**: Data annotation tools and sample files
- ✅ **NEW**: Image post-processing utilities

---

## Features

- **Multi-dataset, multi-task**: Supports Sim3, Sim5, Sim6, Real2 datasets, covering 4 major tasks (CD, OU, PA, SU) and 16 subtasks
- **Unified API**: All scripts use OpenAI GPT-4o API for question generation
- **Plug-and-play**: All input paths are relative to `datasets/`, no manual path editing required
- **Flexible**: Both integrated and single-script running supported
- **Complete Pipeline**: From data collection to annotation to VQA generation

---

## Task & Subtask List

![image](figures/task.pdf)

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

## Project Structure

```
AirCopBench/
├── Data_Collection/                    # Data collection module
│   ├── Derived_Collection/             # Image post-processing tools
│   │   ├── apply_noise_to_image.py    # Image noise addition
│   │   └── export_to_excel.py         # JSON to Excel export
│   └── Simulator_Collection/           # Simulator data collection
│       └── EmbodiedCity_Collection/    # EmbodiedCity simulator collection
│           ├── main.py                 # Main collection script
│           ├── config.py               # Configuration parsing
│           ├── uav_manager.py          # UAV management
│           ├── motion.py               # Motion pattern definitions
│           ├── recorder.py             # Video recording
│           ├── manual_trajectory_recorder.py  # Manual trajectory recording
│           ├── print_point.py          # Point viewing tool
│           ├── scenarios/              # Scenario configuration files
│           ├── requirements.txt        # Dependencies
│           └── utils.py                # Utility functions
├── Data_Annotation/                    # Data annotation module
│   ├── Real2_Sample.json              # Real2 dataset annotation example
│   ├── Sim3_Sample.json               # Sim3 dataset annotation example
│   ├── Sim5_Sample.json               # Sim5 dataset annotation example
│   └── Sim6_Sample.json               # Sim6 dataset annotation example
├── VQA_Generation/                     # VQA generation module
│   ├── integrated_vqa.py              # Integrated VQA generation script
│   ├── VQA_Sim3/                      # Sim3 VQA generation
│   ├── VQA_Sim5/                      # Sim5 VQA generation
│   ├── VQA_Sim6/                      # Sim6 VQA generation
│   └── VQA_Real2/                     # Real2 VQA generation
├── AirCopBench_evaluation/            # Evaluation Code for AirCopBench
│   ├── evaluation.py                  # Evaluation code example using gpt-4o
├── AirCopBench_sft/                   # Configuration of SFT on AirCopBench
│   ├── llava13b_vqa_sft.yaml          # Configuration for fine-tuning llava-next-13b
│   ├── qwen2_5vl_lora_sft.yaml        # Configuration for fine-tuning qwen-2.5-vl/qwen-2-vl
├── requirements.txt                    # Main project dependencies
└── README.md                          # Project documentation
```

---

## VQA Generation

### Dataset Preparation

#### Dataset Organization Structure

1. **Dataset Placement**:
   ```
   VQA_Generation/datasets/
       VQA_Sim3/
       VQA_Sim5/
       VQA_Sim6/
       VQA_Real2/
   ```

2. **Each dataset folder should contain**:
   - `images/`: Image files directory
   - `annotations/`: Annotation files directory
   - `metadata/`: Metadata files (optional)

#### Data Preprocessing Requirements

1. **Image Format**: Supports JPG, PNG formats, recommended to unify as JPG format
2. **Image Size**: Recommended unified resolution, such as 1920x1080
3. **Annotation Format**: JSON format exported from Label Studio
4. **File Naming**: Recommended unified naming convention, such as `{scene_id}_{uav_id}_{frame_id}.jpg`

#### Data Quality Check

Before running VQA generation, it is recommended to perform the following checks:

1. **Image Integrity**: Ensure all image files can be opened normally
2. **Annotation Integrity**: Verify annotation file format is correct
3. **Data Consistency**: Check one-to-one correspondence between images and annotation files
4. **Path Correctness**: Confirm all relative paths are set correctly

### Quick Start

#### 1. Install dependencies

```bash
# Main project dependencies
pip install -r requirements.txt

# Simulator collection dependencies
cd Data_Collection/Simulator_Collection/EmbodiedCity_Collection
pip install -r requirements.txt
```

#### 2. Configure OpenAI API Key

```bash
export OPENAI_API_KEY=your_api_key
```

#### 3. Run the integrated script

```bash
python integrated_vqa.py
```
- Follow the prompts to select dataset, task, and subtask.
- Results will be saved as JSON files in the current directory.

#### 4. Advanced Configuration Options

##### API Configuration Optimization

```python
# Set detailed API configuration in scripts
import openai

openai.api_key = "your_api_key"
openai.api_base = "https://api.openai.com/v1"  # Optional: custom API endpoint
openai.timeout = 60  # Set timeout (seconds)
```

##### Batch Processing Configuration

```bash
# Batch process multiple datasets
python integrated_vqa.py --batch --datasets Sim3,Sim5,Sim6,Real2 --tasks CD,OU,PA,SU
```

##### Output Format Configuration

```bash
# Specify output format and path
python integrated_vqa.py --output-format json --output-dir ./results
```

#### 5. Performance Optimization Recommendations

##### Concurrent Processing
- Use multi-process processing for large amounts of data
- Reasonably set API request frequency to avoid triggering limits
- Utilize caching mechanisms to reduce duplicate requests

##### Memory Management
- Process large datasets in batches
- Release unnecessary data in time
- Use generators to process streaming data

##### Error Handling
- Implement automatic retry mechanisms
- Record detailed error logs
- Support checkpoint resume functionality

### Running a Single Script

You can also run any original task script directly (e.g. Sim3_CD.py, Real2_OU.py):

```bash
cd VQA_Generation/VQA_Sim3
python Sim3_CD.py
```
- All scripts use relative paths based on `datasets/`. No need to edit paths.
- Results are saved as JSON files in the script directory.

---

## Data Collection

### 1. EmbodiedCity Simulator Collection

#### Environment Setup

1. **Start UE Project File**
   ```
   G:\UEProjects\BeiJingSimulation_VR
   Double-click TrafficSimulation.uproject
   ```

2. **Start Project Code**
   ```
   D:\PycharmProject\MultiTaskEmbodiedCity
   Open this project folder with PyCharm
   ```

3. **settings.json Configuration**
   - Path: `C:\Users\11\Documents\AirSim`
   - Ensure configuration for UAV0, UAV1, UAV2, UAV3

#### Install Dependencies

```bash
cd Data_Collection/Simulator_Collection/EmbodiedCity_Collection
pip install -r requirements.txt
```

#### Main Features

1. **RGB Video and Point Cloud Collection**
   - Collect point cloud and RGB video simultaneously:
     ```bash
     python main.py --scene ./scenarios/scene_007.yaml --capture-pointcloud
     ```
   - Collect RGB video only:
     ```bash
     python main.py --scene ./scenarios/scene_007.yaml
     ```

2. **Manual Point or Trajectory Recording**
   - View points:
     ```bash
     python print_point.py
     ```
     - Use W/S for up/down, ↑↓←→ for forward/backward/left/right, F to record points
   - Record trajectory:
     ```bash
     python manual_trajectory_recorder.py --scene ./scenarios/scene_for_recording.yaml
     ```

#### Configuration Modifications

1. **Modify YAML Scene Files (scenarios/)**
   - Change UAV count: modify `observer_drones` and `moving_drones` lists
   - Change UAV positions: modify `pose` and `start` coordinates
   - Change camera settings: modify `camera` parameters
   - Change motion trajectories: modify `motion` parameters

2. **Motion Types**
   - Circular motion (`type: circle`): set radius, height, angular speed, rounds
   - CSV path motion (`type: csv`): specify CSV trajectory file path

#### Code Structure Description

- `config.py`: Parse YAML scene files, generate Scene objects
- `main.py`: Main script for running AirSim multi-UAV YAML scenes
- `manual_trajectory_recorder.py`: Manually control UAV0 and record its trajectory
- `motion.py`: Define UAV motion patterns
- `print_point.py`: Manually control UAV0 and print its coordinates
- `recorder.py`: Responsible for collecting image frames from UAV cameras and writing to video
- `uav_manager.py`: Responsible for UAV initialization and camera orientation setup
- `utils.py`: Contains utility functions

#### Detailed Technical Description

##### YAML Scene Configuration File Details

Scene configuration files use YAML format and contain the following core configurations:

```yaml
scene_name: demo_circle                        # Scene name for log identification and video save path
description: >                                 # Scene description
  Equilateral triangle layout + UAV0 two-circle flight.
  Observer cameras are top-down view, mainly for testing three-camera view fusion.
record_fps: 30                                 # Target frame rate for all UAV image collection (FPS)
center: [5990, -4170, 0]                       # Absolute center coordinates

observer_drones:                               # List of UAVs for shooting/observation, usually stationary camera positions
  - name: UAV1                                 # UAV name (must be unique)
    pose: [5, 0, -25]                          # Displacement coordinates relative to center point (x, y, z), unit: meters
    camera:                                    # Camera orientation configuration
      mode: lookdown                           # Quick direction setting: lookdown=vertical top-down view
      pitch_deg: -90                           # Pitch angle (degrees), overrides default mode setting

  - name: UAV2
    pose: [-5, 0, -25]
    camera:
      mode: oblique                            # oblique=oblique view, can specify pitch/yaw
      pitch_deg: -45                           # Tilt down 45°
      yaw_deg: 30                              # Yaw angle: nose direction, due east is 0°

  - name: UAV3
    pose: [0, 10, -25]
    camera:
      mode: horizon                            # Horizontal view (pitch angle = 0°)
      yaw_deg: 180                             # 180° indicates due west direction

moving_drones:                                 # UAVs executing trajectory motion (supports multiple)
  - name: UAV0
    start: [5.1616, -3.232, -10.16093183]      # Takeoff initial position (relative to center), unit: meters
    motion:                                    # Motion trajectory definition
      type: circle                             # Type is circular motion (supports circle / csv / figure8 etc.)
      radius: 10                               # Circle radius (meters)
      height: -5                               # Flight height (z-axis value, negative is high altitude)
      angular_speed_dps: 20                    # Angular speed: 20° per second (unit: degrees/second)
      rounds: 2                                # Number of flight rounds
```

##### Camera Mode Details

- **lookdown**: Vertical top-down mode, camera vertically downward, pitch angle -90 degrees
- **horizon**: Horizontal view mode, camera horizontally forward, pitch angle 0 degrees
- **oblique**: Oblique view mode, can customize pitch and yaw angles

##### Motion Mode Details

1. **Circular Motion (CircleMotion)**
   - Parameter description:
     - `radius`: Circle radius (meters)
     - `height`: Flight height (negative Z-axis values in AirSim represent high altitude)
     - `angular_speed_dps`: Angular speed (degrees/second)
     - `rounds`: Number of flight rounds
   - Implementation principle: Calculate circular trajectory based on trigonometric functions, update UAV position and orientation in real-time

2. **CSV Path Motion (CSVPathMotion)**
   - Support reading predefined trajectories from CSV files
   - CSV format requirements: contains timestamp, x, y, z, pitch, roll, yaw columns
   - Suitable for precise reproduction of complex trajectories

##### Multi-Process Architecture

The system uses multi-process architecture to achieve parallel data collection:

1. **Main Process**: Responsible for scene initialization, UAV setup, motion control
2. **Recording Process**: Each observer UAV starts an independent process for image collection
3. **Motion Thread**: Each moving UAV starts a thread to execute trajectory motion

##### Video Recording Technical Details

- **Frame Rate Control**: Achieve target frame rate through time interval control
- **Image Format**: Support RGB image collection, optional point cloud data
- **Storage Format**: Video files saved as MP4 format, support high-resolution recording
- **Error Handling**: Comprehensive exception handling mechanism to ensure recording stability

##### Coordinate System Description

- **AirSim Coordinate System**: Right-handed coordinate system, X-axis forward, Y-axis right, Z-axis down
- **Relative Coordinates**: All UAV positions are defined relative to scene center point
- **Absolute Coordinates**: World coordinates calculated through center point offset

##### Performance Optimization

1. **Multi-Process Parallelism**: Different UAVs use independent processes for collection, avoiding mutual blocking
2. **Memory Management**: Release image data in time to avoid memory leaks
3. **Network Optimization**: Optimize AirSim communication to reduce latency
4. **Error Recovery**: Automatic reconnection and error recovery mechanisms

### 2. Image Post-processing (Derived_Collection)

#### Image Noise Addition

```bash
cd Data_Collection/Derived_Collection
python apply_noise_to_image.py
```

**Feature Details**:
- **Gaussian Noise**: Add Gaussian noise with mean 0 and standard deviation 25 to simulate random noise in real environments
- **Salt and Pepper Noise**: Add 5% salt and pepper noise to simulate pixel corruption during image transmission or storage
- **Image Size Adjustment**: Unify images to 3840x2160 resolution to ensure data consistency
- **Batch Processing**: Support batch processing of entire folders of image files

**Technical Implementation**:
```python
def add_gaussian_noise(image, mean=0, std=25):
    gauss = np.random.normal(mean, std, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def add_salt_and_pepper_noise(image, amount=0.05):
    noisy = image.copy()
    num_pixels = image.shape[0] * image.shape[1]
    num_salt = int(num_pixels * amount / 2)
    num_pepper = int(num_pixels * amount / 2)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 255
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 0
    return noisy
```

**Use Cases**:
- Data augmentation: Improve model robustness to noise
- Real environment simulation: Simulate image quality degradation in actual deployment
- Testing and validation: Verify algorithm performance under different image qualities

#### JSON to Excel Export

```bash
cd Data_Collection/Derived_Collection
python export_to_excel.py
```

**Feature Details**:
- **Format Conversion**: Convert JSON format VQA data to Excel format for manual review and analysis
- **Structured Output**: Contains complete fields such as question ID, type, question, image information, options, correct answer
- **Batch Processing**: Support batch conversion of large-scale datasets

**Output Format**:
| Question ID | Question Type | Question Info | Image Info (UAV1/UAV2) | Question Options (A/B/C/D) | Correct Answer | Answer Error | Error Type |
|-------------|---------------|---------------|------------------------|---------------------------|----------------|--------------|------------|
| 001         | CD            | Question desc | UAV1: path1.jpg        | A: Option1               | A              |              |            |
|             |               |               | UAV2: path2.jpg        | B: Option2               |                |              |            |

**Technical Features**:
- **Encoding Processing**: Automatically handle UTF-8 encoding, support Chinese content
- **Error Handling**: Comprehensive exception handling mechanism to ensure stable conversion process
- **Format Compatibility**: Output standard Excel format, compatible with various office software

---

## Data Annotation

### Annotation Tool: Label Studio

#### Installation

```bash
pip install label-studio
```

#### Startup

```bash
label-studio start
```

Visit `http://localhost:8080` to start annotation.

#### Annotation Content Structure Details

Annotation files use JSON format and contain the following core fields:

```json
{
  "img1": "23-00000001-UAV1.jpg",           // Image filename
  "id": 1,                                  // Annotation ID
  "Quality": "Excellent (5/5) - Sharp, clean, balanced colors, no artifacts.",  // Image quality assessment
  "Usibility": "1 (Available)",             // Usability assessment
  "Object_type": {                          // Target type classification
    "choices": ["Vehicles", "Pedestrians"]  // Multi-select labels
  },
  "PerceptionIssues": [                     // Perception issue annotations (rectangle boxes)
    {
      "x": 65.68516421291045,               // Rectangle center X coordinate (percentage)
      "y": 51.541462186988795,              // Rectangle center Y coordinate (percentage)
      "width": 1.4722536806342097,          // Rectangle width (percentage)
      "height": 3.0200075500188754,         // Rectangle height (percentage)
      "rotation": 0,                        // Rotation angle
      "rectanglelabels": ["Too small"],     // Rectangle labels
      "original_width": 1920,               // Original image width
      "original_height": 1080               // Original image height
    }
  ]
}
```

#### Annotation Field Details

##### 1. Image Quality Assessment (Quality)
- **Scoring Standard**: 1-5 scale, 5 is highest quality
- **Assessment Dimensions**: 
  - Sharpness
  - Color Balance
  - Artifacts
  - Exposure
- **Example**: "Excellent (5/5) - Sharp, clean, balanced colors, no artifacts."

##### 2. Usability Assessment (Usibility)
- **Assessment Criteria**: 
  - "1 (Available)": Can be used for tasks
  - "0 (Unavailable)": Cannot be used for tasks
- **Considerations**: Occlusion level, target visibility, image completeness

##### 3. Target Type Classification (Object_type)
- **Supported Types**: 
  - Vehicles: Vehicles
  - Pedestrians: Pedestrians
  - Buildings: Buildings
  - Traffic Signs: Traffic signs
- **Annotation Method**: Multi-select labels, support simultaneous annotation of multiple types

##### 4. Perception Issue Annotation (PerceptionIssues)
- **Coordinate System**: Use percentage coordinates to adapt to different resolutions
- **Issue Types**:
  - "Too small": Target too small
  - "Occluded": Target occluded
  - "Blurry": Target blurry
  - "Poor lighting": Insufficient lighting
- **Annotation Tool**: Rectangle box annotation, supports rotation

#### Annotation Workflow

1. **Project Creation**
   ```bash
   label-studio init my_project
   cd my_project
   label-studio start
   ```

2. **Data Import**
   - Support multiple formats: JSON, CSV, image folders
   - Batch import functionality
   - Automatic task generation

3. **Annotation Interface**
   - Intuitive web interface
   - Keyboard shortcut support
   - Real-time saving
   - Collaborative annotation functionality

4. **Quality Control**
   - Annotation consistency checking
   - Duplicate annotation verification
   - Annotation quality scoring

#### Annotation Example File Details

##### Real2_Sample.json
- **Data Source**: Real environment collected data
- **Characteristics**: Contains real traffic scenarios with large lighting variations
- **Annotation Focus**: Vehicle recognition, pedestrian detection, traffic sign recognition

##### Sim3_Sample.json  
- **Data Source**: Simulator environment data
- **Characteristics**: Urban street scenarios with multi-vehicle interactions
- **Annotation Focus**: Vehicle trajectories, traffic flow analysis, scene understanding

##### Sim5_Sample.json
- **Data Source**: Simulator environment data
- **Characteristics**: Complex intersection scenarios
- **Annotation Focus**: Traffic rule compliance, conflict detection, decision analysis

##### Sim6_Sample.json
- **Data Source**: Simulator environment data
- **Characteristics**: Highway scenarios
- **Annotation Focus**: High-speed moving targets, lane detection, safe distance assessment

#### Annotation Best Practices

1. **Consistency Principles**
   - Establish unified annotation standards
   - Regular annotation training
   - Use annotation guide documents

2. **Quality Control**
   - Set annotation quality thresholds
   - Perform cross-validation
   - Regular review of annotation results

3. **Efficiency Optimization**
   - Use keyboard shortcuts to improve annotation speed
   - Batch process similar tasks
   - Utilize pre-annotation functionality

4. **Data Management**
   - Version control annotation data
   - Backup important annotation results
   - Establish annotation data indexing

---

## Training and Evaluation

### Training
  Please refer to [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

### Evaluation
  We have offered an example code using gpt-4o to conduct evaluation on our benchmark.
```bash
python AirCopBench_evaluation/evaluation.py # remember to set the api and dataset path in the code.
```


## Acknowledgements

Thanks to all contributors and the open-source community for inspiration and support.


