# Emotion-Aware NPCs for Real-Time Adaptive Dialogue in Games

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


## 🎯 Abstract
This project is real-time computer-vision system that senses a player's affect from a webcam and adapts an NPC's dialogue in-game. Beyond a standard Facial Emotion Recognition (FER) network that works together with an LLM (Ollama), we add novel modules grounded in core CV concepts: image mapping, layering, filtering, and temporal stacking.


## 📁 Repository Structure
```
EmotionAwareNPCs/
├── cv/                                                             # Computer vision models and inference
│   ├── inference/                                                  # Real-time inference scripts
|   │   ├── modules/                                                # Modules for real-time connection with unity and LLM
│   ├── output/                                                     # Face detection and output metrics
|                                          
├── .gitignore                                                      # Git ignore rules
├── Emotion-Aware NPCs for Real-Time Adaptive Dialogue in Games.md  # Complete long report
└── README.md                                                       # This file
```

## 🚀 Quick Start

### Prerequisites
- **Python 3.9** with pip
- **Webcam access** for real-time emotion detection
- **Git** for version control

### Running the Game

Although the project’s executable is lightweight (< 1 GB) and the game world is optimized, we recommend meeting the following system specifications to ensure stable performance, smooth post-processing, and real-time communication with the CV backend.
Minimum Specs (Comparable to Genshin Impact Low Settings)

- CPU: Intel Core i5 (7th gen) or Ryzen 5
- GPU: NVIDIA GTX 1060 / AMD RX 580
- RAM: 8 GB
- Storage: 1 GB free space
- OS: Windows 10 or later

## Recommended Specs (Tested Development Environment)

- CPU: Intel i9-10900K
- GPU: NVIDIA RTX 3080
- RAM: 32 GB
- Storage: SSD recommended for fast loading
- OS: Windows 10/11

Notes

- The Unity game executable is < 1 GB, but smooth gameplay under heavy post-processing (TAA/SMAA, AO, GI, bloom) benefits from a mid-tier GPU.
- FER backend does not require GPU acceleration — runs on CPU.
- Network ports must be available (default: 6000) for Unity ↔ backend communication.
- Webcams that support low-light correction perform best with the CV pipeline.

---

## Unity Game & Graphics Factors

- *Advanced Rendering Pipeline:*
  The Unity build uses URP (unity rendering pipeline UNITY 6.2 LTS)
  AAA-style graphic settings (FXAA, SMAA, TAA, ambient occlusion, bloom, tone mapping, HDR exposure control, and global illumination)
  which directly influence image formation, shading, and pixel reconstruction and utilize topics central to computer vision.
- *Optimized Terrain & Environment:*
  The world includes high-quality terrain, volumetric clouds, and post-processing effects that create stable lighting and reduce visual noise.
- *Consistent Illumination:*
  Lighting, contrast, and exposure were tuned to maintain clear global illumination and minimize shading artifacts.
- *Smooth Animation & Interaction:*
  Characters (Asterine and Limi) use Mixamo animations, spline-based navigation, NavMesh AI and real-time UI triggers to ensure responsive gameplay during emotion-driven interactions.
- *Performance-Aware Design:*
  All visual settings are scalable, allowing the game to run on mid-range GPUs while still maintaining the fidelity needed for CV stability.


---

## SETUP

### 1. Clone the Repository and install LLama 3.2
```bash
git clone https://github.com/MukulRay1603/EmotionAwareNPCs.git
cd EmotionAwareNPCs
```
Install ollama from https://ollama.com/download
```bash
ollama pull llama3.2:1b
#To confirm installation
ollama list
```

### 2. CV env setup
```bash
cd cv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. FER setup with unity
Open 4 different terminals and follow the exact order of execution given below
In terminal 1
```bash
#If venv is already not activated
source venv/bin/activate  # On Windows: venv\Scripts\activate
python inference/background_fusion_service.py #or cv\inference\background_fusion_video.py if you want to see the live video
```
In terminal 2
```bash
#If venv is already not activated
source venv/bin/activate  # On Windows: venv\Scripts\activate
python cv/inference/modules/module_b_retriever.py
```
In terminal 3
```bash
#If venv is already not activated
source venv/bin/activate  # On Windows: venv\Scripts\activate
python cv/inference/modules/module_c_llm.py
```
In terminal 4
```bash
#If venv is already not activated
source venv/bin/activate  # On Windows: venv\Scripts\activate
python cv/inference/modules/module_a_unity.py
```


### 4. Unity Setup
1. Download TestRun3.zip from the link https://drive.google.com/drive/folders/1tXYDaiRDb7n9dPrqzdXDa_mjBzvLyOn8?usp=sharing
2. Extract the file into the same folder as the github repository cloned
3. After finishing the CV and FER setup mentioined above run the .exe file inside the extracted folder

## 📄 License
This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.
