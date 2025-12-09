# Emotion-Aware NPCs for Real-Time Adaptive Dialogue in Games

## 🎯 Abstract
We propose a real-time computer-vision system that senses a player's affect from a webcam and adapts an NPC's dialogue in-game. Beyond a standard Facial Emotion Recognition (FER) network, we add novel modules grounded in core CV concepts: image mapping, layering, filtering, and temporal stacking.

## 🚀 System Overview
This project implements a complete pipeline from webcam input to NPC dialogue adaptation, featuring:
- **Continuous Affect Mapping**: Converts categorical FER outputs into 2D valence-arousal space
- **Temporal Emotion Dynamics**: EMA smoothing and spike detection for emotional buildup
- **Multi-Signal Fusion**: Combines FER with Eye Aspect Ratio (EAR), head pose, and optical flow
- **Real-Time Performance**: Target latency ≤400-600ms and 10-15 FPS inference

## 📁 Repository Structure
```
EmotionAwareNPCs/
├── cv/                   # Computer vision models and inference
│   ├── inference/        # Real-time inference scripts
|   │   ├── modules/      # Modules for real-time connection with unity and LLM
│   ├── output/           # Face detection and output metrics
|
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## 🚀 Quick Start

### Prerequisites
- **Python 3.9** with pip
- **Unity 2022.3 LTS** or higher
- **Webcam access** for real-time emotion detection
- **Git** for version control

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
1. Open Unity Hub
2. Open project from `unity/` folder
3. Import required packages (TextMeshPro, Newtonsoft.Json)
4. Load main scene and press Play

