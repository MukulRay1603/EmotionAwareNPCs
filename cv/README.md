# Computer Vision Pipeline

## Overview
This directory contains the computer vision components for emotion recognition and feature extraction.

## Directory Structure
```
cv/
├── models/              # Trained models (ONNX format)
├── inference/           # Inference scripts
├── preprocessing/       # Face detection and alignment
├── features/            # Novel feature extraction modules
├── output/              # Output JSON files
│   └── emotion.json    # Latest emotion prediction
└── tests/              # Unit tests
```

## Components (To be implemented by P1)

### 1. FER Baseline Model
- Model architecture: MobileNetV2 / MiniXception
- Training dataset: FER-2013 or RAF-DB
- Output format: ONNX
- Classes: 7 emotions

### 2. Webcam Inference Script
- Real-time capture: 10-15 FPS
- Face detection: BlazeFace/MediaPipe
- Preprocessing: alignment, normalization
- Output: JSON updates every 1 second

### 3. Novel Feature Modules
- **Affect Vector**: Valence-arousal mapping
- **EAR**: Eye Aspect Ratio for fatigue
- **Head Pose**: 3DOF estimation
- **Optical Flow**: Motion analysis
- **EMA Smoothing**: Temporal stability

## Expected Output Format

### emotion.json
```json
{
  "emotion": "happy",
  "confidence": 0.86,
  "timestamp": 1729584000.0,
  "features": {
    "valence": 0.5,
    "arousal": 0.7,
    "stress_level": 0.2,
    "fatigue_level": 0.1,
    "head_pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
    "eye_aspect_ratio": 0.25,
    "motion_intensity": 0.05
  }
}
```

## Setup Instructions

### Prerequisites
```bash
pip install opencv-python numpy onnxruntime mediapipe
```

### Running Inference (Once implemented)
```bash
cd cv/inference
python webcam_inference.py
```

## Integration Points
- Outputs `emotion.json` to `cv/output/`
- Unity reads this file every 1 second
- Alternative: Call backend API endpoint

## Performance Requirements
- Inference latency: <100ms per frame
- Frame rate: 10-15 FPS
- Model size: <10MB (mobile deployment)

## TODO (P1 Tasks)
- [ ] Implement FER baseline training
- [ ] Export model to ONNX
- [ ] Create webcam inference script
- [ ] Implement novel feature extraction
- [ ] Test on sample images
- [ ] Record demo video

