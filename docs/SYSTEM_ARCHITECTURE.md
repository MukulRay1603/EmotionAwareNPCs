# System Architecture - Emotion-Aware NPCs

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Player's Webcam                          │
└────────────────────────────┬────────────────────────────────────┘
                             │ Video Stream (10-15 FPS)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Computer Vision Pipeline                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Face         │→ │ Preprocessing│→ │ FER Model (ONNX)     │  │
│  │ Detection    │  │ & Alignment  │  │ (MobileNetV2)        │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│                                                ↓                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │         Novel Feature Extraction Modules                  │   │
│  │  • Continuous Affect Vector (Valence-Arousal)           │   │
│  │  • Eye Aspect Ratio (EAR) - Fatigue Detection           │   │
│  │  • Head Pose Estimation (solvePnP)                      │   │
│  │  • Optical Flow - Motion Analysis                       │   │
│  │  • Temporal EMA Smoothing                               │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │ JSON Response
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                             │
│  Endpoints:                                                      │
│  • GET  /health  - Health check                                 │
│  • GET  /infer   - Get latest emotion                           │
│  • POST /infer   - Process new frame                            │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP/REST API
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Unity Game Client                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Dialogue Policy Engine                       │   │
│  │  • Rule-based decision making                            │   │
│  │  • Context flags: STRESS, RUSH, FATIGUE                  │   │
│  │  • Cooldown management (2s between responses)            │   │
│  │  • Debouncing (6-10 frame agreement)                     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                             ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              NPC Response System                          │   │
│  │  • Subtitle rendering                                     │   │
│  │  • Animation triggers                                     │   │
│  │  • Voice synthesis (optional)                            │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
Frame Capture → Face Detection → Preprocessing → FER Model
                                                      ↓
                                              Emotion Probabilities
                                                      ↓
                                         ┌────────────┴────────────┐
                                         ▼                         ▼
                                  Affect Vector              Feature Fusion
                                  (2D Mapping)               ┌─────────────┐
                                         │                   │ • EAR       │
                                         │                   │ • Head Pose │
                                         │                   │ • Optical   │
                                         │                   │   Flow      │
                                         └─────────┬─────────┴─────────────┘
                                                   ▼
                                          Temporal Processing
                                          (EMA Smoothing)
                                                   ↓
                                          Context Flag Generation
                                          {stress, rush, fatigue}
                                                   ↓
                                          Dialogue Policy
                                                   ↓
                                          NPC Response
```

## Component Details

### 1. Face Detection Module
- **Technology**: BlazeFace / MediaPipe
- **Output**: Face bounding box + 68 landmarks
- **Performance**: 10-15 FPS

### 2. FER Backbone
- **Model**: MobileNetV2
- **Classes**: 7 emotions (happy, sad, angry, fear, surprise, disgust, neutral)
- **Format**: ONNX for cross-platform inference
- **Target Latency**: <100ms per frame

### 3. Novel Feature Modules
- **Affect Vector**: Maps softmax → (valence, arousal) space
- **EAR**: Eye Aspect Ratio for blink/fatigue detection
- **Head Pose**: 3DOF rotation (yaw, pitch, roll)
- **Motion**: Optical flow magnitude

### 4. Dialogue Policy
- **Input**: Emotion + context flags + game state
- **Logic**: Rule-based with cooldowns and hysteresis
- **Output**: NPC dialogue line selection

## Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| End-to-End Latency (p50) | ≤350ms | Frame to dialogue |
| End-to-End Latency (p90) | ≤600ms | Frame to dialogue |
| Inference FPS | 10-15 | CV pipeline |
| Dialogue Update Rate | 3-5 Hz | Unity client |
| Debounce Window | 6-10 frames | State stability |

## API Contract

### Emotion Response Schema
```json
{
  "emotion": "string",           // Primary emotion
  "confidence": 0.0-1.0,         // Model confidence
  "timestamp": 1234567890.123,   // Unix timestamp
  "features": {
    "valence": -1.0 to 1.0,      // Positive/negative
    "arousal": 0.0 to 1.0,       // Energy level
    "stress_level": 0.0 to 1.0,
    "fatigue_level": 0.0 to 1.0,
    "head_pose": {
      "yaw": -180 to 180,
      "pitch": -90 to 90,
      "roll": -180 to 180
    },
    "eye_aspect_ratio": 0.0 to 0.4,
    "motion_intensity": 0.0 to 1.0
  }
}
```

## Integration Modes

### Mode A: Unity-Only (Barracuda)
- Single runtime, lowest latency
- ONNX model runs in Unity
- No network overhead

### Mode B: Client-Server (FastAPI)
- Python backend for CV pipeline
- HTTP API for Unity client
- Easier debugging and iteration
- **Current implementation for Phase 1**

## Deployment Architecture

```
Development:
  localhost:8000 (FastAPI) ← Unity Client

Production:
  Cloud/Edge Server (FastAPI) ← Unity Build
  OR
  Embedded in Unity (Barracuda)
```

