# Phase 1 Report - Emotion-Aware NPCs

## Team Members
- **P1 (Mukul Ray)**: Unity Lead - Webcam capture, UI, game logicP2 
- **P2 (Taner Bulbul)**: Data & Eval Lead - Datasets, user study
- **P3 (Rahul Sharma)**: CV Lead - FER baseline, core CV modulesP3 
- **P4 (Karthik Ramanathan)**: MLOps & Release Lead - Repo management, CI/CD

## Phase 1 Objectives
Demonstrate an end-to-end working prototype: webcam → emotion → NPC reaction

## Deliverables Structure
```
phase1_report/
├── README.md                    # This file
├── screenshots/                 # Visual documentation
│   ├── unity_scene.png         # Unity NPC interface
│   ├── cv_inference.png        # Live emotion detection
│   └── api_testing.png         # Backend API responses
├── scripts/                     # Code samples
│   ├── cv_inference.py         # Computer vision pipeline
│   ├── unity_emotion_reader.cs # Unity dialogue logic
│   └── api_client.py           # API integration
├── dataset_summary.md          # Dataset overview
├── baseline_metrics.md         # Model performance
└── videos/                      # Demo recordings
    └── end_to_end_demo.mp4     # 10-second working demo
```

## Component Status

### ✅ P4 - MLOps & Integration (Complete)
- [x] GitHub repository structure created
- [x] FastAPI backend with emotion inference endpoints
- [x] .gitignore, requirements.txt configured
- [x] Setup instructions documented
- [x] System architecture diagram

### ⏳ P1 - CV Lead (In Progress)
- [ ] FER baseline model (MobileNetV2)
- [ ] ONNX export
- [ ] Webcam inference script
- [ ] emotion.json output generation
- [ ] Live prediction demo

### ⏳ P2 - Unity Lead (In Progress)
- [ ] Minimal Unity scene with NPC
- [ ] Coroutine for emotion.json reading
- [ ] Emotion-to-dialogue mapping
- [ ] Cooldown implementation (2s)
- [ ] 10-second demo video

### ⏳ P3 - Data & Eval Lead (In Progress)
- [ ] 500 images from FER-2013 organized
- [ ] Dataset summary table
- [ ] Baseline accuracy metrics
- [ ] Report draft sections

## Quick Start Guide

### 1. Clone Repository
```bash
git clone https://github.com/MukulRay1603/EmotionAwareNPCs.git
cd EmotionAwareNPCs
```

### 2. Setup Backend
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### 3. Test API
```bash
curl http://localhost:8000/health
curl http://localhost:8000/infer
```

### 4. Setup Unity
- Open `unity/` folder in Unity 2022.3+
- Load main scene
- Press Play

## API Endpoints

### GET /health
Health check for backend service

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1729584000.0,
  "service": "emotion-inference-api"
}
```

### GET /infer
Get latest emotion prediction

**Response:**
```json
{
  "emotion": "happy",
  "confidence": 0.86,
  "timestamp": 1729584000.0,
  "features": {
    "valence": 0.5,
    "arousal": 0.7,
    "stress_level": 0.2,
    "fatigue_level": 0.1
  }
}
```

### POST /infer
Process new frame and return emotion

**Request:**
```json
{
  "frame_data": "base64_encoded_image",
  "timestamp": 1729584000.0
}
```

## Integration Workflow

1. **CV Pipeline**: Captures webcam frame at 10-15 FPS
2. **Inference**: Processes through FER model (ONNX)
3. **Feature Extraction**: Computes affect vector, EAR, head pose
4. **API Response**: Returns JSON with emotion + features
5. **Unity Client**: Polls API every 1 second
6. **Dialogue Policy**: Maps emotion to NPC response
7. **NPC Display**: Shows subtitle with 2s cooldown

## Next Steps for Each Team Member

### P1 (Karthik) - CV Lead
1. Train/load FER baseline model
2. Export to ONNX format
3. Create webcam inference script
4. Test on sample images
5. Record demo of live predictions

### P2 (Mukul) - Unity Lead
1. Build Unity scene with NPC character
2. Implement HTTP client for API calls
3. Create emotion-to-dialogue mapping logic
4. Add cooldown timer and debouncing
5. Record 10-second integration demo

### P3 (Taner) - Data & Eval
1. Download and organize FER-2013 subset
2. Create dataset summary table
3. Compute baseline accuracy metrics
4. Draft report sections
5. Prepare evaluation criteria

### P4 (Rahul) - MLOps
1. ✅ Repository structure complete
2. ✅ API backend complete
3. Monitor team integration
4. Merge all deliverables to `/docs/phase1_report/`
5. Prepare final demo video

## Performance Targets (Phase 1)

| Metric | Target | Status |
|--------|--------|--------|
| API Response Time | <100ms | ✅ Stub ready |
| End-to-End Latency | <600ms | ⏳ Pending integration |
| Inference FPS | 10-15 | ⏳ Pending CV model |
| Dialogue Update Rate | 1 Hz | ⏳ Pending Unity |

## Known Issues & Limitations

1. **Stub Implementation**: Current backend returns mock emotions, needs real model integration
2. **Unity Scene**: Pending P2 development
3. **CV Pipeline**: Pending P1 model training and inference script
4. **Dataset**: Pending P3 organization and metrics

## Timeline

- **Oct 21**: Repository and API setup ✅
- **Oct 22**: CV model + Unity scene integration ⏳
- **Oct 22 EOD**: Phase 1 demonstration and report submission

## Contact & Collaboration

- **GitHub Issues**: For bug reports and feature requests
- **Pull Requests**: For code contributions (all team members)
- **Documentation**: Update as you complete tasks

---

**Last Updated**: October 21, 2025  
**Phase**: 1 (Prototype)  
**Status**: In Progress

