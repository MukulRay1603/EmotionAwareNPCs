# Emotion-Aware NPCs for Real-Time Adaptive Dialogue in Games

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Unity 2022.3+](https://img.shields.io/badge/Unity-2022.3+-blue.svg)](https://unity.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)

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
├── unity/                 # Unity game client and NPC logic
│   ├── Assets/           # Unity assets and scripts
│   └── README.md         # Unity setup guide
├── cv/                   # Computer vision models and inference
│   ├── inference/        # Real-time inference scripts
│   ├── models/           # ONNX model files
│   ├── preprocessing/    # Face detection and alignment
│   └── README.md         # CV pipeline guide
├── backend/              # FastAPI server and MLOps pipeline
│   ├── main.py          # FastAPI application
│   ├── requirements.txt # Python dependencies
│   └── SETUP.md         # Backend setup guide
├── docs/                 # Documentation and reports
│   ├── phase1_report/   # Phase 1 deliverables
│   ├── system_diagram/  # Architecture diagrams
│   └── SYSTEM_ARCHITECTURE.md
├── .gitignore           # Git ignore rules
├── CONTRIBUTING.md      # Contribution guidelines
└── README.md            # This file
```

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** with pip
- **Unity 2022.3 LTS** or higher
- **Webcam access** for real-time emotion detection
- **Git** for version control

### 1. Clone the Repository
```bash
git clone https://github.com/MukulRay1603/EmotionAwareNPCs.git
cd EmotionAwareNPCs
```

### 2. Backend Setup
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

The API will be available at `http://localhost:8000`

### 3. Test the API
```bash
# Health check
curl http://localhost:8000/health

# Get emotion prediction
curl http://localhost:8000/infer
```

### 4. Unity Setup
1. Open Unity Hub
2. Open project from `unity/` folder
3. Import required packages (TextMeshPro, Newtonsoft.Json)
4. Load main scene and press Play

## 📊 Performance Targets
- **End-to-End Latency**: ≤400ms (ideal), ≤600ms (acceptable)
- **Inference Rate**: 10-15 FPS on face crops
- **Dialogue Update Rate**: 3-5 Hz
- **Stability**: Dialogue triggers only if N=6-10 consecutive frames agree

## 👥 Team Roles
- **P1 (Rahul Sharma)**: CV Lead - FER baseline, core CV modules
- **P2 (Mukul Ray)**: Unity Lead - Webcam capture, UI, game logic
- **P3 (Karthik Ramanathan)**: Data & Eval Lead - Datasets, user study, robustness tests
- **P4 (Taner Bulbul)**: MLOps & Release Lead - Repo management, CI/CD, ONNX pipeline

## 🔌 API Endpoints
- `GET /health` - Health check
- `GET /infer` - Get latest emotion prediction
- `POST /infer` - Process new frame and return emotion
- `GET /status` - System status and performance metrics

## 📚 Documentation
- [System Architecture](docs/SYSTEM_ARCHITECTURE.md) - Detailed system design
- [Phase 1 Report](docs/phase1_report/README.md) - Current progress and deliverables
- [Contributing Guidelines](CONTRIBUTING.md) - How to contribute to the project
- [Backend Setup](backend/SETUP.md) - Backend configuration guide
- [Unity Setup](unity/README.md) - Unity client setup guide
- [CV Pipeline](cv/README.md) - Computer vision pipeline guide

## 🛠️ Development Status

### Phase 1 (Current) - Prototype
- [x] Repository structure and documentation
- [x] FastAPI backend with mock emotion inference
- [x] System architecture and API design
- [ ] CV pipeline with real FER model
- [ ] Unity client with NPC dialogue system
- [ ] End-to-end integration testing

### Future Phases
- **Phase 2**: Enhanced features and performance optimization
- **Phase 3**: Production deployment and scaling
- **Phase 4**: Advanced ML features and user studies

## 🐛 Troubleshooting

### Common Issues
1. **API not responding**: Check if backend is running on port 8000
2. **Unity connection failed**: Verify API endpoint in Unity scripts
3. **Webcam not detected**: Check camera permissions and device availability
4. **Model loading errors**: Ensure ONNX models are in correct directory

### Getting Help
- Check [Issues](https://github.com/MukulRay1603/EmotionAwareNPCs/issues) for known problems
- Create a new issue for bugs or feature requests
- Contact team members for specific component issues

## 📄 License
This project is licensed under the Apache-2.0 License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing
We welcome contributions! Please read our [Contributing Guidelines](CONTRIBUTING.md) and [Code of Conduct](CONTRIBUTING.md#code-of-conduct) before submitting pull requests.

## 🙏 Acknowledgments
- Thanks to the open-source community for excellent tools and libraries
- Inspiration from existing emotion recognition research
- Special thanks to all team members and contributors

---

**Last Updated**: October 21, 2025  
**Version**: 1.0.0  
**Status**: Phase 1 - In Development
