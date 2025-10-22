#!/usr/bin/env python3
"""
Setup script for Computer Vision module
Downloads required models and sets up the environment
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_requirements():
    """Install required Python packages"""
    logger.info("Installing CV requirements...")
    
    requirements = [
        "opencv-python>=4.5.0",
        "numpy>=1.21.0",
        "onnxruntime>=1.12.0",
        "requests>=2.28.0",
        "Pillow>=9.0.0"
    ]
    
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            logger.info(f"Installed {package}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {package}: {e}")

def create_directories():
    """Create necessary directories"""
    logger.info("Creating directories...")
    
    directories = [
        "models",
        "output",
        "preprocessing",
        "features",
        "tests",
        "data/samples"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def download_sample_model():
    """Download a sample ONNX model for testing"""
    logger.info("Downloading sample model...")
    
    # This would be replaced with actual model download
    # For now, create a placeholder
    model_path = "models/fer_model.onnx"
    
    if not os.path.exists(model_path):
        logger.info("Creating placeholder model file...")
        with open(model_path, 'w') as f:
            f.write("# Placeholder ONNX model file\n")
            f.write("# Replace with actual trained model\n")
        logger.info(f"Created placeholder: {model_path}")
    else:
        logger.info(f"Model already exists: {model_path}")

def create_sample_data():
    """Create sample data files"""
    logger.info("Creating sample data...")
    
    # Create sample emotion.json
    sample_emotion = {
        "emotion": "neutral",
        "confidence": 0.85,
        "timestamp": 1729584000.0,
        "features": {
            "valence": 0.0,
            "arousal": 0.1,
            "stress_level": 0.2,
            "fatigue_level": 0.1,
            "head_pose": {
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0
            },
            "eye_aspect_ratio": 0.25,
            "motion_intensity": 0.05
        }
    }
    
    import json
    with open("output/emotion.json", 'w') as f:
        json.dump(sample_emotion, f, indent=2)
    
    logger.info("Created sample emotion.json")

def create_test_script():
    """Create a test script for the CV module"""
    test_script = '''#!/usr/bin/env python3
"""
Test script for CV module
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference.webcam_inference import EmotionInference

def test_mock_inference():
    """Test mock emotion inference"""
    print("Testing mock emotion inference...")
    
    inference = EmotionInference()
    
    # Test mock prediction
    emotion_data = inference._mock_emotion_prediction()
    print(f"Mock emotion: {emotion_data['emotion']}")
    print(f"Confidence: {emotion_data['confidence']:.2f}")
    print(f"Valence: {emotion_data['features']['valence']:.2f}")
    print(f"Arousal: {emotion_data['features']['arousal']:.2f}")
    
    print("Mock inference test passed!")

if __name__ == "__main__":
    test_mock_inference()
'''
    
    with open("test_cv.py", 'w') as f:
        f.write(test_script)
    
    os.chmod("test_cv.py", 0o755)
    logger.info("Created test_cv.py")

def main():
    """Main setup function"""
    logger.info("Setting up Computer Vision module...")
    
    try:
        install_requirements()
        create_directories()
        download_sample_model()
        create_sample_data()
        create_test_script()
        
        logger.info("CV module setup completed successfully!")
        logger.info("Run 'python test_cv.py' to test the setup")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
