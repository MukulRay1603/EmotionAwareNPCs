#!/usr/bin/env python3
"""
Real-time emotion inference from webcam feed
This script captures webcam frames, processes them through FER model,
and outputs emotion predictions to JSON file and API endpoint.
"""

import cv2
import numpy as np
import json
import time
import os
import logging
from typing import Dict, Any, Optional
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionInference:
    """Real-time emotion inference from webcam"""
    
    def __init__(self, 
                 model_path: str = "models/fer_model.onnx",
                 output_path: str = "output/emotion.json",
                 api_url: str = "http://localhost:8000/infer",
                 target_fps: int = 15):
        """
        Initialize emotion inference system
        
        Args:
            model_path: Path to ONNX model file
            output_path: Path to output JSON file
            api_url: API endpoint URL for real-time updates
            target_fps: Target inference frame rate
        """
        self.model_path = model_path
        self.output_path = output_path
        self.api_url = api_url
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        
        # Initialize components
        self.cap = None
        self.model = None
        self.face_cascade = None
        self.last_inference_time = 0
        
        # Emotion mapping
        self.emotion_labels = [
            'angry', 'disgust', 'fear', 'happy', 
            'sad', 'surprise', 'neutral'
        ]
        
        # Initialize model and face detection
        self._initialize_model()
        self._initialize_face_detection()
        
    def _initialize_model(self):
        """Initialize ONNX model for emotion recognition"""
        try:
            import onnxruntime as ort
            
            if not os.path.exists(self.model_path):
                logger.warning(f"Model file not found: {self.model_path}")
                logger.info("Using mock emotion predictions for demo")
                self.model = None
                return
                
            # Load ONNX model
            self.model = ort.InferenceSession(self.model_path)
            logger.info(f"Loaded ONNX model: {self.model_path}")
            
        except ImportError:
            logger.error("ONNX Runtime not installed. Install with: pip install onnxruntime")
            self.model = None
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
    
    def _initialize_face_detection(self):
        """Initialize OpenCV face detection"""
        try:
            # Load Haar cascade for face detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                logger.error("Failed to load face cascade classifier")
            else:
                logger.info("Face detection initialized")
                
        except Exception as e:
            logger.error(f"Error initializing face detection: {e}")
            self.face_cascade = None
    
    def _detect_faces(self, frame: np.ndarray) -> list:
        """Detect faces in frame"""
        if self.face_cascade is None:
            return []
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(48, 48)
        )
        return faces
    
    def _preprocess_face(self, face_roi: np.ndarray) -> np.ndarray:
        """Preprocess face region for emotion recognition"""
        # Resize to model input size (typically 48x48 for FER)
        face_resized = cv2.resize(face_roi, (48, 48))
        
        # Convert to grayscale if needed
        if len(face_resized.shape) == 3:
            face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        
        # Normalize to [0, 1]
        face_normalized = face_resized.astype(np.float32) / 255.0
        
        # Add batch dimension and channel dimension
        face_batch = np.expand_dims(face_normalized, axis=0)
        face_batch = np.expand_dims(face_batch, axis=0)
        
        return face_batch
    
    def _predict_emotion(self, face_batch: np.ndarray) -> Dict[str, Any]:
        """Predict emotion from preprocessed face"""
        if self.model is None:
            # Return mock prediction for demo
            return self._mock_emotion_prediction()
        
        try:
            # Run inference
            input_name = self.model.get_inputs()[0].name
            output_name = self.model.get_outputs()[0].name
            
            outputs = self.model.run([output_name], {input_name: face_batch})
            emotion_probs = outputs[0][0]
            
            # Get predicted emotion
            emotion_idx = np.argmax(emotion_probs)
            emotion = self.emotion_labels[emotion_idx]
            confidence = float(emotion_probs[emotion_idx])
            
            # Calculate additional features
            features = self._calculate_features(emotion_probs)
            
            return {
                "emotion": emotion,
                "confidence": confidence,
                "timestamp": time.time(),
                "features": features
            }
            
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return self._mock_emotion_prediction()
    
    def _mock_emotion_prediction(self) -> Dict[str, Any]:
        """Generate mock emotion prediction for demo"""
        import random
        
        emotion = random.choice(self.emotion_labels)
        confidence = random.uniform(0.7, 0.95)
        
        return {
            "emotion": emotion,
            "confidence": confidence,
            "timestamp": time.time(),
            "features": {
                "valence": random.uniform(-1.0, 1.0),
                "arousal": random.uniform(0.0, 1.0),
                "stress_level": random.uniform(0.0, 1.0),
                "fatigue_level": random.uniform(0.0, 0.5),
                "head_pose": {
                    "yaw": random.uniform(-30, 30),
                    "pitch": random.uniform(-20, 20),
                    "roll": random.uniform(-10, 10)
                },
                "eye_aspect_ratio": random.uniform(0.2, 0.3),
                "motion_intensity": random.uniform(0.0, 0.2)
            }
        }
    
    def _calculate_features(self, emotion_probs: np.ndarray) -> Dict[str, Any]:
        """Calculate additional features from emotion probabilities"""
        # Map emotions to valence-arousal space
        valence_map = {
            'happy': 0.8, 'surprise': 0.6, 'neutral': 0.0,
            'sad': -0.8, 'angry': -0.6, 'fear': -0.4, 'disgust': -0.7
        }
        
        arousal_map = {
            'happy': 0.7, 'surprise': 0.9, 'angry': 0.8, 'fear': 0.9,
            'sad': 0.2, 'neutral': 0.1, 'disgust': 0.3
        }
        
        # Calculate weighted valence and arousal
        valence = sum(prob * valence_map[label] for prob, label in zip(emotion_probs, self.emotion_labels))
        arousal = sum(prob * arousal_map[label] for prob, label in zip(emotion_probs, self.emotion_labels))
        
        # Calculate stress level (high arousal + negative valence)
        stress_level = max(0, arousal * (1 - valence) / 2)
        
        # Calculate fatigue level (low arousal + negative valence)
        fatigue_level = max(0, (1 - arousal) * (1 - valence) / 2)
        
        return {
            "valence": float(valence),
            "arousal": float(arousal),
            "stress_level": float(stress_level),
            "fatigue_level": float(fatigue_level),
            "head_pose": {
                "yaw": 0.0,  # Placeholder - would need head pose estimation
                "pitch": 0.0,
                "roll": 0.0
            },
            "eye_aspect_ratio": 0.25,  # Placeholder - would need eye detection
            "motion_intensity": 0.05   # Placeholder - would need optical flow
        }
    
    def _save_emotion_json(self, emotion_data: Dict[str, Any]):
        """Save emotion data to JSON file"""
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            
            with open(self.output_path, 'w') as f:
                json.dump(emotion_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving emotion JSON: {e}")
    
    def _send_to_api(self, emotion_data: Dict[str, Any]):
        """Send emotion data to API endpoint"""
        try:
            import requests
            
            response = requests.post(
                self.api_url,
                json=emotion_data,
                timeout=1.0
            )
            
            if response.status_code == 200:
                logger.debug("Successfully sent emotion data to API")
            else:
                logger.warning(f"API request failed with status {response.status_code}")
                
        except ImportError:
            logger.warning("Requests library not available. Install with: pip install requests")
        except Exception as e:
            logger.debug(f"API request failed: {e}")
    
    def run_inference(self):
        """Main inference loop"""
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logger.error("Could not open webcam")
            return
        
        logger.info("Starting emotion inference...")
        logger.info("Press 'q' to quit, 's' to save screenshot")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to read frame from webcam")
                    break
                
                current_time = time.time()
                
                # Check if it's time for inference
                if current_time - self.last_inference_time >= self.frame_interval:
                    # Detect faces
                    faces = self._detect_faces(frame)
                    
                    if len(faces) > 0:
                        # Use the largest face
                        largest_face = max(faces, key=lambda x: x[2] * x[3])
                        x, y, w, h = largest_face
                        
                        # Extract face region
                        face_roi = frame[y:y+h, x:x+w]
                        
                        # Preprocess and predict
                        face_batch = self._preprocess_face(face_roi)
                        emotion_data = self._predict_emotion(face_batch)
                        
                        # Save and send results
                        self._save_emotion_json(emotion_data)
                        self._send_to_api(emotion_data)
                        
                        # Draw bounding box and emotion
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, 
                                  f"{emotion_data['emotion']} ({emotion_data['confidence']:.2f})",
                                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        self.last_inference_time = current_time
                        logger.info(f"Emotion: {emotion_data['emotion']} (confidence: {emotion_data['confidence']:.2f})")
                    else:
                        # No face detected
                        emotion_data = {
                            "emotion": "neutral",
                            "confidence": 0.0,
                            "timestamp": current_time,
                            "features": {
                                "valence": 0.0,
                                "arousal": 0.0,
                                "stress_level": 0.0,
                                "fatigue_level": 0.0,
                                "head_pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
                                "eye_aspect_ratio": 0.25,
                                "motion_intensity": 0.0
                            }
                        }
                        self._save_emotion_json(emotion_data)
                
                # Display frame
                cv2.imshow('Emotion Recognition', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    screenshot_path = f"screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(screenshot_path, frame)
                    logger.info(f"Screenshot saved: {screenshot_path}")
        
        except KeyboardInterrupt:
            logger.info("Inference interrupted by user")
        
        finally:
            # Cleanup
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            logger.info("Emotion inference stopped")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Real-time emotion inference from webcam')
    parser.add_argument('--model', default='models/fer_model.onnx', help='Path to ONNX model')
    parser.add_argument('--output', default='output/emotion.json', help='Output JSON file path')
    parser.add_argument('--api-url', default='http://localhost:8000/infer', help='API endpoint URL')
    parser.add_argument('--fps', type=int, default=15, help='Target inference FPS')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create inference system
    inference = EmotionInference(
        model_path=args.model,
        output_path=args.output,
        api_url=args.api_url,
        target_fps=args.fps
    )
    
    # Run inference
    inference.run_inference()

if __name__ == "__main__":
    main()
