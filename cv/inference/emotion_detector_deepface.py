"""
Real-time Emotion Detection using DeepFace
Outputs emotion.json every 1 second for Unity integration
Phase 1 - Complete Implementation
"""

import cv2
import json
import time
from datetime import datetime
import os
import sys

# Try to import DeepFace
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("✅ DeepFace loaded successfully")
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("⚠️  DeepFace not available")
    print("   Install with: pip install deepface")
    sys.exit(1)

import numpy as np

# Configuration
OUTPUT_JSON = '../output/emotion.json'
UPDATE_INTERVAL = 1.0  # Update JSON every 1 second
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Face detector
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class EmotionDetector:
    def __init__(self):
        """Initialize emotion detector with DeepFace"""
        
        # Create output directories
        os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
        os.makedirs('../models', exist_ok=True)
        
        self.last_save_time = 0
        self.frame_count = 0
        self.fps = 0
        self.last_emotion = "neutral"
        self.last_confidence = 0.0
        self.last_all_emotions = {}
        
        print(f"📁 Output file: {OUTPUT_JSON}")
        print(f"🤖 Using DeepFace AI model")
        print(f"⏱️  Update interval: {UPDATE_INTERVAL}s\n")
    
    def analyze_emotion(self, frame):
        """Analyze frame for emotions using DeepFace"""
        try:
            # DeepFace analysis
            result = DeepFace.analyze(
                frame, 
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv',
                silent=True
            )
            
            # Handle both single and multiple face results
            if isinstance(result, list):
                if len(result) == 0:
                    return None, None, None
                result = result[0]
            
            # Extract emotion data
            emotions = result['emotion']
            dominant_emotion = result['dominant_emotion']
            confidence = emotions[dominant_emotion] / 100.0
            
            # Normalize all emotions to 0-1 range
            normalized_emotions = {k: v/100.0 for k, v in emotions.items()}
            
            return dominant_emotion, confidence, normalized_emotions
            
        except Exception as e:
            print(f"⚠️  DeepFace error: {e}")
            return None, None, None
    
    def save_json(self, emotion, confidence, all_emotions):
        """Save emotion data to JSON file for Unity"""
        
        data = {
            "emotion": emotion,
            "confidence": round(float(confidence), 3),
            "timestamp": datetime.now().isoformat(),
            "frame": self.frame_count,
            "all_predictions": {
                k: round(float(v), 3) 
                for k, v in all_emotions.items()
            },
            "model": "DeepFace",
            "fps": round(self.fps, 1)
        }
        
        try:
            with open(OUTPUT_JSON, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving JSON: {e}")
    
    def run(self):
        """Main inference loop"""
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ ERROR: Cannot open webcam")
            print("   Make sure no other app is using the camera")
            return
        
        # Set camera resolution for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("="*60)
        print("🎥 WEBCAM STARTED - Emotion Detection Active")
        print("="*60)
        print("\n📊 Real-time emotion analysis running...")
        print("💾 Updating emotion.json every 1 second")
        print("👉 Press 'Q' to quit\n")
        print("-"*60)
        
        fps_start = time.time()
        analysis_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break
            
            self.frame_count += 1
            current_time = time.time()
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(
                gray, 
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(100, 100)
            )
            
            display_frame = frame.copy()
            
            # Analyze emotion every UPDATE_INTERVAL seconds
            if current_time - self.last_save_time >= UPDATE_INTERVAL:
                if len(faces) > 0:
                    # Analyze with DeepFace
                    emotion, confidence, all_emotions = self.analyze_emotion(frame)
                    
                    if emotion is not None:
                        # Save results
                        self.save_json(emotion, confidence, all_emotions)
                        self.last_emotion = emotion
                        self.last_confidence = confidence
                        self.last_all_emotions = all_emotions
                        analysis_count += 1
                        
                        # Console output
                        print(f"💾 Frame {self.frame_count:5d} | {emotion:10s} {confidence*100:5.1f}% | Faces: {len(faces)}")
                    else:
                        print(f"⚠️  Frame {self.frame_count:5d} | Analysis failed")
                else:
                    print(f"👤 Frame {self.frame_count:5d} | No face detected")
                
                self.last_save_time = current_time
            
            # Draw face rectangles and emotion labels
            for (x, y, w, h) in faces:
                # Color based on confidence
                if self.last_confidence > 0.6:
                    color = (0, 255, 0)  # Green - high confidence
                elif self.last_confidence > 0.4:
                    color = (0, 165, 255)  # Orange - medium confidence
                else:
                    color = (0, 0, 255)  # Red - low confidence
                
                # Face rectangle
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 3)
                
                # Emotion label background
                label = f"{self.last_emotion}: {self.last_confidence*100:.0f}%"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                
                cv2.rectangle(display_frame, 
                            (x, y-40), 
                            (x + label_size[0] + 10, y), 
                            color, -1)
                
                # Emotion text
                cv2.putText(display_frame, label, (x + 5, y - 12),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            # Calculate FPS
            if self.frame_count % 30 == 0:
                self.fps = 30 / (time.time() - fps_start)
                fps_start = time.time()
            
            # Display HUD info
            hud_y = 30
            cv2.putText(display_frame, f"FPS: {self.fps:.1f}", (10, hud_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            hud_y += 40
            cv2.putText(display_frame, f"Faces: {len(faces)}", (10, hud_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            hud_y += 40
            cv2.putText(display_frame, f"Analyses: {analysis_count}", (10, hud_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            hud_y += 40
            cv2.putText(display_frame, "DeepFace AI", (10, hud_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Emotion Detection - Press Q to Quit', display_frame)
            
            # Quit on 'Q' or 'q'
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("✅ INFERENCE STOPPED")
        print("="*60)
        print(f"📊 Total frames processed: {self.frame_count}")
        print(f"🎯 Total analyses: {analysis_count}")
        print(f"📁 JSON output saved to: {OUTPUT_JSON}")
        print("="*60 + "\n")

def main():
    """Main entry point"""
    
    print("\n" + "="*60)
    print("  EMOTION DETECTION SYSTEM - PHASE 1")
    print("  Powered by DeepFace AI")
    print("="*60 + "\n")
    
    if not DEEPFACE_AVAILABLE:
        print("❌ DeepFace is required but not installed")
        print("   Run: pip install deepface")
        return
    
    detector = EmotionDetector()
    detector.run()

if __name__ == "__main__":
    main()