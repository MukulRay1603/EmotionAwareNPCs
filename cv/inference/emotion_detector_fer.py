"""
Real-time Emotion Detection using Custom FER-2013 Model
Downloads pre-trained Keras model and runs inference
Outputs emotion.json every 1 second for Unity integration
"""

import cv2
import json
import time
from datetime import datetime
import os
import numpy as np
import urllib.request

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
    print("✅ TensorFlow loaded successfully")
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not available")
    print("   Install with: pip install tensorflow")

# Configuration
OUTPUT_JSON = '../output/emotion_fer.json'
MODEL_DIR = '../models'
MODEL_PATH = os.path.join(MODEL_DIR, 'fer2013_mini_xception.h5')
UPDATE_INTERVAL = 1.0

# Emotion labels for FER-2013
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Face detector
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class FEREmotionDetector:
    def __init__(self):
        """Initialize FER emotion detector"""
        
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
        
        self.model = None
        self.last_save_time = 0
        self.frame_count = 0
        self.fps = 0
        self.last_emotion = "neutral"
        self.last_confidence = 0.0
        
        # Load or download model
        self.load_model()
        
        print(f"📁 Output file: {OUTPUT_JSON}")
        print(f"🤖 Using Custom FER-2013 Model\n")
    
    def download_model(self):
        """Download pre-trained FER model"""
        
        # Using a publicly available FER-2013 model
        print("📥 Downloading pre-trained FER-2013 model...")
        print("   This may take a few minutes...")
        
        # Model URL (using a well-known pre-trained FER model)
        model_url = "https://github.com/oarriaga/face_classification/raw/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5"
        
        try:
            urllib.request.urlretrieve(model_url, MODEL_PATH)
            print("✅ Model downloaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to download model: {e}")
            print("   Creating a simple fallback model instead...")
            return False
    
    def create_simple_model(self):
        """Create a simple CNN model as fallback"""
        
        print("🔧 Creating simple CNN model...")
        
        model = keras.Sequential([
            keras.layers.Input(shape=(48, 48, 1)),
            
            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Dropout(0.25),
            
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Dropout(0.25),
            
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Dropout(0.25),
            
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(len(EMOTIONS), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Save the untrained model
        model.save(MODEL_PATH)
        print("✅ Simple model created (untrained - for demo purposes)")
        
        return model
    
    def load_model(self):
        """Load or download the emotion model"""
        
        if not TF_AVAILABLE:
            print("❌ TensorFlow required for FER model")
            return
        
        # Check if model exists
        if os.path.exists(MODEL_PATH):
            print(f"📦 Loading existing model from {MODEL_PATH}...")
            try:
                self.model = keras.models.load_model(MODEL_PATH)
                print("✅ Model loaded successfully!")
                return
            except Exception as e:
                print(f"⚠️  Failed to load existing model: {e}")
        
        # Download or create model
        if not self.download_model():
            self.model = self.create_simple_model()
        else:
            try:
                self.model = keras.models.load_model(MODEL_PATH)
                print("✅ Downloaded model loaded successfully!")
            except Exception as e:
                print(f"⚠️  Failed to load downloaded model: {e}")
                self.model = self.create_simple_model()
    
    def preprocess_face(self, face_img):
        """Preprocess face for model input"""
        
        # Resize to 48x48
        face = cv2.resize(face_img, (48, 48))
        
        # Convert to grayscale
        if len(face.shape) == 3:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # Normalize
        face = face.astype('float32') / 255.0
        
        # Add channel dimension
        face = np.expand_dims(face, axis=-1)
        
        # Add batch dimension
        face = np.expand_dims(face, axis=0)
        
        return face
    
    def predict_emotion(self, face_roi):
        """Predict emotion from face ROI"""
        
        if self.model is None:
            return None, None, None
        
        try:
            # Preprocess
            preprocessed = self.preprocess_face(face_roi)
            
            # Predict
            predictions = self.model.predict(preprocessed, verbose=0)[0]
            
            # Get dominant emotion
            emotion_idx = np.argmax(predictions)
            emotion = EMOTIONS[emotion_idx]
            confidence = float(predictions[emotion_idx])
            
            # All emotions dict
            all_emotions = {
                EMOTIONS[i]: float(predictions[i])
                for i in range(len(EMOTIONS))
            }
            
            return emotion, confidence, all_emotions
            
        except Exception as e:
            print(f"⚠️  Prediction error: {e}")
            return None, None, None
    
    def save_json(self, emotion, confidence, all_emotions):
        """Save emotion data to JSON"""
        
        data = {
            "emotion": emotion,
            "confidence": round(float(confidence), 3),
            "timestamp": datetime.now().isoformat(),
            "frame": self.frame_count,
            "all_predictions": {
                k: round(float(v), 3)
                for k, v in all_emotions.items()
            },
            "model": "Custom FER-2013",
            "fps": round(self.fps, 1)
        }
        
        try:
            with open(OUTPUT_JSON, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving JSON: {e}")
    
    def run(self):
        """Main inference loop"""
        
        if self.model is None:
            print("❌ No model available. Cannot start inference.")
            return
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ ERROR: Cannot open webcam")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("="*60)
        print("🎥 WEBCAM STARTED - FER Model Emotion Detection")
        print("="*60)
        print("\n📊 Real-time emotion analysis running...")
        print("💾 Updating emotion_fer.json every 1 second")
        print("👉 Press 'Q' to quit\n")
        print("-"*60)
        
        fps_start = time.time()
        analysis_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            current_time = time.time()
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
            
            display_frame = frame.copy()
            
            # Process faces
            if current_time - self.last_save_time >= UPDATE_INTERVAL:
                if len(faces) > 0:
                    # Get largest face
                    (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
                    face_roi = frame[y:y+h, x:x+w]
                    
                    # Predict
                    emotion, confidence, all_emotions = self.predict_emotion(face_roi)
                    
                    if emotion is not None:
                        self.save_json(emotion, confidence, all_emotions)
                        self.last_emotion = emotion
                        self.last_confidence = confidence
                        analysis_count += 1
                        
                        print(f"💾 Frame {self.frame_count:5d} | {emotion:10s} {confidence*100:5.1f}% | Faces: {len(faces)}")
                else:
                    print(f"👤 Frame {self.frame_count:5d} | No face detected")
                
                self.last_save_time = current_time
            
            # Draw face rectangles
            for (x, y, w, h) in faces:
                color = (0, 255, 0) if self.last_confidence > 0.6 else (0, 165, 255)
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 3)
                
                label = f"{self.last_emotion}: {self.last_confidence*100:.0f}%"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                
                cv2.rectangle(display_frame, (x, y-40), (x+label_size[0]+10, y), color, -1)
                cv2.putText(display_frame, label, (x+5, y-12),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            # Calculate FPS
            if self.frame_count % 30 == 0:
                self.fps = 30 / (time.time() - fps_start)
                fps_start = time.time()
            
            # Display HUD
            cv2.putText(display_frame, f"FPS: {self.fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Faces: {len(faces)}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Analyses: {analysis_count}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display_frame, "FER-2013 Model", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow('FER Model - Emotion Detection (Press Q to Quit)', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("✅ INFERENCE STOPPED")
        print("="*60)
        print(f"📊 Total frames: {self.frame_count}")
        print(f"🎯 Total analyses: {analysis_count}")
        print(f"📁 Output: {OUTPUT_JSON}")
        print("="*60 + "\n")

def main():
    print("\n" + "="*60)
    print("  CUSTOM FER-2013 EMOTION DETECTOR")
    print("="*60 + "\n")
    
    if not TF_AVAILABLE:
        print("❌ TensorFlow required. Install with: pip install tensorflow")
        return
    
    detector = FEREmotionDetector()
    detector.run()

if __name__ == "__main__":
    main()