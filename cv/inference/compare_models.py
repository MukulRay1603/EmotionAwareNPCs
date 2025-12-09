"""
Compare DeepFace vs Custom FER Model Performance
Side-by-side comparison with metrics and visualizations
"""

import cv2
import json
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime

# Try imports
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except:
    DEEPFACE_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except:
    TF_AVAILABLE = False

OUTPUT_DIR = '../output'
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class ModelComparison:
    def __init__(self):
        """Initialize both models for comparison"""
        
        self.deepface_ready = DEEPFACE_AVAILABLE
        self.fer_ready = False
        self.fer_model = None
        
        # Load FER model if available
        if TF_AVAILABLE:
            fer_model_path = '../models/fer2013_mini_xception.h5'
            if os.path.exists(fer_model_path):
                try:
                    self.fer_model = keras.models.load_model(fer_model_path)
                    self.fer_ready = True
                    print("✅ FER model loaded")
                except:
                    print("⚠️  FER model failed to load")
        
        print(f"DeepFace: {'✅ Ready' if self.deepface_ready else '❌ Not available'}")
        print(f"FER Model: {'✅ Ready' if self.fer_ready else '❌ Not available'}")
    
    def predict_deepface(self, frame):
        """Get DeepFace prediction"""
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], 
                                     enforce_detection=False, silent=True)
            if isinstance(result, list):
                result = result[0]
            emotions = result['emotion']
            dominant = result['dominant_emotion']
            return dominant, emotions[dominant]/100.0, {k: v/100.0 for k, v in emotions.items()}
        except:
            return None, None, None
    
    def predict_fer(self, face_roi):
        """Get FER model prediction"""
        try:
            face = cv2.resize(face_roi, (48, 48))
            if len(face.shape) == 3:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = face.astype('float32') / 255.0
            face = np.expand_dims(face, axis=-1)
            face = np.expand_dims(face, axis=0)
            
            preds = self.fer_model.predict(face, verbose=0)[0]
            emotion_idx = np.argmax(preds)
            emotion = EMOTIONS[emotion_idx]
            confidence = float(preds[emotion_idx])
            all_emotions = {EMOTIONS[i]: float(preds[i]) for i in range(len(EMOTIONS))}
            
            return emotion, confidence, all_emotions
        except:
            return None, None, None
    
    def run_comparison(self, duration=30):
        """Run side-by-side comparison"""
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return
        
        print(f"\n🎥 Running {duration}s comparison...")
        print("Make different expressions!\n")
        
        results = {
            'deepface': {'predictions': [], 'times': [], 'matches': 0},
            'fer': {'predictions': [], 'times': [], 'matches': 0}
        }
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
            
            if len(faces) > 0:
                (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
                face_roi = frame[y:y+h, x:x+w]
                
                # DeepFace prediction
                if self.deepface_ready:
                    t1 = time.time()
                    df_emotion, df_conf, _ = self.predict_deepface(frame)
                    df_time = time.time() - t1
                    if df_emotion:
                        results['deepface']['predictions'].append(df_emotion)
                        results['deepface']['times'].append(df_time)
                
                # FER prediction
                if self.fer_ready:
                    t1 = time.time()
                    fer_emotion, fer_conf, _ = self.predict_fer(face_roi)
                    fer_time = time.time() - t1
                    if fer_emotion:
                        results['fer']['predictions'].append(fer_emotion)
                        results['fer']['times'].append(fer_time)
                
                # Check if both agree
                if df_emotion and fer_emotion:
                    if df_emotion == fer_emotion:
                        results['deepface']['matches'] += 1
                        results['fer']['matches'] += 1
            
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                print(f"  Progress: {elapsed:.1f}s / {duration}s")
        
        cap.release()
        
        # Generate comparison report
        self.generate_report(results, duration)
    
    def generate_report(self, results, duration):
        """Generate comparison report"""
        
        print("\n" + "="*70)
        print("  MODEL COMPARISON RESULTS")
        print("="*70)
        
        # Calculate statistics
        stats = {}
        for model_name in ['deepface', 'fer']:
            preds = results[model_name]['predictions']
            times = results[model_name]['times']
            
            if len(preds) > 0:
                stats[model_name] = {
                    'total_predictions': len(preds),
                    'avg_time': np.mean(times) * 1000,  # ms
                    'std_time': np.std(times) * 1000,
                    'fps': 1 / np.mean(times) if np.mean(times) > 0 else 0,
                    'emotion_dist': {e: preds.count(e) for e in set(preds)},
                    'matches': results[model_name]['matches']
                }
        
        # Print comparison
        print("\nDEEPFACE:")
        if 'deepface' in stats:
            s = stats['deepface']
            print(f"  Total Predictions: {s['total_predictions']}")
            print(f"  Avg Inference Time: {s['avg_time']:.1f}ms ± {s['std_time']:.1f}ms")
            print(f"  FPS: {s['fps']:.1f}")
            print(f"  Agreement with FER: {s['matches']}/{s['total_predictions']}")
        else:
            print("  Not available")
        
        print("\nCUSTOM FER MODEL:")
        if 'fer' in stats:
            s = stats['fer']
            print(f"  Total Predictions: {s['total_predictions']}")
            print(f"  Avg Inference Time: {s['avg_time']:.1f}ms ± {s['std_time']:.1f}ms")
            print(f"  FPS: {s['fps']:.1f}")
            print(f"  Agreement with DeepFace: {s['matches']}/{s['total_predictions']}")
        else:
            print("  Not available")
        
        print("="*70)
        
        # Create visualizations
        self.create_comparison_charts(stats)
        
        # Save JSON report
        self.save_comparison_json(stats, duration)
    
    def create_comparison_charts(self, stats):
        """Create comparison visualizations"""
        
        if not stats:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Inference Time Comparison
        ax = axes[0, 0]
        models = list(stats.keys())
        times = [stats[m]['avg_time'] for m in models]
        colors = ['#3498db', '#e74c3c']
        
        bars = ax.bar(models, times, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Inference Time (ms)', fontweight='bold')
        ax.set_title('Average Inference Speed', fontweight='bold', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        for bar, t in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2, t + 5,
                   f'{t:.1f}ms', ha='center', fontweight='bold')
        
        # 2. FPS Comparison
        ax = axes[0, 1]
        fps_vals = [stats[m]['fps'] for m in models]
        bars = ax.bar(models, fps_vals, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Frames Per Second', fontweight='bold')
        ax.set_title('Processing Speed (FPS)', fontweight='bold', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        for bar, fps in zip(bars, fps_vals):
            ax.text(bar.get_x() + bar.get_width()/2, fps + 0.5,
                   f'{fps:.1f} FPS', ha='center', fontweight='bold')
        
        # 3. Emotion Distribution - DeepFace
        if 'deepface' in stats:
            ax = axes[1, 0]
            emotion_dist = stats['deepface']['emotion_dist']
            ax.pie(emotion_dist.values(), labels=emotion_dist.keys(),
                  autopct='%1.1f%%', startangle=90)
            ax.set_title('DeepFace - Emotion Distribution', fontweight='bold', fontsize=12)
        
        # 4. Emotion Distribution - FER
        if 'fer' in stats:
            ax = axes[1, 1]
            emotion_dist = stats['fer']['emotion_dist']
            ax.pie(emotion_dist.values(), labels=emotion_dist.keys(),
                  autopct='%1.1f%%', startangle=90)
            ax.set_title('FER Model - Emotion Distribution', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        
        output_path = os.path.join(OUTPUT_DIR, 'model_comparison.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Comparison charts saved: {output_path}")
        plt.close()
    
    def save_comparison_json(self, stats, duration):
        """Save comparison data as JSON"""
        
        report = {
            "comparison_date": datetime.now().isoformat(),
            "duration_seconds": duration,
            "models_compared": list(stats.keys()),
            "statistics": stats,
            "conclusion": self.generate_conclusion(stats)
        }
        
        json_path = os.path.join(OUTPUT_DIR, 'model_comparison.json')
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Comparison JSON saved: {json_path}")
    
    def generate_conclusion(self, stats):
        """Generate comparison conclusion"""
        
        if len(stats) < 2:
            return "Incomplete comparison - need both models"
        
        deepface_fps = stats.get('deepface', {}).get('fps', 0)
        fer_fps = stats.get('fer', {}).get('fps', 0)
        
        faster = "FER" if fer_fps > deepface_fps else "DeepFace"
        speed_diff = abs(fer_fps - deepface_fps)
        
        return {
            "faster_model": faster,
            "speed_difference_fps": round(speed_diff, 1),
            "recommendation": "DeepFace for accuracy, FER for speed" if faster == "FER" else "DeepFace recommended for both accuracy and reasonable speed"
        }

def main():
    print("\n" + "="*70)
    print("  MODEL COMPARISON TOOL")
    print("  DeepFace vs Custom FER-2013 Model")
    print("="*70 + "\n")
    
    comparison = ModelComparison()
    
    if not comparison.deepface_ready and not comparison.fer_ready:
        print("❌ No models available for comparison")
        return
    
    print("\nStarting 30-second comparison...")
    print("Make different facial expressions during the test!\n")
    
    comparison.run_comparison(duration=30)
    
    print("\n✅ Comparison complete!")
    print("📊 Check output folder for results\n")

if __name__ == "__main__":
    main()