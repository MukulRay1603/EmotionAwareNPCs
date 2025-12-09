"""
Task 2: Continuous Affect Vector
- Map softmax probabilities to 2D valence-arousal space
- Compare raw vs smoothed affect trajectories
- Measure variance reduction
"""

import cv2
import numpy as np
import time
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from collections import deque
import warnings
warnings.filterwarnings('ignore')

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except:
    DEEPFACE_AVAILABLE = False
    print("❌ DeepFace not available")

# Configuration
OUTPUT_DIR = 'cv/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Emotion coordinates in valence-arousal space (from proposal)
EMOTION_COORDS = {
    'happy': (0.9, 0.6),
    'surprise': (0.6, 0.9),
    'neutral': (0.0, 0.0),
    'sad': (-0.8, -0.6),
    'angry': (-0.7, 0.7),
    'fear': (-0.8, 0.5),
    'disgust': (-0.7, -0.2)
}

class ContinuousAffectMapper:
    def __init__(self):
        """Initialize continuous affect mapper"""
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Storage for analysis
        self.raw_affect_history = []
        self.smoothed_affect_history = []
        self.emotion_probs_history = []
        self.timestamps = []
        
        print(f"\n{'='*70}")
        print(f"  TASK 2: CONTINUOUS AFFECT VECTOR")
        print(f"{'='*70}\n")
        print("Emotion Coordinates (Valence, Arousal):")
        for emotion, (v, a) in EMOTION_COORDS.items():
            print(f"  {emotion:10s}: ({v:+.1f}, {a:+.1f})")
        print()
    
    def detect_face(self, frame):
        """Detect largest face"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )
        if len(faces) > 0:
            return max(faces, key=lambda f: f[2] * f[3])
        return None
    
    def predict_emotion_probs(self, frame):
        """Get emotion probabilities from DeepFace"""
        if not DEEPFACE_AVAILABLE:
            return None
        
        try:
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            if isinstance(result, list):
                result = result[0]
            
            emotions = result['emotion']
            
            # Normalize to probabilities (0-1 range)
            probs = {k: v/100.0 for k, v in emotions.items()}
            
            return probs
            
        except Exception as e:
            return None
    
    def compute_affect_vector(self, emotion_probs):
        """
        Map emotion probabilities to continuous valence-arousal space
        Formula: E(t) = Σ p_i(t) × e_i
        
        Args:
            emotion_probs: dict of {emotion: probability}
        
        Returns:
            (valence, arousal) tuple
        """
        valence = 0.0
        arousal = 0.0
        
        for emotion, prob in emotion_probs.items():
            if emotion in EMOTION_COORDS:
                v, a = EMOTION_COORDS[emotion]
                valence += prob * v
                arousal += prob * a
        
        return valence, arousal
    
    def collect_affect_data(self, duration=30, fps_target=10):
        """Collect affect vectors over time"""
        print(f"📸 Collecting affect data for {duration}s")
        print("   Make different expressions to see affect trajectory!")
        print("   Press Q to quit early\n")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return False
        
        start_time = time.time()
        frame_interval = 1.0 / fps_target
        last_process_time = 0
        sample_count = 0
        
        # For smoothing comparison
        smoothed_valence = 0.0
        smoothed_arousal = 0.0
        alpha = 0.3  # Smoothing factor (lower = more smoothing)
        
        while (time.time() - start_time) < duration:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = time.time()
            display_frame = frame.copy()
            
            # Process at target FPS
            if current_time - last_process_time >= frame_interval:
                face = self.detect_face(frame)
                
                if face is not None:
                    x, y, w, h = face
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Get emotion probabilities
                    probs = self.predict_emotion_probs(frame)
                    
                    if probs is not None:
                        # Compute raw affect vector
                        valence, arousal = self.compute_affect_vector(probs)
                        
                        # Apply exponential smoothing for comparison
                        if sample_count == 0:
                            smoothed_valence = valence
                            smoothed_arousal = arousal
                        else:
                            smoothed_valence = alpha * valence + (1 - alpha) * smoothed_valence
                            smoothed_arousal = alpha * arousal + (1 - alpha) * smoothed_arousal
                        
                        # Store data
                        self.raw_affect_history.append((valence, arousal))
                        self.smoothed_affect_history.append((smoothed_valence, smoothed_arousal))
                        self.emotion_probs_history.append(probs.copy())
                        self.timestamps.append(current_time - start_time)
                        
                        sample_count += 1
                        last_process_time = current_time
                        
                        # Display on frame
                        cv2.putText(display_frame, f"Valence: {valence:+.2f}", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(display_frame, f"Arousal: {arousal:+.2f}", 
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(display_frame, f"Samples: {sample_count}", 
                                  (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
                        # Get dominant emotion for display
                        dominant = max(probs.items(), key=lambda x: x[1])[0]
                        cv2.putText(display_frame, f"Emotion: {dominant}", 
                                  (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
                        print(f"  Sample {sample_count:3d} | "
                              f"V: {valence:+.2f}, A: {arousal:+.2f} | "
                              f"Smoothed V: {smoothed_valence:+.2f}, A: {smoothed_arousal:+.2f}")
            
            # Display info
            elapsed = current_time - start_time
            cv2.putText(display_frame, f"Time: {elapsed:.1f}s / {duration}s", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Task 2: Continuous Affect Mapping', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n✅ Collected {sample_count} affect samples")
        return sample_count > 0
    
    def analyze_results(self):
        """Analyze and compare raw vs smoothed affect"""
        print(f"\n{'='*70}")
        print(f"  ANALYSIS RESULTS")
        print(f"{'='*70}\n")
        
        if len(self.raw_affect_history) == 0:
            print("❌ No data to analyze")
            return
        
        # Convert to numpy arrays
        raw_affect = np.array(self.raw_affect_history)
        smoothed_affect = np.array(self.smoothed_affect_history)
        
        # Calculate statistics
        raw_valence = raw_affect[:, 0]
        raw_arousal = raw_affect[:, 1]
        smoothed_valence = smoothed_affect[:, 0]
        smoothed_arousal = smoothed_affect[:, 1]
        
        # Variance comparison
        raw_val_var = np.var(raw_valence)
        raw_aro_var = np.var(raw_arousal)
        smoothed_val_var = np.var(smoothed_valence)
        smoothed_aro_var = np.var(smoothed_arousal)
        
        val_var_reduction = (raw_val_var - smoothed_val_var) / raw_val_var * 100
        aro_var_reduction = (raw_aro_var - smoothed_aro_var) / raw_aro_var * 100
        
        print("RAW AFFECT STATISTICS:")
        print(f"  Valence:")
        print(f"    Mean:     {np.mean(raw_valence):+.3f}")
        print(f"    Std Dev:  {np.std(raw_valence):.3f}")
        print(f"    Variance: {raw_val_var:.3f}")
        print(f"    Range:    [{np.min(raw_valence):+.2f}, {np.max(raw_valence):+.2f}]")
        print(f"  Arousal:")
        print(f"    Mean:     {np.mean(raw_arousal):+.3f}")
        print(f"    Std Dev:  {np.std(raw_arousal):.3f}")
        print(f"    Variance: {raw_aro_var:.3f}")
        print(f"    Range:    [{np.min(raw_arousal):+.2f}, {np.max(raw_arousal):+.2f}]")
        print()
        
        print("SMOOTHED AFFECT STATISTICS:")
        print(f"  Valence:")
        print(f"    Mean:     {np.mean(smoothed_valence):+.3f}")
        print(f"    Std Dev:  {np.std(smoothed_valence):.3f}")
        print(f"    Variance: {smoothed_val_var:.3f}")
        print(f"    Reduction: {val_var_reduction:.1f}%")
        print(f"  Arousal:")
        print(f"    Mean:     {np.mean(smoothed_arousal):+.3f}")
        print(f"    Std Dev:  {np.std(smoothed_arousal):.3f}")
        print(f"    Variance: {smoothed_aro_var:.3f}")
        print(f"    Reduction: {aro_var_reduction:.1f}%")
        print()
        
        print(f"{'='*70}\n")
        
        # Create visualizations
        self.create_visualizations(raw_affect, smoothed_affect)
        
        # Save results
        self.save_results(raw_val_var, raw_aro_var, smoothed_val_var, 
                         smoothed_aro_var, val_var_reduction, aro_var_reduction)
    
    def create_visualizations(self, raw_affect, smoothed_affect):
        """Create visualization plots"""
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Valence-Arousal Trajectory (2D space)
        ax1 = plt.subplot(2, 3, 1)
        
        # Plot emotion coordinates
        for emotion, (v, a) in EMOTION_COORDS.items():
            ax1.scatter(v, a, s=200, alpha=0.3, label=emotion)
            ax1.text(v, a, emotion, fontsize=8, ha='center', va='center')
        
        # Plot trajectory
        ax1.plot(raw_affect[:, 0], raw_affect[:, 1], 
                'b-', alpha=0.3, linewidth=1, label='Raw trajectory')
        ax1.plot(smoothed_affect[:, 0], smoothed_affect[:, 1], 
                'r-', linewidth=2, label='Smoothed trajectory')
        
        # Mark start and end
        ax1.scatter(raw_affect[0, 0], raw_affect[0, 1], 
                   s=100, c='green', marker='o', label='Start', zorder=5)
        ax1.scatter(raw_affect[-1, 0], raw_affect[-1, 1], 
                   s=100, c='red', marker='x', label='End', zorder=5)
        
        ax1.set_xlabel('Valence', fontweight='bold')
        ax1.set_ylabel('Arousal', fontweight='bold')
        ax1.set_title('Affect Trajectory in Valence-Arousal Space', fontweight='bold')
        ax1.grid(alpha=0.3)
        ax1.axhline(y=0, color='k', linewidth=0.5)
        ax1.axvline(x=0, color='k', linewidth=0.5)
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-1, 1)
        ax1.legend(fontsize=7, loc='upper right')
        
        # 2. Valence over time
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(self.timestamps, raw_affect[:, 0], 
                'b-', alpha=0.5, linewidth=1, label='Raw')
        ax2.plot(self.timestamps, smoothed_affect[:, 0], 
                'r-', linewidth=2, label='Smoothed')
        ax2.set_xlabel('Time (s)', fontweight='bold')
        ax2.set_ylabel('Valence', fontweight='bold')
        ax2.set_title('Valence Over Time', fontweight='bold')
        ax2.grid(alpha=0.3)
        ax2.axhline(y=0, color='k', linewidth=0.5)
        ax2.legend()
        
        # 3. Arousal over time
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(self.timestamps, raw_affect[:, 1], 
                'b-', alpha=0.5, linewidth=1, label='Raw')
        ax3.plot(self.timestamps, smoothed_affect[:, 1], 
                'r-', linewidth=2, label='Smoothed')
        ax3.set_xlabel('Time (s)', fontweight='bold')
        ax3.set_ylabel('Arousal', fontweight='bold')
        ax3.set_title('Arousal Over Time', fontweight='bold')
        ax3.grid(alpha=0.3)
        ax3.axhline(y=0, color='k', linewidth=0.5)
        ax3.legend()
        
        # 4. Variance comparison
        ax4 = plt.subplot(2, 3, 4)
        raw_vars = [np.var(raw_affect[:, 0]), np.var(raw_affect[:, 1])]
        smoothed_vars = [np.var(smoothed_affect[:, 0]), np.var(smoothed_affect[:, 1])]
        
        x = np.arange(2)
        width = 0.35
        ax4.bar(x - width/2, raw_vars, width, label='Raw', color='skyblue', edgecolor='black')
        ax4.bar(x + width/2, smoothed_vars, width, label='Smoothed', color='salmon', edgecolor='black')
        ax4.set_ylabel('Variance', fontweight='bold')
        ax4.set_title('Variance Comparison', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(['Valence', 'Arousal'])
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
        # 5. Emotion probability heatmap (sample)
        ax5 = plt.subplot(2, 3, 5)
        if len(self.emotion_probs_history) > 0:
            # Create matrix of emotion probabilities over time
            emotions_list = list(EMOTION_COORDS.keys())
            prob_matrix = np.array([
                [probs.get(emotion, 0) for emotion in emotions_list]
                for probs in self.emotion_probs_history
            ]).T
            
            im = ax5.imshow(prob_matrix, aspect='auto', cmap='hot', interpolation='nearest')
            ax5.set_yticks(range(len(emotions_list)))
            ax5.set_yticklabels(emotions_list)
            ax5.set_xlabel('Time Sample', fontweight='bold')
            ax5.set_title('Emotion Probabilities Over Time', fontweight='bold')
            plt.colorbar(im, ax=ax5, label='Probability')
        
        # 6. Quadrant distribution
        ax6 = plt.subplot(2, 3, 6)
        quadrants = {
            'Happy (V+,A+)': 0,
            'Calm (V+,A-)': 0,
            'Sad (V-,A-)': 0,
            'Angry (V-,A+)': 0
        }
        
        for v, a in smoothed_affect:
            if v >= 0 and a >= 0:
                quadrants['Happy (V+,A+)'] += 1
            elif v >= 0 and a < 0:
                quadrants['Calm (V+,A-)'] += 1
            elif v < 0 and a < 0:
                quadrants['Sad (V-,A-)'] += 1
            else:
                quadrants['Angry (V-,A+)'] += 1
        
        ax6.bar(quadrants.keys(), quadrants.values(), color='lightgreen', edgecolor='black')
        ax6.set_ylabel('Count', fontweight='bold')
        ax6.set_title('Time Spent in Each Quadrant', fontweight='bold')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        output_path = os.path.join(OUTPUT_DIR, 'task2_affect_vector.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualizations saved: {output_path}")
        plt.close()
    
    def save_results(self, raw_val_var, raw_aro_var, smoothed_val_var, 
                    smoothed_aro_var, val_var_reduction, aro_var_reduction):
        """Save results to JSON"""
        results = {
            "task": "Task 2: Continuous Affect Vector",
            "timestamp": datetime.now().isoformat(),
            "emotion_coordinates": EMOTION_COORDS,
            "total_samples": len(self.raw_affect_history),
            "duration_seconds": self.timestamps[-1] if self.timestamps else 0,
            "raw_affect": {
                "valence_variance": round(raw_val_var, 4),
                "arousal_variance": round(raw_aro_var, 4),
                "valence_mean": round(np.mean([v for v, a in self.raw_affect_history]), 4),
                "arousal_mean": round(np.mean([a for v, a in self.raw_affect_history]), 4),
                "valence_std": round(np.std([v for v, a in self.raw_affect_history]), 4),
                "arousal_std": round(np.std([a for v, a in self.raw_affect_history]), 4)
            },
            "smoothed_affect": {
                "valence_variance": round(smoothed_val_var, 4),
                "arousal_variance": round(smoothed_aro_var, 4),
                "valence_mean": round(np.mean([v for v, a in self.smoothed_affect_history]), 4),
                "arousal_mean": round(np.mean([a for v, a in self.smoothed_affect_history]), 4),
                "valence_std": round(np.std([v for v, a in self.smoothed_affect_history]), 4),
                "arousal_std": round(np.std([a for v, a in self.smoothed_affect_history]), 4)
            },
            "variance_reduction": {
                "valence_percent": round(val_var_reduction, 2),
                "arousal_percent": round(aro_var_reduction, 2)
            }
        }
        
        json_path = os.path.join(OUTPUT_DIR, 'task2_metrics.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results JSON saved: {json_path}")
    
    def run(self):
        """Run full Task 2 pipeline"""
        if not DEEPFACE_AVAILABLE:
            print("❌ DeepFace required for Task 2")
            return
        
        # Collect data
        success = self.collect_affect_data(duration=30, fps_target=10)
        
        if not success:
            print("❌ Data collection failed")
            return
        
        # Analyze results
        self.analyze_results()
        
        print(f"\n{'='*70}")
        print(f"  ✅ TASK 2 COMPLETE")
        print(f"{'='*70}\n")
        print("Outputs for Ray:")
        print(f"  • Valence & arousal values: ✅")
        print(f"  • Smooth vs raw comparison: ✅")
        print(f"  • Variance reduction: ✅")
        print(f"  • Visualizations: cv/output/task2_affect_vector.png")
        print(f"  • Metrics JSON: cv/output/task2_metrics.json")
        print()

def main():
    mapper = ContinuousAffectMapper()
    mapper.run()

if __name__ == "__main__":
    main()