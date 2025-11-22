"""
Task 3: Temporal Smoothing (EMA) + Spike Detection
- Implement EMA smoothing with tunable alpha
- Detect emotion spikes (ΔE) for frustration/stress buildup
- Measure std dev before/after smoothing
- Output ΔE values per frame
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

# Emotion coordinates (from Task 2)
EMOTION_COORDS = {
    'happy': (0.9, 0.6),
    'surprise': (0.6, 0.9),
    'neutral': (0.0, 0.0),
    'sad': (-0.8, -0.6),
    'angry': (-0.7, 0.7),
    'fear': (-0.8, 0.5),
    'disgust': (-0.7, -0.2)
}

class TemporalSmoothingAnalyzer:
    def __init__(self, alpha=0.7, spike_threshold=0.35):
        """
        Initialize temporal smoothing analyzer
        
        Args:
            alpha: EMA smoothing factor (0.6-0.8 recommended, default 0.7 from proposal)
            spike_threshold: Threshold for detecting emotion spikes (default 0.35 from proposal)
        """
        self.alpha = alpha
        self.spike_threshold = spike_threshold
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Tracking variables
        self.E_prev = None  # Previous raw affect vector
        self.E_bar_prev = None  # Previous smoothed affect vector
        
        # Storage for analysis
        self.raw_affect_history = []
        self.smoothed_affect_history = []
        self.delta_E_history = []
        self.spike_detected = []
        self.timestamps = []
        
        print(f"\n{'='*70}")
        print(f"  TASK 3: TEMPORAL SMOOTHING + SPIKE DETECTION")
        print(f"{'='*70}\n")
        print(f"Parameters:")
        print(f"  EMA Alpha (α):      {self.alpha}")
        print(f"  Spike Threshold (γ): {self.spike_threshold}")
        print(f"\nFormulas:")
        print(f"  E_bar(t) = α × E(t) + (1-α) × E_bar(t-1)")
        print(f"  ΔE = ||E(t) - E(t-1)||")
        print(f"  Spike detected if ΔE > {self.spike_threshold}\n")
    
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
            probs = {k: v/100.0 for k, v in emotions.items()}
            
            return probs
            
        except Exception as e:
            return None
    
    def compute_affect_vector(self, emotion_probs):
        """Map emotion probabilities to valence-arousal space"""
        valence = 0.0
        arousal = 0.0
        
        for emotion, prob in emotion_probs.items():
            if emotion in EMOTION_COORDS:
                v, a = EMOTION_COORDS[emotion]
                valence += prob * v
                arousal += prob * a
        
        return np.array([valence, arousal])
    
    def apply_ema_smoothing(self, E_current):
        """
        Apply Exponential Moving Average smoothing
        
        Formula: E_bar(t) = α × E(t) + (1-α) × E_bar(t-1)
        
        Args:
            E_current: Current affect vector [valence, arousal]
        
        Returns:
            E_bar: Smoothed affect vector
        """
        if self.E_bar_prev is None:
            # Initialize with first value
            E_bar = E_current
        else:
            # Apply EMA formula
            E_bar = self.alpha * E_current + (1 - self.alpha) * self.E_bar_prev
        
        self.E_bar_prev = E_bar
        return E_bar
    
    def compute_delta_E(self, E_current):
        """
        Compute emotion change magnitude (spike detection)
        
        Formula: ΔE = ||E(t) - E(t-1)||
        
        Args:
            E_current: Current affect vector
        
        Returns:
            delta_E: Magnitude of emotion change
            is_spike: Boolean indicating if spike detected
        """
        if self.E_prev is None:
            delta_E = 0.0
            is_spike = False
        else:
            # Compute Euclidean distance
            delta_E = np.linalg.norm(E_current - self.E_prev)
            
            # Detect spike
            is_spike = delta_E > self.spike_threshold
        
        self.E_prev = E_current.copy()
        return delta_E, is_spike
    
    def collect_temporal_data(self, duration=40, fps_target=10):
        """Collect temporal affect data with smoothing and spike detection"""
        print(f"📸 Collecting temporal data for {duration}s")
        print("   Try sudden expression changes to trigger spikes!")
        print("   Press Q to quit early\n")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return False
        
        start_time = time.time()
        frame_interval = 1.0 / fps_target
        last_process_time = 0
        sample_count = 0
        spike_count = 0
        
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
                    
                    # Get emotion probabilities
                    probs = self.predict_emotion_probs(frame)
                    
                    if probs is not None:
                        # Compute raw affect vector
                        E_current = self.compute_affect_vector(probs)
                        
                        # Apply EMA smoothing
                        E_smoothed = self.apply_ema_smoothing(E_current)
                        
                        # Compute delta E (spike detection)
                        delta_E, is_spike = self.compute_delta_E(E_current)
                        
                        # Store data
                        self.raw_affect_history.append(E_current.copy())
                        self.smoothed_affect_history.append(E_smoothed.copy())
                        self.delta_E_history.append(delta_E)
                        self.spike_detected.append(is_spike)
                        self.timestamps.append(current_time - start_time)
                        
                        if is_spike:
                            spike_count += 1
                        
                        sample_count += 1
                        last_process_time = current_time
                        
                        # Determine box color based on spike
                        box_color = (0, 0, 255) if is_spike else (0, 255, 0)
                        thickness = 4 if is_spike else 2
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), box_color, thickness)
                        
                        # Display info
                        cv2.putText(display_frame, f"Raw V: {E_current[0]:+.2f}, A: {E_current[1]:+.2f}", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(display_frame, f"Smooth V: {E_smoothed[0]:+.2f}, A: {E_smoothed[1]:+.2f}", 
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(display_frame, f"ΔE: {delta_E:.3f}", 
                                  (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        
                        if is_spike:
                            cv2.putText(display_frame, "⚠️ SPIKE DETECTED!", 
                                      (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                        
                        cv2.putText(display_frame, f"Samples: {sample_count} | Spikes: {spike_count}", 
                                  (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Console output
                        spike_marker = "🔥 SPIKE" if is_spike else ""
                        print(f"  Sample {sample_count:3d} | "
                              f"ΔE: {delta_E:.3f} | "
                              f"Raw: ({E_current[0]:+.2f}, {E_current[1]:+.2f}) | "
                              f"Smooth: ({E_smoothed[0]:+.2f}, {E_smoothed[1]:+.2f}) {spike_marker}")
            
            # Display elapsed time
            elapsed = current_time - start_time
            cv2.putText(display_frame, f"Time: {elapsed:.1f}s / {duration}s", 
                       (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Task 3: Temporal Smoothing + Spike Detection', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n✅ Collected {sample_count} samples with {spike_count} spikes detected")
        return sample_count > 0
    
    def analyze_results(self):
        """Analyze smoothing effectiveness and spike patterns"""
        print(f"\n{'='*70}")
        print(f"  ANALYSIS RESULTS")
        print(f"{'='*70}\n")
        
        if len(self.raw_affect_history) == 0:
            print("❌ No data to analyze")
            return
        
        # Convert to numpy arrays
        raw_affect = np.array(self.raw_affect_history)
        smoothed_affect = np.array(self.smoothed_affect_history)
        delta_E = np.array(self.delta_E_history)
        
        # Calculate standard deviations
        raw_std_valence = np.std(raw_affect[:, 0])
        raw_std_arousal = np.std(raw_affect[:, 1])
        smoothed_std_valence = np.std(smoothed_affect[:, 0])
        smoothed_std_arousal = np.std(smoothed_affect[:, 1])
        
        # Calculate overall std (Euclidean)
        raw_std_overall = np.sqrt(raw_std_valence**2 + raw_std_arousal**2)
        smoothed_std_overall = np.sqrt(smoothed_std_valence**2 + smoothed_std_arousal**2)
        
        std_reduction = (raw_std_overall - smoothed_std_overall) / raw_std_overall * 100
        
        # Spike statistics
        spike_count = sum(self.spike_detected)
        spike_rate = spike_count / len(self.spike_detected) * 100
        avg_delta_E = np.mean(delta_E)
        max_delta_E = np.max(delta_E)
        
        print("STANDARD DEVIATION (Before/After Smoothing):")
        print(f"  Raw Affect:")
        print(f"    Valence Std:  {raw_std_valence:.4f}")
        print(f"    Arousal Std:  {raw_std_arousal:.4f}")
        print(f"    Overall Std:  {raw_std_overall:.4f}")
        print(f"  Smoothed Affect:")
        print(f"    Valence Std:  {smoothed_std_valence:.4f}")
        print(f"    Arousal Std:  {smoothed_std_arousal:.4f}")
        print(f"    Overall Std:  {smoothed_std_overall:.4f}")
        print(f"  Reduction:      {std_reduction:.2f}%")
        print()
        
        print("SPIKE DETECTION (ΔE):")
        print(f"  Total Spikes:     {spike_count}")
        print(f"  Spike Rate:       {spike_rate:.2f}%")
        print(f"  Avg ΔE:           {avg_delta_E:.4f}")
        print(f"  Max ΔE:           {max_delta_E:.4f}")
        print(f"  Spike Threshold:  {self.spike_threshold}")
        print()
        
        # Find spike timestamps
        spike_times = [self.timestamps[i] for i, spike in enumerate(self.spike_detected) if spike]
        if spike_times:
            print(f"  Spike Timestamps: {[f'{t:.2f}s' for t in spike_times[:5]]}{'...' if len(spike_times) > 5 else ''}")
        print()
        
        print(f"{'='*70}\n")
        
        # Create visualizations
        self.create_visualizations(raw_affect, smoothed_affect, delta_E)
        
        # Save results
        self.save_results(raw_std_valence, raw_std_arousal, raw_std_overall,
                         smoothed_std_valence, smoothed_std_arousal, smoothed_std_overall,
                         std_reduction, spike_count, spike_rate, avg_delta_E, max_delta_E)
    
    def create_visualizations(self, raw_affect, smoothed_affect, delta_E):
        """Create visualization plots"""
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Raw vs Smoothed Valence
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(self.timestamps, raw_affect[:, 0], 
                'b-', alpha=0.4, linewidth=1, label='Raw')
        ax1.plot(self.timestamps, smoothed_affect[:, 0], 
                'r-', linewidth=2, label=f'Smoothed (α={self.alpha})')
        ax1.set_xlabel('Time (s)', fontweight='bold')
        ax1.set_ylabel('Valence', fontweight='bold')
        ax1.set_title('Valence: Raw vs Smoothed', fontweight='bold')
        ax1.grid(alpha=0.3)
        ax1.axhline(y=0, color='k', linewidth=0.5)
        ax1.legend()
        
        # 2. Raw vs Smoothed Arousal
        ax2 = plt.subplot(3, 2, 2)
        ax2.plot(self.timestamps, raw_affect[:, 1], 
                'b-', alpha=0.4, linewidth=1, label='Raw')
        ax2.plot(self.timestamps, smoothed_affect[:, 1], 
                'r-', linewidth=2, label=f'Smoothed (α={self.alpha})')
        ax2.set_xlabel('Time (s)', fontweight='bold')
        ax2.set_ylabel('Arousal', fontweight='bold')
        ax2.set_title('Arousal: Raw vs Smoothed', fontweight='bold')
        ax2.grid(alpha=0.3)
        ax2.axhline(y=0, color='k', linewidth=0.5)
        ax2.legend()
        
        # 3. Delta E over time with spike markers
        ax3 = plt.subplot(3, 2, 3)
        ax3.plot(self.timestamps, delta_E, 'b-', linewidth=1.5, label='ΔE')
        ax3.axhline(y=self.spike_threshold, color='r', linestyle='--', 
                   linewidth=2, label=f'Spike Threshold ({self.spike_threshold})')
        
        # Mark spikes
        spike_times = [self.timestamps[i] for i, spike in enumerate(self.spike_detected) if spike]
        spike_values = [delta_E[i] for i, spike in enumerate(self.spike_detected) if spike]
        ax3.scatter(spike_times, spike_values, color='red', s=100, 
                   marker='x', linewidths=3, label='Detected Spikes', zorder=5)
        
        ax3.set_xlabel('Time (s)', fontweight='bold')
        ax3.set_ylabel('ΔE (Emotion Change)', fontweight='bold')
        ax3.set_title('Delta E (Spike Detection)', fontweight='bold')
        ax3.grid(alpha=0.3)
        ax3.legend()
        
        # 4. Standard Deviation Comparison
        ax4 = plt.subplot(3, 2, 4)
        raw_stds = [np.std(raw_affect[:, 0]), np.std(raw_affect[:, 1])]
        smoothed_stds = [np.std(smoothed_affect[:, 0]), np.std(smoothed_affect[:, 1])]
        
        x = np.arange(2)
        width = 0.35
        ax4.bar(x - width/2, raw_stds, width, label='Raw', color='skyblue', edgecolor='black')
        ax4.bar(x + width/2, smoothed_stds, width, label='Smoothed', color='salmon', edgecolor='black')
        ax4.set_ylabel('Standard Deviation', fontweight='bold')
        ax4.set_title('Std Dev Reduction', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(['Valence', 'Arousal'])
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
        # Add reduction percentages
        for i, (raw, smooth) in enumerate(zip(raw_stds, smoothed_stds)):
            reduction = (raw - smooth) / raw * 100
            ax4.text(i, max(raw, smooth) + 0.01, f'-{reduction:.1f}%', 
                    ha='center', fontweight='bold', color='green')
        
        # 5. Histogram of Delta E
        ax5 = plt.subplot(3, 2, 5)
        ax5.hist(delta_E, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax5.axvline(self.spike_threshold, color='r', linestyle='--', 
                   linewidth=2, label=f'Threshold ({self.spike_threshold})')
        ax5.axvline(np.mean(delta_E), color='g', linestyle='--', 
                   linewidth=2, label=f'Mean ({np.mean(delta_E):.3f})')
        ax5.set_xlabel('ΔE Value', fontweight='bold')
        ax5.set_ylabel('Frequency', fontweight='bold')
        ax5.set_title('Delta E Distribution', fontweight='bold')
        ax5.legend()
        ax5.grid(alpha=0.3)
        
        # 6. 2D Trajectory with spikes
        ax6 = plt.subplot(3, 2, 6)
        
        # Plot smoothed trajectory
        ax6.plot(smoothed_affect[:, 0], smoothed_affect[:, 1], 
                'b-', linewidth=2, alpha=0.6, label='Trajectory')
        
        # Mark spike locations
        spike_indices = [i for i, spike in enumerate(self.spike_detected) if spike]
        if spike_indices:
            spike_points = smoothed_affect[spike_indices]
            ax6.scatter(spike_points[:, 0], spike_points[:, 1], 
                       color='red', s=200, marker='*', 
                       edgecolors='black', linewidths=2, 
                       label='Spikes', zorder=5)
        
        # Mark start and end
        ax6.scatter(smoothed_affect[0, 0], smoothed_affect[0, 1], 
                   s=150, c='green', marker='o', label='Start', zorder=5)
        ax6.scatter(smoothed_affect[-1, 0], smoothed_affect[-1, 1], 
                   s=150, c='purple', marker='s', label='End', zorder=5)
        
        ax6.set_xlabel('Valence', fontweight='bold')
        ax6.set_ylabel('Arousal', fontweight='bold')
        ax6.set_title('Affect Trajectory with Spike Markers', fontweight='bold')
        ax6.grid(alpha=0.3)
        ax6.axhline(y=0, color='k', linewidth=0.5)
        ax6.axvline(x=0, color='k', linewidth=0.5)
        ax6.set_xlim(-1, 1)
        ax6.set_ylim(-1, 1)
        ax6.legend()
        
        plt.tight_layout()
        
        output_path = os.path.join(OUTPUT_DIR, 'task3_temporal_smoothing.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualizations saved: {output_path}")
        plt.close()
    
    def save_results(self, raw_std_val, raw_std_aro, raw_std_overall,
                    smooth_std_val, smooth_std_aro, smooth_std_overall,
                    std_reduction, spike_count, spike_rate, avg_delta_E, max_delta_E):
        """Save results to JSON"""
        results = {
            "task": "Task 3: Temporal Smoothing + Spike Detection",
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "ema_alpha": self.alpha,
                "spike_threshold": self.spike_threshold
            },
            "total_samples": len(self.raw_affect_history),
            "duration_seconds": self.timestamps[-1] if self.timestamps else 0,
            "standard_deviation_before_smoothing": {
                "valence": round(raw_std_val, 4),
                "arousal": round(raw_std_aro, 4),
                "overall": round(raw_std_overall, 4)
            },
            "standard_deviation_after_smoothing": {
                "valence": round(smooth_std_val, 4),
                "arousal": round(smooth_std_aro, 4),
                "overall": round(smooth_std_overall, 4)
            },
            "std_reduction_percent": round(std_reduction, 2),
            "spike_detection": {
                "total_spikes": int(spike_count),
                "spike_rate_percent": round(spike_rate, 2),
                "average_delta_E": round(avg_delta_E, 4),
                "max_delta_E": round(max_delta_E, 4),
                "threshold": self.spike_threshold
            },
            "delta_E_values": [round(float(d), 4) for d in self.delta_E_history]
        }
        
        json_path = os.path.join(OUTPUT_DIR, 'task3_metrics.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results JSON saved: {json_path}")
    
    def run(self):
        """Run full Task 3 pipeline"""
        if not DEEPFACE_AVAILABLE:
            print("❌ DeepFace required for Task 3")
            return
        
        # Collect data
        success = self.collect_temporal_data(duration=40, fps_target=10)
        
        if not success:
            print("❌ Data collection failed")
            return
        
        # Analyze results
        self.analyze_results()
        
        print(f"\n{'='*70}")
        print(f"  ✅ TASK 3 COMPLETE")
        print(f"{'='*70}\n")
        print("Outputs for Ray:")
        print(f"  • Standard deviation before/after smoothing: ✅")
        print(f"  • ΔE values per frame: ✅")
        print(f"  • Spike detection with timestamps: ✅")
        print(f"  • EMA smoothing effectiveness: ✅")
        print(f"  • Visualizations: cv/output/task3_temporal_smoothing.png")
        print(f"  • Metrics JSON: cv/output/task3_metrics.json")
        print()

def main():
    # Use alpha=0.7 and gamma=0.35 from proposal
    analyzer = TemporalSmoothingAnalyzer(alpha=0.7, spike_threshold=0.35)
    analyzer.run()

if __name__ == "__main__":
    main()