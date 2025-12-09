"""
Task 4: EAR (Eye Aspect Ratio) - Blink + Fatigue Detection
- Use MediaPipe face mesh for eye landmarks
- Compute EAR for both eyes
- Detect blinks and count blinks/min
- Detect fatigue from sustained low EAR
- Output EAR values, blink count, fatigue confidence
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
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except:
    MEDIAPIPE_AVAILABLE = False
    print("❌ MediaPipe not available")

# Configuration
OUTPUT_DIR = 'cv/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# EAR thresholds (from proposal)
EAR_THRESHOLD_BLINK = 0.20  # Below this = blink
EAR_THRESHOLD_FATIGUE = 0.22  # Sustained below this = fatigue
BLINK_CONSEC_FRAMES = 2  # Frames to confirm blink
FATIGUE_DURATION_SEC = 3.0  # Sustained low EAR for fatigue

# MediaPipe eye landmark indices
# Left eye: [362, 385, 387, 263, 373, 380]
# Right eye: [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

class EARFatigueDetector:
    def __init__(self):
        """Initialize EAR and fatigue detector"""
        
        if not MEDIAPIPE_AVAILABLE:
            print("❌ MediaPipe required for Task 4")
            return
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Tracking variables
        self.ear_history = []
        self.timestamps = []
        self.blink_times = []
        self.fatigue_periods = []
        
        # Blink detection state
        self.blink_counter = 0
        self.total_blinks = 0
        self.frames_below_threshold = 0
        
        # Fatigue detection state
        self.fatigue_start_time = None
        self.is_fatigued = False
        self.fatigue_confidence = 0.0
        
        print(f"\n{'='*70}")
        print(f"  TASK 4: EAR (BLINK + FATIGUE DETECTION)")
        print(f"{'='*70}\n")
        print("EAR Formula:")
        print("  EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)")
        print(f"\nThresholds:")
        print(f"  Blink Threshold:   {EAR_THRESHOLD_BLINK}")
        print(f"  Fatigue Threshold: {EAR_THRESHOLD_FATIGUE}")
        print(f"  Fatigue Duration:  {FATIGUE_DURATION_SEC}s")
        print()
    
    def compute_ear(self, eye_landmarks):
        """
        Compute Eye Aspect Ratio
        
        Formula from proposal:
        EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)
        
        Args:
            eye_landmarks: Array of 6 eye landmark points [p1, p2, p3, p4, p5, p6]
        
        Returns:
            ear: Eye Aspect Ratio value
        """
        # Vertical distances
        A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])  # ||p2-p6||
        B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])  # ||p3-p5||
        
        # Horizontal distance
        C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])  # ||p1-p4||
        
        # EAR calculation
        ear = (A + B) / (2.0 * C)
        
        return ear
    
    def extract_eye_landmarks(self, face_landmarks, image_shape):
        """Extract eye landmark coordinates from MediaPipe results"""
        h, w = image_shape[:2]
        
        # Get left eye landmarks
        left_eye = []
        for idx in LEFT_EYE_INDICES:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            left_eye.append(np.array([x, y]))
        
        # Get right eye landmarks
        right_eye = []
        for idx in RIGHT_EYE_INDICES:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            right_eye.append(np.array([x, y]))
        
        return np.array(left_eye), np.array(right_eye)
    
    def detect_blink(self, ear, current_time):
        """
        Detect blink based on EAR threshold
        
        Returns:
            is_blink: Boolean indicating if blink detected
        """
        is_blink = False
        
        if ear < EAR_THRESHOLD_BLINK:
            self.frames_below_threshold += 1
        else:
            if self.frames_below_threshold >= BLINK_CONSEC_FRAMES:
                # Blink detected!
                is_blink = True
                self.total_blinks += 1
                self.blink_times.append(current_time)
            
            self.frames_below_threshold = 0
        
        return is_blink
    
    def detect_fatigue(self, ear, current_time):
        """
        Detect fatigue based on sustained low EAR
        
        Returns:
            is_fatigued: Boolean indicating fatigue state
            fatigue_confidence: Confidence level (0-1)
        """
        if ear < EAR_THRESHOLD_FATIGUE:
            if self.fatigue_start_time is None:
                self.fatigue_start_time = current_time
            
            # Calculate how long EAR has been low
            fatigue_duration = current_time - self.fatigue_start_time
            
            # Confidence increases with duration
            self.fatigue_confidence = min(1.0, fatigue_duration / FATIGUE_DURATION_SEC)
            
            # Flag as fatigued if sustained
            if fatigue_duration >= FATIGUE_DURATION_SEC:
                self.is_fatigued = True
            else:
                self.is_fatigued = False
        else:
            # Reset fatigue detection
            self.fatigue_start_time = None
            self.is_fatigued = False
            self.fatigue_confidence = max(0.0, self.fatigue_confidence - 0.1)  # Decay
        
        return self.is_fatigued, self.fatigue_confidence
    
    def calculate_blinks_per_minute(self, current_time):
        """Calculate blinks per minute from recent history"""
        # Count blinks in last 60 seconds
        recent_blinks = [t for t in self.blink_times if current_time - t <= 60.0]
        return len(recent_blinks)
    
    def draw_eye_landmarks(self, frame, left_eye, right_eye):
        """Draw eye landmarks on frame for visualization"""
        # Draw left eye
        for point in left_eye:
            cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)
        
        # Draw right eye
        for point in right_eye:
            cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)
        
        # Draw eye contours
        if len(left_eye) == 6:
            pts = left_eye.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 1)
        
        if len(right_eye) == 6:
            pts = right_eye.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 1)
    
    def collect_ear_data(self, duration=40):
        """Collect EAR data with blink and fatigue detection"""
        print(f"📸 Collecting EAR data for {duration}s")
        print("   Instructions:")
        print("   - Blink normally to test blink detection")
        print("   - Keep eyes partially closed for 3s to trigger fatigue")
        print("   Press Q to quit early\n")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return False
        
        start_time = time.time()
        frame_count = 0
        
        while (time.time() - start_time) < duration:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            current_time = time.time() - start_time
            display_frame = frame.copy()
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # Extract eye landmarks
                left_eye, right_eye = self.extract_eye_landmarks(
                    face_landmarks, frame.shape
                )
                
                # Compute EAR for both eyes
                left_ear = self.compute_ear(left_eye)
                right_ear = self.compute_ear(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0
                
                # Detect blink
                is_blink = self.detect_blink(avg_ear, current_time)
                
                # Detect fatigue
                is_fatigued, fatigue_conf = self.detect_fatigue(avg_ear, current_time)
                
                # Calculate blinks per minute
                blinks_per_min = self.calculate_blinks_per_minute(current_time)
                
                # Store data
                self.ear_history.append({
                    'time': current_time,
                    'left_ear': left_ear,
                    'right_ear': right_ear,
                    'avg_ear': avg_ear,
                    'is_blink': is_blink,
                    'is_fatigued': is_fatigued,
                    'fatigue_confidence': fatigue_conf,
                    'blinks_per_min': blinks_per_min
                })
                self.timestamps.append(current_time)
                
                # Draw eye landmarks
                self.draw_eye_landmarks(display_frame, left_eye, right_eye)
                
                # Display EAR values
                color_ear = (0, 0, 255) if avg_ear < EAR_THRESHOLD_BLINK else (0, 255, 0)
                cv2.putText(display_frame, f"EAR: {avg_ear:.3f}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_ear, 2)
                cv2.putText(display_frame, f"Left: {left_ear:.3f} | Right: {right_ear:.3f}", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display blink info
                if is_blink:
                    cv2.putText(display_frame, "👁️ BLINK!", 
                              (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                cv2.putText(display_frame, f"Total Blinks: {self.total_blinks}", 
                          (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(display_frame, f"Blinks/min: {blinks_per_min}", 
                          (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Display fatigue warning
                if is_fatigued:
                    cv2.putText(display_frame, "⚠️ FATIGUE DETECTED!", 
                              (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                
                cv2.putText(display_frame, f"Fatigue: {fatigue_conf:.2f}", 
                          (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                          (0, 0, 255) if is_fatigued else (255, 255, 255), 2)
                
                # Console output (every 10 frames)
                if frame_count % 10 == 0:
                    blink_marker = "👁️ BLINK" if is_blink else ""
                    fatigue_marker = "💤 FATIGUE" if is_fatigued else ""
                    print(f"  Frame {frame_count:4d} | "
                          f"EAR: {avg_ear:.3f} | "
                          f"Blinks: {self.total_blinks:3d} ({blinks_per_min}/min) | "
                          f"Fatigue: {fatigue_conf:.2f} {blink_marker} {fatigue_marker}")
            else:
                cv2.putText(display_frame, "No face detected", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display elapsed time
            cv2.putText(display_frame, f"Time: {current_time:.1f}s / {duration}s", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
            
            cv2.imshow('Task 4: EAR - Blink + Fatigue Detection', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n✅ Collected {len(self.ear_history)} EAR samples")
        print(f"   Total blinks detected: {self.total_blinks}")
        return len(self.ear_history) > 0
    
    def analyze_results(self):
        """Analyze EAR and fatigue patterns"""
        print(f"\n{'='*70}")
        print(f"  ANALYSIS RESULTS")
        print(f"{'='*70}\n")
        
        if len(self.ear_history) == 0:
            print("❌ No data to analyze")
            return
        
        # Extract data
        ear_values = [d['avg_ear'] for d in self.ear_history]
        fatigue_values = [d['fatigue_confidence'] for d in self.ear_history]
        
        # EAR statistics
        avg_ear = np.mean(ear_values)
        std_ear = np.std(ear_values)
        min_ear = np.min(ear_values)
        max_ear = np.max(ear_values)
        
        # Blink statistics
        total_duration_min = self.timestamps[-1] / 60.0
        avg_blinks_per_min = self.total_blinks / total_duration_min if total_duration_min > 0 else 0
        
        # Fatigue statistics
        fatigue_frames = sum(1 for d in self.ear_history if d['is_fatigued'])
        fatigue_percentage = (fatigue_frames / len(self.ear_history)) * 100
        max_fatigue_conf = max(fatigue_values)
        
        print("EAR STATISTICS:")
        print(f"  Average EAR:        {avg_ear:.4f}")
        print(f"  Std Dev:            {std_ear:.4f}")
        print(f"  Min EAR:            {min_ear:.4f}")
        print(f"  Max EAR:            {max_ear:.4f}")
        print(f"  Blink Threshold:    {EAR_THRESHOLD_BLINK}")
        print()
        
        print("BLINK DETECTION:")
        print(f"  Total Blinks:       {self.total_blinks}")
        print(f"  Duration:           {total_duration_min:.2f} minutes")
        print(f"  Avg Blinks/min:     {avg_blinks_per_min:.1f}")
        print(f"  Normal Range:       10-20 blinks/min")
        print()
        
        print("FATIGUE DETECTION:")
        print(f"  Fatigue Frames:     {fatigue_frames} ({fatigue_percentage:.1f}%)")
        print(f"  Max Fatigue Conf:   {max_fatigue_conf:.3f}")
        print(f"  Fatigue Threshold:  {EAR_THRESHOLD_FATIGUE}")
        print(f"  Duration Required:  {FATIGUE_DURATION_SEC}s")
        print()
        
        print(f"{'='*70}\n")
        
        # Create visualizations
        self.create_visualizations(ear_values, fatigue_values)
        
        # Save results
        self.save_results(avg_ear, std_ear, min_ear, max_ear,
                         avg_blinks_per_min, fatigue_percentage, max_fatigue_conf)
    
    def create_visualizations(self, ear_values, fatigue_values):
        """Create visualization plots"""
        fig = plt.figure(figsize=(16, 10))
        
        # 1. EAR over time
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(self.timestamps, ear_values, 'b-', linewidth=1.5, label='EAR')
        ax1.axhline(y=EAR_THRESHOLD_BLINK, color='r', linestyle='--', 
                   linewidth=2, label=f'Blink Threshold ({EAR_THRESHOLD_BLINK})')
        ax1.axhline(y=EAR_THRESHOLD_FATIGUE, color='orange', linestyle='--', 
                   linewidth=2, label=f'Fatigue Threshold ({EAR_THRESHOLD_FATIGUE})')
        
        # Mark blinks
        for blink_time in self.blink_times:
            ax1.axvline(x=blink_time, color='red', alpha=0.3, linewidth=1)
        
        ax1.set_xlabel('Time (s)', fontweight='bold')
        ax1.set_ylabel('EAR Value', fontweight='bold')
        ax1.set_title('Eye Aspect Ratio Over Time', fontweight='bold')
        ax1.grid(alpha=0.3)
        ax1.legend()
        
        # 2. EAR histogram
        ax2 = plt.subplot(3, 2, 2)
        ax2.hist(ear_values, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax2.axvline(EAR_THRESHOLD_BLINK, color='r', linestyle='--', 
                   linewidth=2, label='Blink Threshold')
        ax2.axvline(EAR_THRESHOLD_FATIGUE, color='orange', linestyle='--', 
                   linewidth=2, label='Fatigue Threshold')
        ax2.axvline(np.mean(ear_values), color='g', linestyle='--', 
                   linewidth=2, label=f'Mean ({np.mean(ear_values):.3f})')
        ax2.set_xlabel('EAR Value', fontweight='bold')
        ax2.set_ylabel('Frequency', fontweight='bold')
        ax2.set_title('EAR Distribution', fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Fatigue confidence over time
        ax3 = plt.subplot(3, 2, 3)
        ax3.plot(self.timestamps, fatigue_values, 'r-', linewidth=2, label='Fatigue Confidence')
        ax3.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Full Fatigue')
        ax3.fill_between(self.timestamps, 0, fatigue_values, alpha=0.3, color='red')
        ax3.set_xlabel('Time (s)', fontweight='bold')
        ax3.set_ylabel('Fatigue Confidence', fontweight='bold')
        ax3.set_title('Fatigue Detection Over Time', fontweight='bold')
        ax3.set_ylim(0, 1.1)
        ax3.grid(alpha=0.3)
        ax3.legend()
        
        # 4. Blink events timeline
        ax4 = plt.subplot(3, 2, 4)
        if self.blink_times:
            ax4.scatter(self.blink_times, [1]*len(self.blink_times), 
                       s=100, c='blue', marker='|', linewidths=3)
            ax4.set_yticks([])
            ax4.set_xlabel('Time (s)', fontweight='bold')
            ax4.set_title(f'Blink Events (Total: {self.total_blinks})', fontweight='bold')
            ax4.grid(alpha=0.3, axis='x')
        else:
            ax4.text(0.5, 0.5, 'No blinks detected', 
                    ha='center', va='center', fontsize=12)
            ax4.set_xticks([])
            ax4.set_yticks([])
        
        # 5. Blinks per minute over time
        ax5 = plt.subplot(3, 2, 5)
        blinks_per_min_timeline = [d['blinks_per_min'] for d in self.ear_history]
        ax5.plot(self.timestamps, blinks_per_min_timeline, 'g-', linewidth=2)
        ax5.axhline(y=15, color='orange', linestyle='--', alpha=0.5, label='Normal (~15/min)')
        ax5.set_xlabel('Time (s)', fontweight='bold')
        ax5.set_ylabel('Blinks/Minute', fontweight='bold')
        ax5.set_title('Blink Rate Over Time', fontweight='bold')
        ax5.grid(alpha=0.3)
        ax5.legend()
        
        # 6. Summary statistics
        ax6 = plt.subplot(3, 2, 6)
        ax6.axis('off')
        
        summary_text = f"""
EAR & FATIGUE SUMMARY

EAR Statistics:
  • Average:     {np.mean(ear_values):.4f}
  • Std Dev:     {np.std(ear_values):.4f}
  • Range:       [{np.min(ear_values):.3f}, {np.max(ear_values):.3f}]

Blink Detection:
  • Total Blinks:     {self.total_blinks}
  • Avg Rate:         {self.total_blinks / (self.timestamps[-1]/60.0):.1f}/min
  • Normal Range:     10-20/min

Fatigue:
  • Max Confidence:   {max(fatigue_values):.3f}
  • Fatigue Time:     {sum(1 for d in self.ear_history if d['is_fatigued']) / len(self.ear_history) * 100:.1f}%
        """
        
        ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        
        output_path = os.path.join(OUTPUT_DIR, 'task4_ear_fatigue.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualizations saved: {output_path}")
        plt.close()
    
    def save_results(self, avg_ear, std_ear, min_ear, max_ear,
                    avg_blinks_per_min, fatigue_percentage, max_fatigue_conf):
        """Save results to JSON"""
        results = {
            "task": "Task 4: EAR (Blink + Fatigue Detection)",
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "ear_blink_threshold": EAR_THRESHOLD_BLINK,
                "ear_fatigue_threshold": EAR_THRESHOLD_FATIGUE,
                "fatigue_duration_sec": FATIGUE_DURATION_SEC
            },
            "total_samples": len(self.ear_history),
            "duration_seconds": self.timestamps[-1] if self.timestamps else 0,
            "ear_statistics": {
                "average": round(avg_ear, 4),
                "std_dev": round(std_ear, 4),
                "min": round(min_ear, 4),
                "max": round(max_ear, 4)
            },
            "blink_detection": {
                "total_blinks": self.total_blinks,
                "average_blinks_per_minute": round(avg_blinks_per_min, 2),
                "blink_times": [round(t, 2) for t in self.blink_times]
            },
            "fatigue_detection": {
                "fatigue_percentage": round(fatigue_percentage, 2),
                "max_fatigue_confidence": round(max_fatigue_conf, 3)
            },
            "ear_values_per_frame": [round(d['avg_ear'], 4) for d in self.ear_history]
        }
        
        json_path = os.path.join(OUTPUT_DIR, 'task4_metrics.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results JSON saved: {json_path}")
    
    def run(self):
        """Run full Task 4 pipeline"""
        if not MEDIAPIPE_AVAILABLE:
            print("❌ MediaPipe required for Task 4")
            return
        
        # Collect data
        success = self.collect_ear_data(duration=40)
        
        if not success:
            print("❌ Data collection failed")
            return
        
        # Analyze results
        self.analyze_results()
        
        print(f"\n{'='*70}")
        print(f"  ✅ TASK 4 COMPLETE")
        print(f"{'='*70}\n")
        print("Outputs for Ray:")
        print(f"  • EAR values per frame: ✅")
        print(f"  • Blink count/min: ✅")
        print(f"  • Fatigue confidence: ✅")
        print(f"  • Blink detection timeline: ✅")
        print(f"  • Visualizations: cv/output/task4_ear_fatigue.png")
        print(f"  • Metrics JSON: cv/output/task4_metrics.json")
        print()

def main():
    detector = EARFatigueDetector()
    detector.run()

if __name__ == "__main__":
    main()