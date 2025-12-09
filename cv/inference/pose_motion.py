"""
Task 5: Head Pose + Motion (Optical Flow)
- Use solvePnP for head pose estimation (yaw, pitch, roll)
- Use optical flow for motion/fidgeting detection
- Track head rotation angles
- Measure movement intensity
- Output pose angles and flow magnitude
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

# 3D model points for head pose estimation (generic face model)
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float64)

# MediaPipe landmark indices for pose estimation
POSE_LANDMARKS = [
    1,    # Nose tip
    152,  # Chin
    33,   # Left eye left corner
    263,  # Right eye right corner
    61,   # Left mouth corner
    291   # Right mouth corner
]

class HeadPoseMotionDetector:
    def __init__(self):
        """Initialize head pose and motion detector"""
        
        if not MEDIAPIPE_AVAILABLE:
            print("❌ MediaPipe required for Task 5")
            return
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Camera matrix (approximate - will be refined)
        self.focal_length = 1000
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))
        
        # Previous frame for optical flow
        self.prev_gray = None
        self.prev_landmarks = None
        
        # Storage
        self.pose_history = []
        self.flow_history = []
        self.timestamps = []
        
        print(f"\n{'='*70}")
        print(f"  TASK 5: HEAD POSE + MOTION DETECTION")
        print(f"{'='*70}\n")
        print("Head Pose:")
        print("  • Yaw:   Rotation left/right")
        print("  • Pitch: Rotation up/down")
        print("  • Roll:  Tilt left/right")
        print("\nMotion Detection:")
        print("  • Optical Flow: Tracks movement between frames")
        print("  • Flow Magnitude: Indicates fidgeting/movement intensity")
        print()
    
    def initialize_camera_matrix(self, frame_shape):
        """Initialize camera matrix based on frame dimensions"""
        h, w = frame_shape[:2]
        center = (w / 2, h / 2)
        self.camera_matrix = np.array([
            [self.focal_length, 0, center[0]],
            [0, self.focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
    
    def extract_pose_landmarks(self, face_landmarks, frame_shape):
        """Extract specific landmarks for pose estimation"""
        h, w = frame_shape[:2]
        
        image_points = []
        for idx in POSE_LANDMARKS:
            landmark = face_landmarks.landmark[idx]
            x = landmark.x * w
            y = landmark.y * h
            image_points.append([x, y])
        
        return np.array(image_points, dtype=np.float64)
    
    def estimate_head_pose(self, image_points):
        """
        Estimate head pose using solvePnP
        
        Returns:
            yaw, pitch, roll: Head rotation angles in degrees
        """
        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            MODEL_POINTS,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None, None, None
        
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Extract Euler angles from rotation matrix
        # Based on: https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
        
        # Calculate yaw, pitch, roll
        sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
        
        singular = sy < 1e-6
        
        if not singular:
            pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            yaw = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            yaw = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = 0
        
        # Convert to degrees
        pitch = np.degrees(pitch)
        yaw = np.degrees(yaw)
        roll = np.degrees(roll)
        
        return yaw, pitch, roll, rotation_vector, translation_vector
    
    def compute_optical_flow(self, current_gray, current_landmarks):
        """
        Compute optical flow magnitude around face region
        
        Returns:
            flow_magnitude: Overall movement intensity
        """
        if self.prev_gray is None or self.prev_landmarks is None:
            self.prev_gray = current_gray
            self.prev_landmarks = current_landmarks
            return 0.0
        
        # Compute optical flow using Lucas-Kanade
        try:
            lk_params = dict(
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            
            # Track previous landmarks to current frame
            new_landmarks, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                current_gray,
                self.prev_landmarks,
                None,
                **lk_params
            )
            
            # Calculate displacement magnitude
            if new_landmarks is not None and status is not None:
                # Get valid points
                valid_old = self.prev_landmarks[status == 1]
                valid_new = new_landmarks[status == 1]
                
                if len(valid_old) > 0:
                    # Calculate displacement vectors
                    displacements = valid_new - valid_old
                    
                    # Compute average magnitude
                    magnitudes = np.linalg.norm(displacements, axis=1)
                    flow_magnitude = np.mean(magnitudes)
                else:
                    flow_magnitude = 0.0
            else:
                flow_magnitude = 0.0
            
        except Exception as e:
            flow_magnitude = 0.0
        
        # Update previous frame and landmarks
        self.prev_gray = current_gray.copy()
        self.prev_landmarks = current_landmarks.copy()
        
        return flow_magnitude
    
    def draw_pose_axes(self, frame, rotation_vector, translation_vector):
        """Draw coordinate axes on face to visualize pose"""
        # Axis endpoints in 3D
        axis = np.float32([
            [200, 0, 0],    # X-axis (red)
            [0, 200, 0],    # Y-axis (green)
            [0, 0, 200]     # Z-axis (blue)
        ])
        
        # Project 3D points to 2D
        axis_points, _ = cv2.projectPoints(
            axis,
            rotation_vector,
            translation_vector,
            self.camera_matrix,
            self.dist_coeffs
        )
        
        # Get nose tip (origin)
        origin = tuple(axis_points[0].ravel().astype(int))
        
        # Draw axes
        frame = cv2.line(frame, origin, tuple(axis_points[0].ravel().astype(int)), (0, 0, 255), 3)  # X red
        frame = cv2.line(frame, origin, tuple(axis_points[1].ravel().astype(int)), (0, 255, 0), 3)  # Y green
        frame = cv2.line(frame, origin, tuple(axis_points[2].ravel().astype(int)), (255, 0, 0), 3)  # Z blue
        
        return frame
    
    def get_pose_description(self, yaw, pitch, roll):
        """Get human-readable pose description"""
        descriptions = []
        
        # Yaw (left/right)
        if yaw < -15:
            descriptions.append("Looking RIGHT")
        elif yaw > 15:
            descriptions.append("Looking LEFT")
        else:
            descriptions.append("Facing FORWARD")
        
        # Pitch (up/down)
        if pitch < -15:
            descriptions.append("Looking DOWN")
        elif pitch > 15:
            descriptions.append("Looking UP")
        
        # Roll (tilt)
        if roll < -15:
            descriptions.append("Tilted LEFT")
        elif roll > 15:
            descriptions.append("Tilted RIGHT")
        
        return " | ".join(descriptions)
    
    def collect_pose_motion_data(self, duration=40):
        """Collect head pose and motion data"""
        print(f"📸 Collecting pose and motion data for {duration}s")
        print("   Instructions:")
        print("   - Move your head (left/right, up/down)")
        print("   - Tilt your head")
        print("   - Stay still vs. fidget to see motion detection")
        print("   Press Q to quit early\n")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return False
        
        # Get first frame to initialize camera matrix
        ret, first_frame = cap.read()
        if ret:
            self.initialize_camera_matrix(first_frame.shape)
        
        start_time = time.time()
        frame_count = 0
        
        while (time.time() - start_time) < duration:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            current_time = time.time() - start_time
            display_frame = frame.copy()
            
            # Convert to RGB and grayscale
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Process with MediaPipe
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # Extract landmarks for pose estimation
                image_points = self.extract_pose_landmarks(face_landmarks, frame.shape)
                
                # Estimate head pose
                pose_result = self.estimate_head_pose(image_points)
                
                if pose_result[0] is not None:
                    yaw, pitch, roll, rot_vec, trans_vec = pose_result
                    
                    # Compute optical flow
                    flow_magnitude = self.compute_optical_flow(gray_frame, image_points)
                    
                    # Store data
                    self.pose_history.append({
                        'time': current_time,
                        'yaw': yaw,
                        'pitch': pitch,
                        'roll': roll
                    })
                    self.flow_history.append(flow_magnitude)
                    self.timestamps.append(current_time)
                    
                    # Draw pose axes
                    display_frame = self.draw_pose_axes(display_frame, rot_vec, trans_vec)
                    
                    # Get pose description
                    pose_desc = self.get_pose_description(yaw, pitch, roll)
                    
                    # Display pose angles
                    cv2.putText(display_frame, f"Yaw:   {yaw:+6.1f}°", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(display_frame, f"Pitch: {pitch:+6.1f}°", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(display_frame, f"Roll:  {roll:+6.1f}°", 
                              (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Display motion
                    motion_color = (0, 255, 0) if flow_magnitude < 2.0 else (0, 165, 255)
                    if flow_magnitude > 5.0:
                        motion_color = (0, 0, 255)
                    
                    cv2.putText(display_frame, f"Motion: {flow_magnitude:.2f}", 
                              (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, motion_color, 2)
                    
                    # Display pose description
                    cv2.putText(display_frame, pose_desc, 
                              (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # Indicate high motion
                    if flow_magnitude > 5.0:
                        cv2.putText(display_frame, "🔴 HIGH MOVEMENT", 
                                  (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Console output (every 10 frames)
                    if frame_count % 10 == 0:
                        print(f"  Frame {frame_count:4d} | "
                              f"Yaw: {yaw:+6.1f}° | Pitch: {pitch:+6.1f}° | Roll: {roll:+6.1f}° | "
                              f"Motion: {flow_magnitude:.2f} | {pose_desc}")
            else:
                cv2.putText(display_frame, "No face detected", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display elapsed time
            cv2.putText(display_frame, f"Time: {current_time:.1f}s / {duration}s", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
            
            cv2.imshow('Task 5: Head Pose + Motion Detection', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n✅ Collected {len(self.pose_history)} pose samples")
        return len(self.pose_history) > 0
    
    def analyze_results(self):
        """Analyze pose and motion patterns"""
        print(f"\n{'='*70}")
        print(f"  ANALYSIS RESULTS")
        print(f"{'='*70}\n")
        
        if len(self.pose_history) == 0:
            print("❌ No data to analyze")
            return
        
        # Extract data
        yaw_values = [d['yaw'] for d in self.pose_history]
        pitch_values = [d['pitch'] for d in self.pose_history]
        roll_values = [d['roll'] for d in self.pose_history]
        flow_values = self.flow_history
        
        # Pose statistics
        print("HEAD POSE STATISTICS:")
        print(f"  Yaw (Left/Right):")
        print(f"    Mean:   {np.mean(yaw_values):+6.2f}°")
        print(f"    Std:    {np.std(yaw_values):6.2f}°")
        print(f"    Range:  [{np.min(yaw_values):+6.1f}°, {np.max(yaw_values):+6.1f}°]")
        print(f"  Pitch (Up/Down):")
        print(f"    Mean:   {np.mean(pitch_values):+6.2f}°")
        print(f"    Std:    {np.std(pitch_values):6.2f}°")
        print(f"    Range:  [{np.min(pitch_values):+6.1f}°, {np.max(pitch_values):+6.1f}°]")
        print(f"  Roll (Tilt):")
        print(f"    Mean:   {np.mean(roll_values):+6.2f}°")
        print(f"    Std:    {np.std(roll_values):6.2f}°")
        print(f"    Range:  [{np.min(roll_values):+6.1f}°, {np.max(roll_values):+6.1f}°]")
        print()
        
        # Motion statistics
        print("MOTION STATISTICS (Optical Flow):")
        print(f"  Average Magnitude:  {np.mean(flow_values):.3f}")
        print(f"  Std Dev:            {np.std(flow_values):.3f}")
        print(f"  Max Magnitude:      {np.max(flow_values):.3f}")
        print(f"  High Motion (>5):   {sum(1 for f in flow_values if f > 5.0)} frames")
        print()
        
        # Movement spikes
        spike_threshold = np.mean(flow_values) + 2 * np.std(flow_values)
        spike_count = sum(1 for f in flow_values if f > spike_threshold)
        print(f"  Movement Spikes:    {spike_count} (threshold: {spike_threshold:.2f})")
        print()
        
        print(f"{'='*70}\n")
        
        # Create visualizations
        self.create_visualizations(yaw_values, pitch_values, roll_values, flow_values)
        
        # Save results
        self.save_results(yaw_values, pitch_values, roll_values, flow_values, spike_count)
    
    def create_visualizations(self, yaw_values, pitch_values, roll_values, flow_values):
        """Create visualization plots"""
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Yaw over time
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(self.timestamps, yaw_values, 'b-', linewidth=2)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax1.axhline(y=15, color='r', linestyle='--', alpha=0.3, label='±15° threshold')
        ax1.axhline(y=-15, color='r', linestyle='--', alpha=0.3)
        ax1.fill_between(self.timestamps, -15, 15, alpha=0.1, color='green')
        ax1.set_xlabel('Time (s)', fontweight='bold')
        ax1.set_ylabel('Yaw (degrees)', fontweight='bold')
        ax1.set_title('Head Yaw (Left/Right)', fontweight='bold')
        ax1.grid(alpha=0.3)
        ax1.legend()
        
        # 2. Pitch over time
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(self.timestamps, pitch_values, 'g-', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.axhline(y=15, color='r', linestyle='--', alpha=0.3)
        ax2.axhline(y=-15, color='r', linestyle='--', alpha=0.3)
        ax2.fill_between(self.timestamps, -15, 15, alpha=0.1, color='green')
        ax2.set_xlabel('Time (s)', fontweight='bold')
        ax2.set_ylabel('Pitch (degrees)', fontweight='bold')
        ax2.set_title('Head Pitch (Up/Down)', fontweight='bold')
        ax2.grid(alpha=0.3)
        
        # 3. Roll over time
        ax3 = plt.subplot(3, 3, 3)
        ax3.plot(self.timestamps, roll_values, 'r-', linewidth=2)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax3.axhline(y=15, color='r', linestyle='--', alpha=0.3)
        ax3.axhline(y=-15, color='r', linestyle='--', alpha=0.3)
        ax3.fill_between(self.timestamps, -15, 15, alpha=0.1, color='green')
        ax3.set_xlabel('Time (s)', fontweight='bold')
        ax3.set_ylabel('Roll (degrees)', fontweight='bold')
        ax3.set_title('Head Roll (Tilt)', fontweight='bold')
        ax3.grid(alpha=0.3)
        
        # 4. Optical flow magnitude
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(self.timestamps, flow_values, 'purple', linewidth=2)
        ax4.axhline(y=np.mean(flow_values), color='g', linestyle='--', 
                   linewidth=2, label=f'Mean ({np.mean(flow_values):.2f})')
        spike_threshold = np.mean(flow_values) + 2 * np.std(flow_values)
        ax4.axhline(y=spike_threshold, color='r', linestyle='--', 
                   linewidth=2, label=f'Spike threshold ({spike_threshold:.2f})')
        ax4.set_xlabel('Time (s)', fontweight='bold')
        ax4.set_ylabel('Flow Magnitude', fontweight='bold')
        ax4.set_title('Motion Intensity (Optical Flow)', fontweight='bold')
        ax4.grid(alpha=0.3)
        ax4.legend()
        
        # 5. Pose angle distributions
        ax5 = plt.subplot(3, 3, 5)
        ax5.hist([yaw_values, pitch_values, roll_values], 
                bins=20, label=['Yaw', 'Pitch', 'Roll'], alpha=0.7)
        ax5.set_xlabel('Angle (degrees)', fontweight='bold')
        ax5.set_ylabel('Frequency', fontweight='bold')
        ax5.set_title('Pose Angle Distributions', fontweight='bold')
        ax5.legend()
        ax5.grid(alpha=0.3)
        
        # 6. Flow magnitude distribution
        ax6 = plt.subplot(3, 3, 6)
        ax6.hist(flow_values, bins=30, color='purple', alpha=0.7, edgecolor='black')
        ax6.axvline(np.mean(flow_values), color='g', linestyle='--', 
                   linewidth=2, label='Mean')
        ax6.axvline(spike_threshold, color='r', linestyle='--', 
                   linewidth=2, label='Spike threshold')
        ax6.set_xlabel('Flow Magnitude', fontweight='bold')
        ax6.set_ylabel('Frequency', fontweight='bold')
        ax6.set_title('Motion Distribution', fontweight='bold')
        ax6.legend()
        ax6.grid(alpha=0.3)
        
        # 7. 2D Yaw-Pitch trajectory
        ax7 = plt.subplot(3, 3, 7)
        scatter = ax7.scatter(yaw_values, pitch_values, c=self.timestamps, 
                             cmap='viridis', s=20, alpha=0.6)
        ax7.plot(yaw_values, pitch_values, 'k-', alpha=0.2, linewidth=0.5)
        ax7.scatter(yaw_values[0], pitch_values[0], s=150, c='green', 
                   marker='o', label='Start', zorder=5)
        ax7.scatter(yaw_values[-1], pitch_values[-1], s=150, c='red', 
                   marker='x', label='End', zorder=5)
        ax7.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax7.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax7.set_xlabel('Yaw (degrees)', fontweight='bold')
        ax7.set_ylabel('Pitch (degrees)', fontweight='bold')
        ax7.set_title('Head Movement Trajectory', fontweight='bold')
        ax7.grid(alpha=0.3)
        ax7.legend()
        plt.colorbar(scatter, ax=ax7, label='Time (s)')
        
        # 8. Combined pose magnitude
        ax8 = plt.subplot(3, 3, 8)
        pose_magnitude = np.sqrt(np.array(yaw_values)**2 + 
                                np.array(pitch_values)**2 + 
                                np.array(roll_values)**2)
        ax8.plot(self.timestamps, pose_magnitude, 'orange', linewidth=2)
        ax8.set_xlabel('Time (s)', fontweight='bold')
        ax8.set_ylabel('Total Rotation (degrees)', fontweight='bold')
        ax8.set_title('Overall Head Rotation Magnitude', fontweight='bold')
        ax8.grid(alpha=0.3)
        
        # 9. Summary statistics
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        summary_text = f"""
POSE & MOTION SUMMARY

Head Pose (degrees):
  Yaw:   {np.mean(yaw_values):+6.1f} ± {np.std(yaw_values):5.1f}
  Pitch: {np.mean(pitch_values):+6.1f} ± {np.std(pitch_values):5.1f}
  Roll:  {np.mean(roll_values):+6.1f} ± {np.std(roll_values):5.1f}

Motion:
  Avg Flow:    {np.mean(flow_values):.3f}
  Max Flow:    {np.max(flow_values):.3f}
  High Motion: {sum(1 for f in flow_values if f > 5.0)} frames
  Spikes:      {sum(1 for f in flow_values if f > spike_threshold)}

Total Samples: {len(self.pose_history)}
        """
        
        ax9.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        
        output_path = os.path.join(OUTPUT_DIR, 'task5_pose_motion.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualizations saved: {output_path}")
        plt.close()
    
    def save_results(self, yaw_values, pitch_values, roll_values, flow_values, spike_count):
        """Save results to JSON"""
        results = {
            "task": "Task 5: Head Pose + Motion Detection",
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(self.pose_history),
            "duration_seconds": self.timestamps[-1] if self.timestamps else 0,
            "head_pose_statistics": {
                "yaw": {
                    "mean": round(float(np.mean(yaw_values)), 3),
                    "std": round(float(np.std(yaw_values)), 3),
                    "min": round(float(np.min(yaw_values)), 3),
                    "max": round(float(np.max(yaw_values)), 3)
                },
                "pitch": {
                    "mean": round(float(np.mean(pitch_values)), 3),
                    "std": round(float(np.std(pitch_values)), 3),
                    "min": round(float(np.min(pitch_values)), 3),
                    "max": round(float(np.max(pitch_values)), 3)
                },
                "roll": {
                    "mean": round(float(np.mean(roll_values)), 3),
                    "std": round(float(np.std(roll_values)), 3),
                    "min": round(float(np.min(roll_values)), 3),
                    "max": round(float(np.max(roll_values)), 3)
                }
            },
            "motion_statistics": {
                "average_flow_magnitude": round(float(np.mean(flow_values)), 4),
                "std_flow_magnitude": round(float(np.std(flow_values)), 4),
                "max_flow_magnitude": round(float(np.max(flow_values)), 4),
                "movement_spikes": int(spike_count)
            },
            "head_rotation_angles": [
                {
                    "time": round(d['time'], 2),
                    "yaw": round(d['yaw'], 3),
                    "pitch": round(d['pitch'], 3),
                    "roll": round(d['roll'], 3)
                }
                for d in self.pose_history
            ],
            "flow_magnitude_per_frame": [round(float(f), 4) for f in flow_values]
        }
        
        json_path = os.path.join(OUTPUT_DIR, 'task5_metrics.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results JSON saved: {json_path}")
    
    def run(self):
        """Run full Task 5 pipeline"""
        if not MEDIAPIPE_AVAILABLE:
            print("❌ MediaPipe required for Task 5")
            return
        
        # Collect data
        success = self.collect_pose_motion_data(duration=40)
        
        if not success:
            print("❌ Data collection failed")
            return
        
        # Analyze results
        self.analyze_results()
        
        print(f"\n{'='*70}")
        print(f"  ✅ TASK 5 COMPLETE")
        print(f"{'='*70}\n")
        print("Outputs for Ray:")
        print(f"  • Head rotation angles (yaw, pitch, roll): ✅")
        print(f"  • Flow magnitude per frame: ✅")
        print(f"  • Movement spikes detected: ✅")
        print(f"  • Pose trajectory visualization: ✅")
        print(f"  • Visualizations: cv/output/task5_pose_motion.png")
        print(f"  • Metrics JSON: cv/output/task5_metrics.json")
        print()

def main():
    detector = HeadPoseMotionDetector()
    detector.run()

if __name__ == "__main__":
    main()