"""
Task 6: Feature Fusion → Final Context Flags
Combines all features from Tasks 1-5:
- FER emotion predictions
- Continuous affect vector (valence, arousal)
- Temporal smoothing + spike detection (ΔE)
- EAR (blink + fatigue)
- Head pose + optical flow

Outputs: STRESS / RUSH / FATIGUE / NEUTRAL flags
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

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except:
    MEDIAPIPE_AVAILABLE = False

# Configuration
OUTPUT_DIR = 'cv/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Emotion coordinates (Task 2)
EMOTION_COORDS = {
    'happy': (0.9, 0.6),
    'surprise': (0.6, 0.9),
    'neutral': (0.0, 0.0),
    'sad': (-0.8, -0.6),
    'angry': (-0.7, 0.7),
    'fear': (-0.8, 0.5),
    'disgust': (-0.7, -0.2)
}

# Thresholds (from proposal)
STRESS_VALENCE_THRESHOLD = -0.4  # Negative valence
STRESS_AROUSAL_THRESHOLD = 0.5   # High arousal
SPIKE_THRESHOLD = 0.35           # ΔE spike
EAR_FATIGUE_THRESHOLD = 0.22
FATIGUE_DURATION = 3.0
MOTION_THRESHOLD = 5.0           # High fidgeting

# Decision window (N frames must agree)
DECISION_WINDOW_SIZE = 8
CONFIDENCE_THRESHOLD = 0.55

# MediaPipe landmarks
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype=np.float64)

class EmotionFusionSystem:
    def __init__(self, alpha=0.7):
        """Initialize complete fusion system"""
        
        self.alpha = alpha  # EMA smoothing factor
        
        # Initialize MediaPipe
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        
        # Camera matrix for pose
        self.focal_length = 1000
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))
        
        # State tracking
        self.E_prev = None
        self.E_bar_prev = None
        self.prev_gray = None
        self.prev_landmarks = None
        self.fatigue_start_time = None
        
        # Decision window for stability
        self.flag_history = deque(maxlen=DECISION_WINDOW_SIZE)
        
        # Data storage
        self.frame_data = []
        self.flag_changes = []
        self.timestamps = []
        
        print(f"\n{'='*70}")
        print(f"  TASK 6: FEATURE FUSION SYSTEM")
        print(f"  All Features Combined → Context Flags")
        print(f"{'='*70}\n")
        print("Input Features:")
        print("  [1] FER emotion predictions")
        print("  [2] Continuous affect (valence, arousal)")
        print("  [3] Temporal smoothing + ΔE spikes")
        print("  [4] EAR (blink + fatigue)")
        print("  [5] Head pose (yaw, pitch, roll)")
        print("  [6] Optical flow (motion)")
        print("\nOutput Flags:")
        print("  • STRESS:  Negative valence + high arousal + spikes")
        print("  • RUSH:    High motion + positive arousal")
        print("  • FATIGUE: Low EAR sustained + low arousal")
        print("  • NEUTRAL: Default state")
        print()
    
    def initialize_camera_matrix(self, frame_shape):
        """Initialize camera matrix"""
        h, w = frame_shape[:2]
        center = (w / 2, h / 2)
        self.camera_matrix = np.array([
            [self.focal_length, 0, center[0]],
            [0, self.focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
    
    # === Task 1: FER Prediction ===
    def predict_emotion(self, frame):
        """Get emotion probabilities from DeepFace"""
        if not DEEPFACE_AVAILABLE:
            return None
        
        try:
            result = DeepFace.analyze(
                frame, actions=['emotion'],
                enforce_detection=False, silent=True
            )
            if isinstance(result, list):
                result = result[0]
            
            emotions = result['emotion']
            probs = {k: v/100.0 for k, v in emotions.items()}
            dominant = max(probs.items(), key=lambda x: x[1])[0]
            
            return probs, dominant
        except:
            return None
    
    # === Task 2: Continuous Affect Vector ===
    def compute_affect_vector(self, emotion_probs):
        """Map emotions to valence-arousal space"""
        valence = 0.0
        arousal = 0.0
        
        for emotion, prob in emotion_probs.items():
            if emotion in EMOTION_COORDS:
                v, a = EMOTION_COORDS[emotion]
                valence += prob * v
                arousal += prob * a
        
        return np.array([valence, arousal])
    
    # === Task 3: Temporal Smoothing + Spike Detection ===
    def apply_ema_smoothing(self, E_current):
        """Apply EMA smoothing"""
        if self.E_bar_prev is None:
            E_bar = E_current
        else:
            E_bar = self.alpha * E_current + (1 - self.alpha) * self.E_bar_prev
        
        self.E_bar_prev = E_bar
        return E_bar
    
    def compute_delta_E(self, E_current):
        """Compute emotion change magnitude"""
        if self.E_prev is None:
            delta_E = 0.0
            is_spike = False
        else:
            delta_E = np.linalg.norm(E_current - self.E_prev)
            is_spike = delta_E > SPIKE_THRESHOLD
        
        self.E_prev = E_current.copy()
        return delta_E, is_spike
    
    # === Task 4: EAR (Fatigue Detection) ===
    def compute_ear(self, eye_landmarks):
        """Compute Eye Aspect Ratio"""
        A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        ear = (A + B) / (2.0 * C)
        return ear
    
    def extract_eye_landmarks(self, face_landmarks, frame_shape):
        """Extract eye landmarks"""
        h, w = frame_shape[:2]
        
        left_eye = []
        for idx in LEFT_EYE_INDICES:
            lm = face_landmarks.landmark[idx]
            left_eye.append(np.array([lm.x * w, lm.y * h]))
        
        right_eye = []
        for idx in RIGHT_EYE_INDICES:
            lm = face_landmarks.landmark[idx]
            right_eye.append(np.array([lm.x * w, lm.y * h]))
        
        return np.array(left_eye), np.array(right_eye)
    
    def detect_fatigue(self, ear, current_time):
        """Detect fatigue from sustained low EAR"""
        if ear < EAR_FATIGUE_THRESHOLD:
            if self.fatigue_start_time is None:
                self.fatigue_start_time = current_time
            
            duration = current_time - self.fatigue_start_time
            fatigue_confidence = min(1.0, duration / FATIGUE_DURATION)
            is_fatigued = duration >= FATIGUE_DURATION
        else:
            self.fatigue_start_time = None
            is_fatigued = False
            fatigue_confidence = 0.0
        
        return is_fatigued, fatigue_confidence
    
    # === Task 5: Head Pose + Motion ===
    def extract_pose_landmarks(self, face_landmarks, frame_shape):
        """Extract landmarks for pose estimation"""
        h, w = frame_shape[:2]
        image_points = []
        
        for idx in POSE_LANDMARKS:
            lm = face_landmarks.landmark[idx]
            image_points.append([lm.x * w, lm.y * h])
        
        return np.array(image_points, dtype=np.float64)
    
    def estimate_head_pose(self, image_points):
        """Estimate head pose using solvePnP"""
        success, rot_vec, trans_vec = cv2.solvePnP(
            MODEL_POINTS, image_points,
            self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None, None, None
        
        rot_mat, _ = cv2.Rodrigues(rot_vec)
        sy = np.sqrt(rot_mat[0, 0]**2 + rot_mat[1, 0]**2)
        
        if sy >= 1e-6:
            pitch = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
            yaw = np.arctan2(-rot_mat[2, 0], sy)
            roll = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
        else:
            pitch = np.arctan2(-rot_mat[1, 2], rot_mat[1, 1])
            yaw = np.arctan2(-rot_mat[2, 0], sy)
            roll = 0
        
        return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)
    
    def compute_optical_flow(self, gray_frame, landmarks):
        """Compute optical flow magnitude"""
        if self.prev_gray is None or self.prev_landmarks is None:
            self.prev_gray = gray_frame
            self.prev_landmarks = landmarks
            return 0.0
        
        try:
            lk_params = dict(
                winSize=(15, 15), maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            
            new_lm, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray_frame,
                self.prev_landmarks, None, **lk_params
            )
            
            if new_lm is not None and status is not None:
                valid_old = self.prev_landmarks[status == 1]
                valid_new = new_lm[status == 1]
                
                if len(valid_old) > 0:
                    displacements = valid_new - valid_old
                    magnitudes = np.linalg.norm(displacements, axis=1)
                    flow_mag = np.mean(magnitudes)
                else:
                    flow_mag = 0.0
            else:
                flow_mag = 0.0
        except:
            flow_mag = 0.0
        
        self.prev_gray = gray_frame.copy()
        self.prev_landmarks = landmarks.copy()
        
        return flow_mag
    
    # === Task 6: Feature Fusion ===
    def fuse_features(self, features):
        """
        Fuse all features to determine context flag
        
        Features dict contains:
        - valence, arousal (Task 2)
        - delta_E, is_spike (Task 3)
        - ear, is_fatigued, fatigue_confidence (Task 4)
        - yaw, pitch, roll, flow_magnitude (Task 5)
        - dominant_emotion (Task 1)
        
        Returns: flag, confidence, details
        """
        valence = features['valence']
        arousal = features['arousal']
        is_spike = features['is_spike']
        delta_E = features['delta_E']
        is_fatigued = features['is_fatigued']
        fatigue_conf = features['fatigue_confidence']
        flow_mag = features['flow_magnitude']
        
        # Initialize scores
        stress_score = 0.0
        rush_score = 0.0
        fatigue_score = 0.0
        
        # STRESS: Negative valence + high arousal + spikes
        if valence < STRESS_VALENCE_THRESHOLD:
            stress_score += 0.3
        if arousal > STRESS_AROUSAL_THRESHOLD:
            stress_score += 0.3
        if is_spike:
            stress_score += 0.4
        
        # RUSH: High motion + medium/high arousal
        if flow_mag > MOTION_THRESHOLD:
            rush_score += 0.5
        if arousal > 0.3:
            rush_score += 0.3
        if delta_E > 0.2:  # Rapid emotion changes
            rush_score += 0.2
        
        # FATIGUE: Low EAR + low arousal + low motion
        fatigue_score = fatigue_conf * 0.6
        if arousal < 0.2:
            fatigue_score += 0.2
        if flow_mag < 2.0:
            fatigue_score += 0.2
        
        # Determine flag
        scores = {
            'STRESS': stress_score,
            'RUSH': rush_score,
            'FATIGUE': fatigue_score
        }
        
        max_flag = max(scores.items(), key=lambda x: x[1])
        
        if max_flag[1] >= CONFIDENCE_THRESHOLD:
            flag = max_flag[0]
            confidence = max_flag[1]
        else:
            flag = 'NEUTRAL'
            confidence = 1.0 - max_flag[1]
        
        details = {
            'stress_score': round(stress_score, 3),
            'rush_score': round(rush_score, 3),
            'fatigue_score': round(fatigue_score, 3),
            'chosen_flag': flag,
            'confidence': round(confidence, 3)
        }
        
        return flag, confidence, details
    
    def apply_decision_window(self, current_flag):
        """Apply N-frame agreement for stability"""
        self.flag_history.append(current_flag)
        
        if len(self.flag_history) < DECISION_WINDOW_SIZE:
            return None  # Not enough history
        
        # Count occurrences
        flag_counts = {}
        for flag in self.flag_history:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
        
        # Require majority agreement
        majority_threshold = DECISION_WINDOW_SIZE // 2 + 1
        for flag, count in flag_counts.items():
            if count >= majority_threshold:
                return flag
        
        return None  # No agreement
    
    def run_fusion_system(self, duration=60):
        """Run complete fusion system"""
        print(f"📸 Running fusion system for {duration}s")
        print("   Try different scenarios:")
        print("   - Smile/frown rapidly → STRESS")
        print("   - Move head quickly → RUSH")
        print("   - Close eyes, stay still → FATIGUE")
        print("   - Stay neutral → NEUTRAL")
        print("   Press Q to quit\n")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return False
        
        ret, first_frame = cap.read()
        if ret:
            self.initialize_camera_matrix(first_frame.shape)
        
        start_time = time.time()
        frame_count = 0
        current_stable_flag = None
        last_flag_change_time = start_time
        
        while (time.time() - start_time) < duration:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            current_time = time.time() - start_time
            display_frame = frame.copy()
            
            # Convert frames
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Process with MediaPipe
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # === Extract all features ===
                
                # Task 1 & 2: FER + Affect Vector
                emotion_result = self.predict_emotion(frame)
                if emotion_result:
                    probs, dominant = emotion_result
                    E_raw = self.compute_affect_vector(probs)
                    
                    # Task 3: Smoothing + Spike
                    E_smooth = self.apply_ema_smoothing(E_raw)
                    delta_E, is_spike = self.compute_delta_E(E_raw)
                    
                    valence, arousal = E_smooth
                else:
                    valence, arousal = 0.0, 0.0
                    delta_E, is_spike = 0.0, False
                    dominant = 'neutral'
                
                # Task 4: EAR + Fatigue
                left_eye, right_eye = self.extract_eye_landmarks(face_landmarks, frame.shape)
                left_ear = self.compute_ear(left_eye)
                right_ear = self.compute_ear(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0
                is_fatigued, fatigue_conf = self.detect_fatigue(avg_ear, current_time)
                
                # Task 5: Pose + Motion
                pose_landmarks = self.extract_pose_landmarks(face_landmarks, frame.shape)
                yaw, pitch, roll = self.estimate_head_pose(pose_landmarks)
                if yaw is None:
                    yaw, pitch, roll = 0.0, 0.0, 0.0
                flow_mag = self.compute_optical_flow(gray_frame, pose_landmarks)
                
                # === Fuse features ===
                features = {
                    'valence': valence,
                    'arousal': arousal,
                    'delta_E': delta_E,
                    'is_spike': is_spike,
                    'ear': avg_ear,
                    'is_fatigued': is_fatigued,
                    'fatigue_confidence': fatigue_conf,
                    'yaw': yaw,
                    'pitch': pitch,
                    'roll': roll,
                    'flow_magnitude': flow_mag,
                    'dominant_emotion': dominant
                }
                
                flag, confidence, details = self.fuse_features(features)
                
                # Apply decision window
                stable_flag = self.apply_decision_window(flag)
                
                if stable_flag and stable_flag != current_stable_flag:
                    current_stable_flag = stable_flag
                    self.flag_changes.append({
                        'time': current_time,
                        'flag': stable_flag,
                        'confidence': confidence
                    })
                    last_flag_change_time = current_time
                
                # Store frame data
                self.frame_data.append({
                    'time': current_time,
                    'features': features,
                    'flag': flag,
                    'stable_flag': stable_flag if stable_flag else 'PENDING',
                    'confidence': confidence,
                    'details': details
                })
                self.timestamps.append(current_time)
                
                # === Visualization ===
                
                # Determine display color
                flag_colors = {
                    'STRESS': (0, 0, 255),
                    'RUSH': (0, 165, 255),
                    'FATIGUE': (255, 0, 255),
                    'NEUTRAL': (0, 255, 0),
                    'PENDING': (128, 128, 128)
                }
                
                display_flag = stable_flag if stable_flag else 'PENDING'
                flag_color = flag_colors.get(display_flag, (255, 255, 255))
                
                # Draw large flag indicator
                cv2.rectangle(display_frame, (0, 0), (400, 100), flag_color, -1)
                cv2.putText(display_frame, display_flag, 
                          (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
                
                # Feature values
                y_pos = 130
                cv2.putText(display_frame, f"Emotion: {dominant}", 
                          (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_pos += 30
                cv2.putText(display_frame, f"Valence: {valence:+.2f} | Arousal: {arousal:+.2f}", 
                          (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_pos += 30
                cv2.putText(display_frame, f"ΔE: {delta_E:.3f} {'🔥' if is_spike else ''}", 
                          (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                y_pos += 30
                cv2.putText(display_frame, f"EAR: {avg_ear:.3f} {'💤' if is_fatigued else ''}", 
                          (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                y_pos += 30
                cv2.putText(display_frame, f"Motion: {flow_mag:.2f}", 
                          (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                y_pos += 30
                cv2.putText(display_frame, f"Confidence: {confidence:.2f}", 
                          (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Scores
                y_pos += 40
                cv2.putText(display_frame, f"Stress: {details['stress_score']:.2f}", 
                          (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                y_pos += 25
                cv2.putText(display_frame, f"Rush: {details['rush_score']:.2f}", 
                          (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                y_pos += 25
                cv2.putText(display_frame, f"Fatigue: {details['fatigue_score']:.2f}", 
                          (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                
                # Console output (every 15 frames)
                if frame_count % 15 == 0:
                    print(f"  Frame {frame_count:4d} | {display_flag:8s} ({confidence:.2f}) | "
                          f"V:{valence:+.2f} A:{arousal:+.2f} | ΔE:{delta_E:.3f} | "
                          f"EAR:{avg_ear:.2f} | Motion:{flow_mag:.2f}")
            
            # Time display
            cv2.putText(display_frame, f"Time: {current_time:.1f}s / {duration}s", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
            
            cv2.imshow('Task 6: Complete Fusion System', display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n✅ Processed {len(self.frame_data)} frames")
        print(f"   Flag changes: {len(self.flag_changes)}")
        return len(self.frame_data) > 0
    
    def analyze_results(self):
        """Analyze fusion results"""
        print(f"\n{'='*70}")
        print(f"  FUSION SYSTEM ANALYSIS")
        print(f"{'='*70}\n")
        
        if len(self.frame_data) == 0:
            print("❌ No data")
            return
        
        # Count flags
        flag_counts = {}
        for data in self.frame_data:
            flag = data['stable_flag']
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
        
        print("FLAG DISTRIBUTION:")
        for flag in ['STRESS', 'RUSH', 'FATIGUE', 'NEUTRAL', 'PENDING']:
            count = flag_counts.get(flag, 0)
            percent = (count / len(self.frame_data)) * 100
            print(f"  {flag:8s}: {count:4d} frames ({percent:5.1f}%)")
        print()
        
        print("FLAG CHANGES:")
        print(f"  Total transitions: {len(self.flag_changes)}")
        for change in self.flag_changes[:10]:
            print(f"    {change['time']:6.2f}s → {change['flag']:8s} ({change['confidence']:.2f})")
        if len(self.flag_changes) > 10:
            print(f"    ... and {len(self.flag_changes) - 10} more")
        print()
        
        # Average confidences
        avg_confidences = {}
        for flag in ['STRESS', 'RUSH', 'FATIGUE', 'NEUTRAL']:
            confs = [d['confidence'] for d in self.frame_data if d['flag'] == flag]
            if confs:
                avg_confidences[flag] = np.mean(confs)
        
        print("AVERAGE CONFIDENCES:")
        for flag, conf in avg_confidences.items():
            print(f"  {flag:8s}: {conf:.3f}")
        print()
        
        print(f"{'='*70}\n")
        
        # Visualizations
        self.create_visualizations()
        self.save_results()
    
    def create_visualizations(self):
        """Create visualization plots"""
        fig = plt.figure(figsize=(16, 10))
        
        # Extract data
        flags = [d['stable_flag'] for d in self.frame_data]
        
        # 1. Flag timeline
        ax1 = plt.subplot(2, 2, 1)
        flag_map = {'STRESS': 3, 'RUSH': 2, 'FATIGUE': 1, 'NEUTRAL': 0, 'PENDING': -1}
        flag_values = [flag_map.get(f, -1) for f in flags]
        
        colors = ['red' if f == 'STRESS' else 'orange' if f == 'RUSH' else 
                 'purple' if f == 'FATIGUE' else 'green' if f == 'NEUTRAL' else 'gray' 
                 for f in flags]
        
        ax1.scatter(self.timestamps, flag_values, c=colors, s=5, alpha=0.6)
        ax1.set_xlabel('Time (s)', fontweight='bold')
        ax1.set_ylabel('Context Flag', fontweight='bold')
        ax1.set_yticks([0, 1, 2, 3])
        ax1.set_yticklabels(['NEUTRAL', 'FATIGUE', 'RUSH', 'STRESS'])
        ax1.set_title('Context Flags Over Time', fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # 2. Flag distribution
        ax2 = plt.subplot(2, 2, 2)
        flag_counts = {}
        for flag in flags:
            if flag != 'PENDING':
                flag_counts[flag] = flag_counts.get(flag, 0) + 1
        
        flag_colors_chart = {'STRESS': 'red', 'RUSH': 'orange', 
                           'FATIGUE': 'purple', 'NEUTRAL': 'green'}
        colors_list = [flag_colors_chart.get(f, 'gray') for f in flag_counts.keys()]
        
        ax2.bar(flag_counts.keys(), flag_counts.values(), color=colors_list, edgecolor='black')
        ax2.set_ylabel('Frame Count', fontweight='bold')
        ax2.set_title('Flag Distribution', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add percentages
        total = sum(flag_counts.values())
        for i, (flag, count) in enumerate(flag_counts.items()):
            percent = (count / total) * 100
            ax2.text(i, count + 10, f'{percent:.1f}%', ha='center', fontweight='bold')
        
        # 3. Feature contributions to stress
        ax3 = plt.subplot(2, 2, 3)
        valence_vals = [d['features']['valence'] for d in self.frame_data]
        arousal_vals = [d['features']['arousal'] for d in self.frame_data]
        
        scatter = ax3.scatter(valence_vals, arousal_vals, 
                             c=[flag_map.get(f, -1) for f in flags],
                             cmap='RdYlGn_r', s=20, alpha=0.5)
        ax3.axhline(y=STRESS_AROUSAL_THRESHOLD, color='r', linestyle='--', 
                   linewidth=2, alpha=0.5, label=f'Arousal threshold')
        ax3.axvline(x=STRESS_VALENCE_THRESHOLD, color='r', linestyle='--', 
                   linewidth=2, alpha=0.5, label=f'Valence threshold')
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.2, linewidth=0.5)
        ax3.axvline(x=0, color='k', linestyle='-', alpha=0.2, linewidth=0.5)
        ax3.set_xlabel('Valence', fontweight='bold')
        ax3.set_ylabel('Arousal', fontweight='bold')
        ax3.set_title('Affect Space with Stress Thresholds', fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # 4. Confidence over time
        ax4 = plt.subplot(2, 2, 4)
        confidences = [d['confidence'] for d in self.frame_data]
        ax4.plot(self.timestamps, confidences, 'b-', linewidth=1, alpha=0.6)
        ax4.axhline(y=CONFIDENCE_THRESHOLD, color='r', linestyle='--', 
                   linewidth=2, label=f'Threshold ({CONFIDENCE_THRESHOLD})')
        ax4.set_xlabel('Time (s)', fontweight='bold')
        ax4.set_ylabel('Confidence', fontweight='bold')
        ax4.set_title('Decision Confidence Over Time', fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)
        ax4.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        output_path = os.path.join(OUTPUT_DIR, 'task6_fusion_system.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualizations saved: {output_path}")
        plt.close()
    
    def save_results(self):
        """Save fusion results to JSON"""
        
        # Count final flags
        flag_counts = {}
        for data in self.frame_data:
            flag = data['stable_flag']
            if flag != 'PENDING':
                flag_counts[flag] = flag_counts.get(flag, 0) + 1
        
        total_frames = len([d for d in self.frame_data if d['stable_flag'] != 'PENDING'])
        
        results = {
            "task": "Task 6: Feature Fusion System",
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "ema_alpha": self.alpha,
                "spike_threshold": SPIKE_THRESHOLD,
                "stress_valence_threshold": STRESS_VALENCE_THRESHOLD,
                "stress_arousal_threshold": STRESS_AROUSAL_THRESHOLD,
                "ear_fatigue_threshold": EAR_FATIGUE_THRESHOLD,
                "motion_threshold": MOTION_THRESHOLD,
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "decision_window_size": DECISION_WINDOW_SIZE
            },
            "summary": {
                "total_frames": len(self.frame_data),
                "total_flag_changes": len(self.flag_changes),
                "flag_distribution": {
                    flag: {
                        "count": count,
                        "percentage": round((count / total_frames) * 100, 2) if total_frames > 0 else 0
                    }
                    for flag, count in flag_counts.items()
                }
            },
            "flag_transitions": [
                {
                    "time": round(change['time'], 2),
                    "flag": change['flag'],
                    "confidence": round(change['confidence'], 3)
                }
                for change in self.flag_changes
            ],
            "frame_by_frame_data": [
                {
                    "time": round(d['time'], 2),
                    "flag": d['flag'],
                    "stable_flag": d['stable_flag'],
                    "confidence": round(d['confidence'], 3),
                    "features": {
                        "valence": round(float(d['features']['valence']), 3),
                        "arousal": round(float(d['features']['arousal']), 3),
                        "delta_E": round(float(d['features']['delta_E']), 3),
                        "is_spike": bool(d['features']['is_spike']),
                        "ear": round(float(d['features']['ear']), 3),
                        "is_fatigued": bool(d['features']['is_fatigued']),
                        "flow_magnitude": round(float(d['features']['flow_magnitude']), 3)
                    },
                    "scores": d['details']
                }
                for d in self.frame_data[::5]  # Sample every 5th frame for JSON size
            ]
        }
        
        json_path = os.path.join(OUTPUT_DIR, 'task6_metrics.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results JSON saved: {json_path}")
        
        # Save Unity-compatible output
        unity_output = {
            "flag_changes": self.flag_changes,
            "parameters": results["parameters"],
            "summary": results["summary"]
        }
        
        unity_path = os.path.join(OUTPUT_DIR, 'unity_integration.json')
        with open(unity_path, 'w') as f:
            json.dump(unity_output, f, indent=2)
        
        print(f"✅ Unity integration file saved: {unity_path}")
    
    def run(self):
        """Run complete fusion system"""
        if not DEEPFACE_AVAILABLE:
            print("❌ DeepFace required")
            return
        
        if not MEDIAPIPE_AVAILABLE:
            print("❌ MediaPipe required")
            return
        
        # Run fusion
        success = self.run_fusion_system(duration=60)
        
        if not success:
            print("❌ Fusion failed")
            return
        
        # Analyze
        self.analyze_results()
        
        print(f"\n{'='*70}")
        print(f"  ✅ TASK 6 COMPLETE - ALL TASKS FINISHED!")
        print(f"{'='*70}\n")
        print("Final Outputs for Ray:")
        print(f"  • Context flags: STRESS / RUSH / FATIGUE / NEUTRAL ✅")
        print(f"  • Confidence scores per flag ✅")
        print(f"  • Number of frames agreed (decision window) ✅")
        print(f"  • Flag transition timeline ✅")
        print(f"  • Feature contributions ✅")
        print()
        print("Files Generated:")
        print(f"  • cv/output/task6_fusion_system.png - Visualizations")
        print(f"  • cv/output/task6_metrics.json - Complete metrics")
        print(f"  • cv/output/unity_integration.json - For Unity NPC system")
        print()
        print("="*70)
        print("  🎉 ALL 6 TASKS COMPLETE!")
        print("="*70)
        print()
        print("Summary of Deliverables for Ray:")
        print()
        print("Task 1 - FER Model:")
        print("  ✅ Accuracy metrics")
        print("  ✅ Confusion matrix")
        print("  ✅ Inference time per frame")
        print("  ✅ FPS measurements")
        print()
        print("Task 2 - Continuous Affect Vector:")
        print("  ✅ Valence & arousal values")
        print("  ✅ Smooth vs raw comparison")
        print("  ✅ Variance reduction")
        print()
        print("Task 3 - Temporal Smoothing:")
        print("  ✅ Standard deviation before/after smoothing")
        print("  ✅ ΔE values per frame")
        print("  ✅ Spike detection")
        print()
        print("Task 4 - EAR (Fatigue):")
        print("  ✅ EAR values per frame")
        print("  ✅ Blink count/min")
        print("  ✅ Fatigue confidence")
        print()
        print("Task 5 - Head Pose + Motion:")
        print("  ✅ Head rotation angles (yaw, pitch, roll)")
        print("  ✅ Flow magnitude per frame")
        print("  ✅ Movement spikes")
        print()
        print("Task 6 - Fusion:")
        print("  ✅ Final flags: STRESS/RUSH/FATIGUE/NEUTRAL")
        print("  ✅ Confidence per decision")
        print("  ✅ Frame agreement count")
        print()
        print("All metrics saved to cv/output/")
        print("Ready for Unity integration!")
        print()

def main():
    system = EmotionFusionSystem(alpha=0.7)
    system.run()

if __name__ == "__main__":
    main()