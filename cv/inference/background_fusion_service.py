"""
Background Fusion Service
Continuously runs emotion detection and saves to real-time database
Cleans up old data (>2 minutes)
"""

import cv2
import numpy as np
import time
import json
import os
from datetime import datetime, timedelta
from collections import deque
import threading
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
REALTIME_DB = 'cv/output/realtime_emotions.json'
RETENTION_SECONDS = 120  # Keep data for 2 minutes
UPDATE_INTERVAL = 0.5  # Update every 0.5 seconds

# Emotion coordinates
EMOTION_COORDS = {
    'happy': (0.9, 0.6), 'surprise': (0.6, 0.9), 'neutral': (0.0, 0.0),
    'sad': (-0.8, -0.6), 'angry': (-0.7, 0.7), 'fear': (-0.8, 0.5),
    'disgust': (-0.7, -0.2)
}

# Thresholds
STRESS_VALENCE_THRESHOLD = -0.4
STRESS_AROUSAL_THRESHOLD = 0.5
SPIKE_THRESHOLD = 0.35
EAR_FATIGUE_THRESHOLD = 0.22
FATIGUE_DURATION = 3.0
MOTION_THRESHOLD = 5.0
DECISION_WINDOW_SIZE = 8
CONFIDENCE_THRESHOLD = 0.55

# MediaPipe landmarks
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
], dtype=np.float64)

class BackgroundFusionService:
    def __init__(self):
        """Initialize background fusion service"""
        
        self.running = False
        self.emotion_database = {}  # {timestamp: emotion_data}
        self.db_lock = threading.Lock()
        
        # Initialize MediaPipe
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5
            )
        
        # Camera matrix
        self.focal_length = 1000
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))
        
        # State tracking
        self.E_prev = None
        self.E_bar_prev = None
        self.prev_gray = None
        self.prev_landmarks = None
        self.fatigue_start_time = None
        self.flag_history = deque(maxlen=DECISION_WINDOW_SIZE)
        self.alpha = 0.7
        
        os.makedirs(os.path.dirname(REALTIME_DB), exist_ok=True)
        
        print(f"✅ Background Fusion Service initialized")
        print(f"   Database: {REALTIME_DB}")
        print(f"   Retention: {RETENTION_SECONDS}s")
        print(f"   Update interval: {UPDATE_INTERVAL}s")
    
    def initialize_camera_matrix(self, frame_shape):
        h, w = frame_shape[:2]
        center = (w / 2, h / 2)
        self.camera_matrix = np.array([
            [self.focal_length, 0, center[0]],
            [0, self.focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
    
    # === All feature extraction methods (same as Task 6) ===
    
    def predict_emotion(self, frame):
        if not DEEPFACE_AVAILABLE:
            return None
        try:
            result = DeepFace.analyze(frame, actions=['emotion'],
                                     enforce_detection=False, silent=True)
            if isinstance(result, list):
                result = result[0]
            emotions = result['emotion']
            probs = {k: v/100.0 for k, v in emotions.items()}
            dominant = max(probs.items(), key=lambda x: x[1])[0]
            return probs, dominant
        except:
            return None
    
    def compute_affect_vector(self, emotion_probs):
        valence = arousal = 0.0
        for emotion, prob in emotion_probs.items():
            if emotion in EMOTION_COORDS:
                v, a = EMOTION_COORDS[emotion]
                valence += prob * v
                arousal += prob * a
        return np.array([valence, arousal])
    
    def apply_ema_smoothing(self, E_current):
        if self.E_bar_prev is None:
            E_bar = E_current
        else:
            E_bar = self.alpha * E_current + (1 - self.alpha) * self.E_bar_prev
        self.E_bar_prev = E_bar
        return E_bar
    
    def compute_delta_E(self, E_current):
        if self.E_prev is None:
            delta_E, is_spike = 0.0, False
        else:
            delta_E = np.linalg.norm(E_current - self.E_prev)
            is_spike = delta_E > SPIKE_THRESHOLD
        self.E_prev = E_current.copy()
        return delta_E, is_spike
    
    def compute_ear(self, eye_landmarks):
        A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        return (A + B) / (2.0 * C)
    
    def extract_eye_landmarks(self, face_landmarks, frame_shape):
        h, w = frame_shape[:2]
        left_eye = [np.array([face_landmarks.landmark[idx].x * w,
                              face_landmarks.landmark[idx].y * h])
                    for idx in LEFT_EYE_INDICES]
        right_eye = [np.array([face_landmarks.landmark[idx].x * w,
                               face_landmarks.landmark[idx].y * h])
                     for idx in RIGHT_EYE_INDICES]
        return np.array(left_eye), np.array(right_eye)
    
    def detect_fatigue(self, ear, current_time):
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
    
    def extract_pose_landmarks(self, face_landmarks, frame_shape):
        h, w = frame_shape[:2]
        return np.array([[face_landmarks.landmark[idx].x * w,
                         face_landmarks.landmark[idx].y * h]
                        for idx in POSE_LANDMARKS], dtype=np.float64)
    
    def estimate_head_pose(self, image_points):
        success, rot_vec, _ = cv2.solvePnP(
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
        if self.prev_gray is None or self.prev_landmarks is None:
            self.prev_gray = gray_frame
            self.prev_landmarks = landmarks
            return 0.0
        try:
            lk_params = dict(winSize=(15, 15), maxLevel=2,
                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            new_lm, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray_frame, self.prev_landmarks, None, **lk_params
            )
            if new_lm is not None and status is not None:
                valid_old = self.prev_landmarks[status == 1]
                valid_new = new_lm[status == 1]
                if len(valid_old) > 0:
                    displacements = valid_new - valid_old
                    flow_mag = np.mean(np.linalg.norm(displacements, axis=1))
                else:
                    flow_mag = 0.0
            else:
                flow_mag = 0.0
        except:
            flow_mag = 0.0
        self.prev_gray = gray_frame.copy()
        self.prev_landmarks = landmarks.copy()
        return flow_mag
    
    def fuse_features(self, features):
        """Fuse features to determine flag"""
        valence = features['valence']
        arousal = features['arousal']
        is_spike = features['is_spike']
        delta_E = features['delta_E']
        fatigue_conf = features['fatigue_confidence']
        flow_mag = features['flow_magnitude']
        
        stress_score = rush_score = fatigue_score = 0.0
        
        if valence < STRESS_VALENCE_THRESHOLD:
            stress_score += 0.3
        if arousal > STRESS_AROUSAL_THRESHOLD:
            stress_score += 0.3
        if is_spike:
            stress_score += 0.4
        
        if flow_mag > MOTION_THRESHOLD:
            rush_score += 0.5
        if arousal > 0.3:
            rush_score += 0.3
        if delta_E > 0.2:
            rush_score += 0.2
        
        fatigue_score = fatigue_conf * 0.6
        if arousal < 0.2:
            fatigue_score += 0.2
        if flow_mag < 2.0:
            fatigue_score += 0.2
        
        scores = {'STRESS': stress_score, 'RUSH': rush_score, 'FATIGUE': fatigue_score}
        max_flag = max(scores.items(), key=lambda x: x[1])
        
        if max_flag[1] >= CONFIDENCE_THRESHOLD:
            flag = max_flag[0]
            confidence = max_flag[1]
        else:
            flag = 'NEUTRAL'
            confidence = 1.0 - max_flag[1]
        
        return flag, confidence, scores
    
    def apply_decision_window(self, current_flag):
        """Apply N-frame agreement"""
        self.flag_history.append(current_flag)
        if len(self.flag_history) < DECISION_WINDOW_SIZE:
            return None
        flag_counts = {}
        for flag in self.flag_history:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
        majority_threshold = DECISION_WINDOW_SIZE // 2 + 1
        for flag, count in flag_counts.items():
            if count >= majority_threshold:
                return flag
        return None
    
    def cleanup_old_data(self):
        """Remove data older than RETENTION_SECONDS"""
        current_time = time.time()
        cutoff_time = current_time - RETENTION_SECONDS
        
        with self.db_lock:
            keys_to_delete = [ts for ts in self.emotion_database.keys()
                            if ts < cutoff_time]
            for key in keys_to_delete:
                del self.emotion_database[key]
    
    def save_to_database(self, emotion_data):
        """Save emotion data with timestamp"""
        timestamp = time.time()
        
        with self.db_lock:
            self.emotion_database[timestamp] = emotion_data
            
            # Also save to JSON file for persistence
            try:
                with open(REALTIME_DB, 'w') as f:
                    json.dump({
                        str(k): v for k, v in self.emotion_database.items()
                    }, f, indent=2)
            except Exception as e:
                print(f"Error saving to file: {e}")
    
    def run(self):
        """Main background service loop"""
        print(f"\n{'='*70}")
        print(f"  🚀 BACKGROUND FUSION SERVICE RUNNING")
        print(f"{'='*70}\n")
        print("Service is now capturing emotions in real-time...")
        print("Press CTRL+C to stop\n")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return
        
        ret, first_frame = cap.read()
        if ret:
            self.initialize_camera_matrix(first_frame.shape)
        
        self.running = True
        last_update_time = 0
        last_cleanup_time = 0
        frame_count = 0
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                frame_count += 1
                current_time = time.time()
                
                # Update emotion detection at specified interval
                if current_time - last_update_time >= UPDATE_INTERVAL:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    results = self.face_mesh.process(rgb_frame)
                    
                    if results.multi_face_landmarks:
                        face_landmarks = results.multi_face_landmarks[0]
                        
                        # Extract all features
                        emotion_result = self.predict_emotion(frame)
                        if emotion_result:
                            probs, dominant = emotion_result
                            E_raw = self.compute_affect_vector(probs)
                            E_smooth = self.apply_ema_smoothing(E_raw)
                            delta_E, is_spike = self.compute_delta_E(E_raw)
                            valence, arousal = E_smooth
                        else:
                            valence, arousal = 0.0, 0.0
                            delta_E, is_spike = 0.0, False
                            dominant = 'neutral'
                        
                        left_eye, right_eye = self.extract_eye_landmarks(face_landmarks, frame.shape)
                        avg_ear = (self.compute_ear(left_eye) + self.compute_ear(right_eye)) / 2.0
                        is_fatigued, fatigue_conf = self.detect_fatigue(avg_ear, current_time)
                        
                        pose_landmarks = self.extract_pose_landmarks(face_landmarks, frame.shape)
                        yaw, pitch, roll = self.estimate_head_pose(pose_landmarks)
                        if yaw is None:
                            yaw, pitch, roll = 0.0, 0.0, 0.0
                        flow_mag = self.compute_optical_flow(gray_frame, pose_landmarks)
                        
                        features = {
                            'valence': float(valence),
                            'arousal': float(arousal),
                            'delta_E': float(delta_E),
                            'is_spike': bool(is_spike),
                            'ear': float(avg_ear),
                            'is_fatigued': bool(is_fatigued),
                            'fatigue_confidence': float(fatigue_conf),
                            'flow_magnitude': float(flow_mag)
                        }
                        
                        flag, confidence, scores = self.fuse_features(features)
                        stable_flag = self.apply_decision_window(flag)
                        
                        # Create emotion data entry
                        emotion_data = {
                            'timestamp': current_time,
                            'datetime': datetime.fromtimestamp(current_time).isoformat(),
                            'emotion': dominant,
                            'flag': stable_flag if stable_flag else 'PENDING',
                            'confidence': float(confidence),
                            'valence': float(valence),
                            'arousal': float(arousal),
                            'ear': float(avg_ear),
                            'is_fatigued': bool(is_fatigued),
                            'fatigue_confidence': float(fatigue_conf),
                            'is_spike': bool(is_spike),
                            'delta_E': float(delta_E),
                            'flow_magnitude': float(flow_mag),
                            'scores': {
                                'stress': float(scores['STRESS']),
                                'rush': float(scores['RUSH']),
                                'fatigue': float(scores['FATIGUE'])
                            }
                        }
                        
                        # Save to database
                        self.save_to_database(emotion_data)
                        
                        # Console output
                        if frame_count % 10 == 0:
                            db_size = len(self.emotion_database)
                            print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                                  f"Emotion: {dominant:10s} | Flag: {emotion_data['flag']:8s} | "
                                  f"Confidence: {confidence:.2f} | DB Size: {db_size}")
                    
                    last_update_time = current_time
                
                # Cleanup old data every 10 seconds
                if current_time - last_cleanup_time >= 10:
                    self.cleanup_old_data()
                    last_cleanup_time = current_time
                
                # Small delay to not overload CPU
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Shutting down service...")
        finally:
            self.running = False
            cap.release()
            print("✅ Background service stopped")

def main():
    if not DEEPFACE_AVAILABLE or not MEDIAPIPE_AVAILABLE:
        print("❌ Required libraries not available")
        return
    
    service = BackgroundFusionService()
    service.run()

if __name__ == "__main__":
    main()