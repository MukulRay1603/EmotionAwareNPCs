"""
Module B: Emotion Retriever
Receives timestamp from Module A, retrieves closest emotions,
averages them, sends to Module C
"""

import socket
import time
import json
import os
from datetime import datetime
import numpy as np

# Configuration
LISTEN_PORT = 5555
MODULE_C_HOST = 'localhost'
MODULE_C_PORT = 5556
REALTIME_DB = 'cv/output/realtime_emotions.json'
NUM_CLOSEST = 5  # Number of closest predictions to average

class ModuleB:
    def __init__(self):
        """Initialize Module B"""
        self.server_socket = None
        self.module_c_socket = None
        
        print(f"\n{'='*70}")
        print(f"  MODULE B: Emotion Retriever")
        print(f"{'='*70}\n")
        print(f"Listening on port {LISTEN_PORT}")
        print(f"Will retrieve {NUM_CLOSEST} closest emotions")
        print(f"Database: {REALTIME_DB}\n")
    
    def load_emotion_database(self):
        """Load emotion database from JSON file"""
        try:
            if os.path.exists(REALTIME_DB):
                with open(REALTIME_DB, 'r') as f:
                    data = json.load(f)
                    # Convert string keys back to float timestamps
                    return {float(k): v for k, v in data.items()}
            else:
                print(f"⚠️  Database file not found: {REALTIME_DB}")
                return {}
        except Exception as e:
            print(f"❌ Error loading database: {e}")
            return {}
    
    def find_closest_emotions(self, target_timestamp):
        """Find N closest emotion entries to target timestamp"""
        database = self.load_emotion_database()
        
        if not database:
            print("⚠️  No emotions in database")
            return []
        
        # Calculate time differences
        timestamps = list(database.keys())
        time_diffs = [abs(ts - target_timestamp) for ts in timestamps]
        
        # Get indices of N closest
        sorted_indices = np.argsort(time_diffs)[:NUM_CLOSEST]
        
        closest_emotions = []
        for idx in sorted_indices:
            ts = timestamps[idx]
            emotion_data = database[ts]
            time_diff = time_diffs[idx]
            
            closest_emotions.append({
                'timestamp': ts,
                'time_diff': time_diff,
                'data': emotion_data
            })
        
        return closest_emotions
    
    def average_emotions(self, closest_emotions):
        """Average the closest emotion predictions"""
        if not closest_emotions:
            return None
        
        # Extract values
        emotions = [e['data']['emotion'] for e in closest_emotions]
        flags = [e['data']['flag'] for e in closest_emotions]
        confidences = [e['data']['confidence'] for e in closest_emotions]
        valences = [e['data']['valence'] for e in closest_emotions]
        arousals = [e['data']['arousal'] for e in closest_emotions]
        ears = [e['data']['ear'] for e in closest_emotions]
        fatigues = [e['data']['is_fatigued'] for e in closest_emotions]
        
        # Calculate averages
        avg_confidence = np.mean(confidences)
        avg_valence = np.mean(valences)
        avg_arousal = np.mean(arousals)
        avg_ear = np.mean(ears)
        
        # Most common emotion and flag
        from collections import Counter
        most_common_emotion = Counter(emotions).most_common(1)[0][0]
        most_common_flag = Counter(flags).most_common(1)[0][0]
        fatigue_count = sum(fatigues)
        
        averaged_result = {
            'emotion': most_common_emotion,
            'flag': most_common_flag,
            'confidence': float(avg_confidence),
            'valence': float(avg_valence),
            'arousal': float(avg_arousal),
            'ear': float(avg_ear),
            'is_fatigued': fatigue_count >= (NUM_CLOSEST // 2),
            'num_samples': len(closest_emotions),
            'time_window': {
                'oldest': closest_emotions[-1]['time_diff'],
                'newest': closest_emotions[0]['time_diff']
            },
            'individual_emotions': [e['data']['emotion'] for e in closest_emotions],
            'individual_flags': [e['data']['flag'] for e in closest_emotions]
        }
        
        return averaged_result
    
    def send_to_module_c(self, data):
        """Send processed data to Module C"""
        try:
            if self.module_c_socket is None:
                # Connect to Module C
                self.module_c_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.module_c_socket.connect((MODULE_C_HOST, MODULE_C_PORT))
                print(f"✅ Connected to Module C at {MODULE_C_HOST}:{MODULE_C_PORT}")
            
            # Send data
            message = json.dumps(data)
            self.module_c_socket.sendall(message.encode('utf-8') + b'\n')
            print(f"📤 Sent to Module C: {data['combined_emotion']}")
            
        except Exception as e:
            print(f"⚠️  Could not send to Module C: {e}")
            self.module_c_socket = None
    
    def handle_client(self, client_socket):
        """Handle incoming connection from Module A"""
        print(f"\n{'='*50}")
        print(f"📥 New connection from Module A")
        print(f"{'='*50}")
        
        try:
            # Receive data
            data = client_socket.recv(4096).decode('utf-8')
            if not data:
                return
            
            message = json.loads(data.strip())
            text = message['text']
            timestamp = message['timestamp']
            datetime_str = message['datetime']
            
            print(f"\n📨 Received from Module A:")
            print(f"   Text: {text}")
            print(f"   Timestamp: {timestamp}")
            print(f"   DateTime: {datetime_str}")
            
            # Find closest emotions
            print(f"\n🔍 Finding {NUM_CLOSEST} closest emotions...")
            closest_emotions = self.find_closest_emotions(timestamp)
            
            if closest_emotions:
                print(f"\n✅ Found {len(closest_emotions)} emotions:")
                for i, e in enumerate(closest_emotions, 1):
                    print(f"   {i}. {e['data']['emotion']:10s} | "
                          f"Flag: {e['data']['flag']:8s} | "
                          f"Δt: {e['time_diff']:6.3f}s ago")
                
                # Average emotions
                print(f"\n📊 Averaging emotions...")
                averaged = self.average_emotions(closest_emotions)
                
                print(f"\n🎯 Averaged Result:")
                print(f"   Emotion: {averaged['emotion']}")
                print(f"   Flag: {averaged['flag']}")
                print(f"   Confidence: {averaged['confidence']:.3f}")
                print(f"   Valence: {averaged['valence']:+.3f}")
                print(f"   Arousal: {averaged['arousal']:+.3f}")
                print(f"   EAR: {averaged['ear']:.3f}")
                print(f"   Fatigued: {averaged['is_fatigued']}")
                
                # Prepare response for Module A
                response = {
                    'status': 'success',
                    'text': text,
                    'timestamp': timestamp,
                    **averaged
                }
                
                # Send response back to Module A
                response_json = json.dumps(response)
                client_socket.sendall(response_json.encode('utf-8'))
                
                # Prepare data for Module C
                combined_data = {
                    'text': text,
                    'timestamp': timestamp,
                    'datetime': datetime_str,
                    'combined_emotion': averaged['emotion'],
                    'combined_flag': averaged['flag'],
                    'emotion_metrics': averaged
                }
                
                # Send to Module C
                self.send_to_module_c(combined_data)
                
            else:
                print(f"⚠️  No emotions found in database")
                response = {
                    'status': 'no_data',
                    'message': 'No emotions in database'
                }
                client_socket.sendall(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            print(f"❌ Error handling client: {e}")
            error_response = {
                'status': 'error',
                'message': str(e)
            }
            try:
                client_socket.sendall(json.dumps(error_response).encode('utf-8'))
            except:
                pass
        finally:
            client_socket.close()
            print(f"\n{'='*50}")
    
    def run(self):
        """Main server loop"""
        try:
            # Create server socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('', LISTEN_PORT))
            self.server_socket.listen(5)
            
            print(f"✅ Module B listening on port {LISTEN_PORT}")
            print(f"   Waiting for connections from Module A...")
            print(f"   Press CTRL+C to stop\n")
            
            while True:
                client_socket, address = self.server_socket.accept()
                self.handle_client(client_socket)
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Shutting down Module B...")
        finally:
            if self.server_socket:
                self.server_socket.close()
            if self.module_c_socket:
                self.module_c_socket.close()
            print("✅ Module B stopped")

def main():
    module_b = ModuleB()
    module_b.run()

if __name__ == "__main__":
    main()