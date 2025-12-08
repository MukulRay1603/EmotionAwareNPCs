
"""
Module C: Data Receiver (Placeholder)
Receives combined text + emotion data from Module B
Can be extended for Unity integration, database storage, etc.
"""

import socket
import json
from datetime import datetime

# Configuration
LISTEN_PORT = 5556

class ModuleC:
    def __init__(self):
        """Initialize Module C"""
        self.server_socket = None
        self.received_count = 0
        
        print(f"\n{'='*70}")
        print(f"  MODULE C: Data Receiver (Placeholder)")
        print(f"{'='*70}\n")
        print(f"Listening on port {LISTEN_PORT}")
        print(f"Ready to receive combined text + emotion data\n")
    
    def process_data(self, data):
        """Process received data - placeholder for future functionality"""
        self.received_count += 1
        
        print(f"\n{'='*70}")
        print(f"  📨 RECEIVED DATA #{self.received_count}")
        print(f"{'='*70}")
        
        print(f"\n📝 Text Input:")
        print(f"   {data['text']}")
        
        print(f"\n📅 Timestamp:")
        print(f"   {data['datetime']}")
        
        print(f"\n😊 Combined Emotion Analysis:")
        metrics = data['emotion_metrics']
        print(f"   Emotion: {data['combined_emotion']}")
        print(f"   Flag: {data['combined_flag']}")
        print(f"   Confidence: {metrics['confidence']:.3f}")
        print(f"   Valence: {metrics['valence']:+.3f}")
        print(f"   Arousal: {metrics['arousal']:+.3f}")
        print(f"   EAR: {metrics['ear']:.3f}")
        print(f"   Fatigued: {metrics['is_fatigued']}")
        
        print(f"\n📊 Averaging Details:")
        print(f"   Samples averaged: {metrics['num_samples']}")
        print(f"   Time window: {metrics['time_window']['newest']:.3f}s - {metrics['time_window']['oldest']:.3f}s")
        print(f"   Individual emotions: {', '.join(metrics['individual_emotions'])}")
        print(f"   Individual flags: {', '.join(metrics['individual_flags'])}")
        
        # Future functionality placeholders
        print(f"\n🔮 Future Actions (Not Yet Implemented):")
        print(f"   [ ] Send to Unity NPC system")
        print(f"   [ ] Store in database")
        print(f"   [ ] Generate NPC dialogue")
        print(f"   [ ] Trigger game events")
        print(f"   [ ] Log for analytics")
        
        print(f"\n{'='*70}\n")
        
        # === PLACEHOLDER: Add your custom logic here ===
        # Example integrations:
        #
        # 1. Unity Integration:
        #    Send to Unity via HTTP/WebSocket
        #    unity_api.send_emotion(data)
        #
        # 2. Database Storage:
        #    db.store_emotion_log(data)
        #
        # 3. NPC Dialogue Generation:
        #    dialogue = npc_system.generate_response(
        #        text=data['text'],
        #        emotion=data['combined_emotion'],
        #        flag=data['combined_flag']
        #    )
        #
        # 4. Analytics:
        #    analytics.track_emotion_event(data)
        
        return True
    
    def handle_client(self, client_socket):
        """Handle incoming connection from Module B"""
        try:
            # Receive data
            data = client_socket.recv(8192).decode('utf-8')
            if not data:
                return
            
            message = json.loads(data.strip())
            
            # Process the data
            self.process_data(message)
            
        except Exception as e:
            print(f"❌ Error handling data: {e}")
        finally:
            client_socket.close()
    
    def run(self):
        """Main server loop"""
        try:
            # Create server socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('', LISTEN_PORT))
            self.server_socket.listen(5)
            
            print(f"✅ Module C listening on port {LISTEN_PORT}")
            print(f"   Waiting for data from Module B...")
            print(f"   Press CTRL+C to stop\n")
            
            while True:
                client_socket, address = self.server_socket.accept()
                self.handle_client(client_socket)
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Shutting down Module C...")
        finally:
            if self.server_socket:
                self.server_socket.close()
            print(f"✅ Module C stopped")
            print(f"   Total messages received: {self.received_count}")

def main():
    module_c = ModuleC()
    module_c.run()

if __name__ == "__main__":
    main()