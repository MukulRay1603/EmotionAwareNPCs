"""
Module A: Text Input with Timestamp
Gets user input, timestamps it, sends to Module B via socket
"""

import socket
import time
import json
from datetime import datetime

# Configuration
MODULE_B_HOST = 'localhost'
MODULE_B_PORT = 5555

class ModuleA:
    def __init__(self):
        """Initialize Module A"""
        self.socket = None
        print(f"\n{'='*70}")
        print(f"  MODULE A: Text Input")
        print(f"{'='*70}\n")
        print(f"Connecting to Module B at {MODULE_B_HOST}:{MODULE_B_PORT}...")
    
    def connect_to_module_b(self):
        """Establish connection to Module B"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((MODULE_B_HOST, MODULE_B_PORT))
            print(f"✅ Connected to Module B\n")
            # Close initial connection - will reconnect for each message
            self.socket.close()
            self.socket = None
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Module B: {e}")
            print("   Make sure Module B is running first!")
            return False
    
    def send_message(self, text):
        """Send text with timestamp to Module B"""
        timestamp = time.time()
        
        message = {
            'text': text,
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp).isoformat()
        }
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Check if socket is still connected, reconnect if needed
                if self.socket is None or retry_count > 0:
                    print("🔄 Reconnecting to Module B...")
                    if self.socket:
                        try:
                            self.socket.close()
                        except:
                            pass
                    
                    self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.socket.connect((MODULE_B_HOST, MODULE_B_PORT))
                    print("✅ Reconnected!")
                
                # Send JSON message
                message_json = json.dumps(message)
                self.socket.sendall(message_json.encode('utf-8') + b'\n')
                
                print(f"\n📤 Sent to Module B:")
                print(f"   Text: {text}")
                print(f"   Timestamp: {timestamp}")
                print(f"   DateTime: {message['datetime']}")
                
                # Wait for response from Module B
                response = self.socket.recv(4096).decode('utf-8')
                if response:
                    response_data = json.loads(response)
                    print(f"\n📥 Response from Module B:")
                    print(f"   Status: {response_data.get('status')}")
                    print(f"   Emotion: {response_data.get('emotion')}")
                    print(f"   Flag: {response_data.get('flag')}")
                    print(f"   Confidence: {response_data.get('confidence'):.3f}")
                    print(f"   Valence: {response_data.get('valence'):+.3f}")
                    print(f"   Arousal: {response_data.get('arousal'):+.3f}")
                    return True
                    
            except (ConnectionRefusedError, ConnectionResetError, 
                    ConnectionAbortedError, BrokenPipeError, OSError) as e:
                retry_count += 1
                print(f"❌ Connection error (attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    time.sleep(0.5)
                else:
                    print("❌ Failed to send message after retries")
                    return False
            except Exception as e:
                print(f"❌ Error: {e}")
                return False
        
        return False
    
    def run(self):
        """Main loop"""
        if not self.connect_to_module_b():
            return
        
        print("="*70)
        print("Ready to send messages!")
        print("Type your message and press Enter")
        print("Type 'quit' to exit")
        print("="*70 + "\n")
        
        try:
            while True:
                user_input = input("Enter text: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                self.send_message(user_input)
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
        finally:
            if self.socket:
                self.socket.close()
            print("✅ Module A stopped")

def main():
    module_a = ModuleA()
    module_a.run()

if __name__ == "__main__":
    main()