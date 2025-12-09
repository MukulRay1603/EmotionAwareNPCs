"""
Module A (Unity Mode)
Listens for Unity on Port 6000 -> Forwards to Module B -> Returns response to Unity
"""
import socket
import json
import time
from datetime import datetime

# CONFIGURATION
UNITY_PORT = 6000        # Unity connects here
MODULE_B_PORT = 5555     # We talk to your AI Brain here (Module B)

def send_to_brain(text):
    """Wraps text in JSON with timestamp (just like module_a_input.py) and sends to Module B"""
    timestamp = time.time()
    
    # EXACT same structure as your original module_a_input.py
    message = {
        'text': text,
        'timestamp': timestamp,
        'datetime': datetime.fromtimestamp(timestamp).isoformat()
    }
    
    try:
        # Connect to Module B
        sock_b = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock_b.connect(('localhost', MODULE_B_PORT))
        
        # Send Data
        sock_b.sendall(json.dumps(message).encode('utf-8') + b'\n')
        print(f"   --> Forwarded to Brain: {text}")

        # Wait for Response
        response = sock_b.recv(8192).decode('utf-8')
        sock_b.close()
        return response
    except Exception as e:
        print(f"❌ Error talking to Module B: {e}")
        return json.dumps({"npc_response": "My brain is offline."})

def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server.bind(('localhost', UNITY_PORT))
        server.listen(1)
        print(f"✅ MODULE A (UNITY) READY on Port {UNITY_PORT}")
        
        while True:
            # Wait for Unity to connect
            conn, addr = server.accept()
            print(f"🔗 Unity Connected: {addr}")
            
            try:
                while True:
                    data = conn.recv(4096)
                    if not data: break
                    
                    user_text = data.decode('utf-8').strip()
                    if not user_text: continue
                    
                    print(f"\n📥 From Unity: {user_text}")

                    # 1. Send to Module B (Your AI Chain)
                    ai_response = send_to_brain(user_text)

                    # 2. Send result back to Unity
                    conn.sendall(ai_response.encode('utf-8') + b'\n')
                    print("   <-- Sent answer back to Unity")
                    
            except Exception as e:
                print(f"⚠️ Connection interrupted: {e}")
            finally:
                conn.close()
    except Exception as e:
        print(f"❌ Server Error: {e}")
    finally:
        server.close()

if __name__ == "__main__":
    start_server()