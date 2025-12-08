"""
Module C: Data Receiver with Local LLM Integration
Receives combined text + emotion data from Module B
Generates NPC dialogue using local Llama model via Ollama
"""

import socket
import json
from datetime import datetime
import requests

# Configuration
LISTEN_PORT = 5556
OLLAMA_API = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:1b"  # Lightweight 1B parameter model

class ModuleC:
    def __init__(self):
        """Initialize Module C"""
        self.server_socket = None
        self.received_count = 0
        
        print(f"\n{'='*70}")
        print(f"  MODULE C: NPC Dialogue Generator with Local LLM")
        print(f"{'='*70}\n")
        print(f"Listening on port {LISTEN_PORT}")
        print(f"Using Ollama model: {OLLAMA_MODEL}")
        print(f"API: {OLLAMA_API}\n")
        
        # Check if Ollama is running
        self.check_ollama()
    
    def check_ollama(self):
        """Check if Ollama is running and model is available"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                if any(OLLAMA_MODEL in name for name in model_names):
                    print(f"✅ Ollama is running")
                    print(f"✅ Model '{OLLAMA_MODEL}' is available\n")
                else:
                    print(f"⚠️  Ollama is running but model '{OLLAMA_MODEL}' not found")
                    print(f"   Available models: {', '.join(model_names)}")
                    print(f"\n   To install: ollama pull {OLLAMA_MODEL}\n")
            else:
                print(f"⚠️  Ollama responded with status {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"❌ Ollama is not running!")
            print(f"   Please install and start Ollama:")
            print(f"   1. Download from: https://ollama.com/download")
            print(f"   2. Install and run Ollama")
            print(f"   3. Run: ollama pull {OLLAMA_MODEL}")
            print(f"\n   Module C will continue but LLM features will be disabled.\n")
        except Exception as e:
            print(f"⚠️  Could not check Ollama: {e}\n")
    
    def create_npc_prompt(self, data):
        """Create prompt for NPC dialogue generation"""
        emotion = data['combined_emotion']
        flag = data['combined_flag']
        text = data['text']
        metrics = data['emotion_metrics']
        
        valence = metrics['valence']
        arousal = metrics['arousal']
        is_fatigued = metrics['is_fatigued']
        
        # Describe emotional state in natural language
        emotion_desc = f"The player seems {emotion}"
        
        if flag == "STRESS":
            emotion_desc += " and stressed (negative feelings with high intensity)"
        elif flag == "RUSH":
            emotion_desc += " and rushed (moving quickly, high energy)"
        elif flag == "FATIGUE":
            emotion_desc += " and fatigued (tired, low energy)"
        elif flag == "NEUTRAL":
            emotion_desc += " and calm"
        
        if valence > 0.3:
            emotion_desc += ", with positive feelings"
        elif valence < -0.3:
            emotion_desc += ", with negative feelings"
        
        if arousal > 0.5:
            emotion_desc += ", and high energy"
        elif arousal < -0.2:
            emotion_desc += ", and low energy"
        
        if is_fatigued:
            emotion_desc += ". They look tired"
        
        # Create prompt
        prompt = f"""You are a supportive and empathetic NPC in a video game. Based on the player's message and emotional state, generate a short, natural response (1-2 sentences).

Player's message: "{text}"

Player's emotional state: {emotion_desc}

As a supportive NPC, respond in a way that:
- Acknowledges their emotional state appropriately
- Is helpful and encouraging if they're stressed or struggling
- Celebrates with them if they're happy
- Suggests a break if they're fatigued
- Keeps the tone natural and game-appropriate
- IMPORTANT: Keep response very short - maximum 1-2 brief sentences

NPC response:"""
        
        return prompt
    
    def generate_npc_dialogue(self, prompt):
        """Generate NPC dialogue using local Llama model"""
        try:
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 50  # Reduced for shorter responses
                }
            }
            
            print(f"\n🤖 Generating NPC response with {OLLAMA_MODEL}...")
            
            response = requests.post(
                OLLAMA_API,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                npc_response = result.get('response', '').strip()
                return npc_response
            else:
                print(f"❌ LLM API error: {response.status_code}")
                return None
                
        except requests.exceptions.ConnectionError:
            print(f"❌ Could not connect to Ollama. Make sure it's running.")
            return None
        except Exception as e:
            print(f"❌ Error generating dialogue: {e}")
            return None
    
    def process_data(self, data):
        """Process received data and generate NPC dialogue"""
        self.received_count += 1
        
        print(f"\n{'='*70}")
        print(f"  📨 RECEIVED DATA #{self.received_count}")
        print(f"{'='*70}")
        
        print(f"\n📝 Player Input:")
        print(f"   \"{data['text']}\"")
        
        print(f"\n📅 Timestamp:")
        print(f"   {data['datetime']}")
        
        print(f"\n😊 Emotion Analysis:")
        metrics = data['emotion_metrics']
        print(f"   Emotion: {data['combined_emotion']}")
        print(f"   Context Flag: {data['combined_flag']}")
        print(f"   Confidence: {metrics['confidence']:.3f}")
        print(f"   Valence: {metrics['valence']:+.3f} (negative ← 0 → positive)")
        print(f"   Arousal: {metrics['arousal']:+.3f} (calm ← 0 → excited)")
        print(f"   Fatigue: {'Yes' if metrics['is_fatigued'] else 'No'}")
        
        print(f"\n📊 Details:")
        print(f"   Samples averaged: {metrics['num_samples']}")
        print(f"   Individual emotions: {', '.join(metrics['individual_emotions'])}")
        
        # Generate NPC dialogue
        print(f"\n{'─'*70}")
        print(f"  🎮 GENERATING NPC DIALOGUE")
        print(f"{'─'*70}")
        
        prompt = self.create_npc_prompt(data)
        
        print(f"\n📋 Prompt:")
        print(f"   (Emotion context + player message)")
        
        npc_dialogue = self.generate_npc_dialogue(prompt)
        
        if npc_dialogue:
            # Truncate if too long (safety)
            sentences = npc_dialogue.split('.')
            if len(sentences) > 2:
                npc_dialogue = '. '.join(sentences[:2]) + '.'
            
            print(f"\n💬 NPC SAYS:")
            print(f"   \"{npc_dialogue}\"")
            print(f"\n{'='*70}")
            
            # Prepare response
            response_data = {
                'status': 'success',
                'player_text': data['text'],
                'npc_response': npc_dialogue,
                'emotion': data['combined_emotion'],
                'flag': data['combined_flag'],
                'confidence': metrics['confidence'],
                'valence': metrics['valence'],
                'arousal': metrics['arousal'],
                'is_fatigued': metrics['is_fatigued'],
                'timestamp': data['timestamp']
            }
            
            # Save for Unity integration
            try:
                with open('cv/output/npc_dialogue.json', 'w') as f:
                    json.dump(response_data, f, indent=2)
                print(f"💾 Saved to: cv/output/npc_dialogue.json")
            except Exception as e:
                print(f"⚠️  Could not save dialogue: {e}")
            
            return response_data
        else:
            print(f"\n⚠️  Could not generate NPC dialogue")
            print(f"   Using fallback response based on emotion...")
            
            # Fallback responses
            fallback = self.generate_fallback_response(data)
            print(f"\n💬 NPC SAYS (Fallback):")
            print(f"   \"{fallback}\"")
            
            response_data = {
                'status': 'fallback',
                'player_text': data['text'],
                'npc_response': fallback,
                'emotion': data['combined_emotion'],
                'flag': data['combined_flag'],
                'confidence': metrics['confidence'],
                'valence': metrics['valence'],
                'arousal': metrics['arousal'],
                'is_fatigued': metrics['is_fatigued'],
                'timestamp': data['timestamp']
            }
            
            return response_data
        
        print(f"\n{'='*70}\n")
    
    def generate_fallback_response(self, data):
        """Generate simple rule-based response if LLM unavailable"""
        emotion = data['combined_emotion']
        flag = data['combined_flag']
        
        responses = {
            'STRESS': [
                "Take a deep breath. You've got this!",
                "Don't give up! Sometimes the toughest challenges bring the greatest rewards.",
                "Hey, it's okay to feel frustrated. Want to take a quick break?"
            ],
            'RUSH': [
                "Whoa, slow down a bit! No need to rush.",
                "You're moving fast! Make sure to think things through.",
                "High energy! Channel that into focused action."
            ],
            'FATIGUE': [
                "You look tired. Maybe take a break and come back refreshed?",
                "Rest is important too. Your health matters!",
                "Feeling weary? There's no shame in resting at the inn."
            ],
            'NEUTRAL': {
                'happy': "Great to see you in good spirits!",
                'sad': "Everything alright? I'm here if you need anything.",
                'angry': "I sense some tension. Want to talk about it?",
                'neutral': "How can I help you today?",
                'surprise': "Something caught you off guard?",
                'fear': "Don't worry, you're safe here.",
                'disgust': "Not your favorite, I take it?"
            }
        }
        
        import random
        
        if flag in responses and isinstance(responses[flag], list):
            return random.choice(responses[flag])
        elif flag == 'NEUTRAL' and emotion in responses['NEUTRAL']:
            return responses['NEUTRAL'][emotion]
        else:
            return "I'm here to help. What do you need?"
    
    def handle_client(self, client_socket):
        """Handle incoming connection from Module B"""
        try:
            # Receive data
            buffer = b''
            while True:
                chunk = client_socket.recv(1024)
                if not chunk:
                    break
                buffer += chunk
                if b'\n' in buffer:
                    break
            
            if not buffer:
                return
            
            data_str = buffer.decode('utf-8').strip()
            message = json.loads(data_str)
            
            # Process the data and generate response
            response_data = self.process_data(message)
            
            # Send response back
            if response_data:
                response_json = json.dumps(response_data) + '\n'
                client_socket.sendall(response_json.encode('utf-8'))
                print(f"📤 Sent response back to Module B")
            
        except Exception as e:
            print(f"❌ Error handling data: {e}")
            import traceback
            traceback.print_exc()
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