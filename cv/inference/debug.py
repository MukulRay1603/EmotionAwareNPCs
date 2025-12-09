import traceback
import sys

print(f"Python executable: {sys.executable}")

print("\n--- TEST 1: MEDIAPIPE ---")
try:
    import mediapipe as mp
    print("✅ MediaPipe imported successfully")
except Exception:
    print("❌ MediaPipe Failed:")
    traceback.print_exc()

print("\n--- TEST 2: DEEPFACE ---")
try:
    # Deepface depends on TensorFlow/Keras
    from deepface import DeepFace
    print("✅ DeepFace imported successfully")
except Exception:
    print("❌ DeepFace Failed:")
    traceback.print_exc()