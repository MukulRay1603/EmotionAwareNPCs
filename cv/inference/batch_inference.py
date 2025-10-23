import os
import numpy as np
import onnxruntime as ort
import cv2
from tqdm import tqdm

# Paths
MODEL_PATH = 'c:/MSML640/Project/EmotionAwareNPCs/models/fer_model.onnx'
TEST_DIR = 'C:/MSML640/Project/data/fer2013/test'
Y_PRED_PATH = 'C:/MSML640/Project/data/fer2013/y_pred.txt'
Y_TRUE_PATH = 'C:/MSML640/Project/data/fer2013/y_true.txt'

# FER-2013 label order
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load ONNX model
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

y_true = []
y_pred = []

for class_idx, class_name in enumerate(EMOTION_LABELS):
    class_dir = os.path.join(TEST_DIR, class_name)
    if not os.path.isdir(class_dir):
        print(f"Warning: No directory found for class '{class_name}' at {class_dir}")
        continue
    image_count = 0
    for fname in tqdm(os.listdir(class_dir), desc=f"{class_name}"):
        fpath = os.path.join(class_dir, fname)
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (64, 64))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # Add channel
        img = np.expand_dims(img, axis=0)  # Add batch
        # ONNX expects shape (batch, channel, height, width)
        outputs = session.run([output_name], {input_name: img})
        emotion_probs = outputs[0][0]
        pred_idx = int(np.argmax(emotion_probs))
        y_pred.append(pred_idx)
        y_true.append(class_idx)
        image_count += 1
    print(f"Processed {image_count} images for class '{class_name}'")

print(f"y_pred length: {len(y_pred)}")
print(f"y_true length: {len(y_true)}")
print("First 10 predicted labels:", y_pred[:10])
print("First 10 true labels:", y_true[:10])

# Ensure output directory exists
os.makedirs(os.path.dirname(Y_PRED_PATH), exist_ok=True)

with open(Y_PRED_PATH, 'w') as f:
    for label in y_pred:
        f.write(f"{label}\n")
with open(Y_TRUE_PATH, 'w') as f:
    for label in y_true:
        f.write(f"{label}\n")

print(f"Saved predictions to {Y_PRED_PATH} and true labels to {Y_TRUE_PATH}")
