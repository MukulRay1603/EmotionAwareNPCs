import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import random
import matplotlib.pyplot as plt

def preprocess_custom_data(input_dir='C:/MSML640/Project/data/custom_labeled', 
                          output_dir='C:/MSML640/Project/data/custom_processed'):
    """
    Preprocess collected frames: detect faces, crop, resize, and augment.
    
    Args:
        input_dir: Directory with raw frames
        output_dir: Directory for processed images
    """
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    stats = {}
    all_base_images = []  # For preview grid
    
    for label_dir in input_path.iterdir():
        if not label_dir.is_dir():
            continue
        
        label = label_dir.name
        print(f"\nProcessing label: {label}")
        
        # Create output directory for this label
        label_output = output_path / label
        label_output.mkdir(exist_ok=True)
        
        processed_count = 0
        image_files = list(label_dir.glob('*.jpg'))
        
        for img_path in image_files:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            results = face_detection.process(rgb_frame)
            
            if not results.detections:
                print(f"No face detected in {img_path.name}, skipping...")
                continue
            
            # Get the largest face
            detection = max(results.detections, 
                          key=lambda d: d.location_data.relative_bounding_box.width * 
                                       d.location_data.relative_bounding_box.height)
            
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            
            # Convert relative coordinates to absolute with padding
            padding = 0.1
            x = int(max(0, (bbox.xmin - padding) * w))
            y = int(max(0, (bbox.ymin - padding) * h))
            width = int(min(w - x, (bbox.width + 2 * padding) * w))
            height = int(min(h - y, (bbox.height + 2 * padding) * h))
            
            # Crop face
            face_crop = frame[y:y+height, x:x+width]
            
            if face_crop.size == 0:
                continue
            
            # Resize to 224x224
            face_resized = cv2.resize(face_crop, (224, 224))
            
            # Normalize to [0, 1]
            face_normalized = face_resized.astype(np.float32) / 255.0
            
            # Convert back to uint8 for saving
            face_to_save = (face_normalized * 255).astype(np.uint8)
            
            # Save base image
            img_idx = processed_count
            base_filename = f"img_{img_idx:04d}_base.jpg"
            base_path = label_output / base_filename
            cv2.imwrite(str(base_path), face_to_save)
            
            # Store for preview
            all_base_images.append((str(base_path), label))
            
            # Generate augmentations
            augmentations = generate_augmentations(face_to_save)
            
            for aug_name, aug_img in augmentations.items():
                aug_filename = f"img_{img_idx:04d}_{aug_name}.jpg"
                cv2.imwrite(str(label_output / aug_filename), aug_img)
            
            processed_count += 1
            
            if processed_count % 10 == 0:
                print(f"Processed {processed_count}/{len(image_files)} images")
        
        total_with_aug = processed_count * (1 + len(augmentations))
        stats[label] = {
            'base_images': processed_count,
            'total_with_augmentations': total_with_aug
        }
        
        print(f"Completed {label}: {processed_count} base + {total_with_aug} total with augmentations")
    
    # Create preview grid
    create_preview_grid(all_base_images, output_dir='C:/MSML640/Project/data/previews')
    
    return stats

def generate_augmentations(image):
    """Generate brightness, blur, and noise augmentations."""
    augmentations = {}
    
    # Brightness adjustment
    brightness_factor = random.uniform(0.7, 1.3)
    bright_img = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)
    augmentations['bright'] = bright_img
    
    # Gaussian blur
    blur_img = cv2.GaussianBlur(image, (5, 5), 0)
    augmentations['blur'] = blur_img
    
    # Gaussian noise
    noise = np.random.normal(0, 10, image.shape).astype(np.float32)
    noisy_img = np.clip(image + noise, 0, 255).astype(np.uint8)
    augmentations['noise'] = noisy_img
    
    return augmentations

def create_preview_grid(all_images, output_dir='C:/MSML640/Project/data/previews'):
    """Create a preview grid of 5 sample images."""
    
    if not all_images:
        print("No images to preview!")
        return
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Select 5 random images
    sample_images = random.sample(all_images, min(5, len(all_images)))
    
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    fig.suptitle('Sample Processed Images (224x224, Normalized)', fontsize=14)
    
    for idx, (img_path, label) in enumerate(sample_images):
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if len(sample_images) == 1:
            axes.imshow(img_rgb)
            axes.axis('off')
            axes.set_title(f'{label}', fontsize=12)
        else:
            axes[idx].imshow(img_rgb)
            axes[idx].axis('off')
            axes[idx].set_title(f'{label}', fontsize=12)
    
    # Hide unused subplots if less than 5 images
    if len(sample_images) < 5:
        for idx in range(len(sample_images), 5):
            axes[idx].axis('off')
    
    plt.tight_layout()
    preview_path = f"{output_dir}/preview_grid.jpg"
    plt.savefig(preview_path, dpi=150, bbox_inches='tight')
    print(f"\nPreview grid saved to {preview_path}")
    plt.close()

if __name__ == "__main__":
    stats = preprocess_custom_data()
    
    print("\n=== Preprocessing Summary ===")
    for label, counts in stats.items():
        print(f"{label}:")
        print(f"  Base images: {counts['base_images']}")
        print(f"  Total with augmentations: {counts['total_with_augmentations']}")
    
    # Note: Preview grid is already generated inside preprocess_custom_data()