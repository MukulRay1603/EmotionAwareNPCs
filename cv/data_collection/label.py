import cv2
import csv
from pathlib import Path
import shutil

class ImageLabeler:
    def __init__(self, input_dir='C:/MSML640/Project/data/custom/custom_raw', 
                 output_dir='C:/MSML640/Project/data/custom_labeled'):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define emotion labels
        self.labels = {
            '1': 'happy',
            '2': 'stress',
            '3': 'fatigue',
            '4': 'neutral',
            '5': 'angry',
            '6': 'sad',
            '7': 'surprise',
            '0': 'skip'  # Skip this image
        }
        
        # Collect all images
        self.images = []
        for video_dir in self.input_dir.iterdir():
            if video_dir.is_dir():
                for img_path in video_dir.glob('*.jpg'):
                    self.images.append(img_path)
        
        self.current_idx = 0
        self.labeled_data = []
        
        print(f"Found {len(self.images)} images to label")
        print("\nControls:")
        print("  1 = happy")
        print("  2 = stress")
        print("  3 = fatigue")
        print("  4 = neutral")
        print("  5 = angry")
        print("  6 = sad")
        print("  7 = surprise")
        print("  0 = skip image")
        print("  'n' = next image (without labeling)")
        print("  'p' = previous image")
        print("  'q' = quit and save")
        print("\nPress any key to start...")
    
    def run(self):
        """Main labeling loop"""
        if not self.images:
            print("No images found!")
            return
        
        cv2.namedWindow('Image Labeler', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Image Labeler', 800, 600)
        
        while self.current_idx < len(self.images):
            img_path = self.images[self.current_idx]
            img = cv2.imread(str(img_path))
            
            if img is None:
                print(f"Could not load {img_path}")
                self.current_idx += 1
                continue
            
            # Display image with info
            display_img = img.copy()
            info_text = f"Image {self.current_idx + 1}/{len(self.images)}: {img_path.name}"
            cv2.putText(display_img, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_img, "Press 1-7 to label, 0 to skip, n=next, p=prev, q=quit", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Image Labeler', display_img)
            
            key = cv2.waitKey(0) & 0xFF
            
            # Handle key presses
            if key == ord('q'):
                print("\nQuitting and saving...")
                break
            elif key == ord('n'):
                self.current_idx += 1
            elif key == ord('p'):
                self.current_idx = max(0, self.current_idx - 1)
            elif chr(key) in self.labels:
                label = self.labels[chr(key)]
                if label != 'skip':
                    self.save_labeled_image(img_path, label)
                    print(f"Labeled {img_path.name} as '{label}'")
                else:
                    print(f"Skipped {img_path.name}")
                self.current_idx += 1
        
        cv2.destroyAllWindows()
        self.save_metadata()
        self.print_summary()
    
    def save_labeled_image(self, img_path, label):
        """Copy image to labeled directory with proper organization"""
        # Create label directory
        label_dir = self.output_dir / label
        label_dir.mkdir(exist_ok=True)
        
        # Copy image to labeled directory
        video_source = img_path.parent.name
        new_filename = f"{video_source}_{img_path.name}"
        dest_path = label_dir / new_filename
        shutil.copy(str(img_path), str(dest_path))
        
        # Record metadata
        self.labeled_data.append({
            'original_path': str(img_path),
            'labeled_path': str(dest_path),
            'label': label,
            'video_source': video_source
        })
    
    def save_metadata(self):
        """Save labeling metadata to CSV"""
        if not self.labeled_data:
            print("No images were labeled.")
            return
        
        csv_path = self.output_dir / 'labeling_metadata.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['original_path', 'labeled_path', 'label', 'video_source'])
            writer.writeheader()
            writer.writerows(self.labeled_data)
        
        print(f"\nMetadata saved to {csv_path}")
    
    def print_summary(self):
        """Print summary of labeling session"""
        if not self.labeled_data:
            return
        
        print("\n" + "="*60)
        print("Labeling Summary:")
        print("="*60)
        
        # Count labels
        label_counts = {}
        for item in self.labeled_data:
            label = item['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        for label, count in sorted(label_counts.items()):
            print(f"{label}: {count} images")
        
        print(f"\nTotal labeled: {len(self.labeled_data)}")
        print(f"Output directory: {self.output_dir}")

if __name__ == "__main__":
    labeler = ImageLabeler()
    labeler.run()