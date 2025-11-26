import cv2
import mediapipe as mp
import os
import csv
from pathlib import Path

def collect_frames(video_path, num_frames=50, output_dir='data/custom_raw'):
    """
    Extract frames with faces from video and save them.
    
    Args:
        video_path: Path to video file
        num_frames: Target number of frames to collect
        output_dir: Base directory for saving frames
    """
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    
    # Create output directory using video filename
    video_name = Path(video_path).stem
    video_dir = Path(output_dir) / video_name
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    frame_count = 0
    saved_count = 0
    csv_rows = []
    
    print(f"Processing video: {video_path}")
    
    while saved_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached")
            break
        
        frame_count += 1
        
        # Skip frames to avoid duplicates (process every 10th frame)
        if frame_count % 10 != 0:
            continue
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = face_detection.process(rgb_frame)
        
        if results.detections:
            # Save frame
            frame_filename = f"frame_{saved_count:04d}.jpg"
            frame_path = video_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)
            
            # Add CSV row (with placeholders for manual annotation)
            csv_rows.append({
                'image_path': str(frame_path),
                'video_source': video_name,
                'label': '?',
                'lighting': '?',
                'occlusion': '?'
            })
            
            saved_count += 1
            if saved_count % 10 == 0:
                print(f"Saved {saved_count}/{num_frames} frames")
    
    cap.release()
    
    # Write CSV
    csv_path = video_dir / "metadata.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_path', 'video_source', 'label', 'lighting', 'occlusion'])
        writer.writeheader()
        writer.writerows(csv_rows)
    
    print(f"\nCompleted: Saved {saved_count} frames to {video_dir}")
    print(f"Metadata saved to {csv_path}")
    print(f"Please manually annotate label, lighting, and occlusion in the CSV file")

def batch_process_videos(clips_dir='data/custom/raw_clips', output_dir='data/custom_raw', num_frames=50):
    """
    Process all videos in a directory, creating a folder for each video.
    """
    clips_path = Path(clips_dir)
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4']
    
    processed_count = 0
    
    for video_file in clips_path.iterdir():
        if video_file.suffix in video_extensions:
            print(f"\n{'='*60}")
            print(f"Processing: {video_file.name}")
            print(f"{'='*60}")
            
            collect_frames(str(video_file), num_frames, output_dir)
            processed_count += 1
    
    print(f"\n{'='*60}")
    print(f"Batch processing complete: {processed_count} videos processed")
    print(f"{'='*60}")

def count_images_per_video(base_dir='data/custom_raw'):
    """Count images in each video folder."""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Directory {base_dir} does not exist yet.")
        return {}
    
    summary = {}
    
    for video_dir in base_path.iterdir():
        if video_dir.is_dir():
            count = len(list(video_dir.glob('*.jpg')))
            summary[video_dir.name] = count
    
    return summary

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect frames from streamer videos')
    parser.add_argument('--video', help='Path to single video file')
    parser.add_argument('--batch', action='store_true', help='Batch process all videos in clips directory')
    parser.add_argument('--clips_dir', default='data/custom/raw_clips', help='Directory containing video clips')
    parser.add_argument('--num_frames', type=int, default=50, help='Number of frames to collect per video')
    parser.add_argument('--output_dir', default='data/custom_raw', help='Output directory')
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch process all videos
        batch_process_videos(args.clips_dir, args.output_dir, args.num_frames)
    elif args.video:
        # Process single video
        collect_frames(args.video, args.num_frames, args.output_dir)
    else:
        parser.error("Either provide --batch flag or --video argument")
    
    # Print summary
    print("\n" + "="*60)
    print("=== Collection Summary ===")
    print("="*60)
    summary = count_images_per_video(args.output_dir)
    if summary:
        for video_name, count in summary.items():
            print(f"{video_name}: {count} images")
    else:
        print("No images collected yet or output directory doesn't exist.")