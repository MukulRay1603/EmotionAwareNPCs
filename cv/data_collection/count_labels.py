from pathlib import Path

def count_all_labeled_images(labeled_dir='C:/MSML640/Project/data/custom_labeled'):
    """Count all images currently in the labeled directory."""
    base_path = Path(labeled_dir)
    
    if not base_path.exists():
        print(f"Directory {labeled_dir} does not exist yet.")
        return
    
    print("\n" + "="*60)
    print("=== Total Labeled Images ===")
    print("="*60)
    
    total = 0
    label_counts = {}
    
    for label_dir in sorted(base_path.iterdir()):
        if label_dir.is_dir():
            count = len(list(label_dir.glob('*.jpg')))
            label_counts[label_dir.name] = count
            total += count
            print(f"{label_dir.name}: {count} images")
    
    print("="*60)
    print(f"Total: {total} images")
    print("="*60)
    
    # Check status for required categories
    print("\n=== Status (Target: 30-50 per category) ===")
    needed = ['happy', 'stress', 'fatigue', 'neutral']
    for category in needed:
        count = label_counts.get(category, 0)
        if count < 30:
            print(f"⚠️  {category}: {count}/30 - Need {30 - count} more")
        elif count < 50:
            print(f"✓  {category}: {count}/50 - Good!")
        else:
            print(f"✓✓ {category}: {count}/50 - Complete!")
    
    return label_counts

if __name__ == "__main__":
    count_all_labeled_images()
