# dataset_validator.py

import os
from pathlib import Path

def validate_dataset_structure():
    """Validate the Amharic dataset structure and print detailed information."""
    
    # Get project root and dataset path
    project_root = Path.cwd()
    dataset_path = project_root / "amharic_dataset"
    
    print(f"Checking dataset at: {dataset_path}")
    
    if not dataset_path.exists():
        print(f"ERROR: Dataset directory not found at {dataset_path}")
        print("Please run the dataset generator first:")
        print("python src/training/generated_dataset.py")
        return False
    
    # Check main subdirectories
    train_dir = dataset_path / "train"
    val_dir = dataset_path / "val"
    
    if not train_dir.exists() or not val_dir.exists():
        print("ERROR: Missing required directories:")
        print(f"train directory exists: {train_dir.exists()}")
        print(f"val directory exists: {val_dir.exists()}")
        return False
    
    # Analyze directory structure
    def analyze_directory(directory):
        family_count = 0
        char_count = 0
        image_count = 0
        empty_dirs = []
        
        for family_path in directory.iterdir():
            if family_path.is_dir():
                family_count += 1
                family_has_images = False
                
                for char_path in family_path.iterdir():
                    if char_path.is_dir():
                        char_count += 1
                        images = list(char_path.glob("*.png"))
                        current_images = len(images)
                        image_count += current_images
                        
                        if current_images == 0:
                            empty_dirs.append(str(char_path.relative_to(dataset_path)))
                        else:
                            family_has_images = True
                
                if not family_has_images:
                    empty_dirs.append(str(family_path.relative_to(dataset_path)))
        
        return family_count, char_count, image_count, empty_dirs
    
    # Analyze train and val directories
    print("\nAnalyzing training directory...")
    train_families, train_chars, train_images, train_empty = analyze_directory(train_dir)
    
    print("\nAnalyzing validation directory...")
    val_families, val_chars, val_images, val_empty = analyze_directory(val_dir)
    
    # Print summary
    print("\nDataset Summary:")
    print(f"Training:")
    print(f"- Families: {train_families}")
    print(f"- Characters: {train_chars}")
    print(f"- Images: {train_images}")
    
    print(f"\nValidation:")
    print(f"- Families: {val_families}")
    print(f"- Characters: {val_chars}")
    print(f"- Images: {val_images}")
    
    if train_empty or val_empty:
        print("\nWARNING: Empty directories found:")
        for empty_dir in train_empty + val_empty:
            print(f"- {empty_dir}")
    
    # Check if dataset is valid
    is_valid = (train_images > 0 and val_images > 0)
    
    if is_valid:
        print("\nDataset structure appears to be valid.")
    else:
        print("\nERROR: Dataset is invalid - no images found in one or both splits.")
        print("Please run the dataset generator to create the dataset:")
        print("python src/training/generated_dataset.py")
    
    return is_valid

if __name__ == "__main__":
    validate_dataset_structure()