# src/utils/analysis/generate_char_mapping.py

import torch # type: ignore
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

def create_character_mappings(dataset_path: Path) -> tuple:
    """Create character to index mappings from the dataset"""
    char_to_idx = {}
    idx_to_char = {}
    idx = 0
    
    train_path = dataset_path / 'train'
    print(f"Scanning directory: {train_path}")
    
    # Go through all family folders
    for family_dir in sorted(train_path.glob("*/")):
        print(f"Processing family: {family_dir.name}")
        
        # Go through character folders in each family
        for char_dir in sorted(family_dir.glob("*/")):
            char = char_dir.name
            if char not in char_to_idx:
                char_to_idx[char] = idx
                idx_to_char[idx] = char
                idx += 1
    
    print(f"Found {len(char_to_idx)} unique characters")
    return char_to_idx, idx_to_char

def main():
    dataset_path = project_root.parent / 'amharic_dataset' 
    output_path = project_root.parent / 'char_mapping.pt'
    
    print("Generating character mappings...")
    char_to_idx, idx_to_char = create_character_mappings(dataset_path)
    
    # Save the mappings
    torch.save({
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char
    }, output_path)
    
    print(f"Character mappings saved to: {output_path}")
    print(f"Total characters: {len(char_to_idx)}")
    
    # Print some example mappings
    print("\nExample mappings:")
    for idx, (char, index) in enumerate(char_to_idx.items()):
        if idx < 5:  # Show first 5 mappings
            print(f"Character: {char} -> Index: {index}")

if __name__ == "__main__":
    main()