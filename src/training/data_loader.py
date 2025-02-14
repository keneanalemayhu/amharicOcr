# src/training/data_loader.py

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
from typing import Tuple, Dict, List

class AmharicDataset(Dataset):
    def __init__(self, data_dir: str, transform=None, train: bool = True):
        self.data_dir = Path(data_dir) / ('train' if train else 'val')
        self.transform = transform
        self.samples = []
        self.char_to_idx = {}
        self.idx_to_char = {}
        
        # Load or create character mappings first
        self._setup_character_mapping()
        
        # Then load dataset
        self._load_dataset()

    def _setup_character_mapping(self):
        """Setup character to index mapping"""
        # Store characters in a list to maintain order
        unique_chars = []
        
        # First pass: collect all unique characters
        for family_dir in sorted(self.data_dir.iterdir()):
            if family_dir.is_dir():
                for char_dir in sorted(family_dir.iterdir()):
                    if char_dir.is_dir():
                        char = char_dir.name
                        if char not in unique_chars:
                            unique_chars.append(char)
        
        # Create mappings
        self.char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(unique_chars)}
        
        # Debug print
        print("\nCharacter mapping:")
        for idx, char in sorted(self.idx_to_char.items())[:10]:  # Print first 10 mappings
            print(f"{char} -> {idx}")

    def _load_dataset(self):
        """Load dataset paths and labels"""
        print(f"\nLoading dataset from: {self.data_dir}")
        for family_dir in sorted(self.data_dir.iterdir()):
            if family_dir.is_dir():
                for char_dir in sorted(family_dir.iterdir()):
                    if char_dir.is_dir():
                        char = char_dir.name
                        if char in self.char_to_idx:
                            char_idx = self.char_to_idx[char]
                            
                            # Get all image files in the character folder
                            for img_path in char_dir.glob('*.png'):
                                self.samples.append((str(img_path), char_idx))
        
        print(f"Found {len(self.samples)} samples")
        
        # Debug print some samples
        print("\nFirst few samples:")
        for i, (path, idx) in enumerate(self.samples[:5]):
            char = self.idx_to_char[idx]
            print(f"Sample {i}: {path} -> {char} (index: {idx})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Load image
        with Image.open(img_path).convert('L') as img:  # Convert to grayscale
            if self.transform:
                img = self.transform(img)
            else:
                # Default transform if none provided
                transform = transforms.Compose([
                    transforms.Resize((96, 96)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])
                img = transform(img)
        
        return img, label

def create_data_loaders(
    dataset_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_transform = None,
    val_transform = None
) -> Tuple[DataLoader, DataLoader, Dict, Dict]:
    """
    Create data loaders for training and validation sets.
    """
    
    # Default transforms if none provided
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.RandomRotation(2),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                fill=255
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    if val_transform is None:
        val_transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    # Create datasets
    train_dataset = AmharicDataset(
        data_dir=dataset_path,
        transform=train_transform,
        train=True
    )
    
    val_dataset = AmharicDataset(
        data_dir=dataset_path,
        transform=val_transform,
        train=False
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return (
        train_loader,
        val_loader,
        train_dataset.char_to_idx,
        train_dataset.idx_to_char
    )