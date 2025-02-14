# src/utils/analysis/analyze_errors.py

import matplotlib.pyplot as plt # type: ignore
from PIL import Image # type: ignore
import os
import torch # type: ignore
import numpy as np # type: ignore
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parents[3]
sys.path.append(str(project_root))

from src.inference.inference import AmharicPredictor

def visualize_misclassifications():
    predictor = AmharicPredictor(
        model_path=str(project_root / 'models' / 'best_model.pt'),
        char_mapping_path=str(project_root / 'char_mapping.pt')
    )
    
    # Create directory for error analysis
    error_dir = project_root / "error_analysis"
    error_dir.mkdir(exist_ok=True)
    
    print("Starting error analysis...")
    print(f"Project root: {project_root}")
    
    val_dir = project_root / "amharic_dataset" / "val"
    error_count = 0
    
    # Create a figure with multiple subplots
    plt.figure(figsize=(15, 10))
    
    for family in sorted(os.listdir(val_dir)):
        family_path = val_dir / family
        if family_path.is_dir():
            print(f"\nAnalyzing family: {family}")
            
            for char in sorted(os.listdir(family_path)):
                char_path = family_path / char
                if char_path.is_dir():
                    images = [f for f in os.listdir(char_path) if f.endswith('.png')]
                    for img in images[:1]:  # Look at first image of each character
                        test_image = char_path / img
                        pred_idx, pred_char = predictor.predict(str(test_image))
                        
                        if pred_char != char:  # Misclassification
                            error_count += 1
                            plt.subplot(5, 5, error_count % 25 + 1)
                            image = Image.open(test_image)
                            plt.imshow(np.array(image), cmap='gray')
                            plt.title(f'True: {char}\nPred: {pred_char}', fontsize=8)
                            plt.axis('off')
                            
                            if error_count % 25 == 0:
                                plt.savefig(error_dir / f'errors_batch_{error_count//25}.png')
                                plt.clf()
                                print(f"Saved batch {error_count//25} of error images")
    
    if error_count % 25 > 0:
        plt.savefig(error_dir / f'errors_batch_{error_count//25 + 1}.png')
    
    print(f"\nTotal misclassifications visualized: {error_count}")
    print(f"Error visualizations saved in {error_dir} directory")

def analyze_error_patterns():
    """Analyze and print patterns in misclassifications"""
    predictor = AmharicPredictor(
        model_path=str(project_root / 'models' / 'best_model.pt'),
        char_mapping_path=str(project_root / 'char_mapping.pt')
    )
    
    val_dir = project_root / "amharic_dataset" / "val"
    error_patterns = {}
    total_predictions = 0
    
    print("\nAnalyzing error patterns...")
    
    for family in sorted(os.listdir(val_dir)):
        family_path = val_dir / family
        if family_path.is_dir():
            family_errors = 0
            family_total = 0
            
            for char in sorted(os.listdir(family_path)):
                char_path = family_path / char
                if char_path.is_dir():
                    images = [f for f in os.listdir(char_path) if f.endswith('.png')]
                    for img in images:
                        total_predictions += 1
                        family_total += 1
                        test_image = char_path / img
                        pred_idx, pred_char = predictor.predict(str(test_image))
                        
                        if pred_char != char:
                            family_errors += 1
                            error_key = f"{char}->{pred_char}"
                            error_patterns[error_key] = error_patterns.get(error_key, 0) + 1
            
            if family_total > 0:
                error_rate = (family_errors / family_total) * 100
                print(f"Family {family}: Error rate = {error_rate:.2f}% ({family_errors}/{family_total})")
    
    # Print most common error patterns
    print("\nMost common misclassifications:")
    sorted_errors = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)
    for pattern, count in sorted_errors[:10]:
        print(f"{pattern}: {count} times")
    
    total_errors = sum(error_patterns.values())
    print(f"\nOverall error rate: {(total_errors/total_predictions)*100:.2f}% ({total_errors}/{total_predictions})")

if __name__ == "__main__":
    visualize_misclassifications()
    analyze_error_patterns()