# src/utils/analysis/character_analysis.py

from inference import AmharicPredictor
from pathlib import Path
from collections import defaultdict

def analyze_character_difficulty():
    project_root = Path(__file__).resolve().parent.parent.parent
    predictor = AmharicPredictor(
        model_path=str(project_root / 'models' / 'best_model.pt'),
        char_mapping_path=str(project_root / 'char_mapping.pt')
    )
    
    print("Starting character analysis...")
    
    stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'mistakes': defaultdict(int)})
    family_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    val_dir = project_root / "amharic_dataset" / "val"
    total_chars = sum(1 for _ in val_dir.glob('*/*'))
    analyzed_chars = 0
    
    for family_path in sorted(val_dir.iterdir()):
        if not family_path.is_dir():
            continue
        print(f"\nAnalyzing family: {family_path.name}")
        
        for char_path in sorted(family_path.iterdir()):
            if not char_path.is_dir():
                continue
            
            images = [image_file for image_file in char_path.iterdir() if image_file.suffix == '.png']
            
            for image_file in images:
                pred_idx, pred_char = predictor.predict(str(image_file))
                
                stats[char_path.name]['total'] += 1
                family_stats[family_path.name]['total'] += 1
                
                if pred_char == char_path.name:
                    stats[char_path.name]['correct'] += 1
                    family_stats[family_path.name]['correct'] += 1
                else:
                    stats[char_path.name]['mistakes'][pred_char] += 1
            
            analyzed_chars += 1
            print(f"Analyzed {analyzed_chars}/{total_chars} characters", end='\r')
    
    print("\n\n=== Character Analysis ===")
    print("\nMost Difficult Characters (Lowest Accuracy):")
    char_accuracies = []
    for char, data in stats.items():
        accuracy = (data['correct'] / data['total']) * 100 if data['total'] > 0 else 0
        char_accuracies.append((char, accuracy, data))
    
    for char, accuracy, data in sorted(char_accuracies, key=lambda x: x[1])[:10]:
        print(f"\n{char}:")
        print(f"Accuracy: {accuracy:.2f}% ({data['correct']}/{data['total']})")
        if data['mistakes']:
            print("Common misclassifications:")
            for wrong_char, count in sorted(data['mistakes'].items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"  → {wrong_char}: {count} times")
    
    print("\n=== Family Analysis ===")
    for family, data in sorted(family_stats.items()):
        accuracy = (data['correct'] / data['total']) * 100 if data['total'] > 0 else 0
        print(f"\n{family}:")
        print(f"Accuracy: {accuracy:.2f}% ({data['correct']}/{data['total']})")

if __name__ == "__main__":
    analyze_character_difficulty()