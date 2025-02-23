# src/inference/inference.py

import torch
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

from model import AmharicCNN

class AmharicPredictor:
    def __init__(
        self,
        model_path='models/best_model.pt',
        char_mapping_path='char_mapping.pt',
        image_size=(96, 96)
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        
        # Load character mappings
        print(f"Loading character mappings from {char_mapping_path}")
        mappings = torch.load(char_mapping_path)
        self.char_to_idx = mappings['char_to_idx']
        self.idx_to_char = mappings['idx_to_char']
        print(f"Loaded {len(self.char_to_idx)} characters")
        
        # Load checkpoint first to get num_classes
        print(f"Loading checkpoint from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get num_classes from the checkpoint's state dict
        fc_weight = checkpoint['model_state_dict']['fc_layers.10.weight']
        num_classes = fc_weight.shape[0]
        print(f"Detected {num_classes} classes in checkpoint")
        
        # Initialize model with correct number of classes
        print("Initializing model...")
        self.model = AmharicCNN(num_classes=num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        print("Model initialized successfully")
        
        # Verify character mapping matches model output size
        if len(self.char_to_idx) != num_classes:
            print(f"Warning: Character mapping size ({len(self.char_to_idx)}) "
                  f"doesn't match model output size ({num_classes})")
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def predict(self, image_path):
        """
        Predict character from image file
        
        Args:
            image_path (str): Path to image file
        
        Returns:
            tuple: (predicted_index, predicted_character)
        """
        # Load and preprocess image
        with Image.open(image_path).convert('L') as img:
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            pred_idx = probabilities.argmax().item()
            pred_char = self.idx_to_char[pred_idx]
        
        return pred_idx, pred_char
    
    def predict_with_confidence(self, image_path):
        """
        Predict character with confidence score
        
        Args:
            image_path (str): Path to image file
        
        Returns:
            tuple: (predicted_index, predicted_character, confidence)
        """
        # Load and preprocess image
        with Image.open(image_path).convert('L') as img:
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            pred_idx = probabilities.argmax().item()
            confidence = probabilities[pred_idx].item()
            pred_char = self.idx_to_char[pred_idx]
        
        return pred_idx, pred_char, confidence
    
    def predict_with_correction(self, image_path):
        """Predict with error correction based on known error patterns"""
        # Get initial prediction with confidence
        pred_idx, pred_char, confidence = self.predict_with_confidence(image_path)
        
        # Get top-k predictions for context
        top_k = self.get_top_k_predictions(image_path, k=5)
        
        # Apply error correction
        corrected_char, corrected_conf = self.error_corrector.correct_prediction(
            pred_char, confidence, top_k
        )
        
        return corrected_char, corrected_conf
    
    def get_top_k_predictions(self, image_path, k=5):
        """
        Get top k predictions for an image
        
        Args:
            image_path (str): Path to image file
            k (int): Number of top predictions to return
        
        Returns:
            list: List of tuples (character, probability)
        """
        # Load and preprocess image
        with Image.open(image_path).convert('L') as img:
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # Get top k predictions
            top_probs, top_indices = probabilities.topk(k)
            
            predictions = [
                (self.idx_to_char[idx.item()], prob.item())
                for idx, prob in zip(top_indices, top_probs)
            ]
        
        return predictions

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Amharic OCR Inference')
    parser.add_argument('--image_path', type=str, required=True, help='Path to single image or directory of images')
    parser.add_argument('--model_path', type=str, default='models/best_model.pt', help='Path to model weights')
    parser.add_argument('--char_mapping', type=str, default='char_mapping.pt', help='Path to character mapping file')
    parser.add_argument('--output', type=str, default='predictions.txt', help='Output file for predictions')

    try:
        # If no arguments provided, show help
        if len(sys.argv) == 1:
            print("Please provide an image path to process. Example usage:")
            print("python src/inference/inference.py --image_path test_images/your_image.png")
            sys.exit(1)

        args = parser.parse_args()
        
        # Initialize predictor
        predictor = AmharicPredictor(
            model_path=args.model_path,
            char_mapping_path=args.char_mapping
        )
        
        # Setup paths
        image_path = Path(args.image_path)
        output_file = Path(args.output)
        
        if not image_path.exists():
            print(f"Error: Image path {image_path} does not exist!")
            sys.exit(1)
        
        # Process images
        with open(output_file, 'w', encoding='utf-8') as f:
            if image_path.is_file():
                # Single image processing
                images = [image_path]
            else:
                # Directory processing
                images = list(image_path.glob('*.png')) + list(image_path.glob('*.jpg'))
            
            print(f"Found {len(images)} images to process")
            
            for img_path in images:
                try:
                    # Get predictions
                    regular_pred = predictor.predict_with_confidence(img_path)
                    top_k = predictor.get_top_k_predictions(img_path, k=3)
                    
                    # Write results
                    f.write(f"\nImage: {img_path.name}\n")
                    f.write(f"Prediction: {regular_pred[1]} (confidence: {regular_pred[2]:.2%})\n")
                    f.write("Top 3 predictions:\n")
                    for char, prob in top_k:
                        f.write(f"  {char}: {prob:.2%}\n")
                    
                    # Print to console
                    print(f"\nProcessed: {img_path.name}")
                    print(f"Prediction: {regular_pred[1]} (confidence: {regular_pred[2]:.2%})")
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    f.write(f"\nError processing {img_path}: {e}\n")
                    continue

        print(f"\nResults saved to: {output_file}")

    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)