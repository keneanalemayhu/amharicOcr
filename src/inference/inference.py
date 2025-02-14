# src/inference/inference.py

import torch # type: ignore
from PIL import Image # type: ignore
import torchvision.transforms as transforms # type: ignore
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
        
        # Load character mappings with weights_only=True for safety
        mappings = torch.load(char_mapping_path, weights_only=False)  # Will switch to True in future
        self.char_to_idx = mappings['char_to_idx']
        self.idx_to_char = mappings['idx_to_char']
        
        # Initialize model
        self.model = AmharicCNN(num_classes=len(self.char_to_idx))
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
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
    # Example usage
    predictor = AmharicPredictor()
    test_image = "path_to_test_image.png"
    
    # Single prediction
    pred_idx, pred_char = predictor.predict(test_image)
    print(f"Predicted character: {pred_char}")
    
    # Prediction with confidence
    pred_idx, pred_char, conf = predictor.predict_with_confidence(test_image)
    print(f"Predicted character: {pred_char} with confidence: {conf:.2%}")
    
    # Top k predictions
    predictions = predictor.get_top_k_predictions(test_image, k=3)
    print("\nTop 3 predictions:")
    for char, prob in predictions:
        print(f"{char}: {prob:.2%}")