# src/inference/error_correction.py

import torch
from typing import Dict, Tuple, List

class AmharicErrorCorrector:
    """
    Handles post-processing corrections based on known error patterns
    """
    def __init__(self):
        # Known problematic character mappings from error analysis
        self.number_corrections = {
            '3': ['1', '6', '9', '4', '8'],  # Numbers commonly misclassified as 3
            'ፓ': ['ፖ'],
            'ሸ': ['ሽ'],
            'ሽ': ['ሸ'],
            'ሙ': ['ው'],
            'ጎ': ['ኀ']
        }
        
        # Character families with high error rates
        self.high_error_families = {
            'ጨ': 0.1105,  # 11.05% error rate
            'ፐ': 0.1118,  # 11.18% error rate
            '፩': 0.2979,  # 29.79% error rate for numbers
            '፪': 0.2759,
            '፫': 0.0909,
            '፬': 0.2632,
            '፭': 0.1538,
            '፮': 0.3333,
            '፯': 0.3158,
            '፰': 0.3235,
            '፱': 0.3030
        }

    def correct_prediction(self, char: str, confidence: float, top_k: List[Tuple[str, float]]) -> Tuple[str, float]:
        """
        Apply correction rules based on known error patterns
        
        Args:
            char: Predicted character
            confidence: Prediction confidence
            top_k: List of top-k predictions with confidences
        
        Returns:
            Tuple of (corrected_char, new_confidence)
        """
        # Special handling for numbers due to high error rates
        if char in '፩፪፫፬፭፮፯፰፱':
            return self._handle_number_prediction(char, confidence, top_k)
            
        # Handle known problematic character families
        if any(char.startswith(family) for family in ['ጨ', 'ፐ']):
            return self._handle_error_prone_family(char, confidence, top_k)
            
        return char, confidence

    def _handle_number_prediction(self, char: str, confidence: float, top_k: List[Tuple[str, float]]) -> Tuple[str, float]:
        """Special handling for numeric predictions"""
        # If confidence is very high, trust the prediction
        if confidence > 0.95:
            return char, confidence
            
        # Look for alternative predictions with similar confidence
        for alt_char, alt_conf in top_k[1:]:  # Skip first as it's the current prediction
            if alt_char in '፩፪፫፬፭፮፯፰፱':
                conf_diff = confidence - alt_conf
                if conf_diff < 0.1:  # If confidences are close
                    # Prefer the alternative if current char is commonly misclassified
                    if char == '፫' and alt_char in ['፩', '፮', '፱', '፬', '፰']:
                        return alt_char, alt_conf
                        
        return char, confidence

    def _handle_error_prone_family(self, char: str, confidence: float, top_k: List[Tuple[str, float]]) -> Tuple[str, float]:
        """Handle predictions for character families with high error rates"""
        if confidence < 0.8:  # Lower confidence threshold for known problematic families
            # Check alternative predictions
            for alt_char, alt_conf in top_k[1:]:
                if char in self.number_corrections and alt_char in self.number_corrections[char]:
                    if (confidence - alt_conf) < 0.15:  # More lenient confidence difference
                        return alt_char, alt_conf
                        
        return char, confidence


# Example usage in AmharicPredictor class:
"""
from .error_correction import AmharicErrorCorrector

class AmharicPredictor:
    def __init__(self, ...):
        ...
        self.error_corrector = AmharicErrorCorrector()
    
    def predict_with_correction(self, image_path):
        # Get initial prediction and top-k predictions
        pred_idx, pred_char, confidence = self.predict_with_confidence(image_path)
        top_k = self.get_top_k_predictions(image_path, k=5)
        
        # Apply error correction
        corrected_char, corrected_conf = self.error_corrector.correct_prediction(
            pred_char, confidence, top_k
        )
        
        return corrected_char, corrected_conf
"""