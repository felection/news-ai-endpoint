# app/service/emotion_extraction.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from ..utils.logging_utils import logger

class EmotionAnalysisService:
    """Service for detecting emotions in text."""
    
    def __init__(self, model_name="cardiffnlp/twitter-roberta-large-emotion-latest"):
        try:
            logger.info(f"Loading emotion analysis model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Emotion analysis model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading emotion analysis model: {str(e)}")
            raise
    
    def detect_emotions(self, text):
        """
        Detect emotions in the provided text.
        
        Args:
            text (str): Text to analyze for emotions
            
        Returns:
            dict: Dictionary with emotions and dominant emotion
        """
        try:
            # Prepare text inputs
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)
                scores = scores.cpu().numpy()[0]
            
            # Get emotion labels
            emotion_labels = self.model.config.id2label
            
            # Create dictionary of emotions and scores
            emotions = {emotion_labels[i]: float(scores[i]) for i in range(len(scores))}
            
            # Sort emotions by score in descending order
            sorted_emotions = {k: v for k, v in sorted(emotions.items(), key=lambda item: item[1], reverse=True)}
            
            # Get dominant emotion
            dominant_emotion = next(iter(sorted_emotions))
            
            return {
                "emotions": sorted_emotions,
                "dominant_emotion": dominant_emotion
            }
        except Exception as e:
            logger.error(f"Error detecting emotions: {str(e)}")
            raise

# Create singleton instance
emotion_service = EmotionAnalysisService()