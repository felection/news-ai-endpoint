# app/service/sentiment_analysis.py
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from ..utils.logging_utils import logger

class SentimentAnalysisService:
    """Service for performing sentiment analysis on text."""
    
    SENTIMENT_LABELS = {
        0: "negative",
        1: "positive"
    }
    
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        try:
            logger.info(f"Loading sentiment analysis model: {model_name}")
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            self.model = DistilBertForSequenceClassification.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Sentiment analysis model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading sentiment analysis model: {str(e)}")
            raise
    
    def analyze_text(self, text):
        """
        Analyze the sentiment of the provided text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment analysis result with label and confidence score
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Process results
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get predicted class and confidence
            predicted_class_id = logits.argmax().item()
            confidence = probabilities[0][predicted_class_id].item()
            sentiment = self.SENTIMENT_LABELS[predicted_class_id]
            
            return {
                "sentiment": sentiment,
                "confidence": confidence,
                "probabilities": {
                    "positive": probabilities[0][1].item(),
                    "negative": probabilities[0][0].item()
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            raise

# Create singleton instance
sentiment_service = SentimentAnalysisService()