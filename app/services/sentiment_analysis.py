# app/service/sentiment_analysis.py
import torch
from typing import Dict, List, Union
from transformers import DistilBertForSequenceClassification
from ..utils.logging_utils import logger
from .base_service import BaseService

class SentimentAnalysisService(BaseService[str, Dict[str, Union[str, float, Dict[str, float]]]]):
    """Service for performing sentiment analysis on text."""
    
    SENTIMENT_LABELS = {
        0: "negative",
        1: "positive"
    }
    
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english", quantize=True):
        """
        Initialize the sentiment analysis service.
        
        Args:
            model_name: Name or path of the model to use
            quantize: Whether to quantize the model for faster inference
        """
        super().__init__(
            model_name=model_name,
            model_type=DistilBertForSequenceClassification,
            quantize=quantize
        )
        logger.info(f"Sentiment analysis service initialized with model: {model_name}")
    
    def analyze_text(self, text: str) -> Dict[str, Union[str, float, Dict[str, float]]]:
        """
        Analyze the sentiment of the provided text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with sentiment analysis results
        """
        return self.process(text)
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Union[str, float, Dict[str, float]]]]:
        """
        Analyze the sentiment of multiple texts in a batch.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment analysis results
        """
        return self.process_batch(texts)
    
    def _process_implementation(self, text: str) -> Dict[str, Union[str, float, Dict[str, float]]]:
        """
        Implementation of sentiment analysis for a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with sentiment analysis results
        """
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
    
    def _process_batch_implementation(self, texts: List[str]) -> List[Dict[str, Union[str, float, Dict[str, float]]]]:
        """
        Implementation of sentiment analysis for a batch of texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment analysis results
        """
        # Tokenize batch input
        batch_inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
        batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**batch_inputs)
        
        # Process results
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        results = []
        for i in range(len(texts)):
            # Get predicted class and confidence for each item in batch
            predicted_class_id = logits[i].argmax().item()
            confidence = probabilities[i][predicted_class_id].item()
            sentiment = self.SENTIMENT_LABELS[predicted_class_id]
            
            results.append({
                "sentiment": sentiment,
                "confidence": confidence,
                "probabilities": {
                    "positive": probabilities[i][1].item(),
                    "negative": probabilities[i][0].item()
                }
            })
        
        return results

# Create singleton instance
sentiment_service = SentimentAnalysisService()
