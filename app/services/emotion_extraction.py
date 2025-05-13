# app/service/emotion_extraction.py
import torch
from typing import Dict, List, Union
from transformers import AutoModelForSequenceClassification
from ..utils.logging_utils import logger
from .base_service import BaseService


class EmotionAnalysisService(BaseService[str, Dict[str, Union[str, Dict[str, float]]]]):
    """Service for detecting emotions in text."""

    def __init__(
        self,
        model_name="cardiffnlp/twitter-roberta-large-emotion-latest",
        quantize=True,
    ):
        """
        Initialize the emotion analysis service.

        Args:
            model_name: Name or path of the model to use
            quantize: Whether to quantize the model for faster inference
        """
        super().__init__(
            model_name=model_name,
            model_type=AutoModelForSequenceClassification,
            quantize=quantize,
        )
        logger.info(f"Emotion analysis service initialized with model: {model_name}")

    def detect_emotions(self, text: str) -> Dict[str, Union[str, Dict[str, float]]]:
        """
        Detect emotions in the provided text.

        Args:
            text: Text to analyze for emotions

        Returns:
            Dict with detected emotions and dominant emotion
        """
        return self.process(text)

    def detect_emotions_batch(
        self, texts: List[str]
    ) -> List[Dict[str, Union[str, Dict[str, float]]]]:
        """
        Detect emotions in multiple texts in a batch.

        Args:
            texts: List of texts to analyze for emotions

        Returns:
            List of detected emotions and dominant emotions
        """
        return self.process_batch(texts)

    def _process_implementation(
        self, text: str
    ) -> Dict[str, Union[str, Dict[str, float]]]:
        """
        Implementation of emotion detection for a single text.

        Args:
            text: Text to analyze for emotions

        Returns:
            Dict with detected emotions and dominant emotion
        """
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
        sorted_emotions = {
            k: v
            for k, v in sorted(emotions.items(), key=lambda item: item[1], reverse=True)
        }

        # Get dominant emotion
        dominant_emotion = next(iter(sorted_emotions))

        return {"emotions": sorted_emotions, "dominant_emotion": dominant_emotion}

    def _process_batch_implementation(
        self, texts: List[str]
    ) -> List[Dict[str, Union[str, Dict[str, float]]]]:
        """
        Implementation of emotion detection for a batch of texts.

        Args:
            texts: List of texts to analyze for emotions

        Returns:
            List of detected emotions and dominant emotions
        """
        # Prepare batch inputs
        batch_inputs = self.tokenizer(
            texts, return_tensors="pt", truncation=True, padding=True
        )
        batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}

        # Make predictions
        with torch.no_grad():
            outputs = self.model(**batch_inputs)
            all_scores = torch.nn.functional.softmax(outputs.logits, dim=1)
            all_scores = all_scores.cpu().numpy()

        # Get emotion labels
        emotion_labels = self.model.config.id2label

        results = []
        for i in range(len(texts)):
            scores = all_scores[i]

            # Create dictionary of emotions and scores
            emotions = {emotion_labels[j]: float(scores[j]) for j in range(len(scores))}

            # Sort emotions by score in descending order
            sorted_emotions = {
                k: v
                for k, v in sorted(
                    emotions.items(), key=lambda item: item[1], reverse=True
                )
            }

            # Get dominant emotion
            dominant_emotion = next(iter(sorted_emotions))

            results.append(
                {"emotions": sorted_emotions, "dominant_emotion": dominant_emotion}
            )

        return results


# Create singleton instance
emotion_service = EmotionAnalysisService()
