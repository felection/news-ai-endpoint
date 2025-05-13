# app/service/named_entity_recognition.py
from transformers import AutoModelForTokenClassification, pipeline
from typing import Dict, List, Union
from ..utils.logging_utils import logger
from .base_service import BaseService


class NamedEntityRecognitionService(
    BaseService[str, Dict[str, List[Dict[str, Union[str, float]]]]]
):
    """Service for performing named entity recognition on text."""

    # Entity type mapping from model tags to readable categories
    ENTITY_TYPE_MAP = {
        "B-MISC": "Miscellaneous",
        "I-MISC": "Miscellaneous",
        "B-PER": "Person",
        "I-PER": "Person",
        "B-ORG": "Organization",
        "I-ORG": "Organization",
        "B-LOC": "Location",
        "I-LOC": "Location",
    }

    # Default entity categories - defined as class constant
    DEFAULT_CATEGORIES = {
        "Person": [],
        "Organization": [],
        "Location": [],
        "Miscellaneous": [],
    }

    def __init__(self, model_name="dslim/bert-base-NER", quantize=False):
        """
        Initialize the NER service.

        Args:
            model_name: Name or path of the model to use
            quantize: Whether to quantize the model for faster inference
        """
        super().__init__(
            model_name=model_name,
            model_type=AutoModelForTokenClassification,
            quantize=quantize,
        )
        logger.info(f"NER service initialized with model: {model_name}")
        self.min_score = 0.7  # Default threshold

    def process_text(
        self, text: str, min_score=0.7
    ) -> Dict[str, List[Dict[str, Union[str, float]]]]:
        """
        Process text to identify named entities.

        Args:
            text: Text to process for named entities
            min_score: Minimum confidence score for entity extraction

        Returns:
            Dictionary with extracted entities by category
        """
        self.min_score = min_score
        return self.process(text)

    def _process_implementation(
        self, text: str
    ) -> Dict[str, List[Dict[str, Union[str, float]]]]:
        """
        Implementation of the NER processing logic.

        Args:
            text: The text to process

        Returns:
            Dictionary with extracted entities by category
        """
        # Create NER pipeline using the model and tokenizer
        nlp = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
        )

        # Get raw entities
        raw_entities = nlp(text)

        # Create a fresh copy of the default categories
        categories = {category: [] for category in self.DEFAULT_CATEGORIES}

        # Process entities
        grouped_entities = self._group_entities(raw_entities, categories)
        return self._cleanup_entities(grouped_entities)

    def _group_entities(self, entities: List[Dict], categories: Dict) -> Dict:
        """
        Group raw entity tokens into complete named entities.

        Args:
            entities: Raw entity tokens from the model
            categories: Dictionary with empty lists for each entity category

        Returns:
            Dictionary with grouped entities by category
        """
        grouped_entities = categories
        current_entity = {"text": "", "type": None, "score": 0.0}

        for entity in entities:
            word = entity["word"]
            entity_tag = entity["entity"]
            label = self.ENTITY_TYPE_MAP.get(entity_tag, "Miscellaneous")
            score = float(entity["score"])
            is_subword = word.startswith("##")

            # Handle new entity (B- prefix)
            if entity_tag.startswith("B-"):
                # Save previous entity if it exists
                if current_entity["type"]:
                    current_entity["text"] = current_entity["text"].strip()
                    grouped_entities[current_entity["type"]].append(current_entity)

                # Start new entity
                current_entity = {
                    "text": word,
                    "type": label,
                    "score": score,
                }

            # Handle continuation of entity (I- prefix)
            elif entity_tag.startswith("I-") and current_entity["type"] == label:
                if is_subword:
                    # Append without space
                    current_entity["text"] += word.replace("##", "")
                else:
                    # New full word -> add space
                    current_entity["text"] += " " + word
                # Keep tracking the score
                current_entity["score"] = max(current_entity["score"], score)

            # Handle other cases (entity type change or non-entity)
            else:
                if current_entity["type"]:
                    current_entity["text"] = current_entity["text"].strip()
                    grouped_entities[current_entity["type"]].append(current_entity)
                current_entity = {"text": "", "type": None, "score": 0.0}

        # Add the final entity if one exists
        if current_entity["type"]:
            current_entity["text"] = current_entity["text"].strip()
            grouped_entities[current_entity["type"]].append(current_entity)

        return grouped_entities

    def _cleanup_entities(self, entity_dict: Dict) -> Dict:
        """
        Clean up entities by removing duplicates and low-confidence entities.

        Args:
            entity_dict: Dictionary of categorized entities

        Returns:
            Cleaned dictionary of entities
        """
        result = {}

        for category, entities in entity_dict.items():
            # Skip empty categories
            if not entities:
                result[category] = []
                continue

            # Remove duplicates while preserving order
            seen_texts = set()
            unique_entities = []

            for entity in entities:
                # Skip low-quality entities
                if (
                    entity["text"].startswith("##")
                    or len(entity["text"]) <= 1
                    or entity["score"] < self.min_score
                ):
                    continue

                if entity["text"] not in seen_texts:
                    seen_texts.add(entity["text"])
                    unique_entities.append(entity)

            result[category] = unique_entities

        return result


# Create singleton instance
ner_service = NamedEntityRecognitionService()
