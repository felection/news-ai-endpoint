# app/service/named_entity_recognition.py
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from ..utils.logging_utils import logger

class NamedEntityRecognitionService:
    """Service for performing named entity recognition on text."""
    
    ENTITY_TYPE_MAP = {
        'B-MISC': 'Miscellaneous',
        'I-MISC': 'Miscellaneous',
        'B-PER': 'Person',
        'I-PER': 'Person',
        'B-ORG': 'Organization',
        'I-ORG': 'Organization',
        'B-LOC': 'Location',
        'I-LOC': 'Location'
    }
    
 
    
    def __init__(self, model_name="dslim/bert-base-NER"):
        try:
            logger.info(f"Loading NER model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)
            logger.info("NER model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading NER model: {str(e)}")
            raise

    def process_text(self, text: str, min_score=0.7) -> dict:
        """Process text to identify named entities."""
        DEFAULT_CATEGORIES = {
            "Person": [],
            "Organization": [],
            "Location": [],
            "Miscellaneous": [],
        }
        
        self.min_score = min_score
        
        raw_entities = self.nlp(text)
        grouped_entities = self._group_entities(raw_entities,DEFAULT_CATEGORIES)
        return self._cleanup_entities(grouped_entities)
    
    def _group_entities(self, entities, DEFAULT_CATEGORIES):
        """
        Group raw entity tokens into complete named entities.
        
        Args:
            entities (list): Raw entity tokens from the model
            
        Returns:
            dict: Grouped entities by category
        """
        grouped_entities = DEFAULT_CATEGORIES
        current_entity = {"text": "", "type": None, "score": 0.0}
        
        for entity in entities:
            word = entity["word"]
            entity_tag = entity["entity"]
            label = self.ENTITY_TYPE_MAP.get(entity_tag, "Other")
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
    
    def _cleanup_entities(self, entity_dict):
        """
        Clean up entities by removing duplicates and low-confidence entities.
        
        Args:
            entity_dict (dict): Dictionary of categorized entities
            
        Returns:
            dict: Cleaned dictionary of entities
        """
        for category, entities in entity_dict.items():
            if not entities:  # Skip empty lists
                continue
                
            # Remove duplicates while preserving order
            seen_texts = set()
            unique_entities = []
            for entity in entities:
                # Skip low-quality entities
                if (entity['text'].startswith('##') or 
                    len(entity['text']) <= 1 or 
                    entity['score'] < self.min_score):
                    continue
                    
                if entity['text'] not in seen_texts:
                    seen_texts.add(entity['text'])
                    unique_entities.append(entity)
            
            entity_dict[category] = unique_entities
        
        return entity_dict
# Create singleton instance
ner_service = NamedEntityRecognitionService()