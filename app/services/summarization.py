# app/service/summarization.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from ..utils.logging_utils import logger

class SummarizationService:
    def __init__(self):
        try:
            self.model_name = "T-Systems-onsite/mt5-small-sum-de-en-v2"
            logger.info(f"Loading summarization model: {self.model_name}")
            
            # Safe loading with legacy=False
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, 
                    legacy=False, 
                    use_fast=False
                )
            except TypeError:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            logger.info("Summarization model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading summarization model: {str(e)}")
            raise

    def generate_summary(
        self, 
        text: str, 
        max_length: int = 150, 
        min_length: int = 80
    ) -> str:
        """
        Generates a summary of the input text using mt5-small-sum-de-en-v2 model
        """
        try:
            # Add task prefix for summarization (important for mT5)
            input_text = "summarize: " + text
            
            # Tokenize input with truncation
            inputs = self.tokenizer.encode(
                input_text, 
                return_tensors="pt", 
                max_length=1028, 
                truncation=True
            )
            
            # Generate summary
            summary_ids = self.model.generate(
                inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
            
            # Decode summary
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            raise

# Create singleton instance
summarization_service = SummarizationService()