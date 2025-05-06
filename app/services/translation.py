from transformers import MarianMTModel, MarianTokenizer
from ..utils.logging_utils import logger

class TranslationService:
    def __init__(self):
        try:
            # Define models for both translation directions
            self.models = {
                "de-en": {
                    "model_name": "Helsinki-NLP/opus-mt-de-en",
                    "tokenizer": None,
                    "model": None,
                },
                "en-de": {
                    "model_name": "Helsinki-NLP/opus-mt-en-de",
                    "tokenizer": None,
                    "model": None,
                },
            }

            # Load models and tokenizers
            for direction, config in self.models.items():
                logger.info(f"Loading translation model: {config['model_name']}")
                config["tokenizer"] = MarianTokenizer.from_pretrained(config["model_name"])
                config["model"] = MarianMTModel.from_pretrained(config["model_name"])
                logger.info(f"Translation model for {direction} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading translation models: {str(e)}")
            raise

    def translate_text(self, text, direction="de-en"):
        """
        Translates text using the specified translation direction.
        
        Args:
            text (str): Text to translate.
            direction (str): Translation direction ("de-en" or "en-de").
        
        Returns:
            str: Translated text.
        """
        if direction not in self.models:
            raise ValueError(f"Unsupported translation direction: {direction}")

        # Get the model and tokenizer for the specified direction
        tokenizer = self.models[direction]["tokenizer"]
        model = self.models[direction]["model"]

        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        
        # Generate translation
        translated = model.generate(**inputs)
        
        # Decode the translated tokens
        translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
        
        return translated_text + ' '

    def translate_in_chunks(self, text: str, max_chunk_size: int = 128, direction="de-en") -> str:
        """
        Splits long text into chunks based on token count and translates each chunk separately
        before combining the results.
        
        Args:
            text (str): Text to translate.
            max_chunk_size (int): Maximum size of each chunk in tokens.
            direction (str): Translation direction ("de-en" or "en-de").
        
        Returns:
            str: Translated text.
        """
        # Input validation
        if not text or not text.strip():
            return ""
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be a positive integer.")

        # Get the tokenizer for the specified direction
        if direction not in self.models:
            raise ValueError(f"Unsupported translation direction: {direction}")
        tokenizer = self.models[direction]["tokenizer"]

        # Tokenize the entire text
        tokens = tokenizer.encode(text, add_special_tokens=False)
        num_tokens = len(tokens)

        if num_tokens == 0:
            return ""

        # Handle short text efficiently
        if num_tokens <= max_chunk_size:
            return self.translate_text(text, direction)
        
        # Process longer text in chunks
        translated_chunks = []
        start_index = 0
        
        # Find common sentence terminators for better text splitting
        sentence_terminators = ['.', '!', '?', ':', '"', '»', '"']
        
        while start_index < num_tokens:
            # Calculate tentative end index for this chunk
            end_index = min(start_index + max_chunk_size, num_tokens)
            
            # If we're not at the end of the text, try to find a natural breaking point
            if end_index < num_tokens:
                # Get the text for this potential chunk
                potential_chunk = tokenizer.decode(tokens[start_index:end_index], skip_special_tokens=True)
                
                # Try to find the last sentence terminator in the chunk
                last_terminator_pos = -1
                for terminator in sentence_terminators:
                    pos = potential_chunk.rfind(terminator)
                    if pos > last_terminator_pos:
                        last_terminator_pos = pos
                
                # If we found a sentence terminator, adjust the end index
                if last_terminator_pos > 0 and last_terminator_pos < len(potential_chunk) - 10:
                    break_text = potential_chunk[:last_terminator_pos + 1]
                    break_tokens = tokenizer.encode(break_text, add_special_tokens=False)
                    end_index = start_index + len(break_tokens)
            
            # Get token IDs for the final chunk
            chunk_token_ids = tokens[start_index:end_index]
            chunk_text = tokenizer.decode(chunk_token_ids, skip_special_tokens=True)
            
            if not chunk_text.strip():  # Skip empty chunks
                start_index = end_index
                continue
            
            logger.info(f"Translating chunk {len(translated_chunks) + 1}: tokens {start_index} to {end_index-1} ({end_index - start_index} tokens)")
            
            # Translate the chunk with error handling
            try:
                translated_chunk = self.translate_text(chunk_text, direction)
                translated_chunks.append(translated_chunk)
            except Exception as e:
                error_msg = f"[Translation error in chunk {len(translated_chunks) + 1}: {str(e)}]"
                logger.error(error_msg)
                translated_chunks.append(error_msg)
            
            # Move to the next chunk
            start_index = end_index

        # Combine the translated chunks with proper spacing
        if any('\n' in chunk for chunk in translated_chunks):
            # If chunks contain newlines, join with newlines to preserve paragraph structure
            full_translation = "\n".join(chunk.strip() for chunk in translated_chunks)
        else:
            # Otherwise join with spaces
            full_translation = " ".join(chunk.strip() for chunk in translated_chunks)

        return full_translation

# Create a singleton instance
translation_service = TranslationService()