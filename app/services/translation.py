# app/services/translation.py
from transformers import MarianMTModel, MarianTokenizer
from ..utils.logging_utils import logger

class TranslationService:
    def __init__(self):
        try:
            self.model_name = "Helsinki-NLP/opus-mt-de-en"
            logger.info(f"Loading translation model: {self.model_name}")
            self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
            self.model = MarianMTModel.from_pretrained(self.model_name)
            logger.info("Translation model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading translation model: {str(e)}")
            raise

    def translate_text(self, text):
        """
        Translates German text to English using Helsinki-NLP/opus-mt-de-en model
        """
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        
        # Generate translation
        translated = self.model.generate(**inputs)
        
        # Decode the translated tokens
        translated_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
        
        return translated_text + ' '

    def translate_in_chunks(self, text: str, max_chunk_size: int = 128) -> str:
        """
        Splits long text into chunks based on token count and translates each chunk separately
        before combining the results.
        """
        # Input validation
        if not text or not text.strip():
            return ""
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be a positive integer.")

        # Tokenize the entire text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        num_tokens = len(tokens)

        if num_tokens == 0:
            return ""

        # Handle short text efficiently
        if num_tokens <= max_chunk_size:
            return self.translate_text(text)
        
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
                potential_chunk = self.tokenizer.decode(tokens[start_index:end_index], skip_special_tokens=True)
                
                # Try to find the last sentence terminator in the chunk
                last_terminator_pos = -1
                for terminator in sentence_terminators:
                    pos = potential_chunk.rfind(terminator)
                    if pos > last_terminator_pos:
                        last_terminator_pos = pos
                
                # If we found a sentence terminator, adjust the end index
                if last_terminator_pos > 0 and last_terminator_pos < len(potential_chunk) - 10:
                    break_text = potential_chunk[:last_terminator_pos + 1]
                    break_tokens = self.tokenizer.encode(break_text, add_special_tokens=False)
                    end_index = start_index + len(break_tokens)
            
            # Get token IDs for the final chunk
            chunk_token_ids = tokens[start_index:end_index]
            chunk_text = self.tokenizer.decode(chunk_token_ids, skip_special_tokens=True)
            
            if not chunk_text.strip():  # Skip empty chunks
                start_index = end_index
                continue
            
            logger.info(f"Translating chunk {len(translated_chunks) + 1}: tokens {start_index} to {end_index-1} ({end_index - start_index} tokens)")
            
            # Translate the chunk with error handling
            try:
                translated_chunk = self.translate_text(chunk_text)
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