"""
Base service class for all NLP services.

This module provides a base class for all NLP services with common functionality:
- Model loading and caching via ModelManager
- Batch processing support
- Error handling
- Performance monitoring
"""
import time
from typing import Any, List, TypeVar, Generic
from ..utils.logging_utils import logger
from ..utils.model_manager import model_manager

T = TypeVar('T')  # Input type
R = TypeVar('R')  # Result type

class BaseService(Generic[T, R]):
    """Base class for all NLP services."""
    
    def __init__(self, model_name: str, model_type: Any = None, quantize: bool = False):
        """
        Initialize the service.
        
        Args:
            model_name: Name or path of the model to use
            model_type: Type of model to load (if None, use default)
            quantize: Whether to quantize the model for faster inference
        """
        self.model_name = model_name
        self.model_type = model_type
        self.quantize = quantize
        self.device = model_manager.device
        
        # These will be loaded lazily when needed
        self._model = None
        self._tokenizer = None
    
    @property
    def model(self):
        """Get the model, loading it if not already loaded."""
        if self._model is None:
            self._model = self._load_model()
        return self._model
    
    @property
    def tokenizer(self):
        """Get the tokenizer, loading it if not already loaded."""
        if self._tokenizer is None:
            self._tokenizer = self._load_tokenizer()
        return self._tokenizer
    
    def _load_model(self):
        """
        Load the model using the model manager.
        
        Returns:
            The loaded model
        """
        try:
            print('loading')
            return model_manager.get_model(
                self.model_name, 
                model_type=self.model_type,
                quantize=self.quantize
            )
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {str(e)}")
            raise
    
    def _load_tokenizer(self):
        """
        Load the tokenizer using the model manager.
        
        Returns:
            The loaded tokenizer
        """
        try:
            return model_manager.get_tokenizer(self.model_name)
        except Exception as e:
            logger.error(f"Error loading tokenizer {self.model_name}: {str(e)}")
            raise
    
    def process(self, input_data: T) -> R:
        """
        Process a single input.
        
        Args:
            input_data: The input data to process
            
        Returns:
            The processed result
        """
        try:
            start_time = time.time()
            result = self._process_implementation(input_data)
            process_time = time.time() - start_time
            
            logger.debug(f"Processed input in {process_time:.4f}s")
            return result
        except Exception as e:
            logger.error(f"Error processing input: {str(e)}")
            raise
    
    def process_batch(self, batch_input: List[T]) -> List[R]:
        """
        Process a batch of inputs.
        
        Args:
            batch_input: List of inputs to process
            
        Returns:
            List of processed results
        """
        try:
            start_time = time.time()
            results = self._process_batch_implementation(batch_input)
            process_time = time.time() - start_time
            
            logger.debug(f"Processed batch of {len(batch_input)} inputs in {process_time:.4f}s")
            return results
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            raise
    
    def _process_implementation(self, input_data: T) -> R:
        """
        Implementation of the processing logic.
        
        This method should be overridden by subclasses.
        
        Args:
            input_data: The input data to process
            
        Returns:
            The processed result
        """
        raise NotImplementedError("Subclasses must implement _process_implementation")
    
    def _process_batch_implementation(self, batch_input: List[T]) -> List[R]:
        """
        Implementation of the batch processing logic.
        
        By default, this processes each input individually.
        Subclasses should override this for more efficient batch processing.
        
        Args:
            batch_input: List of inputs to process
            
        Returns:
            List of processed results
        """
        return [self._process_implementation(input_data) for input_data in batch_input]
    
    def clear_cache(self):
        """Clear the model and tokenizer from cache."""
        model_manager.clear_cache(self.model_name)
        self._model = None
        self._tokenizer = None
