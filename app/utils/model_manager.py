"""
Model Manager for centralized model loading and caching.

This module provides a singleton ModelManager class that handles:
- Lazy loading of models (only when needed)
- Caching of models to avoid reloading
- Quantization options for performance
- Centralized error handling for model operations
"""
import os
import torch
from functools import lru_cache
from typing import Dict, Any, Optional, Callable, Type
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from ..utils.logging_utils import logger
from ..config import get_settings

settings = get_settings()

class ModelManager:
    """
    Singleton class for managing ML models.
    
    Handles model loading, caching, and provides a unified interface
    for model operations across the application.
    """
    
    def __init__(self):
        self._models = {}
        self._tokenizers = {}
        #self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = "cpu"
        logger.info(f"ModelManager initialized with device: {self._device}")
        
    @property
    def device(self):
        """Get the current device (CPU or CUDA)."""
        return self._device
    
    def get_model(self, model_name: str, model_type: Type = AutoModel, 
                  quantize: bool = False, **kwargs) -> Any:
        """
        Get a model by name, loading it if not already loaded.
        
        Args:
            model_name: The name or path of the model
            model_type: The model class to use (default: AutoModel)
            quantize: Whether to quantize the model for faster inference
            **kwargs: Additional arguments to pass to the model loading function
            
        Returns:
            The loaded model
        """
        model_key = f"{model_name}_{model_type.__name__}"
        
        if model_key not in self._models:
            try:
                logger.info(f"Loading model: {model_name} (type: {model_type.__name__})")
                
                """ # Handle HF token if needed
                if settings.hf_token:
                    print(f"Using HF token: {settings.hf_token}")
                    kwargs['token'] = settings.hf_token """
                
                # Load the model
                if model_type == SentenceTransformer:
                    model = model_type(model_name, **kwargs)
                else:
                    model = model_type.from_pretrained(model_name, **kwargs)
                
                # Apply quantization if requested and supported
                if quantize and hasattr(model, 'to') and model_type != SentenceTransformer:
                    logger.info(f"Applying quantization to model: {model_name}")
                    model = self._quantize_model(model)
                
                # Move model to device
                if hasattr(model, 'to') and model_type != SentenceTransformer:
                    model.to(self._device)
                    
                # Set to evaluation mode
                if hasattr(model, 'eval'):
                    model.eval()
                
                self._models[model_key] = model
                logger.info(f"Model {model_name} loaded successfully on {self._device}")
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {str(e)}")
                raise
        
        return self._models[model_key]
    
    def get_tokenizer(self, tokenizer_name: str, **kwargs) -> Any:
        """
        Get a tokenizer by name, loading it if not already loaded.
        
        Args:
            tokenizer_name: The name or path of the tokenizer
            **kwargs: Additional arguments to pass to the tokenizer loading function
            
        Returns:
            The loaded tokenizer
        """
        if tokenizer_name not in self._tokenizers:
            try:
                logger.info(f"Loading tokenizer: {tokenizer_name}")
                
                # Handle HF token if needed
                """ if settings.HF_TOKEN:
                    kwargs['token'] = settings.HF_TOKEN """
                
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)
                self._tokenizers[tokenizer_name] = tokenizer
                logger.info(f"Tokenizer {tokenizer_name} loaded successfully")
            except Exception as e:
                logger.error(f"Error loading tokenizer {tokenizer_name}: {str(e)}")
                raise
        
        return self._tokenizers[tokenizer_name]
    
    def _quantize_model(self, model):
        """
        Quantize a model to reduce memory usage and increase inference speed.
        
        Args:
            model: The model to quantize
            
        Returns:
            The quantized model
        """
        try:
            # Check if we can use dynamic quantization
            if hasattr(torch.quantization, 'quantize_dynamic'):
                # Get the list of modules that support quantization
                qconfig_spec = {
                    torch.nn.Linear: torch.quantization.default_dynamic_qconfig,
                    torch.nn.LSTM: torch.quantization.default_dynamic_qconfig,
                    torch.nn.GRU: torch.quantization.default_dynamic_qconfig,
                }
                
                # Apply dynamic quantization
                model = torch.quantization.quantize_dynamic(
                    model, qconfig_spec, dtype=torch.qint8
                )
                logger.info("Applied dynamic quantization to model")
            else:
                logger.warning("Dynamic quantization not available, using original model")
        except Exception as e:
            logger.warning(f"Quantization failed, using original model: {str(e)}")
        
        return model
    
    def clear_cache(self, model_name: Optional[str] = None):
        """
        Clear the model cache.
        
        Args:
            model_name: If provided, clear only this model from cache
        """
        if model_name:
            # Clear specific model
            keys_to_clear = [k for k in self._models.keys() if k.startswith(f"{model_name}_")]
            for key in keys_to_clear:
                del self._models[key]
                logger.info(f"Cleared model {key} from cache")
            
            # Clear specific tokenizer
            if model_name in self._tokenizers:
                del self._tokenizers[model_name]
                logger.info(f"Cleared tokenizer {model_name} from cache")
        else:
            # Clear all models and tokenizers
            self._models = {}
            self._tokenizers = {}
            logger.info("Cleared all models and tokenizers from cache")
        
        # Force garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache")

    def get_memory_usage(self):
        """
        Get the current memory usage of the models.
        
        Returns:
            Dict with memory usage information
        """
        memory_info = {
            "num_models": len(self._models),
            "num_tokenizers": len(self._tokenizers),
        }
        
        if torch.cuda.is_available():
            memory_info["cuda_allocated_memory_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            memory_info["cuda_reserved_memory_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
        
        return memory_info

# Create singleton instance
model_manager = ModelManager()
