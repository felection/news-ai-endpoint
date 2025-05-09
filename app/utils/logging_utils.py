import logging
import sys # To output logs to stdout

# Get log level from settings (app.config)
# This avoids circular import if config itself uses logger
# We will pass the log_level from main.py
# from ..config import get_settings
# settings = get_settings()

def setup_logging(log_level_str: str = "INFO"):
    """
    Set up application-wide logging.
    
    Args:
        log_level_str: The desired log level as a string (e.g., "INFO", "DEBUG")
    """
    # Convert string log level to logging constant
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout) # Log to stdout
        ]
    )
    
    # Get a logger for this module (or a specific name)
    # Using __name__ is common, but for a central util, a fixed name might be better
    # For now, let's use a generic app logger name
    logger = logging.getLogger("app") # Main application logger
    logger.setLevel(log_level) # Ensure this logger also respects the level
    
    # Suppress overly verbose logs from libraries if needed
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING) # Transformers can be very verbose
    
    logger.info(f"Logging configured with level: {logging.getLevelName(log_level)}")
    return logger

# Initialize logger when module is imported
# The actual log level will be set from main.py based on settings
logger = setup_logging()
