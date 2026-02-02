"""
Utility Functions for Indic-OCR
================================

Common utility functions used across the project.
"""

import yaml
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Default config path
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config" / "config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configuration loaded from {config_path}")
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Configuration saved to {config_path}")


def ensure_directories(config: Dict[str, Any] = None):
    """
    Create necessary directories based on configuration.
    
    Args:
        config: Configuration dictionary with paths
    """
    if config is None:
        config = load_config()
    
    paths = config.get('paths', {})
    
    for key, path in paths.items():
        Path(path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {path}")


def get_unicode_range(script: str) -> str:
    """
    Get Unicode range for a script.
    
    Args:
        script: Script name
        
    Returns:
        Unicode range string
    """
    unicode_ranges = {
        "Devanagari": "0900-097F",
        "Malayalam": "0D00-0D7F",
        "Tamil": "0B80-0BFF",
        "Telugu": "0C00-0C7F",
        "Kannada": "0C80-0CFF",
        "Bengali": "0980-09FF",
        "Gujarati": "0A80-0AFF",
        "Oriya": "0B00-0B7F",
        "Punjabi": "0A00-0A7F"
    }
    
    return unicode_ranges.get(script, "")


def is_indic_text(text: str) -> bool:
    """
    Check if text contains Indic script characters.
    
    Args:
        text: Input text
        
    Returns:
        True if text contains Indic characters
    """
    # Unicode ranges for major Indic scripts
    indic_ranges = [
        (0x0900, 0x097F),  # Devanagari
        (0x0980, 0x09FF),  # Bengali
        (0x0A00, 0x0A7F),  # Gurmukhi
        (0x0A80, 0x0AFF),  # Gujarati
        (0x0B00, 0x0B7F),  # Oriya
        (0x0B80, 0x0BFF),  # Tamil
        (0x0C00, 0x0C7F),  # Telugu
        (0x0C80, 0x0CFF),  # Kannada
        (0x0D00, 0x0D7F),  # Malayalam
    ]
    
    for char in text:
        code = ord(char)
        for start, end in indic_ranges:
            if start <= code <= end:
                return True
    
    return False


def detect_script_from_text(text: str) -> Optional[str]:
    """
    Detect the script of text based on Unicode characters.
    
    Args:
        text: Input text
        
    Returns:
        Detected script name or None
    """
    script_ranges = {
        "Devanagari": (0x0900, 0x097F),
        "Bengali": (0x0980, 0x09FF),
        "Punjabi": (0x0A00, 0x0A7F),
        "Gujarati": (0x0A80, 0x0AFF),
        "Oriya": (0x0B00, 0x0B7F),
        "Tamil": (0x0B80, 0x0BFF),
        "Telugu": (0x0C00, 0x0C7F),
        "Kannada": (0x0C80, 0x0CFF),
        "Malayalam": (0x0D00, 0x0D7F),
    }
    
    script_counts = {script: 0 for script in script_ranges}
    
    for char in text:
        code = ord(char)
        for script, (start, end) in script_ranges.items():
            if start <= code <= end:
                script_counts[script] += 1
                break
    
    # Return script with most characters
    max_script = max(script_counts, key=script_counts.get)
    if script_counts[max_script] > 0:
        return max_script
    
    return None


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def get_file_size(file_path: str) -> str:
    """
    Get file size in human-readable format.
    
    Args:
        file_path: Path to file
        
    Returns:
        Formatted file size string
    """
    size = os.path.getsize(file_path)
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    
    return f"{size:.2f} TB"


class Timer:
    """Simple timer for measuring execution time."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start the timer."""
        import time
        self.start_time = time.time()
        return self
    
    def stop(self):
        """Stop the timer."""
        import time
        self.end_time = time.time()
        return self
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        import time
        if self.start_time is None:
            return 0.0
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    def __str__(self) -> str:
        return format_time(self.elapsed)


if __name__ == "__main__":
    # Test utilities
    config = load_config()
    print(f"Loaded config: {config.get('project', {}).get('name', 'Unknown')}")
    
    test_text = "नमस्ते"  # Hindi
    print(f"Is Indic text: {is_indic_text(test_text)}")
    print(f"Detected script: {detect_script_from_text(test_text)}")
