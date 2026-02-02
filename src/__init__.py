"""
Indic-OCR: Multi-Language OCR System for Indian Regional Scripts
================================================================

A machine learning-based OCR solution that can detect, classify,
and accurately recognize text from multiple Indian regional scripts
and convert it into editable Unicode text.

Author: Dhananjayan H
Roll No: AA.SC.P2MCA24070151
"""

__version__ = "1.0.0"
__author__ = "Dhananjayan H"

from .preprocessing import ImagePreprocessor
from .script_classifier import ScriptClassifier
from .ocr_engine import OCREngine
from .utils import load_config

__all__ = [
    "ImagePreprocessor",
    "ScriptClassifier", 
    "OCREngine",
    "load_config"
]
