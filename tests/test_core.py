"""
Unit Tests for Indic-OCR
========================
"""

import unittest
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import ImagePreprocessor
from src.evaluation import (
    character_error_rate,
    word_error_rate,
    levenshtein_distance,
    accuracy,
    precision_recall_f1
)
from src.utils import (
    is_indic_text,
    detect_script_from_text,
    get_unicode_range
)


class TestPreprocessing(unittest.TestCase):
    """Test cases for image preprocessing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = ImagePreprocessor()
        self.test_image = np.ones((100, 200), dtype=np.uint8) * 255
    
    def test_grayscale_conversion(self):
        """Test grayscale conversion."""
        # RGB image
        rgb_image = np.ones((100, 200, 3), dtype=np.uint8) * 128
        gray = self.preprocessor.to_grayscale(rgb_image)
        
        self.assertEqual(len(gray.shape), 2)
        self.assertEqual(gray.shape, (100, 200))
    
    def test_already_grayscale(self):
        """Test that already grayscale images are handled."""
        gray = self.preprocessor.to_grayscale(self.test_image)
        self.assertEqual(gray.shape, self.test_image.shape)
    
    def test_resize(self):
        """Test image resizing."""
        resized = self.preprocessor.resize_image(self.test_image)
        
        self.assertEqual(resized.shape[0], self.preprocessor.target_height)
        self.assertEqual(resized.shape[1], self.preprocessor.target_width)
    
    def test_binarization_otsu(self):
        """Test Otsu binarization."""
        preprocessor = ImagePreprocessor(binarization_method='otsu')
        binary = preprocessor.apply_binarization(self.test_image)
        
        # Should only have 0 and 255 values
        unique_values = np.unique(binary)
        self.assertTrue(all(v in [0, 255] for v in unique_values))
    
    def test_binarization_adaptive(self):
        """Test adaptive binarization."""
        preprocessor = ImagePreprocessor(binarization_method='adaptive')
        binary = preprocessor.apply_binarization(self.test_image)
        
        self.assertEqual(binary.shape, self.test_image.shape)
    
    def test_normalize(self):
        """Test normalization."""
        normalized = self.preprocessor.normalize(self.test_image)
        
        self.assertEqual(normalized.dtype, np.float32)
        self.assertTrue(normalized.min() >= 0.0)
        self.assertTrue(normalized.max() <= 1.0)


class TestEvaluation(unittest.TestCase):
    """Test cases for evaluation metrics."""
    
    def test_levenshtein_distance_identical(self):
        """Test Levenshtein distance for identical strings."""
        self.assertEqual(levenshtein_distance("hello", "hello"), 0)
    
    def test_levenshtein_distance_one_edit(self):
        """Test Levenshtein distance for one edit."""
        self.assertEqual(levenshtein_distance("hello", "hallo"), 1)
        self.assertEqual(levenshtein_distance("hello", "hell"), 1)
        self.assertEqual(levenshtein_distance("hello", "helloo"), 1)
    
    def test_levenshtein_distance_empty(self):
        """Test Levenshtein distance with empty strings."""
        self.assertEqual(levenshtein_distance("", ""), 0)
        self.assertEqual(levenshtein_distance("hello", ""), 5)
        self.assertEqual(levenshtein_distance("", "hello"), 5)
    
    def test_cer_identical(self):
        """Test CER for identical strings."""
        self.assertEqual(character_error_rate("hello", "hello"), 0.0)
    
    def test_cer_different(self):
        """Test CER for different strings."""
        cer = character_error_rate("hallo", "hello")
        self.assertAlmostEqual(cer, 0.2)  # 1 error / 5 characters
    
    def test_cer_empty(self):
        """Test CER with empty strings."""
        self.assertEqual(character_error_rate("", ""), 0.0)
        self.assertEqual(character_error_rate("hello", ""), 1.0)
    
    def test_wer_identical(self):
        """Test WER for identical strings."""
        self.assertEqual(word_error_rate("hello world", "hello world"), 0.0)
    
    def test_wer_one_error(self):
        """Test WER with one word error."""
        wer = word_error_rate("hello wrld", "hello world")
        self.assertAlmostEqual(wer, 0.5)  # 1 error / 2 words
    
    def test_accuracy_perfect(self):
        """Test accuracy with perfect predictions."""
        preds = ["a", "b", "c"]
        truths = ["a", "b", "c"]
        self.assertEqual(accuracy(preds, truths), 1.0)
    
    def test_accuracy_partial(self):
        """Test accuracy with partial matches."""
        preds = ["a", "b", "x"]
        truths = ["a", "b", "c"]
        self.assertAlmostEqual(accuracy(preds, truths), 2/3)
    
    def test_precision_recall_f1(self):
        """Test precision, recall, and F1 score."""
        preds = ["a", "a", "b", "b"]
        truths = ["a", "b", "a", "b"]
        
        precision, recall, f1 = precision_recall_f1(preds, truths)
        
        # Should be between 0 and 1
        self.assertTrue(0 <= precision <= 1)
        self.assertTrue(0 <= recall <= 1)
        self.assertTrue(0 <= f1 <= 1)


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_is_indic_text_hindi(self):
        """Test Indic text detection for Hindi."""
        self.assertTrue(is_indic_text("नमस्ते"))
    
    def test_is_indic_text_tamil(self):
        """Test Indic text detection for Tamil."""
        self.assertTrue(is_indic_text("வணக்கம்"))
    
    def test_is_indic_text_malayalam(self):
        """Test Indic text detection for Malayalam."""
        self.assertTrue(is_indic_text("നമസ്കാരം"))
    
    def test_is_indic_text_english(self):
        """Test that English is not detected as Indic."""
        self.assertFalse(is_indic_text("Hello World"))
    
    def test_is_indic_text_mixed(self):
        """Test mixed text detection."""
        self.assertTrue(is_indic_text("Hello नमस्ते"))
    
    def test_detect_script_devanagari(self):
        """Test script detection for Devanagari."""
        self.assertEqual(detect_script_from_text("नमस्ते"), "Devanagari")
    
    def test_detect_script_tamil(self):
        """Test script detection for Tamil."""
        self.assertEqual(detect_script_from_text("வணக்கம்"), "Tamil")
    
    def test_detect_script_malayalam(self):
        """Test script detection for Malayalam."""
        self.assertEqual(detect_script_from_text("നമസ്കാരം"), "Malayalam")
    
    def test_detect_script_english(self):
        """Test script detection for English (should return None)."""
        self.assertIsNone(detect_script_from_text("Hello World"))
    
    def test_get_unicode_range(self):
        """Test Unicode range retrieval."""
        self.assertEqual(get_unicode_range("Devanagari"), "0900-097F")
        self.assertEqual(get_unicode_range("Tamil"), "0B80-0BFF")
        self.assertEqual(get_unicode_range("Malayalam"), "0D00-0D7F")


class TestIndicText(unittest.TestCase):
    """Test cases for Indic text processing."""
    
    def test_cer_hindi(self):
        """Test CER for Hindi text."""
        cer = character_error_rate("नमस्ते", "नमस्ते")
        self.assertEqual(cer, 0.0)
    
    def test_cer_hindi_error(self):
        """Test CER for Hindi text with error."""
        cer = character_error_rate("नमस्त", "नमस्ते")
        self.assertGreater(cer, 0.0)
    
    def test_cer_tamil(self):
        """Test CER for Tamil text."""
        cer = character_error_rate("வணக்கம்", "வணக்கம்")
        self.assertEqual(cer, 0.0)


if __name__ == '__main__':
    unittest.main()
