"""
Image Preprocessing Module for Indic-OCR
=========================================

This module provides comprehensive image preprocessing functionality
for preparing document images before OCR processing.

Features:
- Grayscale conversion
- Noise reduction (denoising)
- Deskewing (rotation correction)
- Binarization (Otsu, Adaptive, Sauvola)
- Contrast enhancement
- Resizing and normalization
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    A comprehensive image preprocessing class for OCR applications.
    
    This class handles all preprocessing steps required to prepare
    document images for text recognition, including noise removal,
    binarization, deskewing, and normalization.
    """
    
    def __init__(
        self,
        target_height: int = 64,
        target_width: int = 256,
        binarization_method: str = "adaptive",
        denoise: bool = True,
        deskew: bool = True
    ):
        """
        Initialize the ImagePreprocessor.
        
        Args:
            target_height: Target height for resizing
            target_width: Target width for resizing
            binarization_method: Method for binarization ('otsu', 'adaptive', 'sauvola')
            denoise: Whether to apply denoising
            deskew: Whether to apply deskewing
        """
        self.target_height = target_height
        self.target_width = target_width
        self.binarization_method = binarization_method
        self.denoise = denoise
        self.deskew = deskew
        
        logger.info(f"ImagePreprocessor initialized with target size: {target_height}x{target_width}")
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from file path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Loaded image as numpy array
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from: {image_path}")
        logger.info(f"Loaded image: {image_path}, Shape: {image.shape}")
        return image
    
    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale.
        
        Args:
            image: Input image (BGR or already grayscale)
            
        Returns:
            Grayscale image
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        return gray
    
    def remove_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Remove noise from the image using Non-local Means Denoising.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Denoised image
        """
        # Apply Non-local Means Denoising
        denoised = cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7, searchWindowSize=21)
        return denoised
    
    def apply_binarization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply binarization to convert image to binary (black and white).
        
        Args:
            image: Input grayscale image
            
        Returns:
            Binary image
        """
        if self.binarization_method == "otsu":
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif self.binarization_method == "adaptive":
            binary = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, blockSize=11, C=2
            )
        elif self.binarization_method == "sauvola":
            binary = self._sauvola_threshold(image)
        else:
            raise ValueError(f"Unknown binarization method: {self.binarization_method}")
        
        return binary
    
    def _sauvola_threshold(self, image: np.ndarray, window_size: int = 25, k: float = 0.5, r: float = 128) -> np.ndarray:
        """
        Apply Sauvola binarization - good for documents with varying illumination.
        
        Args:
            image: Input grayscale image
            window_size: Size of the local window
            k: Parameter controlling the threshold value
            r: Dynamic range of standard deviation
            
        Returns:
            Binary image
        """
        # Calculate local mean
        mean = cv2.blur(image.astype(np.float64), (window_size, window_size))
        
        # Calculate local standard deviation
        mean_sq = cv2.blur(image.astype(np.float64) ** 2, (window_size, window_size))
        std = np.sqrt(mean_sq - mean ** 2)
        
        # Calculate threshold
        threshold = mean * (1 + k * (std / r - 1))
        
        # Apply threshold
        binary = np.zeros_like(image)
        binary[image > threshold] = 255
        
        return binary.astype(np.uint8)
    
    def deskew_image(self, image: np.ndarray) -> np.ndarray:
        """
        Correct the skew/rotation of the image.
        
        Args:
            image: Input image
            
        Returns:
            Deskewed image
        """
        # Detect edges
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        if lines is None:
            return image
        
        # Calculate angles
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angles.append(angle)
        
        # Get median angle
        median_angle = np.median(angles)
        
        # Limit rotation to avoid extreme corrections
        if abs(median_angle) > 45:
            median_angle = 0
        
        # Rotate image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        logger.info(f"Deskewed image by {median_angle:.2f} degrees")
        return rotated
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Args:
            image: Input grayscale image
            
        Returns:
            Contrast-enhanced image
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        return enhanced
    
    def resize_image(self, image: np.ndarray, maintain_aspect: bool = True) -> np.ndarray:
        """
        Resize image to target dimensions.
        
        Args:
            image: Input image
            maintain_aspect: Whether to maintain aspect ratio
            
        Returns:
            Resized image
        """
        if maintain_aspect:
            h, w = image.shape[:2]
            aspect = w / h
            
            if aspect > (self.target_width / self.target_height):
                new_w = self.target_width
                new_h = int(new_w / aspect)
            else:
                new_h = self.target_height
                new_w = int(new_h * aspect)
            
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Pad to target size
            delta_w = self.target_width - new_w
            delta_h = self.target_height - new_h
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            
            padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)
            return padded
        else:
            return cv2.resize(image, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize pixel values to range [0, 1].
        
        Args:
            image: Input image
            
        Returns:
            Normalized image as float32
        """
        return image.astype(np.float32) / 255.0
    
    def preprocess(self, image: Union[str, np.ndarray], for_training: bool = False) -> np.ndarray:
        """
        Apply full preprocessing pipeline to an image.
        
        Args:
            image: Image path or numpy array
            for_training: If True, return normalized float array
            
        Returns:
            Preprocessed image
        """
        # Load image if path provided
        if isinstance(image, str):
            img = self.load_image(image)
        else:
            img = image.copy()
        
        # Convert to grayscale
        gray = self.to_grayscale(img)
        
        # Denoise
        if self.denoise:
            gray = self.remove_noise(gray)
        
        # Enhance contrast
        gray = self.enhance_contrast(gray)
        
        # Deskew
        if self.deskew:
            gray = self.deskew_image(gray)
        
        # Binarize
        binary = self.apply_binarization(gray)
        
        # Resize
        resized = self.resize_image(binary)
        
        # Normalize for training
        if for_training:
            resized = self.normalize(resized)
        
        logger.info("Preprocessing complete")
        return resized
    
    def preprocess_for_ocr(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """
        Preprocess image specifically for OCR inference.
        
        Args:
            image: Image path or numpy array
            
        Returns:
            Preprocessed image ready for OCR
        """
        # Load image if path provided
        if isinstance(image, str):
            img = self.load_image(image)
        else:
            img = image.copy()
        
        # Convert to grayscale
        gray = self.to_grayscale(img)
        
        # Light denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, h=5)
        
        # Enhance contrast
        enhanced = self.enhance_contrast(denoised)
        
        # Mild sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened
    
    def detect_text_regions(self, image: np.ndarray) -> list:
        """
        Detect text regions in the image using contour detection.
        
        Args:
            image: Input image
            
        Returns:
            List of bounding boxes (x, y, w, h) for text regions
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        dilated = cv2.dilate(cv2.bitwise_not(gray), kernel, iterations=3)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get bounding boxes
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter small boxes
            if w > 50 and h > 10:
                boxes.append((x, y, w, h))
        
        # Sort by y-coordinate (top to bottom)
        boxes.sort(key=lambda b: (b[1], b[0]))
        
        return boxes
    
    def extract_text_regions(self, image: np.ndarray) -> list:
        """
        Extract text regions as separate images.
        
        Args:
            image: Input image
            
        Returns:
            List of cropped text region images
        """
        boxes = self.detect_text_regions(image)
        regions = []
        
        for x, y, w, h in boxes:
            region = image[y:y+h, x:x+w]
            regions.append(region)
        
        return regions


def main():
    """Test the preprocessing module."""
    import os
    
    # Create a test image with some text-like patterns
    test_image = np.ones((200, 400), dtype=np.uint8) * 255
    cv2.putText(test_image, "Test OCR", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 3)
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor()
    
    # Process the test image
    processed = preprocessor.preprocess(test_image)
    
    print(f"Original shape: {test_image.shape}")
    print(f"Processed shape: {processed.shape}")
    print("Preprocessing test completed successfully!")


if __name__ == "__main__":
    main()
