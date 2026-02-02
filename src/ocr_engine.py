"""
OCR Engine Module for Indic-OCR
================================

This module provides a unified OCR interface supporting multiple
OCR backends for text recognition in Indian regional scripts.

Supported Backends:
- PaddleOCR (Primary)
- TrOCR (Transformer-based)
- EasyOCR (Fallback)
"""

import numpy as np
from PIL import Image
import logging
import os
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCREngine:
    """
    Unified OCR Engine supporting multiple backends.
    """
    
    # Language code mapping for different OCR engines
    LANGUAGE_CODES = {
        "PaddleOCR": {
            "Devanagari": "hi",
            "Malayalam": "ml",
            "Tamil": "ta",
            "English": "en"
        },
        "EasyOCR": {
            "Devanagari": "hi",
            "Malayalam": "ml",
            "Tamil": "ta",
            "English": "en"
        },
        "TrOCR": {
            "Devanagari": "devanagari",
            "Malayalam": "malayalam",
            "Tamil": "tamil",
            "English": "english"
        }
    }
    
    def __init__(
        self,
        engine: str = "paddleocr",
        languages: List[str] = None,
        use_gpu: bool = True
    ):
        """
        Initialize the OCR Engine.
        
        Args:
            engine: OCR engine to use ('paddleocr', 'trocr', 'easyocr')
            languages: List of language names to support
            use_gpu: Whether to use GPU acceleration
        """
        self.engine_name = engine.lower()
        self.languages = languages or ["Devanagari", "Malayalam", "Tamil", "English"]
        self.use_gpu = use_gpu
        self.engine = None
        
        self._initialize_engine()
        
        logger.info(f"OCREngine initialized with {self.engine_name}")
    
    def _initialize_engine(self):
        """Initialize the selected OCR engine."""
        if self.engine_name == "paddleocr":
            self._init_paddleocr()
        elif self.engine_name == "easyocr":
            self._init_easyocr()
        elif self.engine_name == "trocr":
            self._init_trocr()
        else:
            raise ValueError(f"Unsupported OCR engine: {self.engine_name}")
    
    def _init_paddleocr(self):
        """Initialize PaddleOCR engine."""
        try:
            from paddleocr import PaddleOCR
            
            # PaddleOCR supports multiple languages
            self.engines = {}
            
            for lang_name in self.languages:
                lang_code = self.LANGUAGE_CODES["PaddleOCR"].get(lang_name, "en")
                try:
                    self.engines[lang_name] = PaddleOCR(
                        use_angle_cls=True,
                        lang=lang_code,
                        use_gpu=self.use_gpu,
                        show_log=False
                    )
                    logger.info(f"PaddleOCR initialized for {lang_name}")
                except Exception as e:
                    logger.warning(f"Could not initialize PaddleOCR for {lang_name}: {e}")
            
            self.engine = "paddleocr"
            
        except ImportError:
            logger.error("PaddleOCR not installed. Install with: pip install paddleocr")
            raise
    
    def _init_easyocr(self):
        """Initialize EasyOCR engine."""
        try:
            import easyocr
            
            # Get language codes for EasyOCR
            lang_codes = []
            for lang_name in self.languages:
                code = self.LANGUAGE_CODES["EasyOCR"].get(lang_name)
                if code:
                    lang_codes.append(code)
            
            if not lang_codes:
                lang_codes = ["en"]
            
            self.reader = easyocr.Reader(
                lang_codes,
                gpu=self.use_gpu,
                verbose=False
            )
            
            self.engine = "easyocr"
            logger.info(f"EasyOCR initialized with languages: {lang_codes}")
            
        except ImportError:
            logger.error("EasyOCR not installed. Install with: pip install easyocr")
            raise
    
    def _init_trocr(self):
        """Initialize TrOCR (Transformer OCR) engine."""
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch
            
            # Use the multilingual TrOCR model
            model_name = "microsoft/trocr-base-handwritten"
            
            self.processor = TrOCRProcessor.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            
            self.device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            
            self.engine = "trocr"
            logger.info(f"TrOCR initialized on {self.device}")
            
        except ImportError:
            logger.error("Transformers not installed. Install with: pip install transformers")
            raise
    
    def recognize(
        self,
        image: Union[str, np.ndarray, Image.Image],
        language: str = None,
        return_confidence: bool = True
    ) -> Dict:
        """
        Perform OCR on an image.
        
        Args:
            image: Image path, numpy array, or PIL Image
            language: Specific language to use (optional)
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary containing recognized text and metadata
        """
        # Convert image to appropriate format
        img = self._prepare_image(image)
        
        if self.engine_name == "paddleocr":
            return self._recognize_paddleocr(img, language, return_confidence)
        elif self.engine_name == "easyocr":
            return self._recognize_easyocr(img, return_confidence)
        elif self.engine_name == "trocr":
            return self._recognize_trocr(img, return_confidence)
        else:
            raise ValueError(f"Engine not initialized: {self.engine_name}")
    
    def _prepare_image(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """Prepare image for OCR processing."""
        if isinstance(image, str):
            img = np.array(Image.open(image).convert('RGB'))
        elif isinstance(image, Image.Image):
            img = np.array(image.convert('RGB'))
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                # Grayscale to RGB
                img = np.stack([image] * 3, axis=-1)
            else:
                img = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        return img
    
    def _recognize_paddleocr(
        self,
        image: np.ndarray,
        language: str = None,
        return_confidence: bool = True
    ) -> Dict:
        """Perform OCR using PaddleOCR."""
        # Select appropriate engine
        if language and language in self.engines:
            ocr = self.engines[language]
        else:
            # Use first available engine
            ocr = list(self.engines.values())[0] if self.engines else None
        
        if ocr is None:
            return {"text": "", "lines": [], "error": "No OCR engine available"}
        
        # Perform OCR
        result = ocr.ocr(image, cls=True)
        
        if result is None or len(result) == 0:
            return {"text": "", "lines": [], "boxes": []}
        
        # Parse results
        lines = []
        boxes = []
        full_text = []
        
        for line in result:
            if line is None:
                continue
            for word_info in line:
                box = word_info[0]
                text = word_info[1][0]
                confidence = word_info[1][1]
                
                lines.append({
                    "text": text,
                    "confidence": confidence,
                    "box": box
                })
                boxes.append(box)
                full_text.append(text)
        
        return {
            "text": " ".join(full_text),
            "lines": lines,
            "boxes": boxes,
            "engine": "paddleocr"
        }
    
    def _recognize_easyocr(
        self,
        image: np.ndarray,
        return_confidence: bool = True
    ) -> Dict:
        """Perform OCR using EasyOCR."""
        result = self.reader.readtext(image)
        
        lines = []
        boxes = []
        full_text = []
        
        for detection in result:
            box, text, confidence = detection
            
            lines.append({
                "text": text,
                "confidence": confidence,
                "box": box
            })
            boxes.append(box)
            full_text.append(text)
        
        return {
            "text": " ".join(full_text),
            "lines": lines,
            "boxes": boxes,
            "engine": "easyocr"
        }
    
    def _recognize_trocr(
        self,
        image: np.ndarray,
        return_confidence: bool = True
    ) -> Dict:
        """Perform OCR using TrOCR."""
        import torch
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image).convert("RGB")
        
        # Process image
        pixel_values = self.processor(images=pil_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        
        # Generate text
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return {
            "text": generated_text,
            "lines": [{"text": generated_text, "confidence": 1.0}],
            "boxes": [],
            "engine": "trocr"
        }
    
    def detect_text_regions(self, image: Union[str, np.ndarray, Image.Image]) -> List[Dict]:
        """
        Detect text regions in an image without recognition.
        
        Args:
            image: Input image
            
        Returns:
            List of detected text regions with bounding boxes
        """
        img = self._prepare_image(image)
        
        if self.engine_name == "paddleocr":
            # PaddleOCR detection
            ocr = list(self.engines.values())[0] if self.engines else None
            if ocr:
                result = ocr.ocr(img, rec=False)
                regions = []
                if result:
                    for line in result:
                        if line:
                            for box in line:
                                regions.append({"box": box})
                return regions
        
        return []
    
    def recognize_with_script_detection(
        self,
        image: Union[str, np.ndarray, Image.Image],
        script_classifier=None
    ) -> Dict:
        """
        Perform OCR with automatic script detection.
        
        Args:
            image: Input image
            script_classifier: Optional ScriptClassifier instance
            
        Returns:
            OCR result with detected script
        """
        img = self._prepare_image(image)
        
        detected_script = None
        confidence = None
        
        # Detect script if classifier provided
        if script_classifier:
            detected_script, confidence = script_classifier.predict(img)
            logger.info(f"Detected script: {detected_script} (confidence: {confidence:.2f})")
        
        # Perform OCR with detected language
        result = self.recognize(img, language=detected_script)
        
        result["detected_script"] = detected_script
        result["script_confidence"] = confidence
        
        return result
    
    def recognize_batch(
        self,
        images: List[Union[str, np.ndarray, Image.Image]],
        language: str = None
    ) -> List[Dict]:
        """
        Perform OCR on multiple images.
        
        Args:
            images: List of images
            language: Specific language to use
            
        Returns:
            List of OCR results
        """
        results = []
        for img in images:
            result = self.recognize(img, language)
            results.append(result)
        return results
    
    def get_unicode_text(self, result: Dict) -> str:
        """
        Extract Unicode text from OCR result.
        
        Args:
            result: OCR result dictionary
            
        Returns:
            Unicode text string
        """
        return result.get("text", "")


class OCRPipeline:
    """
    Complete OCR pipeline combining preprocessing, script detection, and recognition.
    """
    
    def __init__(
        self,
        preprocessor=None,
        script_classifier=None,
        ocr_engine=None
    ):
        """
        Initialize the OCR pipeline.
        
        Args:
            preprocessor: ImagePreprocessor instance
            script_classifier: ScriptClassifier instance
            ocr_engine: OCREngine instance
        """
        self.preprocessor = preprocessor
        self.script_classifier = script_classifier
        self.ocr_engine = ocr_engine
        
        logger.info("OCRPipeline initialized")
    
    def process(
        self,
        image: Union[str, np.ndarray, Image.Image],
        preprocess: bool = True,
        detect_script: bool = True
    ) -> Dict:
        """
        Process an image through the complete OCR pipeline.
        
        Args:
            image: Input image
            preprocess: Whether to apply preprocessing
            detect_script: Whether to detect script before OCR
            
        Returns:
            Complete OCR result dictionary
        """
        result = {
            "original_image": None,
            "preprocessed": False,
            "script_detected": None,
            "script_confidence": None,
            "text": "",
            "lines": [],
            "boxes": [],
            "error": None
        }
        
        try:
            # Load image
            if isinstance(image, str):
                img = np.array(Image.open(image).convert('RGB'))
            elif isinstance(image, Image.Image):
                img = np.array(image.convert('RGB'))
            else:
                img = image
            
            result["original_image"] = img.shape
            
            # Preprocess
            if preprocess and self.preprocessor:
                img = self.preprocessor.preprocess_for_ocr(img)
                result["preprocessed"] = True
            
            # Detect script
            if detect_script and self.script_classifier:
                script, conf = self.script_classifier.predict(img)
                result["script_detected"] = script
                result["script_confidence"] = conf
                
                # Perform OCR with detected language
                ocr_result = self.ocr_engine.recognize(img, language=script)
            else:
                ocr_result = self.ocr_engine.recognize(img)
            
            # Update result
            result["text"] = ocr_result.get("text", "")
            result["lines"] = ocr_result.get("lines", [])
            result["boxes"] = ocr_result.get("boxes", [])
            result["engine"] = ocr_result.get("engine", "unknown")
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"OCR pipeline error: {e}")
        
        return result
    
    def process_batch(
        self,
        images: List[Union[str, np.ndarray, Image.Image]],
        preprocess: bool = True,
        detect_script: bool = True
    ) -> List[Dict]:
        """
        Process multiple images through the OCR pipeline.
        
        Args:
            images: List of images
            preprocess: Whether to apply preprocessing
            detect_script: Whether to detect script
            
        Returns:
            List of OCR result dictionaries
        """
        results = []
        for img in images:
            result = self.process(img, preprocess, detect_script)
            results.append(result)
        return results


def create_default_pipeline(use_gpu: bool = True) -> OCRPipeline:
    """
    Create a default OCR pipeline with all components.
    
    Args:
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Configured OCRPipeline instance
    """
    from .preprocessing import ImagePreprocessor
    from .script_classifier import ScriptClassifier
    
    preprocessor = ImagePreprocessor()
    script_classifier = ScriptClassifier(model_type="cnn")
    ocr_engine = OCREngine(engine="paddleocr", use_gpu=use_gpu)
    
    pipeline = OCRPipeline(
        preprocessor=preprocessor,
        script_classifier=script_classifier,
        ocr_engine=ocr_engine
    )
    
    return pipeline


def main():
    """Test the OCR engine module."""
    print("OCR Engine Module")
    print("=" * 50)
    print("Supported engines: paddleocr, easyocr, trocr")
    print("Supported languages: Devanagari, Malayalam, Tamil, English")
    print("\nTo use, create an OCREngine instance and call recognize()")
    print("Example:")
    print("  engine = OCREngine(engine='paddleocr')")
    print("  result = engine.recognize('image.png')")
    print("  print(result['text'])")


if __name__ == "__main__":
    main()
