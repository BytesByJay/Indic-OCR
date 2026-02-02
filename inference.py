"""
Inference Script for Indic-OCR
===============================

Run OCR on images using trained models.
"""

import argparse
import os
import sys
from pathlib import Path
import logging
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import ImagePreprocessor
from src.script_classifier import ScriptClassifier
from src.ocr_engine import OCREngine, OCRPipeline
from src.utils import Timer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Indic-OCR inference")
    
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to input image or directory of images'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: stdout)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        default='text',
        choices=['text', 'json'],
        help='Output format'
    )
    
    parser.add_argument(
        '--engine',
        type=str,
        default='paddleocr',
        choices=['paddleocr', 'easyocr', 'trocr'],
        help='OCR engine to use'
    )
    
    parser.add_argument(
        '--language',
        type=str,
        default=None,
        choices=['Devanagari', 'Malayalam', 'Tamil', 'English'],
        help='Language/script (auto-detect if not specified)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to trained script classifier model'
    )
    
    parser.add_argument(
        '--no_preprocess',
        action='store_true',
        help='Skip preprocessing'
    )
    
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU acceleration'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    return parser.parse_args()


def process_single_image(image_path: str, pipeline: OCRPipeline, args) -> dict:
    """Process a single image."""
    timer = Timer().start()
    
    result = pipeline.process(
        image_path,
        preprocess=not args.no_preprocess,
        detect_script=args.language is None
    )
    
    timer.stop()
    result['processing_time'] = timer.elapsed
    result['image_path'] = str(image_path)
    
    return result


def process_directory(dir_path: Path, pipeline: OCRPipeline, args) -> list:
    """Process all images in a directory."""
    results = []
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = [f for f in dir_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    logger.info(f"Found {len(image_files)} images to process")
    
    for i, img_path in enumerate(image_files, 1):
        logger.info(f"Processing [{i}/{len(image_files)}]: {img_path.name}")
        result = process_single_image(str(img_path), pipeline, args)
        results.append(result)
    
    return results


def format_output(results: list, format_type: str) -> str:
    """Format results for output."""
    if format_type == 'json':
        # Convert non-serializable items
        for r in results:
            if 'original_image' in r:
                r['original_image'] = str(r['original_image'])
            if 'boxes' in r:
                r['boxes'] = [[list(p) for p in box] if isinstance(box, list) else box 
                             for box in r.get('boxes', [])]
        return json.dumps(results, ensure_ascii=False, indent=2)
    
    else:  # text format
        output_lines = []
        
        for result in results:
            output_lines.append("=" * 60)
            
            if 'image_path' in result:
                output_lines.append(f"Image: {result['image_path']}")
            
            if result.get('script_detected'):
                output_lines.append(f"Script: {result['script_detected']} "
                                  f"(confidence: {result.get('script_confidence', 0):.2%})")
            
            output_lines.append("-" * 40)
            output_lines.append("Extracted Text:")
            output_lines.append(result.get('text', '[No text extracted]'))
            
            if 'processing_time' in result:
                output_lines.append(f"\nProcessing time: {result['processing_time']:.3f}s")
            
            if result.get('error'):
                output_lines.append(f"Error: {result['error']}")
        
        output_lines.append("=" * 60)
        
        return "\n".join(output_lines)


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Initializing Indic-OCR...")
    
    # Initialize components
    preprocessor = ImagePreprocessor() if not args.no_preprocess else None
    
    # Initialize script classifier
    classifier = None
    if args.language is None:
        classifier = ScriptClassifier(model_type="cnn")
        if args.model:
            classifier.load_model(args.model)
            logger.info(f"Loaded classifier model: {args.model}")
    
    # Initialize OCR engine
    ocr_engine = OCREngine(engine=args.engine, use_gpu=args.gpu)
    
    # Create pipeline
    pipeline = OCRPipeline(
        preprocessor=preprocessor,
        script_classifier=classifier,
        ocr_engine=ocr_engine
    )
    
    # Process input
    input_path = Path(args.image)
    
    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        sys.exit(1)
    
    timer = Timer().start()
    
    if input_path.is_file():
        results = [process_single_image(str(input_path), pipeline, args)]
    else:
        results = process_directory(input_path, pipeline, args)
    
    timer.stop()
    
    # Format output
    output_text = format_output(results, args.format)
    
    # Write output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_text)
        
        logger.info(f"Results saved to: {output_path}")
    else:
        print(output_text)
    
    logger.info(f"Total processing time: {timer}")


if __name__ == "__main__":
    main()
