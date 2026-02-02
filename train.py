"""
Training Script for Indic-OCR
==============================

This script handles training of the script classifier model.
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

from src.script_classifier import ScriptClassifier, SCRIPT_LABELS
from src.utils import load_config, ensure_directories, Timer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Indic-OCR models")
    
    parser.add_argument(
        '--task',
        type=str,
        default='script_classifier',
        choices=['script_classifier', 'ocr'],
        help='Training task'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='cnn',
        choices=['cnn', 'resnet'],
        help='Model architecture for script classifier'
    )
    
    parser.add_argument(
        '--train_dir',
        type=str,
        default='data/train',
        help='Training data directory'
    )
    
    parser.add_argument(
        '--val_dir',
        type=str,
        default='data/val',
        help='Validation data directory'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='models',
        help='Directory to save trained models'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training'
    )
    
    return parser.parse_args()


def train_script_classifier(args):
    """Train the script classifier model."""
    logger.info("=" * 60)
    logger.info("TRAINING SCRIPT CLASSIFIER")
    logger.info("=" * 60)
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize classifier
    logger.info(f"Initializing {args.model} model...")
    classifier = ScriptClassifier(
        model_type=args.model,
        num_classes=len(SCRIPT_LABELS),
        pretrained=True
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        classifier.load_model(args.resume)
    
    # Check data directories
    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    
    if not train_dir.exists():
        logger.error(f"Training directory not found: {train_dir}")
        logger.info("Please prepare your dataset first using the dataset utilities.")
        logger.info("Example: python -c \"from src.dataset import DatasetManager; dm = DatasetManager()\"")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate model filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = output_dir / f"script_classifier_{args.model}_{timestamp}.pth"
    
    # Start training
    timer = Timer().start()
    
    logger.info(f"Training configuration:")
    logger.info(f"  - Model: {args.model}")
    logger.info(f"  - Epochs: {args.epochs}")
    logger.info(f"  - Batch Size: {args.batch_size}")
    logger.info(f"  - Learning Rate: {args.learning_rate}")
    logger.info(f"  - Train Dir: {train_dir}")
    logger.info(f"  - Val Dir: {val_dir}")
    logger.info(f"  - Output: {model_path}")
    
    try:
        history = classifier.train(
            train_dir=str(train_dir),
            val_dir=str(val_dir) if val_dir.exists() else None,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            save_path=str(model_path)
        )
        
        timer.stop()
        
        # Save training history
        history_path = output_dir / f"training_history_{timestamp}.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Total time: {timer}")
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"History saved to: {history_path}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def train_ocr(args):
    """Fine-tune OCR model (placeholder for future implementation)."""
    logger.info("OCR fine-tuning is not yet implemented.")
    logger.info("Using pre-trained PaddleOCR/TrOCR models for now.")
    logger.info("For custom training, please refer to:")
    logger.info("  - PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR")
    logger.info("  - TrOCR: https://huggingface.co/docs/transformers/model_doc/trocr")


def main():
    """Main entry point."""
    args = parse_args()
    
    logger.info(f"Starting training task: {args.task}")
    
    if args.task == 'script_classifier':
        train_script_classifier(args)
    elif args.task == 'ocr':
        train_ocr(args)
    else:
        logger.error(f"Unknown task: {args.task}")


if __name__ == "__main__":
    main()
