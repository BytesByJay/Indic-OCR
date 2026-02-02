"""
Dataset Utilities for Indic-OCR
================================

Tools for creating, managing, and preprocessing datasets
for training the script classifier and OCR models.
"""

import os
import json
import random
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

import numpy as np
from PIL import Image
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetManager:
    """
    Manager for creating and organizing OCR datasets.
    """
    
    def __init__(self, base_dir: str = "data"):
        """
        Initialize the dataset manager.
        
        Args:
            base_dir: Base directory for datasets
        """
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.train_dir = self.base_dir / "train"
        self.val_dir = self.base_dir / "val"
        self.test_dir = self.base_dir / "test"
        
        # Create directories
        for dir_path in [self.raw_dir, self.processed_dir, 
                        self.train_dir, self.val_dir, self.test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DatasetManager initialized with base_dir: {base_dir}")
    
    def organize_by_script(self, source_dir: str, annotations_file: str = None):
        """
        Organize images into script-based subdirectories.
        
        Args:
            source_dir: Directory containing source images
            annotations_file: JSON file with image-to-script mappings
        """
        source_path = Path(source_dir)
        
        # Script directories
        scripts = ["devanagari", "malayalam", "tamil"]
        for script in scripts:
            (self.raw_dir / script).mkdir(exist_ok=True)
        
        # Load annotations if provided
        if annotations_file and Path(annotations_file).exists():
            with open(annotations_file, 'r') as f:
                annotations = json.load(f)
        else:
            annotations = {}
        
        # Process images
        for img_path in source_path.glob("*"):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                # Get script from annotations or filename
                img_name = img_path.stem
                script = annotations.get(img_name, self._detect_script_from_name(img_name))
                
                if script:
                    dest = self.raw_dir / script.lower() / img_path.name
                    shutil.copy2(img_path, dest)
                    logger.info(f"Copied {img_path.name} to {script}")
    
    def _detect_script_from_name(self, filename: str) -> Optional[str]:
        """Detect script from filename patterns."""
        filename_lower = filename.lower()
        
        if any(x in filename_lower for x in ['hindi', 'devanagari', 'dev', 'hi']):
            return 'devanagari'
        elif any(x in filename_lower for x in ['malayalam', 'mal', 'ml']):
            return 'malayalam'
        elif any(x in filename_lower for x in ['tamil', 'ta']):
            return 'tamil'
        
        return None
    
    def split_dataset(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ):
        """
        Split raw dataset into train/val/test sets.
        
        Args:
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        
        # Process each script
        for script_dir in self.raw_dir.iterdir():
            if not script_dir.is_dir():
                continue
            
            script_name = script_dir.name
            
            # Create script subdirectories
            (self.train_dir / script_name).mkdir(exist_ok=True)
            (self.val_dir / script_name).mkdir(exist_ok=True)
            (self.test_dir / script_name).mkdir(exist_ok=True)
            
            # Get all images
            images = list(script_dir.glob("*"))
            images = [f for f in images if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
            random.shuffle(images)
            
            # Calculate split indices
            n = len(images)
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            
            # Split and copy
            for i, img in enumerate(images):
                if i < train_end:
                    dest = self.train_dir / script_name / img.name
                elif i < val_end:
                    dest = self.val_dir / script_name / img.name
                else:
                    dest = self.test_dir / script_name / img.name
                
                shutil.copy2(img, dest)
            
            logger.info(f"{script_name}: train={train_end}, val={val_end-train_end}, test={n-val_end}")
    
    def get_dataset_stats(self) -> Dict:
        """
        Get statistics about the dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'total_images': 0,
            'scripts': {},
            'splits': {
                'train': {},
                'val': {},
                'test': {}
            }
        }
        
        # Count images in each split
        for split_name, split_dir in [('train', self.train_dir), 
                                       ('val', self.val_dir), 
                                       ('test', self.test_dir)]:
            if not split_dir.exists():
                continue
                
            for script_dir in split_dir.iterdir():
                if script_dir.is_dir():
                    count = len(list(script_dir.glob("*")))
                    stats['splits'][split_name][script_dir.name] = count
                    
                    # Update script totals
                    if script_dir.name not in stats['scripts']:
                        stats['scripts'][script_dir.name] = 0
                    stats['scripts'][script_dir.name] += count
                    stats['total_images'] += count
        
        return stats
    
    def create_annotations_file(self, images_dir: str, output_file: str):
        """
        Create an annotations template file.
        
        Args:
            images_dir: Directory containing images
            output_file: Output JSON file path
        """
        images_path = Path(images_dir)
        annotations = {}
        
        for img_path in images_path.glob("*"):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                annotations[img_path.stem] = {
                    'script': '',  # To be filled
                    'text': '',    # Ground truth text
                    'type': ''     # 'printed' or 'handwritten'
                }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Created annotations template with {len(annotations)} entries")


class SyntheticDataGenerator:
    """
    Generate synthetic training data for script classification.
    """
    
    # Sample character sets for each script
    SCRIPT_CHARS = {
        'devanagari': 'अआइईउऊऋएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह',
        'malayalam': 'അആഇഈഉഊഎഏഐഒഓഔകഖഗഘങചഛജഝഞടഠഡഢണതഥദധനപഫബഭമയരലവശഷസഹ',
        'tamil': 'அஆஇஈஉஊஎஏஐஒஓஔகஙசஞடணதநபமயரலவழளறனஜஷஸஹ'
    }
    
    # Sample words for each script
    SCRIPT_WORDS = {
        'devanagari': ['नमस्ते', 'भारत', 'हिंदी', 'पुस्तक', 'विद्यालय', 'शिक्षा', 'संस्कृति', 'परिवार'],
        'malayalam': ['നമസ്കാരം', 'കേരളം', 'മലയാളം', 'പുസ്തകം', 'വിദ്യാലയം', 'ശിക്ഷണം', 'സംസ്കാരം'],
        'tamil': ['வணக்கம்', 'தமிழ்', 'இந்தியா', 'புத்தகம்', 'பள்ளிக்கூடம்', 'கல்வி', 'பண்பாடு']
    }
    
    def __init__(self, output_dir: str = "data/synthetic"):
        """
        Initialize the generator.
        
        Args:
            output_dir: Directory to save generated images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create script subdirectories
        for script in self.SCRIPT_CHARS.keys():
            (self.output_dir / script).mkdir(exist_ok=True)
    
    def generate_text_image(
        self,
        text: str,
        font_size: int = 32,
        image_size: Tuple[int, int] = (256, 64),
        add_noise: bool = True
    ) -> np.ndarray:
        """
        Generate an image with the given text.
        
        Args:
            text: Text to render
            font_size: Font size
            image_size: Output image size (width, height)
            add_noise: Whether to add noise
            
        Returns:
            Generated image as numpy array
        """
        # Create blank image
        img = np.ones((image_size[1], image_size[0]), dtype=np.uint8) * 255
        
        # For now, create placeholder pattern (font rendering requires system fonts)
        # In production, use PIL with proper Unicode fonts
        try:
            from PIL import ImageDraw, ImageFont
            
            pil_img = Image.fromarray(img)
            draw = ImageDraw.Draw(pil_img)
            
            # Try to use a Unicode-capable font
            try:
                # Try common system fonts
                font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", font_size)
                except:
                    font = ImageFont.load_default()
            
            # Draw text
            draw.text((10, 10), text, font=font, fill=0)
            img = np.array(pil_img)
            
        except Exception as e:
            logger.warning(f"Could not render text: {e}")
            # Create pattern-based placeholder
            cv2.putText(img, "Text", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
        
        # Add noise if requested
        if add_noise:
            noise = np.random.normal(0, 5, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return img
    
    def generate_dataset(
        self,
        samples_per_script: int = 100,
        add_augmentation: bool = True
    ):
        """
        Generate synthetic dataset for all scripts.
        
        Args:
            samples_per_script: Number of samples to generate per script
            add_augmentation: Whether to apply data augmentation
        """
        for script, words in self.SCRIPT_WORDS.items():
            script_dir = self.output_dir / script
            
            for i in range(samples_per_script):
                # Select random word or create random text
                if random.random() > 0.5:
                    text = random.choice(words)
                else:
                    chars = self.SCRIPT_CHARS[script]
                    text = ''.join(random.choices(chars, k=random.randint(3, 8)))
                
                # Generate image
                img = self.generate_text_image(
                    text,
                    font_size=random.randint(24, 48),
                    add_noise=add_augmentation
                )
                
                # Apply augmentation
                if add_augmentation:
                    img = self._augment_image(img)
                
                # Save image
                img_path = script_dir / f"{script}_{i:04d}.png"
                cv2.imwrite(str(img_path), img)
            
            logger.info(f"Generated {samples_per_script} samples for {script}")
    
    def _augment_image(self, img: np.ndarray) -> np.ndarray:
        """Apply random augmentations to image."""
        # Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-5, 5)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
            img = cv2.warpAffine(img, M, (w, h), borderValue=255)
        
        # Random brightness
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            img = np.clip(img * factor, 0, 255).astype(np.uint8)
        
        # Random blur
        if random.random() > 0.7:
            kernel_size = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        
        return img


class GroundTruthLoader:
    """
    Load and manage ground truth annotations for evaluation.
    """
    
    def __init__(self, annotations_file: str):
        """
        Initialize with annotations file.
        
        Args:
            annotations_file: Path to JSON annotations file
        """
        self.annotations_file = Path(annotations_file)
        self.annotations = {}
        
        if self.annotations_file.exists():
            self.load()
    
    def load(self):
        """Load annotations from file."""
        with open(self.annotations_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        logger.info(f"Loaded {len(self.annotations)} annotations")
    
    def save(self):
        """Save annotations to file."""
        with open(self.annotations_file, 'w', encoding='utf-8') as f:
            json.dump(self.annotations, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(self.annotations)} annotations")
    
    def add_annotation(self, image_id: str, text: str, script: str = None):
        """
        Add or update an annotation.
        
        Args:
            image_id: Image identifier
            text: Ground truth text
            script: Script/language name
        """
        self.annotations[image_id] = {
            'text': text,
            'script': script
        }
    
    def get_annotation(self, image_id: str) -> Optional[Dict]:
        """Get annotation for an image."""
        return self.annotations.get(image_id)
    
    def get_all_texts(self) -> List[str]:
        """Get all ground truth texts."""
        return [ann['text'] for ann in self.annotations.values() if 'text' in ann]
    
    def export_to_csv(self, output_file: str):
        """Export annotations to CSV format."""
        import csv
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['image_id', 'text', 'script'])
            
            for img_id, ann in self.annotations.items():
                writer.writerow([img_id, ann.get('text', ''), ann.get('script', '')])
        
        logger.info(f"Exported to {output_file}")


def main():
    """Test dataset utilities."""
    print("Dataset Utilities for Indic-OCR")
    print("=" * 50)
    
    # Test DatasetManager
    dm = DatasetManager("data")
    stats = dm.get_dataset_stats()
    print(f"Dataset stats: {stats}")
    
    # Test SyntheticDataGenerator
    # generator = SyntheticDataGenerator()
    # generator.generate_dataset(samples_per_script=10)
    
    print("\nDataset utilities test completed!")


if __name__ == "__main__":
    main()
