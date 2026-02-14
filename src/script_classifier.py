"""
Script Classifier Module for Indic-OCR
=======================================

This module provides a CNN-based script classification model
to identify the language/script of text in images.

Supported Scripts:
- Devanagari (Hindi)
- Malayalam
- Tamil
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from PIL import Image
import os
import logging
from typing import Tuple, List, Dict, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Script Labels
SCRIPT_LABELS = {
    0: "Devanagari",
    1: "Malayalam",
    2: "Tamil"
}

SCRIPT_TO_IDX = {v: k for k, v in SCRIPT_LABELS.items()}


class ScriptDataset(Dataset):
    """
    Custom Dataset for loading script images for classification.
    """
    
    def __init__(self, data_dir: str, transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing subdirectories for each script
            transform: Torchvision transforms to apply
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.labels = []
        
        # Load samples from each script directory
        for script_name, script_idx in SCRIPT_TO_IDX.items():
            script_dir = self.data_dir / script_name.lower()
            if script_dir.exists():
                for img_path in script_dir.glob("*.png"):
                    self.samples.append(str(img_path))
                    self.labels.append(script_idx)
                for img_path in script_dir.glob("*.jpg"):
                    self.samples.append(str(img_path))
                    self.labels.append(script_idx)
        
        logger.info(f"Loaded {len(self.samples)} samples from {data_dir}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class ScriptClassifierCNN(nn.Module):
    """
    Custom CNN architecture for script classification.
    """
    
    def __init__(self, num_classes: int = 3, dropout: float = 0.5):
        """
        Initialize the CNN model.
        
        Args:
            num_classes: Number of script classes
            dropout: Dropout probability
        """
        super(ScriptClassifierCNN, self).__init__()
        
        # Convolutional layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ScriptClassifierResNet(nn.Module):
    """
    ResNet-based script classifier using transfer learning.
    """
    
    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        """
        Initialize the ResNet model.
        
        Args:
            num_classes: Number of script classes
            pretrained: Whether to use pretrained weights
        """
        super(ScriptClassifierResNet, self).__init__()
        
        # Load pretrained ResNet18
        self.model = models.resnet18(pretrained=pretrained)
        
        # Modify the final layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ScriptClassifier:
    """
    High-level interface for script classification.
    """
    
    def __init__(
        self,
        model_type: str = "cnn",
        num_classes: int = 3,
        pretrained: bool = True,
        device: str = None
    ):
        """
        Initialize the script classifier.
        
        Args:
            model_type: Type of model ('cnn' or 'resnet')
            num_classes: Number of script classes
            pretrained: Use pretrained weights (for resnet)
            device: Device to use ('cuda' or 'cpu')
        """
        self.model_type = model_type
        self.num_classes = num_classes
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        if model_type == "cnn":
            self.model = ScriptClassifierCNN(num_classes=num_classes)
        elif model_type == "resnet":
            self.model = ScriptClassifierResNet(num_classes=num_classes, pretrained=pretrained)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model = self.model.to(self.device)
        
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"ScriptClassifier initialized with {model_type} on {self.device}")
    
    def train(
        self,
        train_dir: str,
        val_dir: str = None,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        save_path: str = "models/script_classifier.pth"
    ) -> Dict:
        """
        Train the script classifier.
        
        Args:
            train_dir: Directory containing training data
            val_dir: Directory containing validation data
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            save_path: Path to save the best model
            
        Returns:
            Training history dictionary
        """
        # Create datasets
        train_dataset = ScriptDataset(train_dir, transform=self.train_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        
        val_loader = None
        if val_dir:
            val_dataset = ScriptDataset(val_dir, transform=self.val_transform)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = 100.0 * train_correct / train_total
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validation phase
            if val_loader:
                val_loss, val_acc = self.evaluate(val_loader, criterion)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                scheduler.step(val_loss)
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save_model(save_path)
                    logger.info(f"Saved best model with val_acc: {val_acc:.2f}%")
                
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                           f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                           f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                           f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Always save the final model at the end of training
        self.save_model(save_path)
        return history
    
    def evaluate(self, data_loader: DataLoader, criterion=None) -> Tuple[float, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            data_loader: DataLoader for evaluation
            criterion: Loss function
            
        Returns:
            Tuple of (loss, accuracy)
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(data_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Predict the script of a single image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (predicted_script, confidence)
        """
        self.model.eval()
        
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        # Apply transforms
        image_tensor = self.val_transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)
        
        script_name = SCRIPT_LABELS[predicted.item()]
        confidence_score = confidence.item()
        
        return script_name, confidence_score
    
    def predict_batch(self, images: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        Predict scripts for a batch of images.
        
        Args:
            images: List of images as numpy arrays
            
        Returns:
            List of (predicted_script, confidence) tuples
        """
        self.model.eval()
        
        # Convert and transform all images
        tensors = []
        for img in images:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img).convert('RGB')
            tensors.append(self.val_transform(img))
        
        batch = torch.stack(tensors).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(batch)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predictions = probabilities.max(1)
        
        results = []
        for pred, conf in zip(predictions, confidences):
            script_name = SCRIPT_LABELS[pred.item()]
            results.append((script_name, conf.item()))
        
        return results
    
    def save_model(self, path: str):
        """Save model weights to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'num_classes': self.num_classes
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model weights from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")


def main():
    """Test the script classifier module."""
    # Create a dummy classifier
    classifier = ScriptClassifier(model_type="cnn", num_classes=3)
    
    # Create a dummy image
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Make prediction
    script, confidence = classifier.predict(dummy_image)
    
    print(f"Predicted Script: {script}")
    print(f"Confidence: {confidence:.4f}")
    print("Script classifier test completed successfully!")


if __name__ == "__main__":
    main()
