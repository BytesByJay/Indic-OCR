"""
Evaluation Metrics Module for Indic-OCR
========================================

This module provides evaluation metrics for assessing OCR
and script classification performance.

Metrics included:
- Character Error Rate (CER)
- Word Error Rate (WER)
- Accuracy
- Precision, Recall, F1-Score
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from collections import Counter
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein distance between two strings.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Edit distance between the strings
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Calculate costs
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def character_error_rate(predicted: str, ground_truth: str) -> float:
    """
    Calculate Character Error Rate (CER).
    
    CER = (Insertions + Deletions + Substitutions) / Total Characters in Reference
    
    Args:
        predicted: Predicted text from OCR
        ground_truth: Ground truth text
        
    Returns:
        CER value (0.0 to 1.0+, lower is better)
    """
    if len(ground_truth) == 0:
        return 0.0 if len(predicted) == 0 else 1.0
    
    distance = levenshtein_distance(predicted, ground_truth)
    cer = distance / len(ground_truth)
    
    return cer


def word_error_rate(predicted: str, ground_truth: str) -> float:
    """
    Calculate Word Error Rate (WER).
    
    WER = (Word Insertions + Deletions + Substitutions) / Total Words in Reference
    
    Args:
        predicted: Predicted text from OCR
        ground_truth: Ground truth text
        
    Returns:
        WER value (0.0 to 1.0+, lower is better)
    """
    pred_words = predicted.split()
    gt_words = ground_truth.split()
    
    if len(gt_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    
    # Calculate Levenshtein distance at word level
    distance = levenshtein_distance_words(pred_words, gt_words)
    wer = distance / len(gt_words)
    
    return wer


def levenshtein_distance_words(words1: List[str], words2: List[str]) -> int:
    """
    Calculate Levenshtein distance between two word lists.
    
    Args:
        words1: First word list
        words2: Second word list
        
    Returns:
        Edit distance between word lists
    """
    if len(words1) < len(words2):
        return levenshtein_distance_words(words2, words1)
    
    if len(words2) == 0:
        return len(words1)
    
    previous_row = range(len(words2) + 1)
    
    for i, w1 in enumerate(words1):
        current_row = [i + 1]
        for j, w2 in enumerate(words2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (w1 != w2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def accuracy(predictions: List[str], ground_truths: List[str], exact_match: bool = True) -> float:
    """
    Calculate accuracy.
    
    Args:
        predictions: List of predicted texts
        ground_truths: List of ground truth texts
        exact_match: If True, require exact string match
        
    Returns:
        Accuracy value (0.0 to 1.0)
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")
    
    if len(predictions) == 0:
        return 0.0
    
    if exact_match:
        correct = sum(1 for p, g in zip(predictions, ground_truths) if p.strip() == g.strip())
    else:
        # Consider correct if CER < 0.1
        correct = sum(1 for p, g in zip(predictions, ground_truths) 
                     if character_error_rate(p, g) < 0.1)
    
    return correct / len(predictions)


def precision_recall_f1(
    predictions: List[str],
    ground_truths: List[str],
    num_classes: int = None
) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1-score for classification.
    
    Args:
        predictions: List of predicted labels
        ground_truths: List of ground truth labels
        num_classes: Number of classes (optional)
        
    Returns:
        Tuple of (precision, recall, f1_score)
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")
    
    # Calculate confusion matrix values
    true_positives = Counter()
    false_positives = Counter()
    false_negatives = Counter()
    
    for pred, gt in zip(predictions, ground_truths):
        if pred == gt:
            true_positives[gt] += 1
        else:
            false_positives[pred] += 1
            false_negatives[gt] += 1
    
    # Calculate macro-averaged metrics
    classes = set(ground_truths)
    
    precisions = []
    recalls = []
    
    for cls in classes:
        tp = true_positives[cls]
        fp = false_positives[cls]
        fn = false_negatives[cls]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
    
    avg_precision = np.mean(precisions) if precisions else 0.0
    avg_recall = np.mean(recalls) if recalls else 0.0
    
    if avg_precision + avg_recall > 0:
        f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
    else:
        f1 = 0.0
    
    return avg_precision, avg_recall, f1


def confusion_matrix(
    predictions: List[str],
    ground_truths: List[str],
    labels: List[str] = None
) -> np.ndarray:
    """
    Create a confusion matrix.
    
    Args:
        predictions: List of predicted labels
        ground_truths: List of ground truth labels
        labels: List of label names (optional)
        
    Returns:
        Confusion matrix as numpy array
    """
    if labels is None:
        labels = sorted(set(ground_truths) | set(predictions))
    
    label_to_idx = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    matrix = np.zeros((n, n), dtype=int)
    
    for pred, gt in zip(predictions, ground_truths):
        if pred in label_to_idx and gt in label_to_idx:
            matrix[label_to_idx[gt], label_to_idx[pred]] += 1
    
    return matrix


class OCRMetrics:
    """
    Comprehensive OCR evaluation metrics calculator.
    """
    
    def __init__(self):
        """Initialize metrics storage."""
        self.results = []
    
    def add_result(self, predicted: str, ground_truth: str, image_id: str = None):
        """
        Add a single result for evaluation.
        
        Args:
            predicted: Predicted text
            ground_truth: Ground truth text
            image_id: Optional image identifier
        """
        cer = character_error_rate(predicted, ground_truth)
        wer = word_error_rate(predicted, ground_truth)
        
        self.results.append({
            'image_id': image_id,
            'predicted': predicted,
            'ground_truth': ground_truth,
            'cer': cer,
            'wer': wer,
            'exact_match': predicted.strip() == ground_truth.strip()
        })
    
    def calculate_metrics(self) -> Dict:
        """
        Calculate aggregate metrics.
        
        Returns:
            Dictionary with all metrics
        """
        if not self.results:
            return {'error': 'No results to evaluate'}
        
        cers = [r['cer'] for r in self.results]
        wers = [r['wer'] for r in self.results]
        exact_matches = [r['exact_match'] for r in self.results]
        
        metrics = {
            'num_samples': len(self.results),
            'cer': {
                'mean': np.mean(cers),
                'std': np.std(cers),
                'min': np.min(cers),
                'max': np.max(cers),
                'median': np.median(cers)
            },
            'wer': {
                'mean': np.mean(wers),
                'std': np.std(wers),
                'min': np.min(wers),
                'max': np.max(wers),
                'median': np.median(wers)
            },
            'accuracy': {
                'exact_match': np.mean(exact_matches),
                'cer_below_10': np.mean([c < 0.1 for c in cers]),
                'cer_below_20': np.mean([c < 0.2 for c in cers]),
                'cer_below_50': np.mean([c < 0.5 for c in cers])
            }
        }
        
        return metrics
    
    def get_worst_cases(self, n: int = 10) -> List[Dict]:
        """
        Get n worst performing samples by CER.
        
        Args:
            n: Number of samples to return
            
        Returns:
            List of worst performing results
        """
        sorted_results = sorted(self.results, key=lambda x: x['cer'], reverse=True)
        return sorted_results[:n]
    
    def get_best_cases(self, n: int = 10) -> List[Dict]:
        """
        Get n best performing samples by CER.
        
        Args:
            n: Number of samples to return
            
        Returns:
            List of best performing results
        """
        sorted_results = sorted(self.results, key=lambda x: x['cer'])
        return sorted_results[:n]
    
    def save_report(self, output_path: str):
        """
        Save evaluation report to file.
        
        Args:
            output_path: Path to save report
        """
        report = {
            'metrics': self.calculate_metrics(),
            'worst_cases': self.get_worst_cases(10),
            'best_cases': self.get_best_cases(10),
            'all_results': self.results
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Report saved to {output_path}")
    
    def print_summary(self):
        """Print a summary of metrics."""
        metrics = self.calculate_metrics()
        
        print("\n" + "=" * 60)
        print("OCR EVALUATION REPORT")
        print("=" * 60)
        print(f"\nTotal Samples: {metrics['num_samples']}")
        print("\nCharacter Error Rate (CER):")
        print(f"  Mean: {metrics['cer']['mean']:.4f}")
        print(f"  Std:  {metrics['cer']['std']:.4f}")
        print(f"  Min:  {metrics['cer']['min']:.4f}")
        print(f"  Max:  {metrics['cer']['max']:.4f}")
        print("\nWord Error Rate (WER):")
        print(f"  Mean: {metrics['wer']['mean']:.4f}")
        print(f"  Std:  {metrics['wer']['std']:.4f}")
        print("\nAccuracy:")
        print(f"  Exact Match: {metrics['accuracy']['exact_match']*100:.2f}%")
        print(f"  CER < 10%:   {metrics['accuracy']['cer_below_10']*100:.2f}%")
        print(f"  CER < 20%:   {metrics['accuracy']['cer_below_20']*100:.2f}%")
        print("=" * 60)


class ClassificationMetrics:
    """
    Metrics for script classification evaluation.
    """
    
    def __init__(self, class_names: List[str]):
        """
        Initialize with class names.
        
        Args:
            class_names: List of class/script names
        """
        self.class_names = class_names
        self.predictions = []
        self.ground_truths = []
    
    def add_result(self, predicted: str, ground_truth: str):
        """Add a classification result."""
        self.predictions.append(predicted)
        self.ground_truths.append(ground_truth)
    
    def calculate_metrics(self) -> Dict:
        """Calculate classification metrics."""
        precision, recall, f1 = precision_recall_f1(self.predictions, self.ground_truths)
        acc = accuracy(self.predictions, self.ground_truths)
        conf_matrix = confusion_matrix(self.predictions, self.ground_truths, self.class_names)
        
        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix.tolist(),
            'class_names': self.class_names,
            'num_samples': len(self.predictions)
        }
    
    def print_summary(self):
        """Print classification metrics summary."""
        metrics = self.calculate_metrics()
        
        print("\n" + "=" * 60)
        print("SCRIPT CLASSIFICATION REPORT")
        print("=" * 60)
        print(f"\nTotal Samples: {metrics['num_samples']}")
        print(f"\nAccuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"Precision: {metrics['precision']*100:.2f}%")
        print(f"Recall:    {metrics['recall']*100:.2f}%")
        print(f"F1-Score:  {metrics['f1_score']*100:.2f}%")
        print("\nConfusion Matrix:")
        
        # Print header
        header = "       " + " ".join(f"{name[:8]:>8}" for name in self.class_names)
        print(header)
        
        # Print rows
        conf_matrix = np.array(metrics['confusion_matrix'])
        for i, name in enumerate(self.class_names):
            row = f"{name[:6]:>6} " + " ".join(f"{val:>8}" for val in conf_matrix[i])
            print(row)
        
        print("=" * 60)


def evaluate_ocr_results(
    predictions: List[str],
    ground_truths: List[str],
    output_file: str = None
) -> Dict:
    """
    Convenience function to evaluate OCR results.
    
    Args:
        predictions: List of predicted texts
        ground_truths: List of ground truth texts
        output_file: Optional path to save report
        
    Returns:
        Metrics dictionary
    """
    metrics = OCRMetrics()
    
    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        metrics.add_result(pred, gt, image_id=f"sample_{i}")
    
    metrics.print_summary()
    
    if output_file:
        metrics.save_report(output_file)
    
    return metrics.calculate_metrics()


def main():
    """Test evaluation metrics."""
    print("Testing Evaluation Metrics")
    print("=" * 50)
    
    # Test CER and WER
    pred = "hello world"
    gt = "hallo world"
    
    cer = character_error_rate(pred, gt)
    wer = word_error_rate(pred, gt)
    
    print(f"Predicted: '{pred}'")
    print(f"Ground Truth: '{gt}'")
    print(f"CER: {cer:.4f}")
    print(f"WER: {wer:.4f}")
    
    # Test with Indic text
    pred_indic = "नमस्ते भारत"
    gt_indic = "नमस्ते भारत"
    
    cer_indic = character_error_rate(pred_indic, gt_indic)
    print(f"\nIndic CER (exact match): {cer_indic:.4f}")
    
    # Test OCRMetrics
    print("\n" + "=" * 50)
    metrics = OCRMetrics()
    metrics.add_result("hello world", "hello world")
    metrics.add_result("hallo world", "hello world")
    metrics.add_result("test", "testing")
    metrics.print_summary()


if __name__ == "__main__":
    main()
