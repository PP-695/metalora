#!/usr/bin/env python
"""
Evaluation script for class-aware meta-learning.

This script loads a trained model and evaluates it with detailed class-specific metrics:
- Overall accuracy
- Many/Medium/Few (Head/Medium/Tail) accuracy
- G-Mean (geometric mean of per-class accuracies)
- Worst-case accuracy
- Confusion matrix
- Per-class accuracy plots
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate class-aware model with detailed metrics')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Directory containing trained model')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., CIFAR100_IR100)')
    parser.add_argument('--model', type=str, default='clip_vit_b16',
                        help='Model architecture')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for plots (default: same as model-dir)')
    parser.add_argument('--save-confmat', action='store_true',
                        help='Save confusion matrix plot')
    parser.add_argument('--save-per-class', action='store_true',
                        help='Save per-class accuracy plot')
    return parser.parse_args()


def plot_confusion_matrix(y_true, y_pred, num_classes, save_path):
    """Plot and save confusion matrix."""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=True)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def plot_per_class_accuracy(per_class_acc, head_indices, medium_indices, tail_indices, save_path):
    """Plot per-class accuracy with group coloring."""
    num_classes = len(per_class_acc)
    colors = ['gray'] * num_classes
    
    for idx in head_indices:
        colors[idx] = 'green'
    for idx in medium_indices:
        colors[idx] = 'orange'
    for idx in tail_indices:
        colors[idx] = 'red'
    
    plt.figure(figsize=(14, 6))
    plt.bar(range(num_classes), per_class_acc, color=colors, alpha=0.7)
    plt.xlabel('Class Index')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Class Accuracy')
    plt.axhline(y=per_class_acc.mean(), color='k', linestyle='--', alpha=0.5, label=f'Mean: {per_class_acc.mean():.2f}%')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label=f'Head ({len(head_indices)} classes)'),
        Patch(facecolor='orange', alpha=0.7, label=f'Medium ({len(medium_indices)} classes)'),
        Patch(facecolor='red', alpha=0.7, label=f'Tail ({len(tail_indices)} classes)'),
    ]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved per-class accuracy plot to {save_path}")


def main():
    args = parse_args()
    
    # Import after args parsing to avoid torch import issues if just showing help
    try:
        import torch
        from omegaconf import OmegaConf
        from trainer import MetaTrainer, Trainer
        from trainer_class_aware import ClassAwareMetaTrainer
        from utils.config_omega import cfg
        from utils.class_imbalance_utils import (
            compute_balanced_accuracy,
            compute_group_accuracy,
            split_by_imbalance_ratio
        )
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Make sure PyTorch and other dependencies are installed.")
        sys.exit(1)
    
    # Set output directory
    output_dir = args.output_dir or args.model_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Update config
    cfg.dataset = args.dataset
    cfg.backbone = args.model
    cfg.model_dir = args.model_dir
    cfg.test_only = True
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create trainer
    print(f"Loading model from {args.model_dir}")
    if getattr(cfg, 'use_class_aware', False):
        trainer = ClassAwareMetaTrainer(cfg, device)
    elif getattr(cfg, 'use_meta', False):
        trainer = MetaTrainer(cfg, device)
    else:
        trainer = Trainer(cfg, device)
    
    trainer.initialize()
    trainer.load_model(args.model_dir)
    
    # Run evaluation
    print("\nRunning evaluation...")
    results = trainer.test()
    
    # Get class distribution
    cls_num_list = np.array(trainer.cls_num_list)
    head_indices, medium_indices, tail_indices = split_by_imbalance_ratio(
        cls_num_list, head_threshold=100, tail_threshold=20
    )
    
    # Print detailed metrics
    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80)
    
    # Overall metrics
    if 'acc' in results:
        print(f"\nOverall Accuracy: {results['acc']:.2f}%")
    
    # Group metrics
    if hasattr(trainer, 'many_idxs'):
        print(f"\nGroup Accuracies:")
        print(f"  Head (Many):   {results.get('many_acc', 0):.2f}%")
        print(f"  Medium:        {results.get('medium_acc', 0):.2f}%")
        print(f"  Tail (Few):    {results.get('few_acc', 0):.2f}%")
        
        head_tail_gap = results.get('many_acc', 0) - results.get('few_acc', 0)
        print(f"\nHead-Tail Gap: {head_tail_gap:.2f}%")
    
    # Additional metrics if available
    if 'balanced_acc' in results:
        print(f"\nBalanced Accuracy: {results['balanced_acc']:.2f}%")
    if 'geometric_mean' in results:
        print(f"G-Mean: {results['geometric_mean']:.4f}")
    if 'worst_case_acc' in results:
        print(f"Worst-Case Accuracy: {results['worst_case_acc']:.2f}%")
    
    print("\n" + "="*80)
    
    # Generate plots if requested
    # Note: This requires collecting predictions during test, which may need modification
    # to the test() method to return predictions and labels
    if args.save_confmat or args.save_per_class:
        print("\nNote: Confusion matrix and per-class plots require model predictions.")
        print("This feature requires extending the test() method to return predictions.")
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
