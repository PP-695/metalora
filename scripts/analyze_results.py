"""
Advanced visualization and analysis tools for long-tailed OOD learning.

Generates:
1. Confusion matrix heatmaps
2. Per-class accuracy vs sample count scatter plots
3. t-SNE/UMAP embeddings
4. Calibration curves/reliability diagrams
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from typing import Dict, List, Optional, Tuple
import argparse


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    save_path: str,
    normalize: bool = True,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 10)
):
    """
    Plot and save confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes
        save_path: Path to save figure
        normalize: Whether to normalize by true class counts
        class_names: Optional list of class names
        figsize: Figure size
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot heatmap
    if num_classes <= 20:
        # Show all ticks for small number of classes
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                   cmap='Blues', cbar=True,
                   xticklabels=class_names if class_names else range(num_classes),
                   yticklabels=class_names if class_names else range(num_classes))
    else:
        # Skip annotations for many classes
        sns.heatmap(cm, annot=False, cmap='Blues', cbar=True)
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''), fontsize=14)
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {save_path}")


def plot_per_class_accuracy(
    class_accuracies: np.ndarray,
    class_sample_counts: np.ndarray,
    class_groups: Optional[Dict[str, List[int]]] = None,
    save_path: str = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot per-class accuracy vs sample count scatter plot.
    
    Args:
        class_accuracies: Accuracy for each class (num_classes,)
        class_sample_counts: Number of samples per class (num_classes,)
        class_groups: Optional dict mapping group name to class indices
                     e.g., {'head': [0,1,2], 'medium': [3,4], 'tail': [5,6,7]}
        save_path: Path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    if class_groups is not None:
        # Plot with different colors for each group
        colors = {'head': 'blue', 'medium': 'orange', 'tail': 'red'}
        markers = {'head': 'o', 'medium': 's', 'tail': '^'}
        
        for group_name, class_indices in class_groups.items():
            counts = class_sample_counts[class_indices]
            accs = class_accuracies[class_indices]
            plt.scatter(counts, accs, 
                       c=colors.get(group_name, 'gray'),
                       marker=markers.get(group_name, 'o'),
                       s=100, alpha=0.7, label=group_name.capitalize())
    else:
        # Plot all classes with same color
        plt.scatter(class_sample_counts, class_accuracies, 
                   c='blue', s=100, alpha=0.7)
    
    plt.xlabel('Number of Training Samples (log scale)', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title('Per-Class Accuracy vs Sample Count', fontsize=14)
    plt.xscale('log')
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    
    if class_groups is not None:
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Per-class accuracy plot saved to {save_path}")
    else:
        plt.show()


def plot_tsne_embeddings(
    features: np.ndarray,
    labels: np.ndarray,
    class_groups: Optional[Dict[str, List[int]]] = None,
    ood_features: Optional[np.ndarray] = None,
    save_path: str = None,
    figsize: Tuple[int, int] = (10, 8),
    perplexity: int = 30,
    n_iter: int = 1000
):
    """
    Plot t-SNE embeddings of features.
    
    Args:
        features: Feature vectors (N, D)
        labels: Class labels (N,)
        class_groups: Optional dict mapping group name to class indices
        ood_features: Optional OOD features (N_ood, D)
        save_path: Path to save figure
        figsize: Figure size
        perplexity: t-SNE perplexity parameter
        n_iter: Number of t-SNE iterations
    """
    print("Computing t-SNE embeddings...")
    
    # Combine ID and OOD features if provided
    if ood_features is not None:
        all_features = np.vstack([features, ood_features])
        all_labels = np.concatenate([labels, np.full(len(ood_features), -1)])
    else:
        all_features = features
        all_labels = labels
    
    # Compute t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    embeddings = tsne.fit_transform(all_features)
    
    # Split back into ID and OOD
    if ood_features is not None:
        id_embeddings = embeddings[:len(features)]
        ood_embeddings = embeddings[len(features):]
    else:
        id_embeddings = embeddings
        ood_embeddings = None
    
    # Plot
    plt.figure(figsize=figsize)
    
    if class_groups is not None:
        # Plot with different colors for each group
        colors = {'head': 'blue', 'medium': 'orange', 'tail': 'red'}
        
        for group_name, class_indices in class_groups.items():
            mask = np.isin(labels, class_indices)
            plt.scatter(id_embeddings[mask, 0], id_embeddings[mask, 1],
                       c=colors.get(group_name, 'gray'),
                       s=20, alpha=0.6, label=f'{group_name.capitalize()} classes')
    else:
        # Color by class
        num_classes = len(np.unique(labels))
        if num_classes <= 20:
            scatter = plt.scatter(id_embeddings[:, 0], id_embeddings[:, 1],
                                c=labels, s=20, alpha=0.6, cmap='tab20')
            plt.colorbar(scatter, label='Class')
        else:
            plt.scatter(id_embeddings[:, 0], id_embeddings[:, 1],
                       c='blue', s=20, alpha=0.6, label='ID')
    
    # Plot OOD if provided
    if ood_embeddings is not None:
        plt.scatter(ood_embeddings[:, 0], ood_embeddings[:, 1],
                   c='black', s=20, alpha=0.6, marker='x', label='OOD')
    
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.title('t-SNE Embeddings of Features', fontsize=14)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"t-SNE plot saved to {save_path}")
    else:
        plt.show()


def plot_calibration_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    n_bins: int = 10,
    class_groups: Optional[Dict[str, List[int]]] = None,
    save_path: str = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot calibration curve (reliability diagram).
    
    Args:
        y_true: True labels (N,)
        y_probs: Predicted probabilities (N, num_classes)
        n_bins: Number of bins for calibration
        class_groups: Optional dict to plot separate curves per group
        save_path: Path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Get predicted class and confidence
    y_pred = np.argmax(y_probs, axis=1)
    confidences = np.max(y_probs, axis=1)
    accuracies = (y_pred == y_true).astype(float)
    
    if class_groups is not None:
        # Plot separate calibration curves for each group
        colors = {'head': 'blue', 'medium': 'orange', 'tail': 'red'}
        
        for group_name, class_indices in class_groups.items():
            mask = np.isin(y_true, class_indices)
            if mask.sum() == 0:
                continue
            
            group_conf = confidences[mask]
            group_acc = accuracies[mask]
            
            # Compute calibration
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            bin_accs = []
            bin_confs = []
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (group_conf > bin_lower) & (group_conf <= bin_upper)
                if in_bin.sum() > 0:
                    bin_accs.append(group_acc[in_bin].mean())
                    bin_confs.append(group_conf[in_bin].mean())
            
            if len(bin_accs) > 0:
                plt.plot(bin_confs, bin_accs, 'o-', 
                        color=colors.get(group_name, 'gray'),
                        label=f'{group_name.capitalize()} classes',
                        linewidth=2, markersize=8)
    else:
        # Plot overall calibration curve
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_accs = []
        bin_confs = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            if in_bin.sum() > 0:
                bin_accs.append(accuracies[in_bin].mean())
                bin_confs.append(confidences[in_bin].mean())
        
        plt.plot(bin_confs, bin_accs, 'o-', color='blue', 
                label='Overall', linewidth=2, markersize=8)
    
    # Plot perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    plt.xlabel('Confidence', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Calibration Curve (Reliability Diagram)', fontsize=14)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Calibration curve saved to {save_path}")
    else:
        plt.show()


def compute_per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Compute per-class accuracy.
    
    Args:
        y_true: True labels (N,)
        y_pred: Predicted labels (N,)
        num_classes: Number of classes
        
    Returns:
        Per-class accuracies (num_classes,)
    """
    accs = np.zeros(num_classes)
    
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum() > 0:
            accs[c] = (y_pred[mask] == c).mean()
        else:
            accs[c] = 0.0
    
    return accs


def analyze_results(
    logits_path: str,
    labels_path: str,
    features_path: Optional[str] = None,
    ood_features_path: Optional[str] = None,
    class_counts_path: Optional[str] = None,
    class_groups_path: Optional[str] = None,
    output_dir: str = "output/plots",
    save_confmat: bool = True,
    save_per_class: bool = True,
    save_tsne: bool = False,
    save_calibration: bool = True
):
    """
    Main function to analyze and visualize results.
    
    Args:
        logits_path: Path to saved logits (.npy)
        labels_path: Path to saved labels (.npy)
        features_path: Optional path to saved features
        ood_features_path: Optional path to OOD features
        class_counts_path: Optional path to class sample counts
        class_groups_path: Optional path to class groups dict
        output_dir: Directory to save plots
        save_confmat: Whether to save confusion matrix
        save_per_class: Whether to save per-class accuracy plot
        save_tsne: Whether to save t-SNE plot
        save_calibration: Whether to save calibration curve
    """
    # Load data
    print("Loading data...")
    logits = np.load(logits_path)
    labels = np.load(labels_path)
    
    # Get predictions
    y_pred = np.argmax(logits, axis=1)
    y_true = labels
    num_classes = logits.shape[1]
    
    # Compute probabilities
    y_probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    
    # Load optional data
    class_groups = None
    if class_groups_path and os.path.exists(class_groups_path):
        import pickle
        with open(class_groups_path, 'rb') as f:
            class_groups = pickle.load(f)
    
    class_counts = None
    if class_counts_path and os.path.exists(class_counts_path):
        class_counts = np.load(class_counts_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Confusion matrix
    if save_confmat:
        confmat_path = os.path.join(output_dir, "confusion_matrix.png")
        plot_confusion_matrix(y_true, y_pred, num_classes, confmat_path)
    
    # 2. Per-class accuracy
    if save_per_class and class_counts is not None:
        per_class_acc = compute_per_class_accuracy(y_true, y_pred, num_classes)
        per_class_path = os.path.join(output_dir, "per_class_accuracy.png")
        plot_per_class_accuracy(per_class_acc, class_counts, class_groups, per_class_path)
    
    # 3. t-SNE embeddings
    if save_tsne and features_path and os.path.exists(features_path):
        features = np.load(features_path)
        ood_features = None
        if ood_features_path and os.path.exists(ood_features_path):
            ood_features = np.load(ood_features_path)
        
        tsne_path = os.path.join(output_dir, "tsne_embeddings.png")
        plot_tsne_embeddings(features, y_true, class_groups, ood_features, tsne_path)
    
    # 4. Calibration curve
    if save_calibration:
        calib_path = os.path.join(output_dir, "calibration_curve.png")
        plot_calibration_curve(y_true, y_probs, class_groups=class_groups, save_path=calib_path)
    
    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and visualize long-tailed OOD results")
    parser.add_argument("--logits", type=str, required=True, help="Path to logits file (.npy)")
    parser.add_argument("--labels", type=str, required=True, help="Path to labels file (.npy)")
    parser.add_argument("--features", type=str, default=None, help="Path to features file (.npy)")
    parser.add_argument("--ood-features", type=str, default=None, help="Path to OOD features file (.npy)")
    parser.add_argument("--class-counts", type=str, default=None, help="Path to class counts file (.npy)")
    parser.add_argument("--class-groups", type=str, default=None, help="Path to class groups file (.pkl)")
    parser.add_argument("--output-dir", type=str, default="output/plots", help="Output directory for plots")
    parser.add_argument("--save-confmat", action="store_true", default=True, help="Save confusion matrix")
    parser.add_argument("--save-per-class", action="store_true", default=True, help="Save per-class accuracy")
    parser.add_argument("--save-tsne", action="store_true", default=False, help="Save t-SNE embeddings")
    parser.add_argument("--save-calibration", action="store_true", default=True, help="Save calibration curve")
    
    args = parser.parse_args()
    
    analyze_results(
        logits_path=args.logits,
        labels_path=args.labels,
        features_path=args.features,
        ood_features_path=args.ood_features,
        class_counts_path=args.class_counts,
        class_groups_path=args.class_groups,
        output_dir=args.output_dir,
        save_confmat=args.save_confmat,
        save_per_class=args.save_per_class,
        save_tsne=args.save_tsne,
        save_calibration=args.save_calibration
    )
