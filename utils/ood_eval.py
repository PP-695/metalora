"""
OOD (Out-of-Distribution) detection evaluation utilities.

Implements metrics for evaluating OOD detection performance:
- AUROC (Area Under ROC Curve)
- AUPR (Area Under Precision-Recall Curve)  
- FPR@95 (False Positive Rate at 95% True Positive Rate)
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from typing import Tuple, Dict, Optional
from torch.utils.data import DataLoader


def compute_ood_scores_msp(logits: torch.Tensor) -> np.ndarray:
    """
    Compute Maximum Softmax Probability (MSP) scores.
    
    Higher MSP indicates in-distribution (ID).
    
    Args:
        logits: Model logits (N, num_classes)
        
    Returns:
        MSP scores (N,)
    """
    probs = F.softmax(logits, dim=1)
    msp_scores = probs.max(dim=1)[0]
    return msp_scores.cpu().numpy()


def compute_ood_scores_energy(logits: torch.Tensor, temperature: float = 1.0) -> np.ndarray:
    """
    Compute Energy-based scores.
    
    Higher energy indicates in-distribution (ID).
    Energy(x) = T * log(sum(exp(logit_i / T)))
    
    Args:
        logits: Model logits (N, num_classes)
        temperature: Temperature parameter
        
    Returns:
        Energy scores (N,)
    """
    energy = temperature * torch.logsumexp(logits / temperature, dim=1)
    return energy.cpu().numpy()


def compute_ood_scores_odin(
    model: torch.nn.Module,
    images: torch.Tensor,
    temperature: float = 1000.0,
    epsilon: float = 0.0014
) -> np.ndarray:
    """
    Compute ODIN (Out-of-DIstribution detector for Neural networks) scores.
    
    Uses temperature scaling and input preprocessing.
    
    Args:
        model: Neural network model
        images: Input images (N, C, H, W)
        temperature: Temperature for scaling logits
        epsilon: Magnitude of input perturbation
        
    Returns:
        ODIN scores (N,)
    """
    model.eval()
    images = images.clone().detach().requires_grad_(True)
    
    # Forward pass with temperature scaling
    logits = model(images)
    scaled_logits = logits / temperature
    
    # Compute maximum softmax probability
    probs = F.softmax(scaled_logits, dim=1)
    max_probs, _ = probs.max(dim=1)
    
    # Compute gradient
    max_probs.sum().backward()
    
    # Add perturbation in gradient direction
    gradient = images.grad.data
    images_perturbed = images - epsilon * gradient.sign()
    
    # Re-compute with perturbed inputs
    with torch.no_grad():
        logits_perturbed = model(images_perturbed)
        scaled_logits_perturbed = logits_perturbed / temperature
        probs_perturbed = F.softmax(scaled_logits_perturbed, dim=1)
        odin_scores = probs_perturbed.max(dim=1)[0]
    
    return odin_scores.cpu().numpy()


def compute_metrics(
    id_scores: np.ndarray,
    ood_scores: np.ndarray
) -> Dict[str, float]:
    """
    Compute OOD detection metrics.
    
    Args:
        id_scores: Scores for in-distribution samples (higher = more ID)
        ood_scores: Scores for OOD samples (higher = more ID)
        
    Returns:
        Dictionary with AUROC, AUPR-IN, AUPR-OUT, FPR@95
    """
    # Create labels (1 for ID, 0 for OOD)
    y_true = np.concatenate([
        np.ones(len(id_scores)),
        np.zeros(len(ood_scores))
    ])
    y_score = np.concatenate([id_scores, ood_scores])
    
    # AUROC
    auroc = roc_auc_score(y_true, y_score)
    
    # AUPR (for both ID as positive and OOD as positive)
    aupr_in = average_precision_score(y_true, y_score)
    aupr_out = average_precision_score(1 - y_true, -y_score)
    
    # FPR@95 (False Positive Rate at 95% True Positive Rate)
    fpr95 = compute_fpr_at_tpr(y_true, y_score, tpr_threshold=0.95)
    
    return {
        'auroc': auroc,
        'aupr_in': aupr_in,
        'aupr_out': aupr_out,
        'fpr95': fpr95
    }


def compute_fpr_at_tpr(y_true: np.ndarray, y_score: np.ndarray, tpr_threshold: float = 0.95) -> float:
    """
    Compute False Positive Rate at a given True Positive Rate threshold.
    
    Args:
        y_true: True labels (1 for ID, 0 for OOD)
        y_score: Prediction scores (higher = more ID)
        tpr_threshold: TPR threshold (e.g., 0.95 for FPR@95)
        
    Returns:
        FPR at the given TPR threshold
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    
    # Find FPR at TPR >= tpr_threshold
    idx = np.where(tpr >= tpr_threshold)[0]
    if len(idx) == 0:
        return 1.0  # Worst case
    
    return fpr[idx[0]]


def evaluate_ood_detection(
    model: torch.nn.Module,
    id_loader: DataLoader,
    ood_loader: DataLoader,
    device: torch.device,
    score_type: str = 'msp',
    temperature: float = 1.0,
    max_batches: Optional[int] = None
) -> Dict[str, float]:
    """
    Evaluate OOD detection performance.
    
    Args:
        model: Neural network model
        id_loader: DataLoader for in-distribution test data
        ood_loader: DataLoader for OOD test data
        device: Device to run evaluation on
        score_type: Type of OOD score ('msp', 'energy', 'odin')
        temperature: Temperature parameter (for energy/odin)
        max_batches: Maximum number of batches to evaluate (None = all)
        
    Returns:
        Dictionary with OOD detection metrics
    """
    model.eval()
    
    id_scores_list = []
    ood_scores_list = []
    
    # Collect ID scores
    with torch.no_grad():
        for i, (images, _) in enumerate(id_loader):
            if max_batches is not None and i >= max_batches:
                break
            
            images = images.to(device)
            logits = model(images)
            
            if score_type == 'msp':
                scores = compute_ood_scores_msp(logits)
            elif score_type == 'energy':
                scores = compute_ood_scores_energy(logits, temperature)
            elif score_type == 'odin':
                # ODIN requires gradient computation
                scores = compute_ood_scores_odin(model, images, temperature)
            else:
                raise ValueError(f"Unknown score type: {score_type}")
            
            id_scores_list.append(scores)
    
    # Collect OOD scores
    with torch.no_grad():
        for i, (images, _) in enumerate(ood_loader):
            if max_batches is not None and i >= max_batches:
                break
            
            images = images.to(device)
            logits = model(images)
            
            if score_type == 'msp':
                scores = compute_ood_scores_msp(logits)
            elif score_type == 'energy':
                scores = compute_ood_scores_energy(logits, temperature)
            elif score_type == 'odin':
                scores = compute_ood_scores_odin(model, images, temperature)
            else:
                raise ValueError(f"Unknown score type: {score_type}")
            
            ood_scores_list.append(scores)
    
    # Concatenate scores
    id_scores = np.concatenate(id_scores_list)
    ood_scores = np.concatenate(ood_scores_list)
    
    # Compute metrics
    metrics = compute_metrics(id_scores, ood_scores)
    
    return metrics


def evaluate_multiple_ood_datasets(
    model: torch.nn.Module,
    id_loader: DataLoader,
    ood_loaders: Dict[str, DataLoader],
    device: torch.device,
    score_type: str = 'msp',
    temperature: float = 1.0
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate OOD detection on multiple OOD datasets.
    
    Args:
        model: Neural network model
        id_loader: DataLoader for in-distribution test data
        ood_loaders: Dict mapping OOD dataset name to DataLoader
        device: Device to run evaluation on
        score_type: Type of OOD score ('msp', 'energy', 'odin')
        temperature: Temperature parameter
        
    Returns:
        Nested dict: {ood_name: {metric: value}}
    """
    results = {}
    
    for ood_name, ood_loader in ood_loaders.items():
        print(f"Evaluating OOD detection on {ood_name}...")
        metrics = evaluate_ood_detection(
            model, id_loader, ood_loader, device, score_type, temperature
        )
        results[ood_name] = metrics
    
    # Compute average metrics across all OOD datasets
    avg_metrics = {}
    for metric in ['auroc', 'aupr_in', 'aupr_out', 'fpr95']:
        values = [results[ood_name][metric] for ood_name in ood_loaders.keys()]
        avg_metrics[metric] = np.mean(values)
    
    results['average'] = avg_metrics
    
    return results


def print_ood_metrics(metrics: Dict[str, float], dataset_name: str = ""):
    """
    Pretty print OOD detection metrics.
    
    Args:
        metrics: Dictionary of metrics
        dataset_name: Name of OOD dataset (optional)
    """
    if dataset_name:
        print(f"\n{'='*60}")
        print(f"OOD Detection Metrics - {dataset_name}")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"OOD Detection Metrics")
        print(f"{'='*60}")
    
    print(f"AUROC:     {metrics['auroc']:.4f}")
    print(f"AUPR (ID): {metrics['aupr_in']:.4f}")
    print(f"AUPR (OOD):{metrics['aupr_out']:.4f}")
    print(f"FPR@95:    {metrics['fpr95']:.4f}")
    print(f"{'='*60}\n")
