"""
Class-aware loss functions for long-tailed learning with meta-learned parameters.

This module provides loss functions that can adapt their behavior based on
per-class meta-parameters learned during meta-optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class ClassAwareLDAM(nn.Module):
    """
    LDAM Loss with per-class adaptive margins.
    
    Extends the standard LDAM loss to support meta-learned per-class margins
    that can be adjusted during meta-optimization.
    
    Args:
        cls_num_list: List/tensor of sample counts per class
        max_m: Maximum margin value
        s: Scale parameter for logits
        use_class_weights: If True, use learnable per-class margin weights
    """
    
    def __init__(self, cls_num_list, max_m=0.5, s=30, use_class_weights=False):
        super().__init__()
        # Compute base margins as in standard LDAM
        m_list = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(cls_num_list, dtype=torch.float32)))
        m_list = m_list * (max_m / torch.max(m_list))
        
        # Store as buffer (non-trainable by default)
        self.register_buffer('base_m_list', m_list)
        self.s = s
        self.use_class_weights = use_class_weights
        
        # Optional learnable per-class weights
        if use_class_weights:
            # Initialize to 1.0 (no adjustment)
            self.class_margin_weights = nn.Parameter(torch.ones(len(cls_num_list)))
        else:
            self.class_margin_weights = None
    
    def forward(self, logit, target):
        """
        Forward pass with adaptive margins.
        
        Args:
            logit: Model predictions (batch_size, num_classes)
            target: Ground truth labels (batch_size,)
            
        Returns:
            Cross-entropy loss with LDAM margins
        """
        # Get effective margins
        if self.class_margin_weights is not None:
            # Apply learnable weights to base margins
            m_list = self.base_m_list * torch.sigmoid(self.class_margin_weights)
        else:
            m_list = self.base_m_list
        
        # Create index mask for target classes
        index = torch.zeros_like(logit, dtype=torch.bool)
        index.scatter_(1, target.view(-1, 1), True)
        
        # Compute per-sample margins
        batch_m = torch.matmul(m_list[None, :], index.float().t()).t()
        
        # Apply margins to target class logits
        logit_m = logit - batch_m * self.s
        
        # Use adjusted logits for target classes, original for others
        output = torch.where(index, logit_m, logit)
        
        return F.cross_entropy(output, target)
    
    def get_meta_parameters(self):
        """Return meta-learnable parameters."""
        if self.class_margin_weights is not None:
            return [self.class_margin_weights]
        return []


class ClassAwareBalancedSoftmax(nn.Module):
    """
    Balanced Softmax Loss with meta-learned class-specific adjustments.
    
    Args:
        cls_num_list: List/tensor of sample counts per class
        use_class_weights: If True, use learnable per-class adjustment weights
    """
    
    def __init__(self, cls_num_list, use_class_weights=False):
        super().__init__()
        cls_num_list = torch.tensor(cls_num_list, dtype=torch.float32)
        cls_num_ratio = cls_num_list / torch.sum(cls_num_list)
        log_cls_num = torch.log(cls_num_ratio)
        
        self.register_buffer('base_log_cls_num', log_cls_num)
        self.use_class_weights = use_class_weights
        
        if use_class_weights:
            # Learnable adjustment to log probabilities
            self.class_log_weights = nn.Parameter(torch.zeros(len(cls_num_list)))
        else:
            self.class_log_weights = None
    
    def forward(self, logit, target):
        """
        Forward pass with balanced softmax.
        
        Args:
            logit: Model predictions (batch_size, num_classes)
            target: Ground truth labels (batch_size,)
            
        Returns:
            Balanced softmax cross-entropy loss
        """
        # Get effective log adjustments
        if self.class_log_weights is not None:
            log_adjustment = self.base_log_cls_num + self.class_log_weights
        else:
            log_adjustment = self.base_log_cls_num
        
        # Adjust logits
        logit_adjusted = logit + log_adjustment.unsqueeze(0)
        
        return F.cross_entropy(logit_adjusted, target)
    
    def get_meta_parameters(self):
        """Return meta-learnable parameters."""
        if self.class_log_weights is not None:
            return [self.class_log_weights]
        return []


class BalancedAccuracyLoss(nn.Module):
    """
    Loss function that optimizes for balanced accuracy (mean per-class accuracy).
    
    This loss approximates the balanced accuracy objective by computing
    a weighted cross-entropy where each class is weighted equally.
    
    Args:
        num_classes: Total number of classes
        temperature: Temperature for soft weighting (lower = harder)
    """
    
    def __init__(self, num_classes, temperature=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
    
    def forward(self, logit, target):
        """
        Forward pass optimizing for balanced accuracy.
        
        Args:
            logit: Model predictions (batch_size, num_classes)
            target: Ground truth labels (batch_size,)
            
        Returns:
            Balanced cross-entropy loss
        """
        # Compute per-class sample counts in the batch
        batch_size = logit.size(0)
        class_counts = torch.zeros(self.num_classes, device=logit.device)
        
        for c in range(self.num_classes):
            class_counts[c] = (target == c).sum()
        
        # Compute weights: inverse of class frequency in batch
        # Add small epsilon to avoid division by zero
        eps = 1e-8
        class_weights = 1.0 / (class_counts + eps)
        
        # Normalize weights so they sum to num_classes
        class_weights = class_weights * self.num_classes / (class_weights.sum() + eps)
        
        # Apply temperature scaling
        class_weights = class_weights ** (1.0 / self.temperature)
        
        # Compute weighted cross-entropy
        return F.cross_entropy(logit, target, weight=class_weights)


class ClassDistributionRegularizer(nn.Module):
    """
    Regularizer to prevent extreme divergence in class-specific parameters.
    
    This regularizer encourages smoothness in the learned class-specific
    parameters (e.g., ranks, alphas) while allowing adaptation.
    
    Args:
        smoothness_weight: Weight for smoothness regularization
        sparsity_weight: Weight for sparsity regularization
    """
    
    def __init__(self, smoothness_weight=0.01, sparsity_weight=0.0):
        super().__init__()
        self.smoothness_weight = smoothness_weight
        self.sparsity_weight = sparsity_weight
    
    def forward(self, class_params: torch.Tensor) -> torch.Tensor:
        """
        Compute regularization loss for class-specific parameters.
        
        Args:
            class_params: Tensor of shape (num_classes,) or (num_classes, param_dim)
            
        Returns:
            Regularization loss
        """
        loss = 0.0
        
        if self.smoothness_weight > 0:
            # L2 smoothness: penalize large differences between adjacent classes
            # This assumes classes are ordered by frequency
            if class_params.dim() == 1:
                diff = class_params[1:] - class_params[:-1]
                smoothness_loss = (diff ** 2).mean()
            else:
                diff = class_params[1:] - class_params[:-1]
                smoothness_loss = (diff ** 2).sum(dim=-1).mean()
            
            loss = loss + self.smoothness_weight * smoothness_loss
        
        if self.sparsity_weight > 0:
            # L1 sparsity: encourage parameters to be close to zero
            sparsity_loss = torch.abs(class_params).mean()
            loss = loss + self.sparsity_weight * sparsity_loss
        
        return loss


class ClassAwareFocalLoss(nn.Module):
    """
    Focal Loss with per-class adaptive focusing parameters.
    
    Args:
        cls_num_list: List/tensor of sample counts per class
        base_gamma: Base focusing parameter
        use_class_gamma: If True, use learnable per-class gamma values
    """
    
    def __init__(self, cls_num_list, base_gamma=2.0, use_class_gamma=False):
        super().__init__()
        self.base_gamma = base_gamma
        self.use_class_gamma = use_class_gamma
        
        if use_class_gamma:
            # Initialize class-specific gammas around base_gamma
            num_classes = len(cls_num_list)
            self.class_gamma = nn.Parameter(torch.ones(num_classes) * base_gamma)
        else:
            self.class_gamma = None
    
    def forward(self, logit, target):
        """
        Forward pass with class-aware focal loss.
        
        Args:
            logit: Model predictions (batch_size, num_classes)
            target: Ground truth labels (batch_size,)
            
        Returns:
            Focal loss
        """
        # Compute probabilities
        probs = F.softmax(logit, dim=1)
        
        # Get probabilities for target classes
        target_probs = probs.gather(1, target.view(-1, 1)).squeeze(1)
        
        # Compute focal weight
        if self.class_gamma is not None:
            # Use class-specific gamma values
            gamma = self.class_gamma[target]
        else:
            gamma = self.base_gamma
        
        focal_weight = (1 - target_probs) ** gamma
        
        # Compute cross-entropy
        ce_loss = F.cross_entropy(logit, target, reduction='none')
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()
    
    def get_meta_parameters(self):
        """Return meta-learnable parameters."""
        if self.class_gamma is not None:
            return [self.class_gamma]
        return []


class OutlierClassLoss(nn.Module):
    """
    Outlier Class Learning (OCL) loss for OOD detection.
    
    Supports an auxiliary OOD dataset with an explicit outlier class
    appended to the in-distribution classes. The last logit index is
    reserved for the outlier class.
    
    Args:
        num_id_classes: Number of in-distribution classes
        weight: Optional weight for the OOD samples
    """
    
    def __init__(self, num_id_classes: int, weight: float = 1.0):
        super().__init__()
        self.num_id_classes = num_id_classes
        self.weight = weight
        # Outlier class index is num_id_classes (0-indexed)
        self.outlier_class_idx = num_id_classes
    
    def forward(self, logits, labels, is_ood=None):
        """
        Forward pass for outlier class learning.
        
        Args:
            logits: Model predictions (batch_size, num_classes + 1)
                   Last index is outlier class logit
            labels: Ground truth labels (batch_size,)
                   For OOD samples, should be set to num_id_classes
            is_ood: Optional boolean tensor (batch_size,) indicating OOD samples
                   If None, infers from labels
            
        Returns:
            Cross-entropy loss treating last class as outlier
        """
        if is_ood is None:
            # Infer OOD samples from labels
            is_ood = labels >= self.num_id_classes
        
        # Compute standard cross-entropy
        loss = F.cross_entropy(logits, labels, reduction='none')
        
        # Apply weight to OOD samples if specified
        if self.weight != 1.0:
            weights = torch.where(is_ood, 
                                torch.tensor(self.weight, device=loss.device), 
                                torch.tensor(1.0, device=loss.device))
            loss = loss * weights
        
        return loss.mean()


class TailPrototypeLoss(nn.Module):
    """
    OOD-Aware Tail Prototype Learning loss.
    
    Contrasts tail class features with OOD features to push tail representations
    away from OOD distribution. Uses contrastive learning approach.
    
    Args:
        temperature: Temperature for contrastive loss (default: 0.07)
        margin: Margin for pushing tail away from OOD (default: 0.0)
    """
    
    def __init__(self, temperature: float = 0.07, margin: float = 0.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, features_tail, features_ood, labels_tail=None):
        """
        Compute contrastive loss between tail and OOD features.
        
        Args:
            features_tail: Features from tail class samples (N_tail, D)
            features_ood: Features from OOD samples (N_ood, D)
            labels_tail: Optional labels for tail samples (N_tail,)
            
        Returns:
            Contrastive loss pushing tail away from OOD
        """
        # Normalize features
        features_tail = F.normalize(features_tail, dim=1)
        features_ood = F.normalize(features_ood, dim=1)
        
        # Compute similarity matrix: (N_tail, N_ood)
        similarity = torch.matmul(features_tail, features_ood.t()) / self.temperature
        
        # We want to minimize similarity (push tail away from OOD)
        # Use negative of similarity as logits for a "repulsion" objective
        # Create labels where all OOD samples should have low similarity
        
        # Simple approach: minimize max similarity to any OOD sample
        max_similarity = similarity.max(dim=1)[0]  # (N_tail,)
        
        # Apply margin and compute loss
        loss = F.relu(max_similarity - self.margin)
        
        return loss.mean()
    
    def forward_with_prototypes(self, features_tail, features_ood, labels_tail):
        """
        Alternative: Use per-class prototypes for tail classes.
        
        Args:
            features_tail: Features from tail class samples (N_tail, D)
            features_ood: Features from OOD samples (N_ood, D)
            labels_tail: Labels for tail samples (N_tail,)
            
        Returns:
            Prototype-based contrastive loss
        """
        # Normalize features
        features_tail = F.normalize(features_tail, dim=1)
        features_ood = F.normalize(features_ood, dim=1)
        
        # Compute per-class prototypes for tail
        unique_labels = torch.unique(labels_tail)
        prototypes = []
        
        for label in unique_labels:
            mask = labels_tail == label
            if mask.sum() > 0:
                proto = features_tail[mask].mean(dim=0)
                prototypes.append(F.normalize(proto.unsqueeze(0), dim=1))
        
        if len(prototypes) == 0:
            return torch.tensor(0.0, device=features_tail.device)
        
        prototypes = torch.cat(prototypes, dim=0)  # (num_tail_classes, D)
        
        # Compute similarity between prototypes and OOD features
        similarity = torch.matmul(prototypes, features_ood.t()) / self.temperature
        
        # Minimize max similarity
        max_similarity = similarity.max(dim=1)[0]  # (num_tail_classes,)
        loss = F.relu(max_similarity - self.margin)
        
        return loss.mean()


def calibrate_logits(logits, class_priors, tau: float = 1.0):
    """
    Calibrated Logit Adjustment for inference.
    
    Adjusts logits based on class priors to correct for class imbalance.
    This is typically applied at test time.
    
    Args:
        logits: Model predictions (batch_size, num_classes)
        class_priors: Prior probabilities for each class (num_classes,)
                     Should sum to 1.0
        tau: Temperature parameter controlling adjustment strength
             tau=0 means no adjustment, tau=1 means full adjustment
            
    Returns:
        Calibrated logits (batch_size, num_classes)
    """
    if tau == 0.0:
        return logits
    
    # Ensure class_priors is on same device as logits
    if not isinstance(class_priors, torch.Tensor):
        class_priors = torch.tensor(class_priors, device=logits.device, dtype=logits.dtype)
    else:
        class_priors = class_priors.to(device=logits.device, dtype=logits.dtype)
    
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    class_priors = torch.clamp(class_priors, min=eps)
    
    # Adjust logits by subtracting tau * log(prior)
    adjustment = tau * torch.log(class_priors)
    
    return logits - adjustment.unsqueeze(0)


class DebiasedHeadLoss(nn.Module):
    """
    Debiased loss for head classes to prevent over-confidence.
    
    Applies a penalty to head class predictions to reduce their dominance
    and improve calibration for tail classes.
    
    Args:
        cls_num_list: List/tensor of sample counts per class
        head_threshold: Threshold for classifying as head class (default: 100)
        penalty_weight: Weight for the debiasing penalty (default: 0.1)
    """
    
    def __init__(self, cls_num_list, head_threshold: int = 100, penalty_weight: float = 0.1):
        super().__init__()
        cls_num_list = torch.tensor(cls_num_list, dtype=torch.float32)
        
        # Identify head classes
        is_head = cls_num_list >= head_threshold
        self.register_buffer('is_head', is_head)
        self.penalty_weight = penalty_weight
    
    def forward(self, logits, target):
        """
        Compute debiased loss.
        
        Args:
            logits: Model predictions (batch_size, num_classes)
            target: Ground truth labels (batch_size,)
            
        Returns:
            Cross-entropy with debiasing penalty
        """
        # Standard cross-entropy
        ce_loss = F.cross_entropy(logits, target, reduction='none')
        
        # Compute penalty for over-confident head predictions
        probs = F.softmax(logits, dim=1)
        
        # Get head class probabilities
        head_probs = probs[:, self.is_head]  # (batch_size, num_head_classes)
        
        # Penalty: discourage high confidence on head classes
        # Use entropy-based penalty (lower entropy = higher confidence)
        head_entropy = -(head_probs * torch.log(head_probs + 1e-8)).sum(dim=1)
        max_head_entropy = torch.log(torch.tensor(self.is_head.sum().float()))
        
        # Normalize and invert (high entropy = low penalty)
        confidence_penalty = 1.0 - (head_entropy / (max_head_entropy + 1e-8))
        
        # Apply penalty
        loss = ce_loss + self.penalty_weight * confidence_penalty
        
        return loss.mean()
