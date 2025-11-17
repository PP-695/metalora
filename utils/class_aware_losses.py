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
