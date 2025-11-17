"""
Class-Aware Meta-Trainer for long-tailed learning with adaptive PEFT.

This module extends MetaTrainer to optimize for balanced accuracy across
class groups (head/medium/tail) and track class-specific performance.
"""

import os
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from trainer import MetaTrainer
from utils.meter import AverageMeter
from utils.class_imbalance_utils import (
    split_by_imbalance_ratio,
    compute_balanced_accuracy,
    compute_group_accuracy,
    visualize_class_distribution
)


class ClassAwareMetaTrainer(MetaTrainer):
    """
    Meta-trainer with class-aware optimization for long-tailed datasets.
    
    Extends MetaTrainer to:
    - Analyze class distribution and group classes
    - Optimize for balanced accuracy instead of overall accuracy
    - Track per-group (head/medium/tail) performance
    - Visualize rank/alpha evolution during training
    - Apply progressive class weighting
    
    Args:
        cfg: Configuration object
        device: Device for training
    """
    
    def __init__(self, cfg, device):
        # Initialize additional config parameters
        self.use_class_aware = getattr(cfg, 'use_class_aware', False)
        self.meta_objective = getattr(cfg, 'meta_objective', 'balanced_accuracy')
        self.focus_on_tail = getattr(cfg, 'focus_on_tail', True)
        self.tail_loss_weight = getattr(cfg, 'tail_loss_weight', 2.0)
        self.rank_divergence_penalty = getattr(cfg, 'rank_divergence_penalty', 0.01)
        self.alpha_smoothness_penalty = getattr(cfg, 'alpha_smoothness_penalty', 0.005)
        
        # Initialize parent
        super().__init__(cfg, device)
        
        # Class grouping information (initialized in _analyze_class_distribution)
        self.head_indices = None
        self.medium_indices = None
        self.tail_indices = None
        
        # Track metrics history for visualization
        self.metrics_history = defaultdict(list)
        if not hasattr(self, 'tuner'):
            self.tuner = None
    
    def initialize(self):
        """Override initialization to add class distribution analysis"""
        super().initialize()
        
        if self.use_class_aware:
            self._analyze_class_distribution()
            self._initialize_class_aware_modules()
    
    def _analyze_class_distribution(self):
        """
        Analyze the class distribution and split into head/medium/tail groups.
        """
        cls_num_list = np.array(self.cls_num_list)
        
        # Use existing many/med/few indices if available
        if hasattr(self, 'many_idxs') and len(self.many_idxs) > 0:
            self.head_indices = self.many_idxs
            self.medium_indices = self.med_idxs
            self.tail_indices = self.few_idxs
        else:
            # Split based on thresholds
            head_threshold = getattr(self.cfg, 'head_threshold', 100)
            tail_threshold = getattr(self.cfg, 'tail_threshold', 20)
            
            self.head_indices, self.medium_indices, self.tail_indices = \
                split_by_imbalance_ratio(cls_num_list, head_threshold, tail_threshold)
        
        # Log class distribution
        print(f"\nClass Distribution Analysis:")
        print(f"  Total classes: {self.num_classes}")
        print(f"  Head classes: {len(self.head_indices)} (samples >= 100)")
        print(f"  Medium classes: {len(self.medium_indices)} (20 <= samples < 100)")
        print(f"  Tail classes: {len(self.tail_indices)} (samples < 20)")
        print(f"  Imbalance Ratio: {cls_num_list.max() / cls_num_list.min():.2f}")
        
        # Visualize distribution if output_dir exists
        if hasattr(self, 'cfg') and hasattr(self.cfg, 'output_dir'):
            save_path = os.path.join(self.cfg.output_dir, 'class_distribution.png')
            try:
                visualize_class_distribution(
                    cls_num_list,
                    self.head_indices,
                    self.medium_indices,
                    self.tail_indices,
                    save_path=save_path
                )
            except Exception as e:
                print(f"Warning: Could not save class distribution plot: {e}")
    
    def _initialize_class_aware_modules(self):
        """
        Initialize class-aware modules with class distribution information.
        """
        if self.tuner is None:
            return
        
        cls_num_list = np.array(self.cls_num_list)
        
        # Find all ClassAwareMetaLoRA modules and initialize them
        for name, module in self.tuner.named_modules():
            if hasattr(module, 'initialize_from_imbalance'):
                print(f"Initializing class-aware module: {name}")
                
                # Get config parameters
                head_rank_factor = getattr(self.cfg, 'head_rank_factor', 0.5)
                tail_rank_factor = getattr(self.cfg, 'tail_rank_factor', 2.0)
                head_alpha_factor = getattr(self.cfg, 'head_alpha_factor', 0.5)
                tail_alpha_factor = getattr(self.cfg, 'tail_alpha_factor', 2.0)
                
                module.initialize_from_imbalance(
                    cls_num_list,
                    head_rank_factor=head_rank_factor,
                    tail_rank_factor=tail_rank_factor,
                    head_alpha_factor=head_alpha_factor,
                    tail_alpha_factor=tail_alpha_factor
                )
    
    def _meta_optimization_step(self, meters):
        """
        Override meta-optimization to use balanced accuracy objective.
        """
        if not self.use_class_aware:
            # Use standard meta-optimization
            return super()._meta_optimization_step(meters)
        
        # Backup base model parameters
        self._backup_weights()
        
        # Set modules to meta mode
        self._set_meta_mode(True)
        
        # 1. Inner loop: optimize model parameters on meta-train set
        for _ in range(self.meta_inner_steps):
            # Get a batch from meta-train
            try:
                meta_batch = next(iter(self.meta_train_loader))
            except StopIteration:
                meta_loader_iter = iter(self.meta_train_loader)
                meta_batch = next(meta_loader_iter)
            
            images = meta_batch[0].to(self.device)
            labels = meta_batch[1].to(self.device)
            
            # Forward pass and optimization
            self.model.train()
            self.optim.zero_grad()
            
            if self.cfg.prec == "amp":
                from torch.cuda.amp import autocast
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optim.step()
            
            # Update train metrics
            with torch.no_grad():
                preds = outputs.argmax(dim=1)
                acc = (preds == labels).float().mean() * 100.0
            
            meters["meta_train_loss"].update(loss.item())
            meters["meta_train_acc"].update(acc.item())
        
        # 2. Outer loop: evaluate on meta-val with class-aware objective
        self.model.eval()
        self.meta_optim.zero_grad()
        
        # Get validation batch
        try:
            val_batch = next(iter(self.meta_val_loader))
        except StopIteration:
            val_loader_iter = iter(self.meta_val_loader)
            val_batch = next(val_loader_iter)
        
        val_images = val_batch[0].to(self.device)
        val_labels = val_batch[1].to(self.device)
        
        # Compute validation loss with class-aware objective
        with torch.set_grad_enabled(True):  # Need gradients for meta-params
            if self.cfg.prec == "amp":
                from torch.cuda.amp import autocast
                with autocast():
                    val_outputs = self.model(val_images)
                    val_loss = self._compute_class_aware_loss(val_outputs, val_labels)
                self.scaler.scale(val_loss).backward()
                self.scaler.step(self.meta_optim)
                self.scaler.update()
            else:
                val_outputs = self.model(val_images)
                val_loss = self._compute_class_aware_loss(val_outputs, val_labels)
                val_loss.backward()
                self.meta_optim.step()
        
        # Update validation metrics
        with torch.no_grad():
            val_preds = val_outputs.argmax(dim=1)
            val_acc = (val_preds == val_labels).float().mean() * 100.0
            
            # Compute balanced metrics
            balanced_metrics = self._compute_balanced_metrics(val_preds, val_labels)
        
        meters["meta_val_loss"].update(val_loss.item())
        meters["meta_val_acc"].update(val_acc.item())
        
        # Log class-specific metrics
        if self.is_main_process:
            self._log_class_specific_metrics(balanced_metrics)
        
        # Restore original parameters
        self._restore_weights()
        
        # Set modules back to normal mode
        self._set_meta_mode(False)
    
    def _compute_class_aware_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute loss with class-aware weighting.
        
        Args:
            outputs: Model predictions (batch_size, num_classes)
            labels: Ground truth labels (batch_size,)
            
        Returns:
            Weighted loss value
        """
        # Base criterion loss
        base_loss = self.criterion(outputs, labels)
        
        if not self.focus_on_tail:
            return base_loss
        
        # Add tail class weighting
        # Create per-sample weights based on class group
        weights = torch.ones_like(labels, dtype=torch.float32)
        
        if self.tail_indices is not None and len(self.tail_indices) > 0:
            tail_mask = torch.isin(labels, torch.tensor(self.tail_indices, device=labels.device))
            weights[tail_mask] = self.tail_loss_weight
        
        # Compute weighted loss
        per_sample_loss = F.cross_entropy(outputs, labels, reduction='none')
        weighted_loss = (per_sample_loss * weights).mean()
        
        # Add regularization if using class-aware parameters
        reg_loss = self._compute_regularization_loss()
        
        return weighted_loss + reg_loss
    
    def _compute_regularization_loss(self) -> torch.Tensor:
        """
        Compute regularization loss for class-specific parameters.
        
        Returns:
            Regularization loss value
        """
        if self.tuner is None:
            return 0.0
        
        total_reg = 0.0
        
        # Iterate through class-aware modules
        for module in self.tuner.modules():
            if hasattr(module, 'class_rank_weights') and module.class_rank_weights is not None:
                # Smoothness regularization on rank weights
                if self.rank_divergence_penalty > 0:
                    rank_weights = module.get_effective_rank_weights()
                    # Penalize variance from mean
                    rank_var = ((rank_weights - rank_weights.mean()) ** 2).mean()
                    total_reg += self.rank_divergence_penalty * rank_var
            
            if hasattr(module, 'class_alpha_weights') and module.class_alpha_weights is not None:
                # Smoothness regularization on alpha weights
                if self.alpha_smoothness_penalty > 0:
                    alpha_weights = module.get_effective_alpha_weights()
                    # Penalize differences between adjacent classes
                    alpha_diff = (alpha_weights[1:] - alpha_weights[:-1]) ** 2
                    total_reg += self.alpha_smoothness_penalty * alpha_diff.mean()
        
        return total_reg
    
    def _compute_balanced_metrics(self, preds: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Compute balanced accuracy metrics for head/medium/tail groups.
        
        Args:
            preds: Predicted labels
            labels: Ground truth labels
            
        Returns:
            Dictionary with balanced accuracy metrics
        """
        # Overall balanced accuracy
        balanced_acc = compute_balanced_accuracy(preds, labels, self.num_classes)
        
        # Per-group accuracy
        group_acc = compute_group_accuracy(
            preds, labels,
            self.head_indices,
            self.medium_indices,
            self.tail_indices
        )
        
        # Combine metrics
        metrics = {**balanced_acc, **group_acc}
        
        return metrics
    
    def _log_class_specific_metrics(self, metrics: Dict[str, float]):
        """
        Log class-specific metrics during meta-optimization.
        
        Args:
            metrics: Dictionary of computed metrics
        """
        print(f"  Balanced Acc: {metrics.get('balanced_acc', 0):.2f}%")
        print(f"  G-Mean: {metrics.get('geometric_mean', 0):.4f}")
        print(f"  Head: {metrics.get('head_acc', 0):.2f}%, "
              f"Medium: {metrics.get('medium_acc', 0):.2f}%, "
              f"Tail: {metrics.get('tail_acc', 0):.2f}%")
        
        # Track history for visualization
        for key, value in metrics.items():
            self.metrics_history[key].append(value)
    
    def _visualize_rank_evolution(self, epoch: int):
        """
        Visualize how rank and alpha weights evolve during training.
        
        Args:
            epoch: Current epoch number
        """
        if self.tuner is None:
            return
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        
        # Collect rank and alpha weights from all class-aware modules
        all_rank_weights = []
        all_alpha_weights = []
        
        for module in self.tuner.modules():
            if hasattr(module, 'get_effective_rank_weights'):
                rank_weights = module.get_effective_rank_weights().detach().cpu().numpy()
                all_rank_weights.append(rank_weights)
            
            if hasattr(module, 'get_effective_alpha_weights'):
                alpha_weights = module.get_effective_alpha_weights().detach().cpu().numpy()
                all_alpha_weights.append(alpha_weights)
        
        if not all_rank_weights:
            return
        
        # Average across modules
        avg_rank_weights = np.mean(all_rank_weights, axis=0)
        avg_alpha_weights = np.mean(all_alpha_weights, axis=0) if all_alpha_weights else None
        
        # Create visualization
        fig, axes = plt.subplots(1, 2 if avg_alpha_weights is not None else 1, figsize=(12, 5))
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        # Plot rank weights
        colors = ['green'] * len(self.head_indices) + \
                 ['orange'] * len(self.medium_indices) + \
                 ['red'] * len(self.tail_indices)
        
        axes[0].bar(range(self.num_classes), avg_rank_weights, color=colors, alpha=0.7)
        axes[0].set_xlabel('Class Index')
        axes[0].set_ylabel('Rank Weight')
        axes[0].set_title(f'Class-Specific Rank Weights (Epoch {epoch})')
        axes[0].axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
        
        # Plot alpha weights if available
        if avg_alpha_weights is not None:
            axes[1].bar(range(self.num_classes), avg_alpha_weights, color=colors, alpha=0.7)
            axes[1].set_xlabel('Class Index')
            axes[1].set_ylabel('Alpha Weight')
            axes[1].set_title(f'Class-Specific Alpha Weights (Epoch {epoch})')
            axes[1].axhline(y=1.0, color='k', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.cfg.output_dir, f'rank_evolution_epoch_{epoch}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved rank evolution plot to {save_path}")
    
    def test(self, split="test"):
        """
        Override test to include class-aware metrics.
        """
        # Run standard test
        result = super().test(split)
        
        # Add class-aware analysis if enabled
        if self.use_class_aware and self.is_main_process:
            print("\n" + "="*80)
            print("Class-Aware Performance Analysis")
            print("="*80)
            
            # Print group statistics if available
            if self.tuner is not None:
                for name, module in self.tuner.named_modules():
                    if hasattr(module, 'get_group_statistics'):
                        stats = module.get_group_statistics()
                        print(f"\nModule: {name}")
                        for group_name, group_stats in stats.items():
                            print(f"  {group_name.capitalize()}:")
                            print(f"    Rank weight: {group_stats['mean_rank_weight']:.3f} ± {group_stats['std_rank_weight']:.3f}")
                            print(f"    Alpha weight: {group_stats['mean_alpha_weight']:.3f} ± {group_stats['std_alpha_weight']:.3f}")
        
        return result
