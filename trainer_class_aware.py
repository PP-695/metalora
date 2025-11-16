"""
Class-Aware Meta-Trainer for long-tailed learning with adaptive PEFT.

This module extends MetaTrainer to optimize for balanced accuracy across
class groups (head/medium/tail) and track class-specific performance.

Gap 10 Extensions:
- COCL-style loss components (OCL, tail prototype, head debias)
- EAT-style tail augmentation with CutMix
- OOD detection evaluation
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
from utils.class_aware_losses import (
    OutlierClassLoss,
    TailPrototypeLoss,
    DebiasedHeadLoss,
    calibrate_logits
)
from datasets.tail_augmentation import (
    TailAugmentationMixer,
    get_tail_mask,
    get_head_mask
)
from datasets.ood_sampler import OODSampler
from utils.ood_eval import evaluate_ood_detection, evaluate_multiple_ood_datasets, print_ood_metrics


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
        
        # Gap 10: COCL-style loss configuration
        cocl_cfg = getattr(cfg, 'cocl', {})
        self.use_ocl = cocl_cfg.get('use_ocl', False)
        self.use_tail_proto = cocl_cfg.get('use_tail_proto', False)
        self.use_head_debias = cocl_cfg.get('use_head_debias', False)
        self.lambda_ocl = cocl_cfg.get('lambda_ocl', 0.5)
        self.lambda_tail_proto = cocl_cfg.get('lambda_tail_proto', 0.3)
        self.lambda_head_debias = cocl_cfg.get('lambda_head_debias', 0.1)
        self.use_logit_calibration = cocl_cfg.get('use_logit_calibration', False)
        self.tau_calibrate = cocl_cfg.get('tau_calibrate', 1.0)
        
        # Gap 10: Tail augmentation configuration
        tail_aug_cfg = getattr(cfg, 'tail_augmentation', {})
        self.use_tail_cutmix = tail_aug_cfg.get('use_tail_cutmix', False)
        self.tail_cutmix_alpha = tail_aug_cfg.get('tail_cutmix_alpha', 0.9999)
        self.tail_cutmix_prob = tail_aug_cfg.get('tail_cutmix_prob', 0.5)
        self.use_ood_paste = tail_aug_cfg.get('use_ood_paste', False)
        
        # Gap 10: OOD configuration
        ood_cfg = getattr(cfg, 'ood', {})
        self.use_ood = ood_cfg.get('use_ood', False)
        self.ood_dataset = ood_cfg.get('ood_dataset', '')
        self.ood_data_path = ood_cfg.get('ood_data_path', './data')
        
        # Gap 10: OOD eval configuration
        ood_eval_cfg = getattr(cfg, 'ood_eval', {})
        self.enable_ood_eval = ood_eval_cfg.get('enable', False)
        self.ood_test_datasets = ood_eval_cfg.get('ood_test_datasets', [])
        self.ood_metric = ood_eval_cfg.get('ood_metric', 'msp')
        
        # Initialize parent
        super().__init__(cfg, device)
        
        # Class grouping information (initialized in _analyze_class_distribution)
        self.head_indices = None
        self.medium_indices = None
        self.tail_indices = None
        
        # Gap 10: Additional components (initialized later)
        self.ocl_loss = None
        self.tail_proto_loss = None
        self.head_debias_loss = None
        self.tail_augmenter = None
        self.ood_sampler = None
        self.ood_test_loaders = {}
        
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
            self._initialize_gap10_components()
    
    def _initialize_gap10_components(self):
        """Initialize Gap 10 components (COCL losses, tail augmentation, OOD)."""
        cls_num_list = np.array(self.cls_num_list)
        
        # 1. Initialize COCL-style losses
        if self.use_ocl:
            print("Initializing Outlier Class Learning loss...")
            # OCL requires num_classes + 1 (last index for outlier class)
            self.ocl_loss = OutlierClassLoss(
                num_id_classes=self.num_classes,
                weight=getattr(self.cfg, 'cocl', {}).get('ocl_weight', 1.0)
            ).to(self.device)
        
        if self.use_tail_proto:
            print("Initializing Tail Prototype Learning loss...")
            cocl_cfg = getattr(self.cfg, 'cocl', {})
            self.tail_proto_loss = TailPrototypeLoss(
                temperature=cocl_cfg.get('tail_proto_temperature', 0.07),
                margin=cocl_cfg.get('tail_proto_margin', 0.1)
            ).to(self.device)
        
        if self.use_head_debias:
            print("Initializing Debiased Head loss...")
            head_threshold = getattr(self.cfg, 'head_threshold', 100)
            self.head_debias_loss = DebiasedHeadLoss(
                cls_num_list=cls_num_list,
                head_threshold=head_threshold,
                penalty_weight=getattr(self.cfg, 'cocl', {}).get('head_debias_penalty', 0.1)
            ).to(self.device)
        
        # 2. Initialize tail augmentation
        if self.use_tail_cutmix:
            print("Initializing tail CutMix augmentation...")
            self.tail_augmenter = TailAugmentationMixer(
                alpha=self.tail_cutmix_alpha,
                prob=self.tail_cutmix_prob,
                use_ood=self.use_ood_paste
            )
        
        # 3. Initialize OOD sampler
        if self.use_ood and self.ood_dataset:
            print(f"Initializing OOD sampler with dataset: {self.ood_dataset}...")
            try:
                # Get transform from train loader if available
                transform = None
                if hasattr(self, 'train_loader') and hasattr(self.train_loader.dataset, 'transform'):
                    transform = self.train_loader.dataset.transform
                
                ood_cfg = getattr(self.cfg, 'ood', {})
                self.ood_sampler = OODSampler(
                    ood_dataset=self.ood_dataset,
                    data_path=self.ood_data_path,
                    batch_size=ood_cfg.get('ood_batch_size', 32),
                    transform=transform,
                    num_samples=ood_cfg.get('ood_num_samples', 0),
                    num_workers=4
                )
                
                # Pre-fill OOD buffer for augmentation if needed
                if self.tail_augmenter is not None and self.use_ood_paste:
                    print("Filling OOD buffer for tail augmentation...")
                    ood_buffer = self.ood_sampler.get_buffer(buffer_size=256)
                    self.tail_augmenter.set_ood_buffer(ood_buffer.to(self.device))
                    
            except Exception as e:
                print(f"Warning: Failed to initialize OOD sampler: {e}")
                print("Continuing without OOD data...")
                self.use_ood = False
                self.ood_sampler = None
        
        # 4. Initialize OOD evaluation datasets
        if self.enable_ood_eval and len(self.ood_test_datasets) > 0:
            print("Initializing OOD evaluation datasets...")
            self._initialize_ood_eval_loaders()
    
    def _initialize_ood_eval_loaders(self):
        """Initialize data loaders for OOD evaluation datasets."""
        from torch.utils.data import DataLoader
        
        # Get transform from test loader
        transform = None
        if hasattr(self, 'test_loader') and hasattr(self.test_loader.dataset, 'transform'):
            transform = self.test_loader.dataset.transform
        
        for ood_name in self.ood_test_datasets:
            try:
                print(f"  Loading OOD test dataset: {ood_name}")
                sampler = OODSampler(
                    ood_dataset=ood_name,
                    data_path=self.ood_data_path,
                    batch_size=self.cfg.test_batch_size if hasattr(self.cfg, 'test_batch_size') else 64,
                    transform=transform,
                    num_samples=0,  # Use all samples for eval
                    num_workers=4
                )
                self.ood_test_loaders[ood_name] = sampler.loader
            except Exception as e:
                print(f"  Warning: Failed to load OOD dataset {ood_name}: {e}")
    
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
        Compute loss with class-aware weighting and Gap 10 components.
        
        Args:
            outputs: Model predictions (batch_size, num_classes)
            labels: Ground truth labels (batch_size,)
            
        Returns:
            Weighted loss value
        """
        # Base criterion loss
        base_loss = self.criterion(outputs, labels)
        
        if not self.focus_on_tail:
            total_loss = base_loss
        else:
            # Add tail class weighting
            # Create per-sample weights based on class group
            weights = torch.ones_like(labels, dtype=torch.float32)
            
            if self.tail_indices is not None and len(self.tail_indices) > 0:
                tail_mask = torch.isin(labels, torch.tensor(self.tail_indices, device=labels.device))
                weights[tail_mask] = self.tail_loss_weight
            
            # Compute weighted loss
            per_sample_loss = F.cross_entropy(outputs, labels, reduction='none')
            weighted_loss = (per_sample_loss * weights).mean()
            total_loss = weighted_loss
        
        # Add Gap 10: COCL-style losses
        gap10_loss = self._compute_gap10_losses(outputs, labels)
        total_loss = total_loss + gap10_loss
        
        # Add regularization if using class-aware parameters
        reg_loss = self._compute_regularization_loss()
        
        return total_loss + reg_loss
    
    def _compute_gap10_losses(
        self, 
        outputs: torch.Tensor, 
        labels: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        ood_images: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Gap 10 loss components (OCL, tail prototype, head debias).
        
        Args:
            outputs: Model predictions (batch_size, num_classes)
            labels: Ground truth labels (batch_size,)
            features: Optional feature embeddings (batch_size, feature_dim)
            ood_images: Optional OOD images for OCL
            
        Returns:
            Combined Gap 10 loss
        """
        gap10_loss = 0.0
        
        # 1. Head debiasing loss (always applied to ID data)
        if self.use_head_debias and self.head_debias_loss is not None:
            head_debias = self.head_debias_loss(outputs, labels)
            gap10_loss += self.lambda_head_debias * head_debias
        
        # 2. Tail prototype loss (requires features)
        if self.use_tail_proto and self.tail_proto_loss is not None and features is not None:
            # Get tail samples
            if self.tail_indices is not None and len(self.tail_indices) > 0:
                tail_mask = torch.isin(labels, torch.tensor(self.tail_indices, device=labels.device))
                if tail_mask.sum() > 0:
                    tail_features = features[tail_mask]
                    tail_labels = labels[tail_mask]
                    
                    # Get OOD features if available
                    if ood_images is not None and self.ood_sampler is not None:
                        # Extract OOD features
                        with torch.no_grad():
                            ood_features = self._extract_features(ood_images)
                        
                        # Compute tail prototype loss
                        tail_proto = self.tail_proto_loss(tail_features, ood_features, tail_labels)
                        gap10_loss += self.lambda_tail_proto * tail_proto
        
        # 3. OCL loss (requires OOD data and model with outlier class)
        if self.use_ocl and self.ocl_loss is not None and ood_images is not None:
            # Forward pass on OOD data with outlier class logit
            ood_outputs = self.model(ood_images)
            
            # Create labels for OOD (set to outlier class index)
            ood_labels = torch.full((ood_images.size(0),), 
                                   self.num_classes,  # Outlier class index
                                   dtype=torch.long,
                                   device=ood_images.device)
            
            # Compute OCL loss
            ocl = self.ocl_loss(ood_outputs, ood_labels, is_ood=None)
            gap10_loss += self.lambda_ocl * ocl
        
        return gap10_loss
    
    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract feature embeddings from images.
        
        Args:
            images: Input images (batch_size, C, H, W)
            
        Returns:
            Feature embeddings (batch_size, feature_dim)
        """
        # This assumes model has a feature extraction method
        # Adapt based on actual model architecture
        if hasattr(self.model, 'extract_features'):
            return self.model.extract_features(images)
        elif hasattr(self.model, 'encode_image'):
            # For CLIP-like models
            return self.model.encode_image(images)
        else:
            # Fallback: use penultimate layer
            # This may need to be customized based on model architecture
            return self.model(images)  # Return logits as features for now
    
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
        Override test to include class-aware metrics and OOD evaluation.
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
        
        # Gap 10: OOD Detection Evaluation
        if self.enable_ood_eval and len(self.ood_test_loaders) > 0 and self.is_main_process:
            print("\n" + "="*80)
            print("OOD Detection Evaluation")
            print("="*80)
            
            try:
                # Get ID test loader
                id_loader = self.test_loader if hasattr(self, 'test_loader') else None
                
                if id_loader is not None:
                    # Evaluate on each OOD dataset
                    ood_results = evaluate_multiple_ood_datasets(
                        model=self.model,
                        id_loader=id_loader,
                        ood_loaders=self.ood_test_loaders,
                        device=self.device,
                        score_type=self.ood_metric,
                        temperature=1.0
                    )
                    
                    # Print results
                    for ood_name, metrics in ood_results.items():
                        print_ood_metrics(metrics, ood_name)
                    
                    # Save results
                    if hasattr(self.cfg, 'output_dir'):
                        import json
                        results_path = os.path.join(self.cfg.output_dir, 'ood_detection_results.json')
                        with open(results_path, 'w') as f:
                            json.dump(ood_results, f, indent=2)
                        print(f"OOD detection results saved to {results_path}")
                else:
                    print("Warning: No test loader available for OOD evaluation")
                    
            except Exception as e:
                print(f"Warning: OOD evaluation failed: {e}")
                import traceback
                traceback.print_exc()
        
        return result
    
    def _apply_logit_calibration(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply calibrated logit adjustment at inference time.
        
        Args:
            logits: Model predictions (batch_size, num_classes)
            
        Returns:
            Calibrated logits
        """
        if not self.use_logit_calibration or self.tau_calibrate == 0.0:
            return logits
        
        # Compute class priors from training distribution
        cls_num_list = np.array(self.cls_num_list)
        class_priors = cls_num_list / cls_num_list.sum()
        
        # Apply calibration
        return calibrate_logits(logits, class_priors, self.tau_calibrate)
