"""
Tail-focused augmentation for long-tailed learning.

Implements EAT-style CutMix augmentation that helps tail classes by
mixing them with head or OOD samples.
"""

import numpy as np
import torch
from typing import Tuple, Optional


def rand_bbox(size: Tuple[int, int, int, int], lam: float) -> Tuple[int, int, int, int]:
    """
    Generate random bounding box for CutMix.
    
    Args:
        size: Image size (B, C, H, W)
        lam: Lambda value from Beta distribution
        
    Returns:
        Bounding box coordinates (bbx1, bby1, bbx2, bby2)
    """
    W = size[2]
    H = size[3]
    
    # Compute cut ratio
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # Uniform random center point
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    # Bounding box
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2


def augment_tail_with_cutmix(
    tail_images: torch.Tensor,
    tail_labels: torch.Tensor,
    background_images: torch.Tensor,
    background_labels: Optional[torch.Tensor] = None,
    alpha: float = 0.9999,
    is_ood_background: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Apply CutMix augmentation focusing on tail classes.
    
    Pastes tail class patches into background images (head or OOD).
    With high alpha, the tail region is preserved (small background patch).
    
    Args:
        tail_images: Images from tail classes (N_tail, C, H, W)
        tail_labels: Labels for tail images (N_tail,)
        background_images: Background images to paste into (N_bg, C, H, W)
        background_labels: Labels for background images (N_bg,) - None for OOD
        alpha: Beta distribution parameter (high value = preserve more tail)
        is_ood_background: Whether background is OOD (affects labeling)
        
    Returns:
        mixed_images: Augmented images (N_tail, C, H, W)
        mixed_labels: Labels (N_tail,) - always tail labels
        lam: Lambda value used for mixing
    """
    batch_size = tail_images.size(0)
    
    # Sample lambda from Beta distribution
    # High alpha (e.g., 0.9999) means lam is very close to 1.0
    # This preserves most of the tail image
    lam = np.random.beta(alpha, alpha)
    
    # Randomly select background images
    bg_indices = torch.randint(0, background_images.size(0), (batch_size,))
    bg_images = background_images[bg_indices]
    
    # Generate random bounding boxes
    bbx1, bby1, bbx2, bby2 = rand_bbox(tail_images.size(), lam)
    
    # Create mixed images
    mixed_images = tail_images.clone()
    # Paste background patch into tail images
    mixed_images[:, :, bbx1:bbx2, bby1:bby2] = bg_images[:, :, bbx1:bbx2, bby1:bby2]
    
    # For tail-focused augmentation, we always keep tail labels
    # since we preserve most of the tail image
    mixed_labels = tail_labels
    
    return mixed_images, mixed_labels, lam


class TailAugmentationMixer:
    """
    Class to handle tail augmentation during training.
    
    Args:
        alpha: Beta distribution parameter
        prob: Probability of applying augmentation
        use_ood: Whether to mix with OOD samples
    """
    
    def __init__(self, alpha: float = 0.9999, prob: float = 0.5, use_ood: bool = False):
        self.alpha = alpha
        self.prob = prob
        self.use_ood = use_ood
        self.head_buffer = None
        self.ood_buffer = None
    
    def set_head_buffer(self, images: torch.Tensor, labels: torch.Tensor):
        """Store head class samples for mixing."""
        self.head_buffer = (images, labels)
    
    def set_ood_buffer(self, images: torch.Tensor):
        """Store OOD samples for mixing."""
        self.ood_buffer = images
    
    def __call__(
        self, 
        images: torch.Tensor, 
        labels: torch.Tensor, 
        is_tail: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply tail augmentation.
        
        Args:
            images: Batch of images (B, C, H, W)
            labels: Batch of labels (B,)
            is_tail: Boolean mask for tail samples (B,)
            
        Returns:
            Augmented images and labels
        """
        # Check if we should apply augmentation
        if np.random.random() > self.prob:
            return images, labels
        
        # Check if we have tail samples
        if not is_tail.any():
            return images, labels
        
        # Select background source
        if self.use_ood and self.ood_buffer is not None:
            background_images = self.ood_buffer
            is_ood_bg = True
        elif self.head_buffer is not None:
            background_images, _ = self.head_buffer
            is_ood_bg = False
        else:
            # No buffer available, skip augmentation
            return images, labels
        
        # Extract tail samples
        tail_images = images[is_tail]
        tail_labels = labels[is_tail]
        
        # Apply CutMix
        mixed_tail, mixed_labels, _ = augment_tail_with_cutmix(
            tail_images,
            tail_labels,
            background_images,
            alpha=self.alpha,
            is_ood_background=is_ood_bg
        )
        
        # Replace tail samples with augmented versions
        images_out = images.clone()
        images_out[is_tail] = mixed_tail
        labels_out = labels.clone()
        labels_out[is_tail] = mixed_labels
        
        return images_out, labels_out


def get_tail_mask(labels: torch.Tensor, tail_class_indices: list) -> torch.Tensor:
    """
    Get boolean mask for tail class samples.
    
    Args:
        labels: Batch labels (B,)
        tail_class_indices: List of tail class indices
        
    Returns:
        Boolean mask (B,)
    """
    is_tail = torch.zeros(labels.size(0), dtype=torch.bool, device=labels.device)
    for tail_idx in tail_class_indices:
        is_tail |= (labels == tail_idx)
    return is_tail


def get_head_mask(labels: torch.Tensor, head_class_indices: list) -> torch.Tensor:
    """
    Get boolean mask for head class samples.
    
    Args:
        labels: Batch labels (B,)
        head_class_indices: List of head class indices
        
    Returns:
        Boolean mask (B,)
    """
    is_head = torch.zeros(labels.size(0), dtype=torch.bool, device=labels.device)
    for head_idx in head_class_indices:
        is_head |= (labels == head_idx)
    return is_head
