"""
OOD (Out-of-Distribution) data sampler and loader.

Provides infrastructure for loading and batching auxiliary OOD datasets
consistently with the main training pipeline.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from typing import Optional, Tuple, List


class OODDataset(Dataset):
    """
    Wrapper for OOD datasets.
    
    Args:
        dataset: Underlying dataset (e.g., from torchvision)
        transform: Optional transform to apply
        max_samples: Maximum number of samples to use (None = use all)
    """
    
    def __init__(
        self, 
        dataset: Dataset, 
        transform: Optional[transforms.Compose] = None,
        max_samples: Optional[int] = None
    ):
        self.dataset = dataset
        self.transform = transform
        
        # Limit number of samples if specified
        if max_samples is not None and max_samples < len(dataset):
            indices = np.random.choice(len(dataset), max_samples, replace=False)
            self.dataset = Subset(dataset, indices)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if hasattr(self.dataset, '__getitem__'):
            img, _ = self.dataset[idx]  # Ignore original label
        else:
            img = self.dataset[idx]
        
        if self.transform is not None:
            img = self.transform(img)
        
        # Return image with dummy label (-1 for OOD)
        return img, -1


class OODSampler:
    """
    Sampler for auxiliary OOD data during training.
    
    Args:
        ood_dataset: OOD dataset name or path
        data_path: Root path for data
        batch_size: Batch size for OOD data
        transform: Transform to apply to OOD images
        num_samples: Max number of OOD samples (0 = use all)
        num_workers: Number of workers for data loading
    """
    
    def __init__(
        self,
        ood_dataset: str,
        data_path: str = "./data",
        batch_size: int = 32,
        transform: Optional[transforms.Compose] = None,
        num_samples: int = 0,
        num_workers: int = 4
    ):
        self.ood_dataset_name = ood_dataset
        self.data_path = data_path
        self.batch_size = batch_size
        self.transform = transform
        self.num_samples = num_samples if num_samples > 0 else None
        self.num_workers = num_workers
        
        # Load OOD dataset
        self.dataset = self._load_ood_dataset()
        
        # Create data loader
        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        # Create iterator
        self.iterator = iter(self.loader)
    
    def _load_ood_dataset(self) -> Dataset:
        """Load the specified OOD dataset."""
        ood_name = self.ood_dataset_name.lower()
        
        # Common OOD datasets used in long-tailed + OOD literature
        if ood_name in ['tinyimages', 'tinyimages_300k', '300k_random_images']:
            # TinyImages subset (commonly used)
            return self._load_tinyimages()
        
        elif ood_name == 'places365':
            # Places365 (high-resolution scenes)
            return self._load_places365()
        
        elif ood_name == 'lsun':
            # LSUN scenes
            return self._load_lsun()
        
        elif ood_name == 'textures':
            # Describable Textures Dataset
            return self._load_textures()
        
        elif ood_name == 'svhn':
            # SVHN (Street View House Numbers)
            return self._load_svhn()
        
        elif ood_name == 'gaussian':
            # Gaussian noise (simple baseline)
            return self._create_gaussian_dataset()
        
        elif ood_name == 'uniform':
            # Uniform noise (simple baseline)
            return self._create_uniform_dataset()
        
        elif os.path.exists(self.ood_dataset_name):
            # Custom dataset from path
            return self._load_from_path(self.ood_dataset_name)
        
        else:
            raise ValueError(f"Unknown OOD dataset: {ood_name}")
    
    def _load_tinyimages(self) -> Dataset:
        """Load TinyImages subset."""
        # TinyImages is typically stored as a binary file
        # This is a placeholder - actual implementation depends on data format
        path = os.path.join(self.data_path, "tinyimages")
        if not os.path.exists(path):
            raise FileNotFoundError(f"TinyImages not found at {path}")
        
        # For now, return a simple ImageFolder
        base_dataset = datasets.ImageFolder(path, transform=None)
        return OODDataset(base_dataset, self.transform, self.num_samples)
    
    def _load_places365(self) -> Dataset:
        """Load Places365 dataset."""
        path = os.path.join(self.data_path, "places365")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Places365 not found at {path}")
        
        base_dataset = datasets.ImageFolder(path, transform=None)
        return OODDataset(base_dataset, self.transform, self.num_samples)
    
    def _load_lsun(self) -> Dataset:
        """Load LSUN dataset."""
        path = os.path.join(self.data_path, "lsun")
        if not os.path.exists(path):
            raise FileNotFoundError(f"LSUN not found at {path}")
        
        # LSUN has multiple scene categories, use 'test' split
        base_dataset = datasets.LSUN(path, classes=['test'], transform=None)
        return OODDataset(base_dataset, self.transform, self.num_samples)
    
    def _load_textures(self) -> Dataset:
        """Load Describable Textures Dataset."""
        path = os.path.join(self.data_path, "dtd")
        if not os.path.exists(path):
            raise FileNotFoundError(f"DTD not found at {path}")
        
        base_dataset = datasets.ImageFolder(path, transform=None)
        return OODDataset(base_dataset, self.transform, self.num_samples)
    
    def _load_svhn(self) -> Dataset:
        """Load SVHN dataset."""
        path = os.path.join(self.data_path, "svhn")
        base_dataset = datasets.SVHN(path, split='test', download=False, transform=None)
        return OODDataset(base_dataset, self.transform, self.num_samples)
    
    def _create_gaussian_dataset(self) -> Dataset:
        """Create synthetic Gaussian noise dataset."""
        from .synthetic_ood import GaussianNoiseDataset
        return GaussianNoiseDataset(
            num_samples=self.num_samples or 10000,
            transform=self.transform
        )
    
    def _create_uniform_dataset(self) -> Dataset:
        """Create synthetic uniform noise dataset."""
        from .synthetic_ood import UniformNoiseDataset
        return UniformNoiseDataset(
            num_samples=self.num_samples or 10000,
            transform=self.transform
        )
    
    def _load_from_path(self, path: str) -> Dataset:
        """Load OOD dataset from custom path."""
        if os.path.isdir(path):
            base_dataset = datasets.ImageFolder(path, transform=None)
        else:
            raise ValueError(f"Invalid OOD dataset path: {path}")
        
        return OODDataset(base_dataset, self.transform, self.num_samples)
    
    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get next batch of OOD data.
        
        Returns:
            Tuple of (images, labels) where labels are all -1
        """
        try:
            images, labels = next(self.iterator)
        except StopIteration:
            # Reset iterator when exhausted
            self.iterator = iter(self.loader)
            images, labels = next(self.iterator)
        
        return images, labels
    
    def get_buffer(self, buffer_size: int = 256) -> torch.Tensor:
        """
        Get a buffer of OOD images for augmentation.
        
        Args:
            buffer_size: Number of images to buffer
            
        Returns:
            Tensor of OOD images (buffer_size, C, H, W)
        """
        images_list = []
        remaining = buffer_size
        
        while remaining > 0:
            images, _ = self.get_batch()
            take = min(remaining, images.size(0))
            images_list.append(images[:take])
            remaining -= take
        
        return torch.cat(images_list, dim=0)


class SyntheticOODDataset(Dataset):
    """Base class for synthetic OOD datasets."""
    
    def __init__(self, num_samples: int, img_size: Tuple[int, int, int] = (3, 224, 224)):
        self.num_samples = num_samples
        self.img_size = img_size
    
    def __len__(self):
        return self.num_samples


class GaussianNoiseDataset(SyntheticOODDataset):
    """Synthetic OOD dataset with Gaussian noise."""
    
    def __init__(self, num_samples: int = 10000, img_size: Tuple[int, int, int] = (3, 224, 224), transform=None):
        super().__init__(num_samples, img_size)
        self.transform = transform
    
    def __getitem__(self, idx):
        # Generate Gaussian noise image
        img = torch.randn(self.img_size)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, -1


class UniformNoiseDataset(SyntheticOODDataset):
    """Synthetic OOD dataset with uniform noise."""
    
    def __init__(self, num_samples: int = 10000, img_size: Tuple[int, int, int] = (3, 224, 224), transform=None):
        super().__init__(num_samples, img_size)
        self.transform = transform
    
    def __getitem__(self, idx):
        # Generate uniform noise image
        img = torch.rand(self.img_size)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, -1
