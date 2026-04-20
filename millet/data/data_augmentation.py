"""
data_augmentation.py

Simple data augmentation techniques for ECG time series.
Helps handle class imbalance by creating synthetic samples.
"""

import torch
import numpy as np
from typing import Optional, Tuple


class ECGDataAugmentation:
    """
    Simple data augmentation for ECG signals.
    
    Techniques implemented:
    1. Jittering: Add small random noise
    2. Scaling: Multiply by random factor
    3. Time Warping: Stretch/compress time axis slightly
    4. Magnitude Warping: Stretch/compress amplitude
    """
    
    def __init__(self, 
                 jitter_std: float = 0.01,
                 scale_range: Tuple[float, float] = (0.9, 1.1),
                 warp_strength: float = 0.05,
                 seed: Optional[int] = None):
        """
        Args:
            jitter_std: Standard deviation of Gaussian noise to add
            scale_range: (min, max) range for random scaling
            warp_strength: How much to warp time/magnitude (0 to 1)
            seed: Random seed for reproducibility
        """
        self.jitter_std = jitter_std
        self.scale_range = scale_range
        self.warp_strength = warp_strength
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
    
    def jitter(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add small Gaussian noise to the signal.
        
        This mimics slight measurement variations and helps the model
        become more robust to noise.
        """
        noise = torch.randn_like(x) * self.jitter_std
        return x + noise
    
    def scale(self, x: torch.Tensor) -> torch.Tensor:
        """
        Randomly scale the amplitude of the signal.
        
        This helps the model focus on shape rather than exact amplitude.
        """
        factor = np.random.uniform(*self.scale_range)
        return x * factor
    
    def time_warp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Slightly stretch or compress the time axis.
        
        This simulates natural variation in heart rate.
        """
        length = x.shape[0]
        
        # Create random warp curve
        warp_amount = self.warp_strength * length
        warp_curve = np.random.uniform(-warp_amount, warp_amount, size=3)
        warp_curve = np.convolve(warp_curve, [0.25, 0.5, 0.25], mode='same')
        
        # Create new time indices
        old_indices = np.arange(length)
        new_indices = old_indices + warp_curve[0] * np.sin(2 * np.pi * old_indices / length)
        new_indices = np.clip(new_indices, 0, length - 1)
        
        # Interpolate
        new_indices_floor = np.floor(new_indices).astype(int)
        new_indices_ceil = np.ceil(new_indices).astype(int)
        alpha = new_indices - new_indices_floor
        
        # Linear interpolation
        warped = (1 - alpha) * x[new_indices_floor] + alpha * x[new_indices_ceil]
        return torch.as_tensor(warped, dtype=x.dtype)
    
    def magnitude_warp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Slightly vary the amplitude over time.
        
        This simulates natural amplitude variations in ECG signals.
        """
        length = x.shape[0]
        
        # Create smooth random curve
        knots = np.random.uniform(0.9, 1.1, size=5)
        knots = np.convolve(knots, [1/3, 1/3, 1/3], mode='same')
        
        # Interpolate to full length
        x_old = np.linspace(0, length-1, len(knots))
        x_new = np.arange(length)
        warp_curve = np.interp(x_new, x_old, knots)
        
        # Apply warp
        warp_tensor = torch.as_tensor(warp_curve, dtype=x.dtype).unsqueeze(1)
        return x * warp_tensor
    
    def augment(self, x: torch.Tensor, 
                apply_jitter: bool = True,
                apply_scale: bool = True,
                apply_time_warp: bool = True,
                apply_magnitude_warp: bool = True) -> torch.Tensor:
        """
        Apply a random combination of augmentations.
        
        Args:
            x: Input time series (timesteps, channels)
            apply_*: Flags to control which augmentations to apply
            
        Returns:
            Augmented time series
        """
        augmented = x.clone()
        
        if apply_jitter and np.random.random() < 0.5:
            augmented = self.jitter(augmented)
        
        if apply_scale and np.random.random() < 0.5:
            augmented = self.scale(augmented)
        
        if apply_time_warp and np.random.random() < 0.3:  # Less frequent
            augmented = self.time_warp(augmented)
        
        if apply_magnitude_warp and np.random.random() < 0.3:  # Less frequent
            augmented = self.magnitude_warp(augmented)
        
        return augmented
    
    def augment_minority_classes(self, 
                                  dataset: 'ECG5000Dataset',
                                  target_count: int = 100) -> List[torch.Tensor]:
        """
        Generate synthetic samples for minority classes.
        
        Args:
            dataset: ECG5000Dataset instance
            target_count: Desired number of samples per class
            
        Returns:
            List of augmented tensors for minority classes
        """
        augmented_samples = []
        
        for class_idx in range(dataset.n_clz):
            current_count = dataset.class_counts.get(class_idx, 0)
            
            if current_count >= target_count:
                print(f"  Class {class_idx}: {current_count} samples (sufficient)")
                continue
            
            needed = target_count - current_count
            print(f"  Class {class_idx}: {current_count} → augmenting to {target_count} "
                  f"(+{needed} synthetic)")
            
            # Get original samples for this class
            orig_indices = dataset.get_samples_by_class(class_idx)
            
            for i in range(needed):
                # Pick a random original sample
                orig_idx = np.random.choice(orig_indices)
                orig_sample = dataset.get_bag(orig_idx)
                
                # Apply augmentations
                augmented = self.augment(orig_sample)
                augmented_samples.append(augmented)
        
        return augmented_samples