"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0

ECG5000 Dataset class with improvements for handling imbalanced data.
"""

import json
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
from overrides import override
from collections import Counter

from millet.data.mil_tsc_dataset import MILTSCDataset


class ECG5000Dataset(MILTSCDataset):
    """
    MIL TSC Dataset implementation for the ECG5000 dataset.
    
    IMPROVEMENTS MADE:
    - Added class distribution tracking for debugging imbalance
    - Better error handling for file parsing
    - Optional class weighting for loss function
    - Support for getting samples by class (useful for balanced sampling)
    """
    
    # Class names for ECG5000 (for better readability)
    CLASS_NAMES = ["Normal", "R-on-T PVC", "PVC", "SP", "UB"]
    
    def __init__(self, split: str, name: str = "ECG5000", apply_transform: bool = True):
        """
        Args:
            split: "TRAIN" or "TEST"
            name: Dataset name
            apply_transform: Whether to apply data augmentation/transforms
        """
        super().__init__(name, split, apply_transform=apply_transform)
        
        # Store class distribution for debugging and weighted sampling
        self.class_counts = self._compute_class_distribution()
        self.class_weights = self._compute_class_weights()
        
        # Print distribution info (helpful for debugging imbalance)
        self._print_distribution_info()
        
        self._metadata = None
    
    def _compute_class_distribution(self) -> Dict[int, int]:
        """Count samples per class - useful for identifying imbalance."""
        targets = self.targets.numpy()
        return dict(Counter(targets))
    
    def _compute_class_weights(self) -> torch.Tensor:
        """
        Compute inverse frequency weights for each class.
        Used for weighted loss function to handle imbalance.
        
        Returns:
            Tensor of shape (n_classes,) with weight for each class.
            Minority classes get higher weights.
        """
        n_samples = len(self.targets)
        n_classes = self.n_clz
        
        weights = torch.zeros(n_classes)
        for c in range(n_classes):
            count = self.class_counts.get(c, 1)  # Avoid division by zero
            weights[c] = n_samples / (n_classes * count)
        
        return weights
    
    def _print_distribution_info(self):
        """Print class distribution for debugging."""
        print(f"\n{'='*50}")
        print(f"ECG5000 {self.split} Split - Class Distribution")
        print(f"{'='*50}")
        print(f"Total samples: {len(self.targets)}")
        print(f"Classes: {self.n_clz}")
        print("-" * 40)
        
        for c in range(self.n_clz):
            count = self.class_counts.get(c, 0)
            pct = 100 * count / len(self.targets)
            weight = self.class_weights[c].item()
            bar_length = int(pct / 2)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"Class {c} ({self.CLASS_NAMES[c]:12s}): {count:4d} ({pct:5.1f}%) "
                  f"weight={weight:.3f} |{bar}|")
        print(f"{'='*50}\n")
    
    def get_class_weights(self) -> torch.Tensor:
        """Return class weights for weighted loss function."""
        return self.class_weights
    
    def get_samples_by_class(self, class_idx: int, n_samples: Optional[int] = None) -> List[int]:
        """
        Get indices of samples belonging to a specific class.
        Useful for balanced sampling during training.
        
        Args:
            class_idx: Target class index (0-4)
            n_samples: Number of samples to return (None = all)
            
        Returns:
            List of sample indices
        """
        targets = self.targets.numpy()
        indices = np.where(targets == class_idx)[0].tolist()
        
        if n_samples is not None and n_samples < len(indices):
            indices = np.random.choice(indices, n_samples, replace=False).tolist()
        
        return indices
    
    def get_balanced_batch_indices(self, batch_size: int) -> List[int]:
        """
        Generate indices for a balanced batch (equal samples per class).
        Handles minority classes by oversampling.
        
        Args:
            batch_size: Desired batch size
            
        Returns:
            List of sample indices for a balanced batch
        """
        samples_per_class = max(1, batch_size // self.n_clz)
        indices = []
        
        for c in range(self.n_clz):
            class_indices = self.get_samples_by_class(c)
            if len(class_indices) == 0:
                continue
            # Oversample minority classes
            sampled = np.random.choice(class_indices, samples_per_class, replace=True).tolist()
            indices.extend(sampled)
        
        np.random.shuffle(indices)
        return indices[:batch_size]

    def get_time_series_collection_and_targets(self, split: str) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Load time series from ECG5000 TS file.
        
        ECG5000 file format:
        - Header lines starting with @ (metadata)
        - @data marker indicates start of data
        - Each data row: "v1,v2,...,v140:label"
        - Labels are 1-indexed (1-5), we convert to 0-indexed (0-4)
        """
        file_path = f"data/ECG5000/ECG5000_{split}.ts"
        
        data_rows = []
        data_started = False
        
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Check for @data marker
                if line.lower() == '@data':
                    data_started = True
                    continue
                
                # Skip header lines
                if not data_started:
                    continue
                
                # Skip comment lines
                if line.startswith('#'):
                    continue
                
                # Parse data line
                try:
                    if ':' in line:
                        # Format: "values:label"
                        series_part, label_part = line.rsplit(':', 1)
                        timesteps = [float(x) for x in series_part.split(',')]
                        label = int(float(label_part))
                        data_rows.append((timesteps, label))
                    else:
                        # Fallback: first value is label
                        row_values = [float(x) for x in line.split(',')]
                        label = int(row_values[0])
                        timesteps = row_values[1:]
                        data_rows.append((timesteps, label))
                except ValueError as e:
                    print(f"⚠️ Warning: Line {line_num} parse error: {e}")
                    continue
        
        print(f"📂 Loaded {len(data_rows)} rows from {file_path}")
        
        ts_collection = []
        targets = []
        
        for timesteps, label in data_rows:
            # Convert from 1-indexed to 0-indexed labels
            target = label - 1
            
            # Validate label range
            if target < 0 or target >= 5:
                print(f"⚠️ Warning: Invalid label {label} (should be 1-5)")
                continue
            
            # Create tensor with shape (timesteps, channels)
            ts_tensor = torch.as_tensor(timesteps, dtype=torch.float)
            ts_tensor = ts_tensor.unsqueeze(1)  # Add channel dimension
            
            ts_collection.append(ts_tensor)
            targets.append(target)
        
        if len(targets) == 0:
            raise ValueError(f"❌ No valid samples found in {file_path}")
        
        targets_tensor = torch.as_tensor(targets, dtype=torch.int)
        print(f"✅ Processed {len(ts_collection)} valid samples")
        
        return ts_collection, targets_tensor

    @override
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
            - 'bag': time series tensor (timesteps, channels)
            - 'target': class label (0-4)
        """
        bag = self.get_bag(idx)
        
        if self.apply_transform:
            bag = self.apply_bag_transform(bag)
        
        target = self.targets[idx]
        
        return {
            "bag": bag,
            "target": target,
        }