import random
import math
from typing import List, Dict, Set
import numpy as np
from torch.utils.data import Dataset, Sampler


class EpochSamplingDataset(Dataset):
    """Dataset wrapper that implements epoch-wise sampling strategy"""
    
    def __init__(self, base_dataset, sampling_ratio=0.1, shuffle_seed=None):
        """
        Args:
            base_dataset: Original dataset
            sampling_ratio: Fraction of data to use per epoch (default 0.1 = 10%)
            shuffle_seed: Seed for reproducible shuffling
        """
        self.base_dataset = base_dataset
        self.sampling_ratio = sampling_ratio
        self.total_samples = len(base_dataset)
        self.samples_per_epoch = int(self.total_samples * sampling_ratio)
        
        # Calculate total epochs needed to cover all data
        self.total_epochs_needed = math.ceil(1.0 / sampling_ratio)
        
        # Initialize sampling state
        self.current_epoch = 0
        self.used_indices = set()
        self.available_indices = list(range(self.total_samples))
        
        # Set seed for reproducibility
        if shuffle_seed is not None:
            random.seed(shuffle_seed)
            np.random.seed(shuffle_seed)
        
        # Shuffle initial indices
        random.shuffle(self.available_indices)
        
        # Get current epoch indices
        self.current_epoch_indices = self._get_epoch_indices()
        
        print(f"EpochSamplingDataset initialized:")
        print(f"  Total samples: {self.total_samples}")
        print(f"  Samples per epoch: {self.samples_per_epoch}")
        print(f"  Sampling ratio: {sampling_ratio*100:.1f}%")
        print(f"  Epochs to cover all data: {self.total_epochs_needed}")
    
    def _get_epoch_indices(self):
        """Get indices for current epoch"""
        # If we've used all data, reset
        if len(self.available_indices) == 0:
            self.used_indices.clear()
            self.available_indices = list(range(self.total_samples))
            random.shuffle(self.available_indices)
            print(f"Data cycle completed. Reset for new cycle.")
        
        # Sample indices for this epoch
        num_needed = min(self.samples_per_epoch, len(self.available_indices))
        epoch_indices = self.available_indices[:num_needed]
        
        # Remove used indices from available pool
        self.available_indices = self.available_indices[num_needed:]
        self.used_indices.update(epoch_indices)
        
        print(f"Epoch {self.current_epoch}: Using {len(epoch_indices)} samples")
        print(f"  Remaining unused samples: {len(self.available_indices)}")
        print(f"  Total used so far: {len(self.used_indices)}")
        
        return epoch_indices
    
    def new_epoch(self):
        """Call this at the start of each new epoch"""
        self.current_epoch += 1
        self.current_epoch_indices = self._get_epoch_indices()
        print(f"Started epoch {self.current_epoch}")
    
    def __len__(self):
        return len(self.current_epoch_indices)
    
    def __getitem__(self, idx):
        if idx >= len(self.current_epoch_indices):
            raise IndexError(f"Index {idx} out of range for epoch dataset size {len(self.current_epoch_indices)}")
        
        # Map epoch index to actual dataset index
        actual_idx = self.current_epoch_indices[idx]
        return self.base_dataset[actual_idx]
    
    def get_sampling_stats(self):
        """Get current sampling statistics"""
        return {
            "current_epoch": self.current_epoch,
            "samples_this_epoch": len(self.current_epoch_indices),
            "total_used_samples": len(self.used_indices),
            "remaining_unused": len(self.available_indices),
            "coverage_percentage": (len(self.used_indices) / self.total_samples) * 100
        }


def train_with_epoch_sampling(trainer, epoch_dataset, original_epochs):
    """
    Training function that handles epoch transitions for sampling
    
    Args:
        trainer: HuggingFace Trainer instance
        epoch_dataset: EpochSamplingDataset instance
        original_epochs: Original number of epochs requested
    """
    print("Starting training with epoch sampling strategy...")
    
    # Calculate effective epochs (may need more to cover all data)
    total_epochs_for_full_coverage = epoch_dataset.total_epochs_needed
    effective_epochs = max(original_epochs, total_epochs_for_full_coverage)
    
    print(f"Training plan:")
    print(f"  Requested epochs: {original_epochs}")
    print(f"  Epochs for full data coverage: {total_epochs_for_full_coverage}")
    print(f"  Will train for: {effective_epochs} epochs")
    
    # Store original training arguments
    original_train_dataset = trainer.train_dataset
    original_num_epochs = trainer.args.num_train_epochs
    
    # Set to train one epoch at a time
    trainer.args.num_train_epochs = 1
    
    try:
        # Custom training loop
        for epoch in range(effective_epochs):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch + 1}/{effective_epochs}")
            print(f"{'='*60}")
            
            if epoch > 0:
                epoch_dataset.new_epoch()
            
            # Update trainer's dataset
            trainer.train_dataset = epoch_dataset
            
            # Print sampling stats
            stats = epoch_dataset.get_sampling_stats()
            print(f"Sampling stats:")
            print(f"  Current epoch: {stats['current_epoch']}")
            print(f"  Samples this epoch: {stats['samples_this_epoch']}")
            print(f"  Total used samples: {stats['total_used_samples']}")
            print(f"  Remaining unused: {stats['remaining_unused']}")
            print(f"  Coverage: {stats['coverage_percentage']:.1f}%")
            
            # Run single epoch training
            result = trainer.train()
            
            # Print epoch completion stats
            print(f"\nEpoch {epoch + 1} completed:")
            _print_epoch_stats(stats)
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETED WITH EPOCH SAMPLING")
        print(f"{'='*60}")
        final_stats = epoch_dataset.get_sampling_stats()
        print(f"Final sampling stats:")
        print(f"  Total epochs: {effective_epochs}")
        print(f"  Final coverage: {final_stats['coverage_percentage']:.1f}%")
        print(f"  Total samples processed: {final_stats['total_used_samples']}")
        
        return result
        
    finally:
        # Restore original trainer state
        trainer.train_dataset = original_train_dataset
        trainer.args.num_train_epochs = original_num_epochs


def _print_epoch_stats(sampling_stats):
    """Print statistics after each epoch"""
    # Memory stats if available
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"  GPU Memory: {allocated:.2f}GB")
    except:
        pass
    
    # Sampling coverage
    print(f"  Data coverage: {sampling_stats['coverage_percentage']:.1f}%")
    print(f"  Samples used this epoch: {sampling_stats['samples_this_epoch']}")