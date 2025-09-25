#!/usr/bin/env python3
"""
Demo script showing epoch sampling strategy in action
"""

from epoch_sampling_strategy import EpochSamplingDataset


class DemoDataset:
    """Demo dataset with simple data"""
    def __init__(self, size=100):
        self.data = [{"instruction": f"Question {i}", "response": f"Answer {i}"} for i in range(size)]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {"sample_id": idx, **self.data[idx]}


def demonstrate_epoch_sampling():
    """Demonstrate epoch sampling strategy"""
    print("🎲 EPOCH SAMPLING STRATEGY DEMONSTRATION")
    print("=" * 60)
    
    # Create demo dataset
    base_dataset = DemoDataset(100)
    print(f"Created dataset with {len(base_dataset)} samples")
    
    # Create epoch sampling dataset with 20% sampling
    epoch_dataset = EpochSamplingDataset(
        base_dataset, 
        sampling_ratio=0.2,  # 20% per epoch
        shuffle_seed=42
    )
    
    print(f"\nWith 20% sampling ratio:")
    print(f"- Samples per epoch: {len(epoch_dataset)}")
    print(f"- Epochs needed for full coverage: {epoch_dataset.total_epochs_needed}")
    
    # Demonstrate multiple epochs
    print(f"\n{'='*60}")
    print("TRAINING SIMULATION")
    print("=" * 60)
    
    for epoch in range(6):  # Demonstrate 6 epochs
        print(f"\n📅 EPOCH {epoch + 1}")
        print("-" * 30)
        
        # Show some sample IDs from this epoch
        sample_ids = []
        for i in range(min(10, len(epoch_dataset))):  # Show first 10 samples
            sample = epoch_dataset[i]
            sample_ids.append(sample["sample_id"])
        
        print(f"Sample IDs (first 10): {sample_ids}")
        
        # Show statistics
        stats = epoch_dataset.get_sampling_stats()
        print(f"Epoch samples: {stats['samples_this_epoch']}")
        print(f"Total used: {stats['total_used_samples']}")
        print(f"Remaining: {stats['remaining_unused']}")
        print(f"Coverage: {stats['coverage_percentage']:.1f}%")
        
        # Check if all data has been used
        if stats['remaining_unused'] == 0 and epoch < 5:
            print("🔄 All data used! Will reset for next cycle...")
        
        # Prepare for next epoch
        if epoch < 5:
            epoch_dataset.new_epoch()
    
    print(f"\n{'='*60}")
    print("KEY BENEFITS:")
    print("✓ Uses only 20% of data per epoch (reduces overfitting)")
    print("✓ No sample repetition until all data is used")
    print("✓ Automatic reset when full dataset is covered")
    print("✓ Provides detailed statistics for monitoring")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_epoch_sampling()