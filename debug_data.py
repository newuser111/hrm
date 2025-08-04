#!/usr/bin/env python3
"""Debug script to test data loading"""

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data import DatasetLoader

def test_data_loading():
    print("Testing Sudoku data loading...")
    
    dataloader = DatasetLoader.get_dataloader(
        dataset_type='sudoku',
        batch_size=2,
        num_samples=4,
        difficulty='extreme'
    )
    
    print(f"DataLoader created with {len(dataloader)} batches")
    
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"  Inputs shape: {batch['inputs'].shape}")
        print(f"  Targets shape: {batch['targets'].shape}")
        print(f"  Inputs dtype: {batch['inputs'].dtype}")
        print(f"  Targets dtype: {batch['targets'].dtype}")
        
        if i >= 1:  # Only check first 2 batches
            break
    
    print("✅ Data loading test completed")

if __name__ == '__main__':
    test_data_loading()
