#!/usr/bin/env python3
"""Debug training loop to identify hanging issue"""

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src import HRM, HRMTrainer, DatasetLoader

def debug_training_step():
    print("Debugging training step...")
    
    # Create minimal model
    model = HRM(
        input_dim=1,
        h_dim=16,
        l_dim=8,
        n_heads_h=1,
        n_heads_l=1,
        n_layers_h=1,
        n_layers_l=1,
        output_dim=10,
        max_len=81
    )
    
    trainer = HRMTrainer(model, learning_rate=1e-3)
    
    # Create single batch manually
    batch = {
        'inputs': torch.randint(0, 10, (2, 81, 1)).float(),
        'targets': torch.randint(1, 10, (2, 81)).long()
    }
    
    print("Running single training step...")
    try:
        metrics = trainer.train_step(batch)
        print("✅ Training step successful!")
        print(f"Metrics: {metrics}")
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()

def debug_dataloader():
    print("\nDebugging dataloader iteration...")
    
    try:
        dataloader = DatasetLoader.get_dataloader(
            dataset_type='sudoku',
            batch_size=2,
            num_samples=4,
            difficulty='extreme'
        )
        
        print("Iterating through dataloader...")
        for i, batch in enumerate(dataloader):
            print(f"Batch {i}: {batch['inputs'].shape}")
            if i >= 2:  # Only check a few batches
                break
        print("✅ Dataloader iteration successful!")
        
    except Exception as e:
        print(f"❌ Dataloader iteration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    debug_training_step()
    debug_dataloader()
