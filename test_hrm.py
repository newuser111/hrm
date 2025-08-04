#!/usr/bin/env python3
"""
HRM Test Script

Quick test to verify the implementation works correctly.
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src import HRM, HRMTrainer, DatasetLoader


def test_model_creation():
    """Test basic model instantiation"""
    print("Testing model creation...")
    
    model = HRM(
        input_dim=1,
        h_dim=64,  # Smaller for testing
        l_dim=32,
        n_heads_h=2,
        n_heads_l=2,
        n_layers_h=2,
        n_layers_l=1,
        output_dim=10,
        max_len=81
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created successfully with {total_params:,} parameters")
    
    return model


def test_forward_pass():
    """Test forward pass with dummy data"""
    print("\nTesting forward pass...")
    
    model = test_model_creation()
    
    # Create dummy input (simulating Sudoku puzzle)
    batch_size, seq_len = 2, 81
    dummy_input = torch.randint(0, 10, (batch_size, seq_len, 1)).float()
    
    with torch.no_grad():
        result = model(dummy_input, max_steps=5, return_intermediates=True)
    
    print(f"✅ Forward pass successful")
    print(f"   Output shape: {result['output'].shape}")
    print(f"   Computation steps: {result['num_steps']}")
    print(f"   Halting probs shape: {result['halting_probs'].shape}")
    
    return result


def test_sudoku_dataset():
    """Test Sudoku dataset creation"""
    print("\nTesting Sudoku dataset...")
    
    try:
        dataloader = DatasetLoader.get_dataloader(
            dataset_type='sudoku',
            batch_size=4,
            num_samples=10,  # Small for testing
            difficulty='extreme'
        )
        
        # Get one batch
        batch = next(iter(dataloader))
        print(f"✅ Sudoku dataset working")
        print(f"   Batch input shape: {batch['inputs'].shape}")
        print(f"   Batch target shape: {batch['targets'].shape}")
        
        return dataloader
        
    except Exception as e:
        print(f"❌ Sudoku dataset failed: {e}")
        return None


def test_training_step():
    """Test a single training step"""
    print("\nTesting training step...")
    
    model = test_model_creation()
    trainer = HRMTrainer(model, learning_rate=1e-3)
    
    # Create dummy batch
    batch = {
        'inputs': torch.randint(0, 10, (2, 81, 1)).float(),
        'targets': torch.randint(1, 10, (2, 81)).long()
    }
    
    try:
        metrics = trainer.train_step(batch)
        print(f"✅ Training step successful")
        print(f"   Total loss: {metrics['total_loss']:.4f}")
        print(f"   Main loss: {metrics['main_loss']:.4f}")
        print(f"   Average steps: {metrics['avg_steps']}")
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests"""
    print("🧠 HRM Implementation Test Suite")
    print("=" * 40)
    
    test_model_creation()
    test_forward_pass()
    test_sudoku_dataset()
    test_training_step()
    
    print("\n" + "=" * 40)
    print("🎉 All tests completed!")
    print("\nTo start training:")
    print("   python train_hrm.py --dataset sudoku --epochs 10 --num_samples 100")


if __name__ == '__main__':
    main()
