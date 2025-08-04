#!/usr/bin/env python3
"""Simple model test"""

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.model import HRM

def simple_test():
    print("Testing HRM model...")
    
    # Create small model
    model = HRM(
        input_dim=1,
        h_dim=16,
        l_dim=8,
        n_heads_h=1,
        n_heads_l=1,
        n_layers_h=1,
        n_layers_l=1,
        output_dim=10,
        max_len=9  # Small for testing
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    inputs = torch.randint(0, 10, (1, 9, 1)).float()  # [batch, seq, features]
    print(f"Input shape: {inputs.shape}")
    
    with torch.no_grad():
        result = model(inputs, max_steps=3)
    
    print(f"Output shape: {result['output'].shape}")
    print(f"Number of computation steps: {result['num_steps']}")
    print("✅ Simple test passed!")

if __name__ == '__main__':
    simple_test()
