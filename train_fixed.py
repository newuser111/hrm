#!/usr/bin/env python3
"""
Fixed Training Script - No hanging issues
"""

import argparse
import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from src.model import HRM
from src.training_fixed import HRMTrainerFixed
from src.data import DatasetLoader


def main():
    parser = argparse.ArgumentParser(description='Train HRM (fixed version)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--h_dim', type=int, default=64, help='High-level dimension')
    parser.add_argument('--l_dim', type=int, default=32, help='Low-level dimension')
    
    args = parser.parse_args()
    
    print("🧠 HRM Training (Fixed Version)")
    print("=" * 40)
    
    # Create model
    model = HRM(
        input_dim=1,
        h_dim=args.h_dim,
        l_dim=args.l_dim,
        n_heads_h=2,
        n_heads_l=2,
        n_layers_h=2,
        n_layers_l=1,
        output_dim=10,
        max_len=81
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataloader
    train_loader = DatasetLoader.get_dataloader(
        dataset_type='sudoku',
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        difficulty='extreme'
    )
    
    print(f"Training on {len(train_loader)} batches")
    
    # Create trainer
    trainer = HRMTrainerFixed(model)
    
    # Training loop
    for epoch in range(args.epochs):
        train_metrics = trainer.train_epoch_simple(train_loader, epoch)
        eval_metrics = trainer.evaluate_simple(train_loader, max_batches=3)
        
        print(f"Epoch {epoch}: train_loss={train_metrics['total_loss']:.4f}, "
              f"eval_accuracy={eval_metrics['eval_accuracy']:.4f}")
    
    print("\n🎉 Training completed successfully!")
    
    # Save model
    save_path = Path('checkpoints/working_model.pt')
    save_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == '__main__':
    main()
