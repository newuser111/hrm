#!/usr/bin/env python3
"""
HRM Training Script

Train the Hierarchical Reasoning Model on reasoning tasks.
Usage: python train_hrm.py --dataset sudoku --epochs 100
"""

import argparse
import torch
import wandb
from pathlib import Path

from src import HRM, HRMTrainer, DatasetLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Train HRM on reasoning tasks')
    
    # Model architecture
    parser.add_argument('--h_dim', type=int, default=512, help='High-level module dimension')
    parser.add_argument('--l_dim', type=int, default=256, help='Low-level module dimension') 
    parser.add_argument('--n_heads_h', type=int, default=8, help='High-level attention heads')
    parser.add_argument('--n_heads_l', type=int, default=4, help='Low-level attention heads')
    parser.add_argument('--n_layers_h', type=int, default=6, help='High-level layers')
    parser.add_argument('--n_layers_l', type=int, default=3, help='Low-level layers')
    
    # Training
    parser.add_argument('--dataset', type=str, default='sudoku', choices=['arc', 'sudoku'])
    parser.add_argument('--data_path', type=str, help='Path to dataset file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--q_lr', type=float, default=1e-4, help='Q-learning rate for ACT')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of training samples')
    
    # Logging
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--project_name', type=str, default='hrm-reasoning', help='W&B project name')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Model save directory')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=args.project_name,
            config=vars(args)
        )
    
    # Determine input/output dimensions based on dataset
    if args.dataset == 'sudoku':
        input_dim = 1  # Single number per cell
        output_dim = 10  # 0-9 (0 for empty, 1-9 for digits)
        max_len = 81  # 9x9 grid
    elif args.dataset == 'arc':
        input_dim = 1  # Single color per cell
        output_dim = 10  # 0-9 colors in ARC
        max_len = 900  # 30x30 maximum grid size
    
    # Create model
    model = HRM(
        input_dim=input_dim,
        h_dim=args.h_dim,
        l_dim=args.l_dim, 
        n_heads_h=args.n_heads_h,
        n_heads_l=args.n_heads_l,
        n_layers_h=args.n_layers_h,
        n_layers_l=args.n_layers_l,
        output_dim=output_dim,
        max_len=max_len
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create data loaders
    train_loader = DatasetLoader.get_dataloader(
        dataset_type=args.dataset,
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        shuffle=True,
        difficulty='extreme' if args.dataset == 'sudoku' else None
    )
    
    # Create trainer
    trainer = HRMTrainer(
        model=model,
        learning_rate=args.lr,
        q_learning_rate=args.q_lr,
        use_wandb=args.use_wandb
    )
    
    # Training loop
    best_accuracy = 0.0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)
        
        # Evaluate (on same data for now - in practice use separate val set)
        eval_metrics = trainer.evaluate(train_loader, max_batches=10)
        
        # Log metrics
        print(f"Train Loss: {train_metrics['total_loss']:.4f}")
        print(f"Eval Accuracy: {eval_metrics['eval_accuracy']:.4f}")
        print(f"Avg Computation Steps: {eval_metrics['avg_computation_steps']:.2f}")
        
        if args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_metrics['total_loss'],
                'eval_accuracy': eval_metrics['eval_accuracy'],
                'avg_steps': eval_metrics['avg_computation_steps']
            })
        
        # Save best model
        if eval_metrics['eval_accuracy'] > best_accuracy:
            best_accuracy = eval_metrics['eval_accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'accuracy': best_accuracy,
                'args': vars(args)
            }, save_dir / 'best_model.pt')
            print(f"New best accuracy: {best_accuracy:.4f}")
    
    print(f"\nTraining complete! Best accuracy: {best_accuracy:.4f}")


if __name__ == '__main__':
    main()
