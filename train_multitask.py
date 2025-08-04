#!/usr/bin/env python3
"""
Multi-Task Training Script

Train HRM on multiple reasoning tasks.
"""

import argparse
import torch
import json
from pathlib import Path
import sys
from datetime import datetime
sys.path.append(str(Path(__file__).parent))

from src.model import HRM
from src.training_fixed import HRMTrainerFixed
from src.additional_tasks import EnhancedDatasetLoader


def train_on_task(model, task_name, config):
    """Train model on a specific task"""
    
    print(f"\n🎯 Training on {task_name}...")
    
    # Get task-specific config
    task_config = config.get(f'{task_name}_config', {})
    
    # Create dataloader
    train_loader = EnhancedDatasetLoader.get_dataloader(
        dataset_type=task_name,
        batch_size=config['batch_size'],
        num_samples=config['num_samples_per_task'],
        **task_config
    )
    
    # Create trainer
    trainer = HRMTrainerFixed(model, learning_rate=config['learning_rate'])
    
    # Training loop
    best_accuracy = 0.0
    
    for epoch in range(config['epochs_per_task']):
        train_metrics = trainer.train_epoch_simple(train_loader, epoch)
        eval_metrics = trainer.evaluate_simple(train_loader, max_batches=3)
        
        if eval_metrics['eval_accuracy'] > best_accuracy:
            best_accuracy = eval_metrics['eval_accuracy']
        
        print(f"  Epoch {epoch}: loss={train_metrics['total_loss']:.4f}, "
              f"acc={eval_metrics['eval_accuracy']:.4f}")
    
    return best_accuracy


def main():
    parser = argparse.ArgumentParser(description='Multi-task HRM training')
    parser.add_argument('--config', type=str, default='small_test', 
                       choices=['small_test', 'medium_benchmark'],
                       help='Predefined configuration')
    
    args = parser.parse_args()
    
    # Demo configurations
    configs = {
        'small_test': {
            'h_dim': 32, 'l_dim': 16, 'n_heads_h': 2, 'n_heads_l': 1,
            'n_layers_h': 1, 'n_layers_l': 1, 'output_dim': 10, 'max_len': 225,
            'learning_rate': 1e-3, 'batch_size': 4, 'epochs_per_task': 3,
            'num_samples_per_task': 20, 'tasks': ['maze', 'pattern']
        },
        'medium_benchmark': {
            'h_dim': 64, 'l_dim': 32, 'n_heads_h': 2, 'n_heads_l': 2,
            'n_layers_h': 2, 'n_layers_l': 1, 'output_dim': 10, 'max_len': 225,
            'learning_rate': 1e-3, 'batch_size': 8, 'epochs_per_task': 5,
            'num_samples_per_task': 50, 'tasks': ['maze', 'pattern'],
            'maze_config': {'maze_size': 15, 'difficulty': 'medium'}
        }
    }
    
    config = configs[args.config]
    
    print(f"🚀 Multi-task training: {args.config}")
    print(f"📋 Tasks: {config['tasks']}")
    
    # Create model
    model = HRM(
        input_dim=1,
        h_dim=config['h_dim'],
        l_dim=config['l_dim'],
        n_heads_h=config['n_heads_h'],
        n_heads_l=config['n_heads_l'],
        n_layers_h=config['n_layers_h'],
        n_layers_l=config['n_layers_l'],
        output_dim=config['output_dim'],
        max_len=config['max_len']
    )
    
    print(f"🧠 Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train on each task
    results = {}
    
    for task_name in config['tasks']:
        try:
            best_acc = train_on_task(model, task_name, config)
            results[task_name] = best_acc
            print(f"✅ {task_name}: {best_acc:.1%}")
            
        except Exception as e:
            print(f"❌ {task_name}: {e}")
            results[task_name] = 0.0
    
    # Print summary
    print(f"\n🎉 Training completed!")
    print("📊 Final Results:")
    for task, acc in results.items():
        print(f"   {task}: {acc:.1%}")
    
    # Save model
    model_path = Path(f"./models/hrm_multitask_{args.config}.pt")
    model_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved: {model_path}")


if __name__ == '__main__':
    main()
