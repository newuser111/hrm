"""
Fixed Training Module - Debug version without hanging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Dict, List, Tuple, Optional
import numpy as np

from .model import HRM
from .training import AdamAtan2  # Import the custom optimizer


class HRMTrainerFixed:
    """Fixed HRM Training without hanging issues"""
    
    def __init__(
        self,
        model: HRM,
        learning_rate: float = 1e-3,
        q_learning_rate: float = 1e-4,
        supervision_weight: float = 0.1,
        act_loss_weight: float = 0.01,
        use_wandb: bool = False
    ):
        self.model = model
        self.supervision_weight = supervision_weight
        self.act_loss_weight = act_loss_weight
        self.use_wandb = use_wandb
        
        # Main optimizer for model parameters (excluding Q-head)
        model_params = [p for name, p in model.named_parameters() if 'q_head' not in name]
        self.optimizer = AdamAtan2(model_params, lr=learning_rate)
        
        # Separate optimizer for Q-learning (ACT)
        q_params = [p for name, p in model.named_parameters() if 'q_head' in name]
        self.q_optimizer = AdamW(q_params, lr=q_learning_rate)
        
        # Loss functions
        self.criterion = nn.CrossEntropyLoss()
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        device = next(self.model.parameters()).device
        inputs = batch['inputs'].to(device)
        targets = batch['targets'].to(device)
        
        # Forward pass
        result = self.model(inputs, return_intermediates=True)
        
        # Main prediction loss
        main_loss = self.criterion(
            result['output'].view(-1, result['output'].size(-1)),
            targets.view(-1)
        )
        
        # Simple supervision loss (just main loss for now)
        supervision_loss = main_loss * 0.1  # Simplified
        
        # Simple ACT loss
        act_loss = result['halting_probs'].mean() * 0.01  # Simplified
        
        # Combined loss
        total_loss = main_loss + supervision_loss + act_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        self.q_optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update parameters
        self.optimizer.step()
        self.q_optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'main_loss': main_loss.item(),
            'supervision_loss': supervision_loss.item(),
            'act_loss': act_loss.item(),
            'avg_steps': result['num_steps'],
            'avg_halt_prob': result['halting_probs'].mean().item()
        }
    
    def train_epoch_simple(self, dataloader, epoch: int) -> Dict[str, float]:
        """Simplified training epoch without progress bar"""
        self.model.train()
        total_loss = 0.0
        total_batches = 0
        
        print(f"Training epoch {epoch}...")
        
        for i, batch in enumerate(dataloader):
            metrics = self.train_step(batch)
            total_loss += metrics['total_loss']
            total_batches += 1
            
            if i % 5 == 0:  # Print every 5 batches
                print(f"  Batch {i}: loss={metrics['total_loss']:.4f}, steps={metrics['avg_steps']}")
        
        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
        print(f"Epoch {epoch} complete: avg_loss={avg_loss:.4f}")
        
        return {'total_loss': avg_loss}
    
    def evaluate_simple(self, dataloader, max_batches: Optional[int] = None) -> Dict[str, float]:
        """Simplified evaluation"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_batches = 0
        
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if max_batches and i >= max_batches:
                    break
                    
                inputs = batch['inputs'].to(device)
                targets = batch['targets'].to(device)
                
                result = self.model(inputs)
                
                # Compute loss
                loss = self.criterion(
                    result['output'].view(-1, result['output'].size(-1)),
                    targets.view(-1)
                )
                total_loss += loss.item()
                
                # Compute accuracy
                predictions = result['output'].argmax(dim=-1)
                correct = (predictions == targets).sum().item()
                total_correct += correct
                total_samples += targets.numel()
                total_batches += 1
        
        self.model.train()
        
        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return {
            'eval_loss': avg_loss,
            'eval_accuracy': accuracy
        }
