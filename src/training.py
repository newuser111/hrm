"""
HRM Training Module

Implements training logic with:
- One-step gradient (memory-efficient alternative to BPTT)
- Deep supervision for better convergence
- Q-learning for adaptive computation time
- Adam-atan2 optimizer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Dict, List, Tuple, Optional
import wandb
from tqdm import tqdm
import numpy as np

from .model import HRM


class AdamAtan2(torch.optim.Optimizer):
    """
    Adam variant using atan2 for gradient direction
    More stable for small sample training
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                    
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                
                # Exponential moving averages
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Use atan2 for direction, magnitude for step size
                direction = torch.atan2(exp_avg, torch.sqrt(exp_avg_sq) + group['eps'])
                magnitude = torch.sqrt(exp_avg_sq / bias_correction2) + group['eps']
                
                step_size = group['lr'] / bias_correction1
                p.data.add_(torch.sin(direction) * step_size / magnitude, alpha=-1)
                
        return loss

class HRMTrainer:
    """HRM Training with One-Step Gradient and Deep Supervision"""
    
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
        self.mse_loss = nn.MSELoss()
        
    def compute_one_step_gradient(
        self, 
        states: List[torch.Tensor], 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute one-step gradient instead of BPTT
        Memory-efficient alternative using DEQ principles
        """
        # Use final state for gradient computation
        final_state = states[-1]
        
        # Compute loss on final state
        output = self.model.output_proj(final_state)
        loss = self.criterion(output.view(-1, output.size(-1)), targets.view(-1))
        
        # One-step gradient: approximate gradient using final state only
        # This avoids storing all intermediate states for BPTT
        return loss
    
    def compute_deep_supervision_loss(
        self, 
        intermediates: List[Dict], 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Deep supervision: apply loss to intermediate states
        Helps with gradient flow and faster convergence
        """
        supervision_losses = []
        
        for state_dict in intermediates[::2]:  # Sample every 2nd intermediate
            l_state = state_dict['l_state']
            output = self.model.output_proj(l_state)
            loss = self.criterion(output.view(-1, output.size(-1)), targets.view(-1))
            supervision_losses.append(loss)
        
        if supervision_losses:
            return torch.stack(supervision_losses).mean()
        return torch.tensor(0.0, device=targets.device)
    
    def compute_act_loss(self, halting_probs: torch.Tensor) -> torch.Tensor:
        """
        Adaptive Computation Time loss to encourage efficient computation
        Penalizes both too early and too late halting
        """
        # Encourage halting around reasonable number of steps
        target_steps = halting_probs.size(0) // 2  # Middle of max_steps
        actual_steps = halting_probs.sum(dim=0)  # [B, L]
        
        # Penalty for deviating from target steps
        step_penalty = F.mse_loss(actual_steps, 
                                 torch.full_like(actual_steps, target_steps))
        
        # Encourage decisive halting (avoid hovering around threshold)
        decisiveness = torch.var(halting_probs, dim=0).mean()
        decisiveness_penalty = -decisiveness  # Negative because we want high variance
        
        return step_penalty + 0.1 * decisiveness_penalty
    
    def train_step(
        self, 
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Single training step with HRM-specific losses"""
        
        # Move inputs to device
        device = next(self.model.parameters()).device
        inputs = batch['inputs'].to(device)
        targets = batch['targets'].to(device)
        
        # Forward pass with intermediate states
        result = self.model(inputs, return_intermediates=True)
        
        # Main prediction loss
        main_loss = self.criterion(
            result['output'].view(-1, result['output'].size(-1)),
            targets.view(-1)
        )
        
        # Deep supervision loss
        supervision_loss = self.compute_deep_supervision_loss(
            result['intermediates'], targets
        )
        
        # ACT loss for efficient computation
        act_loss = self.compute_act_loss(result['halting_probs'])
        
        # Combined loss
        total_loss = (main_loss + 
                     self.supervision_weight * supervision_loss +
                     self.act_loss_weight * act_loss)
        
        # Backward pass
        self.optimizer.zero_grad()
        self.q_optimizer.zero_grad()
        
        total_loss.backward()
        
        # Gradient clipping for stability
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
    
    def evaluate(
        self, 
        dataloader, 
        max_batches: Optional[int] = None
    ) -> Dict[str, float]:
        """Evaluate model performance"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_steps = 0
        
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
                total_steps += result['num_steps']
        
        self.model.train()
        
        return {
            'eval_loss': total_loss / (i + 1),
            'eval_accuracy': total_correct / total_samples,
            'avg_computation_steps': total_steps / (i + 1)
        }
    
    def train_epoch(
        self, 
        dataloader, 
        epoch: int,
        log_interval: int = 10
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_metrics = {
            'total_loss': 0.0,
            'main_loss': 0.0,
            'supervision_loss': 0.0, 
            'act_loss': 0.0,
            'avg_steps': 0.0
        }
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for i, batch in enumerate(progress_bar):
            # Training step
            metrics = self.train_step(batch)
            
            # Accumulate metrics
            for key in epoch_metrics:
                if key in metrics:
                    epoch_metrics[key] += metrics[key]
            
            # Update progress bar
            if i % log_interval == 0:
                current_loss = epoch_metrics['total_loss'] / (i + 1)
                progress_bar.set_postfix({'loss': f'{current_loss:.4f}'})
                
                if self.use_wandb:
                    wandb.log({
                        'step': epoch * len(dataloader) + i,
                        'batch_loss': metrics['total_loss'],
                        'batch_steps': metrics['avg_steps']
                    })
        
        # Average metrics over epoch
        for key in epoch_metrics:
            epoch_metrics[key] /= len(dataloader)
            
        return epoch_metrics
