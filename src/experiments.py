"""
Experiment Tracking and Configuration Management

Comprehensive experiment tracking with W&B integration,
hyperparameter sweeps, and performance monitoring.
"""

import wandb
import torch
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import hashlib


class ExperimentTracker:
    """Enhanced experiment tracking with W&B and local logging"""
    
    def __init__(
        self,
        project_name: str = "hrm-reasoning",
        experiment_name: Optional[str] = None,
        config: Optional[Dict] = None,
        use_wandb: bool = True,
        save_dir: str = "./experiments"
    ):
        self.project_name = project_name
        self.use_wandb = use_wandb
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Generate experiment ID
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_hash = self._hash_config(config) if config else "default"
            experiment_name = f"hrm_{timestamp}_{config_hash[:8]}"
        
        self.experiment_name = experiment_name
        self.experiment_dir = self.save_dir / experiment_name
        self.experiment_dir.mkdir(exist_ok=True)
        
        # Initialize W&B
        if self.use_wandb:
            wandb.init(
                project=project_name,
                name=experiment_name,
                config=config,
                save_code=True
            )
        
        # Save config locally
        if config:
            self._save_config(config)
        
        self.metrics_log = []
        print(f"🔬 Experiment: {experiment_name}")
    
    def _hash_config(self, config: Dict) -> str:
        """Create hash of config for unique experiment IDs"""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _save_config(self, config: Dict):
        """Save configuration to local file"""
        config_path = self.experiment_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to W&B and local storage"""
        
        # Add timestamp
        metrics['timestamp'] = datetime.now().isoformat()
        if step is not None:
            metrics['step'] = step
        
        # Log to W&B
        if self.use_wandb:
            wandb.log(metrics, step=step)
        
        # Log locally
        self.metrics_log.append(metrics.copy())
        
        # Save to file periodically
        if len(self.metrics_log) % 10 == 0:
            self._save_metrics()
    
    def _save_metrics(self):
        """Save metrics log to file"""
        metrics_path = self.experiment_dir / "metrics.jsonl"
        with open(metrics_path, 'w') as f:
            for metrics in self.metrics_log:
                f.write(json.dumps(metrics) + '\n')
    
    def save_model(self, model: torch.nn.Module, metrics: Dict[str, float], suffix: str = ""):
        """Save model checkpoint with metrics"""
        if suffix:
            filename = f"model_{suffix}.pt"
        else:
            filename = f"model_step_{len(self.metrics_log)}.pt"
        
        checkpoint_path = self.experiment_dir / filename
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'experiment_name': self.experiment_name,
            'step': len(self.metrics_log)
        }, checkpoint_path)
        
        print(f"💾 Model saved: {checkpoint_path}")
        
        # Log model to W&B
        if self.use_wandb:
            wandb.save(str(checkpoint_path))
    
    def finish(self):
        """Finish experiment and cleanup"""
        self._save_metrics()
        
        if self.use_wandb:
            wandb.finish()
        
        print(f"🏁 Experiment completed: {self.experiment_name}")


class HyperparameterSweep:
    """Hyperparameter sweep configuration"""
    
    @staticmethod
    def get_sweep_config() -> Dict:
        """Define hyperparameter sweep for W&B"""
        
        sweep_config = {
            'method': 'bayes',  # or 'grid', 'random'
            'metric': {
                'name': 'eval_accuracy',
                'goal': 'maximize'
            },
            'parameters': {
                # Architecture parameters
                'h_dim': {
                    'values': [64, 128, 256, 512]
                },
                'l_dim': {
                    'values': [32, 64, 128, 256]  
                },
                'n_layers_h': {
                    'values': [1, 2, 3, 4, 6]
                },
                'n_layers_l': {
                    'values': [1, 2, 3]
                },
                'n_heads_h': {
                    'values': [2, 4, 8]
                },
                'n_heads_l': {
                    'values': [1, 2, 4]
                },
                
                # Training parameters
                'learning_rate': {
                    'distribution': 'log_uniform_values',
                    'min': 1e-5,
                    'max': 1e-2
                },
                'q_learning_rate': {
                    'distribution': 'log_uniform_values', 
                    'min': 1e-6,
                    'max': 1e-3
                },
                'batch_size': {
                    'values': [8, 16, 32, 64]
                },
                'supervision_weight': {
                    'distribution': 'uniform',
                    'min': 0.01,
                    'max': 0.5
                },
                'act_loss_weight': {
                    'distribution': 'uniform',
                    'min': 0.001,
                    'max': 0.1
                },
                'act_threshold': {
                    'distribution': 'uniform',
                    'min': 0.3,
                    'max': 0.8
                }
            }
        }
        
        return sweep_config
    
    @staticmethod
    def create_sweep(project_name: str = "hrm-reasoning") -> str:
        """Create W&B sweep and return sweep ID"""
        sweep_config = HyperparameterSweep.get_sweep_config()
        sweep_id = wandb.sweep(sweep_config, project=project_name)
        print(f"🔍 Sweep created: {sweep_id}")
        return sweep_id


class PerformanceMonitor:
    """Monitor model performance and computational efficiency"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.forward_times = []
        self.memory_usage = []
        self.computation_steps = []
        self.halting_patterns = []
    
    def record_forward_pass(
        self, 
        forward_time: float,
        memory_mb: float,
        num_steps: int,
        halting_probs: torch.Tensor
    ):
        """Record metrics from a forward pass"""
        self.forward_times.append(forward_time)
        self.memory_usage.append(memory_mb)
        self.computation_steps.append(num_steps)
        self.halting_patterns.append(halting_probs.cpu().numpy())
    
    def get_summary(self) -> Dict[str, float]:
        """Get performance summary"""
        if not self.forward_times:
            return {}
        
        return {
            'avg_forward_time': np.mean(self.forward_times),
            'max_memory_mb': max(self.memory_usage) if self.memory_usage else 0,
            'avg_computation_steps': np.mean(self.computation_steps),
            'std_computation_steps': np.std(self.computation_steps),
            'min_computation_steps': min(self.computation_steps),
            'max_computation_steps': max(self.computation_steps)
        }
    
    def analyze_halting_patterns(self) -> Dict[str, float]:
        """Analyze adaptive computation patterns"""
        if not self.halting_patterns:
            return {}
        
        # Stack all halting probabilities
        all_halting = np.concatenate(self.halting_patterns, axis=1)  # [steps, all_sequences]
        
        # Analyze patterns
        early_halt_ratio = np.mean(np.argmax(np.cumsum(all_halting, axis=0) > 0.5, axis=0) < 3)
        late_halt_ratio = np.mean(np.argmax(np.cumsum(all_halting, axis=0) > 0.5, axis=0) > 7)
        avg_halt_step = np.mean(np.argmax(np.cumsum(all_halting, axis=0) > 0.5, axis=0))
        
        return {
            'early_halt_ratio': early_halt_ratio,
            'late_halt_ratio': late_halt_ratio, 
            'avg_halt_step': avg_halt_step,
            'halt_variance': np.var(np.argmax(np.cumsum(all_halting, axis=0) > 0.5, axis=0))
        }


def run_experiment_with_tracking(config: Dict[str, Any]):
    """Run a single experiment with full tracking"""
    
    # Initialize tracker
    tracker = ExperimentTracker(
        project_name="hrm-reasoning",
        config=config,
        use_wandb=config.get('use_wandb', False)
    )
    
    try:
        # Create model
        model = HRM(
            input_dim=1,
            h_dim=config['h_dim'],
            l_dim=config['l_dim'],
            n_heads_h=config['n_heads_h'],
            n_heads_l=config['n_heads_l'],
            n_layers_h=config['n_layers_h'],
            n_layers_l=config['n_layers_l'],
            output_dim=10,
            max_len=81,
            act_threshold=config.get('act_threshold', 0.5)
        )
        
        # Create trainer
        trainer = HRMTrainerFixed(
            model=model,
            learning_rate=config['learning_rate'],
            q_learning_rate=config.get('q_learning_rate', 1e-4),
            supervision_weight=config.get('supervision_weight', 0.1),
            act_loss_weight=config.get('act_loss_weight', 0.01),
            use_wandb=False  # Handled by tracker
        )
        
        # Create dataloader
        train_loader = DatasetLoader.get_dataloader(
            dataset_type=config.get('dataset', 'sudoku'),
            batch_size=config['batch_size'],
            num_samples=config.get('num_samples', 100),
            difficulty=config.get('difficulty', 'extreme')
        )
        
        # Training loop with monitoring
        monitor = PerformanceMonitor()
        best_accuracy = 0.0
        
        for epoch in range(config['epochs']):
            # Train epoch
            train_metrics = trainer.train_epoch_simple(train_loader, epoch)
            
            # Evaluate
            eval_metrics = trainer.evaluate_simple(train_loader, max_batches=5)
            
            # Combine metrics
            all_metrics = {**train_metrics, **eval_metrics, 'epoch': epoch}
            
            # Log metrics
            tracker.log_metrics(all_metrics, step=epoch)
            
            # Save best model
            if eval_metrics['eval_accuracy'] > best_accuracy:
                best_accuracy = eval_metrics['eval_accuracy']
                tracker.save_model(model, all_metrics, suffix="best")
            
            print(f"Epoch {epoch}: loss={train_metrics['total_loss']:.4f}, "
                  f"acc={eval_metrics['eval_accuracy']:.4f}")
        
        # Final performance summary
        perf_summary = monitor.get_summary()
        tracker.log_metrics(perf_summary, step=config['epochs'])
        
        return best_accuracy
        
    finally:
        tracker.finish()
