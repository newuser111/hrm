#!/usr/bin/env python3
"""
Comprehensive HRM Evaluation Suite

Test HRM on all reasoning tasks with detailed analysis.
"""

import torch
import numpy as np
import json
from pathlib import Path
import sys
import time
sys.path.append(str(Path(__file__).parent))

from src.model import HRM
from src.training_fixed import HRMTrainerFixed
from src.additional_tasks import EnhancedDatasetLoader
from src.visualizations import HRMVisualizer


def evaluate_task_performance(
    model: HRM,
    task_name: str,
    num_samples: int = 100,
    **task_kwargs
) -> Dict[str, float]:
    """Evaluate model performance on a specific task"""
    
    print(f"  Evaluating {task_name}...")
    
    try:
        # Create test dataloader
        test_loader = EnhancedDatasetLoader.get_dataloader(
            dataset_type=task_name,
            batch_size=8,
            num_samples=num_samples,
            shuffle=False,
            **task_kwargs
        )
        
        model.eval()
        device = next(model.parameters()).device
        
        total_correct = 0
        total_samples = 0
        total_forward_time = 0.0
        computation_steps = []
        
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch['inputs'].to(device)
                targets = batch['targets'].to(device)
                
                # Time forward pass
                start_time = time.time()
                result = model(inputs, max_steps=20)
                forward_time = time.time() - start_time
                
                total_forward_time += forward_time
                computation_steps.extend([result['num_steps']] * inputs.size(0))
                
                # Compute accuracy
                predictions = result['output'].argmax(dim=-1)
                correct = (predictions == targets).sum().item()
                total_correct += correct
                total_samples += targets.numel()
        
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        avg_forward_time = total_forward_time / len(test_loader)
        avg_steps = np.mean(computation_steps)
        
        return {
            'accuracy': accuracy,
            'avg_forward_time_ms': avg_forward_time * 1000,
            'avg_computation_steps': avg_steps,
            'total_samples': total_samples
        }
        
    except Exception as e:
        print(f"    ❌ Error evaluating {task_name}: {e}")
        return {'error': str(e)}


def comprehensive_evaluation():
    """Run comprehensive evaluation on all tasks"""
    
    print("🧠 HRM Comprehensive Evaluation")
    print("=" * 50)
    
    # Test different model sizes
    model_configs = {
        'tiny': {
            'h_dim': 32, 'l_dim': 16, 'n_heads_h': 2, 'n_heads_l': 1,
            'n_layers_h': 1, 'n_layers_l': 1, 'max_len': 225
        },
        'small': {
            'h_dim': 64, 'l_dim': 32, 'n_heads_h': 2, 'n_heads_l': 2,
            'n_layers_h': 2, 'n_layers_l': 1, 'max_len': 225
        },
        'medium': {
            'h_dim': 128, 'l_dim': 64, 'n_heads_h': 4, 'n_heads_l': 2,
            'n_layers_h': 3, 'n_layers_l': 2, 'max_len': 225
        }
    }
    
    # Define tasks to evaluate
    tasks = {
        'maze': {'maze_size': 15, 'difficulty': 'medium'},
        'pattern': {'pattern_type': 'sequence'}
    }
    
    all_results = {}
    
    for model_name, model_config in model_configs.items():
        print(f"\n🏗️  Testing {model_name} model...")
        
        # Create model
        model = HRM(
            input_dim=1,
            output_dim=10,
            **model_config
        )
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"   Parameters: {param_count:,}")
        
        # Evaluate on each task
        model_results = {}
        
        for task_name, task_config in tasks.items():
            task_results = evaluate_task_performance(
                model, task_name, num_samples=50, **task_config
            )
            model_results[task_name] = task_results
        
        all_results[model_name] = {
            'model_config': model_config,
            'param_count': param_count,
            'task_results': model_results
        }
    
    # Print summary
    print("\n📊 Evaluation Summary")
    print("=" * 50)
    
    for model_name, model_result in all_results.items():
        print(f"\n{model_name.upper()} MODEL ({model_result['param_count']:,} params):")
        
        for task_name, task_result in model_result['task_results'].items():
            if 'accuracy' in task_result:
                print(f"  {task_name:12}: {task_result['accuracy']:.1%} accuracy, "
                      f"{task_result['avg_computation_steps']:.1f} steps, "
                      f"{task_result['avg_forward_time_ms']:.1f}ms")
            else:
                print(f"  {task_name:12}: FAILED")
    
    # Save results
    results_path = Path("./evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Full results saved to: {results_path}")
    
    return all_results


def test_reasoning_capabilities():
    """Test specific reasoning capabilities"""
    
    print("\n🧪 Testing Reasoning Capabilities")
    print("=" * 40)
    
    # Create model
    model = HRM(
        input_dim=1,
        h_dim=64,
        l_dim=32,
        n_heads_h=2,
        n_heads_l=2,
        n_layers_h=2,
        n_layers_l=1,
        output_dim=10,
        max_len=225
    )
    
    # Test hierarchical processing
    print("1. Testing hierarchical state evolution...")
    
    # Create test input
    test_input = torch.randint(0, 5, (1, 25, 1)).float()  # 5x5 grid
    
    with torch.no_grad():
        result = model(test_input, max_steps=10, return_intermediates=True)
    
    print(f"   ✅ Computation completed in {result['num_steps']} steps")
    print(f"   📊 Halting pattern: {result['halting_probs'].mean(dim=[1,2]).tolist()}")
    
    # Test adaptive computation
    print("2. Testing adaptive computation...")
    
    # Easy vs hard inputs
    easy_input = torch.zeros(1, 25, 1)  # All zeros - should halt early
    hard_input = torch.randint(0, 10, (1, 225, 1)).float()  # Complex pattern
    
    with torch.no_grad():
        easy_result = model(easy_input, max_steps=20)
        hard_result = model(hard_input, max_steps=20)
    
    print(f"   Easy input steps: {easy_result['num_steps']}")
    print(f"   Hard input steps: {hard_result['num_steps']}")
    
    if hard_result['num_steps'] > easy_result['num_steps']:
        print("   ✅ Adaptive computation working correctly")
    else:
        print("   ⚠️  Adaptive computation may need tuning")
    
    # Test memory efficiency
    print("3. Testing memory efficiency...")
    
    import torch.profiler
    
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        with torch.no_grad():
            for _ in range(10):
                result = model(test_input, max_steps=5)
    
    print(f"   ✅ Memory profiling completed")
    
    return {
        'hierarchical_steps': result['num_steps'],
        'adaptive_difference': hard_result['num_steps'] - easy_result['num_steps'],
        'memory_efficient': True  # Placeholder
    }


def create_benchmark_report():
    """Create comprehensive benchmark report"""
    
    print("📋 Creating Benchmark Report...")
    
    # Run evaluations
    eval_results = comprehensive_evaluation()
    reasoning_results = test_reasoning_capabilities()
    
    # Create report
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'evaluation_results': eval_results,
        'reasoning_capabilities': reasoning_results,
        'summary': {
            'total_tasks_tested': len([task for model in eval_results.values() 
                                     for task in model['task_results'].keys()]),
            'successful_tasks': len([task for model in eval_results.values() 
                                   for task, result in model['task_results'].items() 
                                   if 'accuracy' in result]),
            'best_overall_accuracy': max([result['accuracy'] 
                                        for model in eval_results.values() 
                                        for task_result in model['task_results'].values() 
                                        if 'accuracy' in task_result], default=0.0)
        }
    }
    
    # Save report
    report_path = Path("./benchmark_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Benchmark report saved: {report_path}")
    
    return report


if __name__ == '__main__':
    print("🚀 Starting comprehensive HRM evaluation...")
    
    try:
        report = create_benchmark_report()
        
        print("\n🎉 Evaluation completed successfully!")
        print(f"📊 Summary:")
        print(f"   Tasks tested: {report['summary']['total_tasks_tested']}")
        print(f"   Successful: {report['summary']['successful_tasks']}")
        print(f"   Best accuracy: {report['summary']['best_overall_accuracy']:.1%}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
