# Hierarchical Reasoning Model (HRM) Implementation

A PyTorch implementation of the brain-inspired Hierarchical Reasoning Model that achieved **40.3% accuracy on ARC-AGI-1** with only 1,000 training examples.

## 🧠 Key Features

- **Brain-Inspired Architecture**: Mimics cortical hierarchy with different timescales
- **Hierarchical Processing**: High-level module (H) for abstract planning, Low-level module (L) for detailed computation  
- **Adaptive Computation Time (ACT)**: Dynamic halting mechanism using Q-learning
- **One-Step Gradient**: Memory-efficient alternative to BPTT using DEQ principles
- **Small Sample Learning**: Designed for few-shot reasoning tasks

## 🏆 Performance

| Task | HRM | Comparison |
|------|-----|------------|
| **ARC-AGI-1** | **40.3%** | o3-mini-high: 34.5%, Claude 3.7: 21.2% |
| **Sudoku-Extreme** | **~100%** | With only 1K training samples |
| **Maze-Hard** | **Optimal** | 30x30 pathfinding |

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/newuser111/hrm.git
cd hrm
pip install -r requirements.txt
```

### Test the Implementation

```bash
python test_hrm.py
```

### Train on Sudoku (Paper Configuration)

```bash
python train_hrm.py --dataset sudoku --num_samples 1000 --epochs 100 --difficulty extreme
```

### Train on ARC-AGI (if you have the dataset)

```bash
python train_hrm.py --dataset arc --data_path path/to/arc_data.json --num_samples 1000
```

## 📁 Project Structure

```
hrm/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── model.py             # Core HRM architecture
│   ├── training.py          # Training logic with one-step gradient
│   └── data.py              # Dataset loaders for reasoning tasks
├── train_hrm.py             # Main training script
├── test_hrm.py              # Test suite
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🏗️ Architecture Details

### Core Components

1. **Input Network**: Projects input to low-level dimension
2. **Low-level Module (L)**: Fast, detailed Transformer (3 layers, 256d)
3. **High-level Module (H)**: Slow, abstract Transformer (6 layers, 512d)  
4. **Cross-Module Communication**: H→L and L→H projections
5. **Q-head**: Adaptive computation time with Q-learning
6. **Output Network**: Final prediction layer with StableMax

### Key Innovations

- **Hierarchical Convergence**: L-module converges locally, then H-module updates and resets L-module
- **Multi-Timescale Processing**: Different update frequencies for abstract vs detailed reasoning
- **Memory Efficiency**: One-step gradient avoids storing full computation graph
- **Biologically Plausible**: No backpropagation through time (BPTT)

## 🔬 Implementation Notes

- **Architecture**: Encoder-only Transformers with RMSNorm, RoPE, GLU activations
- **Initialization**: LeCun normal for training stability
- **Optimizer**: Adam-atan2 variant for small sample generalization  
- **Training**: Deep supervision + ACT loss + main task loss
- **Regularization**: StableMax for numerical stability with small datasets

## 📊 Monitoring Training

The implementation includes comprehensive logging:

- **Loss Components**: Main task, deep supervision, ACT penalty
- **Computation Efficiency**: Average steps per forward pass
- **Convergence**: Halting probability distributions
- **Performance**: Task-specific accuracy metrics

## 🎯 Next Steps

- [ ] Add maze navigation dataset
- [ ] Implement more ARC-AGI task variants
- [ ] Add visualization tools for hierarchical states
- [ ] Experiment with different H/L module ratios
- [ ] Add multi-task training capabilities

## 📚 References

Based on the paper: *"Hierarchical Reasoning Model: Brain-Inspired Neural Architecture for Improved Reasoning"*

Key insights:
- Biological inspiration from cortical hierarchy
- Multi-timescale computation for complex reasoning
- Memory-efficient training without BPTT
- Few-shot learning on challenging reasoning tasks

---

**Status**: 🟢 Core implementation complete and ready for training!
