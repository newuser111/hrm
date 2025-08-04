"""
Hierarchical Reasoning Model (HRM) Implementation

A PyTorch implementation of the brain-inspired neural architecture
that achieved 40.3% on ARC-AGI-1 with only 1K training examples.
"""

from .model import HRM
from .training import HRMTrainer
from .data import DatasetLoader

__version__ = "0.1.0"
__all__ = ["HRM", "HRMTrainer", "DatasetLoader"]
