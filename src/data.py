"""
Data Loading Module for HRM

Supports common reasoning tasks:
- ARC-AGI-1 style tasks
- Sudoku puzzles
- Maze navigation
- Custom reasoning datasets
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import random


class ARCDataset(Dataset):
    """ARC-AGI-1 style abstract reasoning tasks"""
    
    def __init__(self, data_path: str, max_examples: Optional[int] = None):
        """
        Load ARC dataset
        
        Args:
            data_path: Path to ARC JSON file
            max_examples: Limit number of examples (for quick testing)
        """
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        if max_examples:
            # Randomly sample examples for small-sample learning
            task_ids = list(self.data.keys())
            selected_ids = random.sample(task_ids, min(max_examples, len(task_ids)))
            self.data = {tid: self.data[tid] for tid in selected_ids}
        
        self.task_ids = list(self.data.keys())
        
    def __len__(self):
        return len(self.task_ids)
    
    def __getitem__(self, idx):
        task_id = self.task_ids[idx]
        task = self.data[task_id]
        
        # Convert grid to tensor representation
        # ARC grids are variable size - pad to fixed size
        train_examples = task['train']
        test_examples = task['test']
        
        # For now, use first training example
        example = train_examples[0]
        input_grid = torch.tensor(example['input'], dtype=torch.float32)
        output_grid = torch.tensor(example['output'], dtype=torch.long)
        
        # Pad to consistent size (30x30 maximum)
        input_padded = F.pad(input_grid, (0, 30-input_grid.size(1), 0, 30-input_grid.size(0)))
        output_padded = F.pad(output_grid, (0, 30-output_grid.size(1), 0, 30-output_grid.size(0)))
        
        # Flatten for sequence processing
        input_seq = input_padded.flatten().unsqueeze(0)  # [1, 900]
        output_seq = output_padded.flatten()  # [900]
        
        return {
            'inputs': input_seq,
            'targets': output_seq,
            'task_id': task_id
        }


class SudokuDataset(Dataset):
    """Sudoku puzzle dataset for extreme generalization testing"""
    
    def __init__(self, num_samples: int = 1000, difficulty: str = 'extreme'):
        """
        Generate Sudoku puzzles
        
        Args:
            num_samples: Number of puzzles to generate
            difficulty: 'easy', 'medium', 'hard', 'extreme'
        """
        self.num_samples = num_samples
        self.difficulty = difficulty
        self.puzzles = self._generate_puzzles()
    
    def _generate_puzzles(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate Sudoku puzzles with solutions"""
        puzzles = []
        
        difficulty_clues = {
            'easy': 40,
            'medium': 30, 
            'hard': 25,
            'extreme': 17  # As used in HRM paper
        }
        
        num_clues = difficulty_clues[self.difficulty]
        
        for _ in range(self.num_samples):
            # Generate complete valid Sudoku
            solution = self._generate_complete_sudoku()
            
            # Remove clues to create puzzle
            puzzle = solution.copy()
            cells_to_remove = 81 - num_clues
            positions = random.sample(range(81), cells_to_remove)
            
            for pos in positions:
                row, col = pos // 9, pos % 9
                puzzle[row, col] = 0
                
            puzzles.append((puzzle, solution))
        
        return puzzles
    
    def _generate_complete_sudoku(self) -> np.ndarray:
        """Generate a complete valid Sudoku solution"""
        # Simple backtracking generator
        grid = np.zeros((9, 9), dtype=int)
        
        def is_valid(grid, row, col, num):
            # Check row
            if num in grid[row]:
                return False
            # Check column  
            if num in grid[:, col]:
                return False
            # Check 3x3 box
            box_row, box_col = 3 * (row // 3), 3 * (col // 3)
            if num in grid[box_row:box_row+3, box_col:box_col+3]:
                return False
            return True
        
        def solve(grid):
            for row in range(9):
                for col in range(9):
                    if grid[row, col] == 0:
                        numbers = list(range(1, 10))
                        random.shuffle(numbers)  # Randomize for variety
                        for num in numbers:
                            if is_valid(grid, row, col, num):
                                grid[row, col] = num
                                if solve(grid):
                                    return True
                                grid[row, col] = 0
                        return False
            return True
        
        solve(grid)
        return grid
    
    def __len__(self):
        return len(self.puzzles)
    
    def __getitem__(self, idx):
        puzzle, solution = self.puzzles[idx]
        
        # Convert to tensors and flatten for sequence processing
        input_seq = torch.tensor(puzzle.flatten(), dtype=torch.float32).unsqueeze(0)  # [1, 81]
        target_seq = torch.tensor(solution.flatten(), dtype=torch.long)  # [81]
        
        return {
            'inputs': input_seq,
            'targets': target_seq,
            'puzzle_idx': idx
        }


class DatasetLoader:
    """Unified data loader for different reasoning tasks"""
    
    @staticmethod
    def get_dataloader(
        dataset_type: str,
        data_path: Optional[str] = None,
        batch_size: int = 32,
        num_samples: Optional[int] = None,
        shuffle: bool = True,
        **kwargs
    ) -> DataLoader:
        """
        Create DataLoader for specified dataset type
        
        Args:
            dataset_type: 'arc', 'sudoku', 'maze'
            data_path: Path to data file (for file-based datasets)
            batch_size: Batch size
            num_samples: Limit number of samples
            shuffle: Whether to shuffle data
        """
        
        if dataset_type == 'arc':
            if not data_path:
                raise ValueError("data_path required for ARC dataset")
            dataset = ARCDataset(data_path, max_examples=num_samples)
            
        elif dataset_type == 'sudoku':
            difficulty = kwargs.get('difficulty', 'extreme')
            num_samples = num_samples or 1000
            dataset = SudokuDataset(num_samples, difficulty)
            
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # Avoid multiprocessing issues with small datasets
            pin_memory=torch.cuda.is_available()
        )
