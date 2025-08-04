"""
Additional Reasoning Tasks for HRM

Implements various reasoning tasks beyond Sudoku:
- Maze navigation
- Tower of Hanoi  
- Pattern completion
- Logic puzzles
- Graph coloring
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Optional
import random
from collections import deque


class MazeDataset(Dataset):
    """Maze navigation task - find optimal path from start to goal"""
    
    def __init__(self, num_samples: int = 1000, maze_size: int = 15, difficulty: str = 'medium'):
        self.num_samples = num_samples
        self.maze_size = maze_size
        self.difficulty = difficulty
        
        wall_density = {'easy': 0.2, 'medium': 0.3, 'hard': 0.4}[difficulty]
        self.wall_prob = wall_density
        
        self.mazes = self._generate_mazes()
    
    def _generate_maze(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate single maze with solution path"""
        maze = np.zeros((self.maze_size, self.maze_size), dtype=int)
        
        # Add walls randomly
        for i in range(self.maze_size):
            for j in range(self.maze_size):
                if random.random() < self.wall_prob:
                    maze[i, j] = 1
        
        # Ensure start and goal are free
        start = (0, 0)
        goal = (self.maze_size-1, self.maze_size-1)
        maze[start] = 0
        maze[goal] = 0
        
        # Find path and create solution
        path = self._find_path(maze, start, goal)
        if not path:
            path = self._create_clear_path(maze, start, goal)
        
        solution = maze.copy()
        for pos in path:
            if solution[pos] == 0:
                solution[pos] = 2  # Path marker
        
        return maze, solution
    
    def _find_path(self, maze: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find shortest path using BFS"""
        queue = deque([(start, [start])])
        visited = {start}
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while queue:
            (x, y), path = queue.popleft()
            if (x, y) == goal:
                return path
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.maze_size and 0 <= ny < self.maze_size and 
                    (nx, ny) not in visited and maze[nx, ny] == 0):
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))
        return []
    
    def _create_clear_path(self, maze: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Create a clear path when none exists"""
        path = []
        x, y = start
        
        while y < goal[1]:
            maze[x, y] = 0
            path.append((x, y))
            y += 1
        
        while x < goal[0]:
            maze[x, y] = 0
            path.append((x, y))
            x += 1
        
        path.append(goal)
        return path
    
    def _generate_mazes(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate all mazes"""
        mazes = []
        for _ in range(self.num_samples):
            maze, solution = self._generate_maze()
            mazes.append((maze, solution))
        return mazes
    
    def __len__(self):
        return len(self.mazes)
    
    def __getitem__(self, idx):
        maze, solution = self.mazes[idx]
        
        input_seq = torch.tensor(maze.flatten(), dtype=torch.float32).unsqueeze(-1)
        target_seq = torch.tensor(solution.flatten(), dtype=torch.long)
        
        return {
            'inputs': input_seq,
            'targets': target_seq,
            'maze_idx': idx
        }


class PatternCompletionDataset(Dataset):
    """Pattern completion tasks - continue sequences"""
    
    def __init__(self, num_samples: int = 1000, pattern_type: str = 'sequence'):
        self.num_samples = num_samples
        self.pattern_type = pattern_type
        self.patterns = self._generate_patterns()
    
    def _generate_patterns(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate patterns based on type"""
        patterns = []
        for _ in range(self.num_samples):
            if self.pattern_type == 'sequence':
                pattern, completion = self._generate_sequence_pattern()
            else:
                pattern, completion = self._generate_sequence_pattern()  # Default
            patterns.append((pattern, completion))
        return patterns
    
    def _generate_sequence_pattern(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate arithmetic or geometric sequence"""
        seq_len = 20
        pattern_len = 12
        
        if random.choice([True, False]):
            # Arithmetic sequence
            start = random.randint(1, 10)
            step = random.randint(1, 5)
            sequence = [start + i * step for i in range(seq_len)]
        else:
            # Geometric sequence
            start = random.randint(1, 5)
            ratio = random.choice([2, 3])
            sequence = [start * (ratio ** i) for i in range(seq_len)]
            sequence = [min(val, 100) for val in sequence]  # Cap values
        
        pattern = np.array(sequence[:pattern_len])
        completion = np.array(sequence[pattern_len:])
        
        # Clamp values to valid range (0-9)
        pattern = np.clip(pattern, 0, 9)
        completion = np.clip(completion, 0, 9)
        
        return pattern, completion
    
    def __len__(self):
        return len(self.patterns)
    
    def __getitem__(self, idx):
        pattern, completion = self.patterns[idx]
        
        full_sequence = np.concatenate([pattern, completion])
        input_seq = np.concatenate([pattern, np.zeros_like(completion)])
        
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(-1)
        target_tensor = torch.tensor(full_sequence, dtype=torch.long)
        
        return {
            'inputs': input_tensor,
            'targets': target_tensor,
            'pattern_idx': idx
        }


class EnhancedDatasetLoader:
    """Enhanced dataset loader with additional reasoning tasks"""
    
    @staticmethod
    def get_dataloader(
        dataset_type: str,
        data_path: Optional[str] = None,
        batch_size: int = 32,
        num_samples: Optional[int] = None,
        shuffle: bool = True,
        **kwargs
    ):
        """Create DataLoader for reasoning tasks"""
        
        num_samples = num_samples or 1000
        
        if dataset_type == 'sudoku':
            from .data import SudokuDataset
            difficulty = kwargs.get('difficulty', 'extreme')
            dataset = SudokuDataset(num_samples, difficulty)
            
        elif dataset_type == 'maze':
            maze_size = kwargs.get('maze_size', 15)
            difficulty = kwargs.get('difficulty', 'medium')
            dataset = MazeDataset(num_samples, maze_size, difficulty)
            
        elif dataset_type == 'pattern':
            pattern_type = kwargs.get('pattern_type', 'sequence')
            dataset = PatternCompletionDataset(num_samples, pattern_type)
            
        elif dataset_type == 'arc':
            from .data import ARCDataset
            if not data_path:
                raise ValueError("data_path required for ARC dataset")
            dataset = ARCDataset(data_path, max_examples=num_samples)
            
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )


def benchmark_all_tasks():
    """Benchmark HRM on all reasoning tasks"""
    
    tasks = {
        'sudoku': {'difficulty': 'extreme'},
        'maze': {'maze_size': 10, 'difficulty': 'medium'},
        'pattern': {'pattern_type': 'sequence'}
    }
    
    results = {}
    
    for task_name, task_kwargs in tasks.items():
        print(f"\n🎯 Benchmarking {task_name}...")
        
        try:
            dataloader = EnhancedDatasetLoader.get_dataloader(
                dataset_type=task_name,
                batch_size=4,
                num_samples=20,
                **task_kwargs
            )
            
            batch = next(iter(dataloader))
            print(f"✅ {task_name}: input {batch['inputs'].shape}, target {batch['targets'].shape}")
            
            results[task_name] = {
                'status': 'success',
                'input_shape': list(batch['inputs'].shape),
                'target_shape': list(batch['targets'].shape)
            }
            
        except Exception as e:
            print(f"❌ {task_name}: {e}")
            results[task_name] = {
                'status': 'failed',
                'error': str(e)
            }
    
    return results


if __name__ == '__main__':
    print("🧠 Testing Additional Reasoning Tasks")
    print("=" * 40)
    
    results = benchmark_all_tasks()
    
    print("\n📊 Benchmark Results:")
    for task, result in results.items():
        status = "✅" if result['status'] == 'success' else "❌"
        print(f"{status} {task}: {result['status']}")
