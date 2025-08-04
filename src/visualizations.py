"""
HRM Visualization Tools

Visualize hierarchical states, computation patterns, and training dynamics.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path

# Set style for better plots
plt.style.use('default')


class HRMVisualizer:
    """Visualization tools for HRM analysis"""
    
    def __init__(self, save_dir: str = "./visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
    
    def plot_sudoku_solving(
        self,
        puzzle: np.ndarray,
        solution: np.ndarray,
        prediction: Optional[np.ndarray] = None,
        save_name: str = "sudoku_example"
    ):
        """Visualize Sudoku puzzle, solution, and prediction"""
        
        n_plots = 3 if prediction is not None else 2
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
        
        if n_plots == 2:
            axes = [axes[0], axes[1]]
        
        # Plot puzzle
        self._plot_sudoku_grid(axes[0], puzzle, "Input Puzzle")
        
        # Plot solution
        self._plot_sudoku_grid(axes[1], solution, "Target Solution")
        
        # Plot prediction if available
        if prediction is not None:
            accuracy = np.mean(prediction == solution)
            self._plot_sudoku_grid(axes[2], prediction, f"Prediction (Acc: {accuracy:.1%})")
        
        plt.tight_layout()
        save_path = self.save_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"🧩 Sudoku visualization saved: {save_path}")
    
    def _plot_sudoku_grid(self, ax, grid: np.ndarray, title: str):
        """Plot a single Sudoku grid"""
        if grid.shape == (81,):
            grid = grid.reshape(9, 9)
        
        # Create heatmap
        ax.imshow(grid, cmap='Blues', vmin=0, vmax=9)
        
        # Add numbers
        for i in range(9):
            for j in range(9):
                value = int(grid[i, j])
                if value > 0:
                    text_color = 'white' if value > 5 else 'black'
                    ax.text(j, i, str(value), ha='center', va='center', 
                           fontsize=12, fontweight='bold', color=text_color)
        
        # Add grid lines
        for i in range(10):
            lw = 2 if i % 3 == 0 else 1
            ax.axhline(i-0.5, color='black', linewidth=lw)
            ax.axvline(i-0.5, color='black', linewidth=lw)
        
        ax.set_title(title, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
    
    def plot_maze_solving(
        self,
        maze: np.ndarray,
        solution: np.ndarray,
        save_name: str = "maze_example"
    ):
        """Visualize maze and solution path"""
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original maze
        maze_size = int(np.sqrt(len(maze)))
        maze_2d = maze.reshape(maze_size, maze_size)
        solution_2d = solution.reshape(maze_size, maze_size)
        
        # Plot maze
        axes[0].imshow(maze_2d, cmap='gray')
        axes[0].set_title("Maze")
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        
        # Plot solution
        solution_display = maze_2d.copy().astype(float)
        solution_display[solution_2d == 2] = 0.5  # Path in gray
        
        axes[1].imshow(solution_display, cmap='RdYlBu')
        axes[1].set_title("Solution Path")
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        
        plt.tight_layout()
        save_path = self.save_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"🔍 Maze visualization saved: {save_path}")
    
    def plot_training_dynamics(
        self,
        metrics: List[Dict],
        save_name: str = "training_dynamics"
    ):
        """Plot training dynamics from metrics list"""
        
        if not metrics:
            print("⚠️  No metrics to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Training Dynamics', fontsize=16)
        
        epochs = [m['epoch'] for m in metrics]
        losses = [m['total_loss'] for m in metrics]
        accuracies = [m.get('eval_accuracy', 0) for m in metrics]
        avg_steps = [m.get('avg_steps', 0) for m in metrics]
        
        # Training loss
        axes[0, 0].plot(epochs, losses, marker='o', color='red')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(epochs, accuracies, marker='s', color='green')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Computation steps
        axes[1, 0].plot(epochs, avg_steps, marker='^', color='purple')
        axes[1, 0].set_title('Average Computation Steps')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary stats
        axes[1, 1].text(0.1, 0.8, f"Final Accuracy: {accuracies[-1]:.1%}", 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.6, f"Final Loss: {losses[-1]:.4f}", 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.4, f"Avg Steps: {avg_steps[-1]:.1f}", 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        save_path = self.save_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Training dynamics saved: {save_path}")


def create_visualization_demo():
    """Create demo visualizations"""
    print("🎨 Creating visualization demos...")
    
    visualizer = HRMVisualizer()
    
    # Demo Sudoku puzzle
    puzzle = np.array([
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0], 
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ])
    
    solution = np.array([
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9]
    ])
    
    visualizer.plot_sudoku_solving(puzzle, solution)
    
    # Demo maze
    maze = np.random.choice([0, 1], size=100, p=[0.7, 0.3])  # 10x10 maze
    solution = maze.copy()
    solution[0] = 2  # Start
    solution[99] = 2  # Goal
    solution[10:20] = 2  # Simple path
    
    visualizer.plot_maze_solving(maze, solution)
    
    print("✅ Demo visualizations created!")


if __name__ == '__main__':
    create_visualization_demo()
