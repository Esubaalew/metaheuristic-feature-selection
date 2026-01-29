"""
Simulated Annealing for Binary Feature Selection.

This module implements Simulated Annealing with exponential cooling
schedule and bit-flip neighborhood generation.
"""
import numpy as np
from typing import Callable, List, Tuple


class SimulatedAnnealing:
    """
    Simulated Annealing for binary feature selection.
    
    Attributes:
        n_features: Number of features in the dataset.
        fitness_func: Function that evaluates a binary feature mask.
        max_iter: Maximum number of iterations.
        initial_temp: Starting temperature.
        final_temp: Ending temperature.
        cooling_rate: Temperature decay factor.
        n_neighbors: Number of neighbors to evaluate per iteration.
    """
    
    def __init__(
        self,
        n_features: int,
        fitness_func: Callable,
        max_iter: int = 100,
        initial_temp: float = 100.0,
        final_temp: float = 0.01,
        cooling_rate: float = 0.95,
        n_neighbors: int = 10,
        seed: int = None
    ):
        """Initialize Simulated Annealing."""
        self.n_features = n_features
        self.fitness_func = fitness_func
        self.max_iter = max_iter
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.n_neighbors = n_neighbors
        
        if seed is not None:
            np.random.seed(seed)
        
        self.convergence_curve = []
        self.best_solution = None
        self.best_fitness = -np.inf
    
    def _initialize_solution(self) -> np.ndarray:
        """Initialize random binary solution."""
        solution = np.random.randint(0, 2, size=self.n_features)
        if np.sum(solution) == 0:
            solution[np.random.randint(0, self.n_features)] = 1
        return solution
    
    def _get_neighbor(self, solution: np.ndarray) -> np.ndarray:
        """Generate neighbor by flipping 1-3 random bits."""
        neighbor = solution.copy()
        n_flips = np.random.randint(1, min(4, self.n_features + 1))
        flip_indices = np.random.choice(self.n_features, n_flips, replace=False)
        
        for idx in flip_indices:
            neighbor[idx] = 1 - neighbor[idx]
        
        if np.sum(neighbor) == 0:
            neighbor[np.random.randint(0, self.n_features)] = 1
        return neighbor
    
    def _acceptance_probability(self, current_fitness: float, neighbor_fitness: float, temperature: float) -> float:
        """Calculate acceptance probability using Metropolis criterion."""
        if neighbor_fitness > current_fitness:
            return 1.0
        return np.exp((neighbor_fitness - current_fitness) / temperature)
    
    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        """
        Run Simulated Annealing optimization.
        
        Returns:
            best_solution: Binary mask of selected features.
            best_fitness: Fitness value of best solution.
            convergence_curve: List of best fitness values per iteration.
        """
        current_solution = self._initialize_solution()
        current_fitness = self.fitness_func(current_solution)
        
        self.best_solution = current_solution.copy()
        self.best_fitness = current_fitness
        self.convergence_curve = [self.best_fitness]
        
        temperature = self.initial_temp
        
        for _ in range(self.max_iter):
            for _ in range(self.n_neighbors):
                neighbor = self._get_neighbor(current_solution)
                neighbor_fitness = self.fitness_func(neighbor)
                
                if np.random.random() < self._acceptance_probability(current_fitness, neighbor_fitness, temperature):
                    current_solution = neighbor
                    current_fitness = neighbor_fitness
                    
                    if current_fitness > self.best_fitness:
                        self.best_solution = current_solution.copy()
                        self.best_fitness = current_fitness
            
            temperature = max(self.final_temp, temperature * self.cooling_rate)
            self.convergence_curve.append(self.best_fitness)
        
        return self.best_solution, self.best_fitness, self.convergence_curve
    
    def get_selected_features(self) -> np.ndarray:
        """Return indices of selected features."""
        if self.best_solution is None:
            raise ValueError("Run optimize() first")
        return np.where(self.best_solution == 1)[0]
