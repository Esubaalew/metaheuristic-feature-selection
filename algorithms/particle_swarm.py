"""
Particle Swarm Optimization for Binary Feature Selection.

This module implements Binary PSO using a sigmoid transfer function
to convert continuous velocities to binary positions.
"""
import numpy as np
from typing import Callable, List, Tuple


class ParticleSwarmOptimization:
    """
    Binary Particle Swarm Optimization for feature selection.
    
    Attributes:
        n_features: Number of features in the dataset.
        fitness_func: Function that evaluates a binary feature mask.
        n_particles: Number of particles in the swarm.
        max_iter: Maximum number of iterations.
        w: Inertia weight.
        c1: Cognitive coefficient (personal best influence).
        c2: Social coefficient (global best influence).
        v_max: Maximum velocity magnitude.
    """
    
    def __init__(
        self,
        n_features: int,
        fitness_func: Callable,
        n_particles: int = 50,
        max_iter: int = 100,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        v_max: float = 6.0,
        seed: int = None
    ):
        """Initialize Particle Swarm Optimization."""
        self.n_features = n_features
        self.fitness_func = fitness_func
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        
        if seed is not None:
            np.random.seed(seed)
        
        self.convergence_curve = []
        self.best_solution = None
        self.best_fitness = -np.inf
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid transfer function."""
        return 1 / (1 + np.exp(-x))
    
    def _initialize_swarm(self) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize particle positions and velocities."""
        positions = np.random.randint(0, 2, size=(self.n_particles, self.n_features))
        for i in range(self.n_particles):
            if np.sum(positions[i]) == 0:
                positions[i, np.random.randint(0, self.n_features)] = 1
        velocities = np.random.uniform(-self.v_max, self.v_max, 
                                       size=(self.n_particles, self.n_features))
        return positions, velocities
    
    def _evaluate_swarm(self, positions: np.ndarray) -> np.ndarray:
        """Evaluate fitness of all particles."""
        return np.array([self.fitness_func(pos) for pos in positions])
    
    def _update_velocity(
        self,
        velocities: np.ndarray,
        positions: np.ndarray,
        p_best: np.ndarray,
        g_best: np.ndarray
    ) -> np.ndarray:
        """Update particle velocities using PSO equations."""
        r1 = np.random.random(size=(self.n_particles, self.n_features))
        r2 = np.random.random(size=(self.n_particles, self.n_features))
        
        cognitive = self.c1 * r1 * (p_best - positions)
        social = self.c2 * r2 * (g_best - positions)
        
        new_velocities = self.w * velocities + cognitive + social
        return np.clip(new_velocities, -self.v_max, self.v_max)
    
    def _update_position(self, velocities: np.ndarray) -> np.ndarray:
        """Update particle positions using sigmoid transfer function."""
        probabilities = self._sigmoid(velocities)
        random_matrix = np.random.random(size=velocities.shape)
        new_positions = (random_matrix < probabilities).astype(int)
        
        for i in range(self.n_particles):
            if np.sum(new_positions[i]) == 0:
                new_positions[i, np.random.randint(0, self.n_features)] = 1
        return new_positions
    
    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        """
        Run PSO optimization.
        
        Returns:
            best_solution: Binary mask of selected features.
            best_fitness: Fitness value of best solution.
            convergence_curve: List of best fitness values per iteration.
        """
        positions, velocities = self._initialize_swarm()
        fitness = self._evaluate_swarm(positions)
        
        p_best = positions.copy()
        p_best_fitness = fitness.copy()
        
        best_idx = np.argmax(fitness)
        g_best = positions[best_idx].copy()
        self.best_fitness = fitness[best_idx]
        self.best_solution = g_best.copy()
        self.convergence_curve = [self.best_fitness]
        
        for _ in range(self.max_iter):
            velocities = self._update_velocity(velocities, positions, p_best, g_best)
            positions = self._update_position(velocities)
            fitness = self._evaluate_swarm(positions)
            
            improved = fitness > p_best_fitness
            p_best[improved] = positions[improved].copy()
            p_best_fitness[improved] = fitness[improved]
            
            best_idx = np.argmax(p_best_fitness)
            if p_best_fitness[best_idx] > self.best_fitness:
                g_best = p_best[best_idx].copy()
                self.best_fitness = p_best_fitness[best_idx]
                self.best_solution = g_best.copy()
            
            self.convergence_curve.append(self.best_fitness)
        
        return self.best_solution, self.best_fitness, self.convergence_curve
    
    def get_selected_features(self) -> np.ndarray:
        """Return indices of selected features."""
        if self.best_solution is None:
            raise ValueError("Run optimize() first")
        return np.where(self.best_solution == 1)[0]
