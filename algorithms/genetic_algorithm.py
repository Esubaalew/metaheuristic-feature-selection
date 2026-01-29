"""
Genetic Algorithm for Binary Feature Selection.

This module implements a standard Genetic Algorithm with binary encoding
for the feature selection problem. The algorithm uses tournament selection,
two-point crossover, and bit-flip mutation.
"""
import numpy as np
from typing import Callable, List, Tuple


class GeneticAlgorithm:
    """
    Genetic Algorithm for binary feature selection.
    
    Attributes:
        n_features: Number of features in the dataset.
        fitness_func: Function that evaluates a binary feature mask.
        pop_size: Population size.
        max_iter: Maximum number of generations.
        crossover_rate: Probability of crossover between parents.
        mutation_rate: Probability of mutation per gene.
        elitism: Number of best individuals preserved each generation.
        tournament_size: Number of individuals in tournament selection.
    """
    
    def __init__(
        self,
        n_features: int,
        fitness_func: Callable,
        pop_size: int = 50,
        max_iter: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elitism: int = 2,
        tournament_size: int = 3,
        seed: int = None
    ):
        """Initialize the Genetic Algorithm."""
        self.n_features = n_features
        self.fitness_func = fitness_func
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.tournament_size = tournament_size
        
        if seed is not None:
            np.random.seed(seed)
        
        self.convergence_curve = []
        self.best_solution = None
        self.best_fitness = -np.inf
    
    def _initialize_population(self) -> np.ndarray:
        """Initialize random binary population ensuring at least one feature selected."""
        population = np.random.randint(0, 2, size=(self.pop_size, self.n_features))
        for i in range(self.pop_size):
            if np.sum(population[i]) == 0:
                population[i, np.random.randint(0, self.n_features)] = 1
        return population
    
    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """Evaluate fitness of all individuals in the population."""
        return np.array([self.fitness_func(ind) for ind in population])
    
    def _tournament_selection(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """Select parent using tournament selection."""
        indices = np.random.randint(0, self.pop_size, size=self.tournament_size)
        best_idx = indices[np.argmax(fitness[indices])]
        return population[best_idx].copy()
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform two-point crossover between parents."""
        if np.random.random() < self.crossover_rate:
            points = sorted(np.random.choice(self.n_features, 2, replace=False))
            child1, child2 = parent1.copy(), parent2.copy()
            child1[points[0]:points[1]] = parent2[points[0]:points[1]]
            child2[points[0]:points[1]] = parent1[points[0]:points[1]]
            return child1, child2
        return parent1.copy(), parent2.copy()
    
    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """Apply bit-flip mutation to an individual."""
        mutant = individual.copy()
        for i in range(self.n_features):
            if np.random.random() < self.mutation_rate:
                mutant[i] = 1 - mutant[i]
        if np.sum(mutant) == 0:
            mutant[np.random.randint(0, self.n_features)] = 1
        return mutant
    
    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        """
        Run the genetic algorithm optimization.
        
        Returns:
            best_solution: Binary mask of selected features.
            best_fitness: Fitness value of best solution.
            convergence_curve: List of best fitness values per generation.
        """
        population = self._initialize_population()
        fitness = self._evaluate_population(population)
        
        best_idx = np.argmax(fitness)
        self.best_solution = population[best_idx].copy()
        self.best_fitness = fitness[best_idx]
        self.convergence_curve = [self.best_fitness]
        
        for _ in range(self.max_iter):
            sorted_indices = np.argsort(fitness)[::-1]
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]
            
            new_population = [population[i].copy() for i in range(self.elitism)]
            
            while len(new_population) < self.pop_size:
                parent1 = self._tournament_selection(population, fitness)
                parent2 = self._tournament_selection(population, fitness)
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                new_population.append(child1)
                if len(new_population) < self.pop_size:
                    new_population.append(child2)
            
            population = np.array(new_population)
            fitness = self._evaluate_population(population)
            
            best_idx = np.argmax(fitness)
            if fitness[best_idx] > self.best_fitness:
                self.best_solution = population[best_idx].copy()
                self.best_fitness = fitness[best_idx]
            
            self.convergence_curve.append(self.best_fitness)
        
        return self.best_solution, self.best_fitness, self.convergence_curve
    
    def get_selected_features(self) -> np.ndarray:
        """Return indices of selected features."""
        if self.best_solution is None:
            raise ValueError("Run optimize() first")
        return np.where(self.best_solution == 1)[0]
