"""
Rigorous Experimental Evaluation of Metaheuristic Feature Selection.

This module implements a rigorous experimental framework comparing three
metaheuristic optimization algorithms (GA, PSO, SA) against SHAP baseline
for feature selection on the Breast Cancer Wisconsin dataset.

Usage:
    python experiment_rigorous.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
import os

from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.particle_swarm import ParticleSwarmOptimization
from algorithms.simulated_annealing import SimulatedAnnealing
from shap_baseline import SHAPFeatureSelector, SHAP_AVAILABLE

warnings.filterwarnings('ignore')


def create_fitness_function(X: np.ndarray, y: np.ndarray, k_neighbors: int = 5) -> callable:
    """
    Create a fitness function for feature selection.
    
    The fitness function evaluates a binary feature mask by training a 
    k-NN classifier on the selected features and computing accuracy via
    5-fold stratified cross-validation.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Target vector of shape (n_samples,).
        k_neighbors: Number of neighbors for k-NN classifier.
        
    Returns:
        Callable that takes a binary mask and returns fitness score.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    def fitness(solution: np.ndarray) -> float:
        selected = np.where(solution == 1)[0]
        if len(selected) == 0:
            return 0.0
        
        X_selected = X_scaled[:, selected]
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in kf.split(X_selected, y):
            X_train, X_val = X_selected[train_idx], X_selected[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            clf = KNeighborsClassifier(n_neighbors=k_neighbors)
            clf.fit(X_train, y_train)
            scores.append(clf.score(X_val, y_val))
        
        accuracy = np.mean(scores)
        n_features_ratio = len(selected) / len(solution)
        
        return 0.99 * accuracy + 0.01 * (1 - n_features_ratio)
    
    return fitness


def run_single_experiment(
    algorithm_class,
    n_features: int,
    fitness_func: callable,
    seed: int,
    max_iter: int = 50,
    pop_size: int = 30
) -> dict:
    """
    Run a single optimization experiment.
    
    Args:
        algorithm_class: The metaheuristic algorithm class.
        n_features: Number of features in the dataset.
        fitness_func: Fitness evaluation function.
        seed: Random seed for reproducibility.
        max_iter: Maximum iterations.
        pop_size: Population/swarm size.
        
    Returns:
        Dictionary with best_fitness, n_selected, selected_features, convergence.
    """
    if algorithm_class == GeneticAlgorithm:
        optimizer = algorithm_class(
            n_features=n_features,
            fitness_func=fitness_func,
            pop_size=pop_size,
            max_iter=max_iter,
            crossover_rate=0.8,
            mutation_rate=0.1,
            elitism=2,
            seed=seed
        )
    elif algorithm_class == ParticleSwarmOptimization:
        optimizer = algorithm_class(
            n_features=n_features,
            fitness_func=fitness_func,
            n_particles=pop_size,
            max_iter=max_iter,
            w=0.7,
            c1=1.5,
            c2=1.5,
            seed=seed
        )
    else:  # SimulatedAnnealing
        optimizer = algorithm_class(
            n_features=n_features,
            fitness_func=fitness_func,
            max_iter=max_iter,
            initial_temp=100.0,
            cooling_rate=0.95,
            n_neighbors=pop_size,
            seed=seed
        )
    
    best_solution, best_fitness, convergence = optimizer.optimize()
    
    return {
        'best_fitness': best_fitness,
        'n_selected': int(np.sum(best_solution)),
        'selected_features': optimizer.get_selected_features(),
        'convergence': convergence
    }


def run_experiment_batch(
    algorithms: dict,
    n_features: int,
    fitness_func: callable,
    n_runs: int = 30,
    max_iter: int = 50,
    pop_size: int = 30
) -> dict:
    """
    Run multiple independent experiments for each algorithm.
    
    Args:
        algorithms: Dictionary mapping algorithm names to classes.
        n_features: Number of features in the dataset.
        fitness_func: Fitness evaluation function.
        n_runs: Number of independent runs per algorithm.
        max_iter: Maximum iterations per run.
        pop_size: Population/swarm size.
        
    Returns:
        Dictionary with results for each algorithm.
    """
    results = {name: {'fitness': [], 'n_selected': [], 'convergences': []} 
               for name in algorithms}
    
    for name, algo_class in algorithms.items():
        print(f"Running {name}...")
        for run in range(n_runs):
            result = run_single_experiment(
                algo_class, n_features, fitness_func,
                seed=run * 100 + 42,
                max_iter=max_iter,
                pop_size=pop_size
            )
            results[name]['fitness'].append(result['best_fitness'])
            results[name]['n_selected'].append(result['n_selected'])
            results[name]['convergences'].append(result['convergence'])
            
            if (run + 1) % 10 == 0:
                print(f"  Completed {run + 1}/{n_runs} runs")
    
    return results


def compute_statistics(results: dict) -> pd.DataFrame:
    """
    Compute descriptive statistics for each algorithm.
    
    Args:
        results: Dictionary with experimental results.
        
    Returns:
        DataFrame with mean, std, min, max for fitness and n_selected.
    """
    stats_data = []
    
    for name, data in results.items():
        fitness = np.array(data['fitness'])
        n_selected = np.array(data['n_selected'])
        
        stats_data.append({
            'Algorithm': name,
            'Fitness Mean': np.mean(fitness),
            'Fitness Std': np.std(fitness),
            'Fitness Min': np.min(fitness),
            'Fitness Max': np.max(fitness),
            'Features Mean': np.mean(n_selected),
            'Features Std': np.std(n_selected)
        })
    
    return pd.DataFrame(stats_data)


def perform_wilcoxon_tests(results: dict) -> pd.DataFrame:
    """
    Perform pairwise Wilcoxon rank-sum tests (Mann-Whitney U).
    
    Tests for statistically significant differences between algorithm pairs.
    
    Args:
        results: Dictionary with experimental results.
        
    Returns:
        DataFrame with algorithm pairs, test statistics, and p-values.
    """
    algorithms = list(results.keys())
    comparisons = []
    
    for i in range(len(algorithms)):
        for j in range(i + 1, len(algorithms)):
            alg1, alg2 = algorithms[i], algorithms[j]
            fitness1 = np.array(results[alg1]['fitness'])
            fitness2 = np.array(results[alg2]['fitness'])
            
            stat, p_value = stats.mannwhitneyu(fitness1, fitness2, alternative='two-sided')
            
            comparisons.append({
                'Comparison': f'{alg1} vs {alg2}',
                'Statistic': stat,
                'P-Value': p_value,
                'Significant': p_value < 0.05
            })
    
    return pd.DataFrame(comparisons)


def plot_convergence(results: dict, output_path: str):
    """
    Plot mean convergence curves with standard deviation bands.
    
    Args:
        results: Dictionary with experimental results.
        output_path: Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    colors = {'GA': '#2E86AB', 'PSO': '#E94F37', 'SA': '#44AF69', 'SHAP': '#9B59B6'}
    
    for name, data in results.items():
        convergences = np.array(data['convergences'])
        mean_conv = np.mean(convergences, axis=0)
        std_conv = np.std(convergences, axis=0)
        iterations = np.arange(len(mean_conv))
        
        linestyle = '--' if name == 'SHAP' else '-'
        plt.plot(iterations, mean_conv, label=name, color=colors.get(name, 'gray'), 
                 linewidth=2, linestyle=linestyle)
        plt.fill_between(iterations, mean_conv - std_conv, mean_conv + std_conv,
                        alpha=0.2, color=colors.get(name, 'gray'))
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Best Fitness', fontsize=12)
    plt.title('Convergence Comparison (Mean ± Std over 30 runs)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved convergence plot to {output_path}")


def plot_accuracy_comparison(results: dict, output_path: str):
    """
    Plot box plots of accuracy distribution for each algorithm.
    
    Args:
        results: Dictionary with experimental results.
        output_path: Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    colors = {'GA': '#2E86AB', 'PSO': '#E94F37', 'SA': '#44AF69', 'SHAP': '#9B59B6'}
    
    data = [results[name]['fitness'] for name in results]
    names = list(results.keys())
    
    bp = plt.boxplot(data, labels=names, patch_artist=True)
    
    for patch, name in zip(bp['boxes'], names):
        patch.set_facecolor(colors.get(name, 'gray'))
        patch.set_alpha(0.7)
    
    plt.ylabel('Fitness Score', fontsize=12)
    plt.title('Fitness Distribution Comparison (30 runs)', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy comparison to {output_path}")


def run_shap_experiment(X: np.ndarray, y: np.ndarray, n_runs: int = 30) -> dict:
    """
    Run SHAP baseline experiments.
    
    Args:
        X: Feature matrix.
        y: Target vector.
        n_runs: Number of independent runs.
        
    Returns:
        Dictionary with fitness scores and feature counts.
    """
    results = {'fitness': [], 'n_selected': [], 'convergences': []}
    
    print("Running SHAP...")
    for run in range(n_runs):
        selector = SHAPFeatureSelector(
            n_features=X.shape[1],
            random_state=run * 100 + 42
        )
        
        mask, accuracy, k = selector.select_features(X, y, cv_folds=5)
        
        n_features_ratio = k / X.shape[1]
        fitness = 0.99 * accuracy + 0.01 * (1 - n_features_ratio)
        
        results['fitness'].append(fitness)
        results['n_selected'].append(k)
        results['convergences'].append([fitness] * 51)
        
        if (run + 1) % 10 == 0:
            print(f"  Completed {run + 1}/{n_runs} runs")
    
    return results


def main():
    """Execute the complete experimental evaluation."""
    print("=" * 60)
    print("Metaheuristic Feature Selection Experiment")
    print("=" * 60)
    
    os.makedirs('results', exist_ok=True)
    os.makedirs('paper', exist_ok=True)
    
    print("\nLoading Breast Cancer Wisconsin dataset...")
    data = load_breast_cancer()
    X, y = data.data, data.target
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {np.bincount(y)}")
    
    fitness_func = create_fitness_function(X, y)
    
    algorithms = {
        'GA': GeneticAlgorithm,
        'PSO': ParticleSwarmOptimization,
        'SA': SimulatedAnnealing
    }
    
    print("\nRunning experiments (30 independent runs per algorithm)...")
    results = run_experiment_batch(
        algorithms=algorithms,
        n_features=X.shape[1],
        fitness_func=fitness_func,
        n_runs=30,
        max_iter=50,
        pop_size=30
    )
    
    if SHAP_AVAILABLE:
        shap_results = run_shap_experiment(X, y, n_runs=30)
        results['SHAP'] = shap_results
    else:
        print("\nWARNING: SHAP not available. Install with: pip install shap")
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    stats_df = compute_statistics(results)
    print("\nDescriptive Statistics:")
    print(stats_df.to_string(index=False))
    stats_df.to_csv('results/summary_table.csv', index=False)
    
    wilcoxon_df = perform_wilcoxon_tests(results)
    print("\nWilcoxon Rank-Sum Tests (Mann-Whitney U):")
    print(wilcoxon_df.to_string(index=False))
    wilcoxon_df.to_csv('results/wilcoxon_test.csv', index=False)
    
    print("\nGenerating plots...")
    plot_convergence(results, 'results/convergence_comparison.png')
    plot_convergence(results, 'paper/convergence_comparison.png')
    plot_accuracy_comparison(results, 'results/accuracy_comparison.png')
    plot_accuracy_comparison(results, 'paper/accuracy_comparison.png')
    
    print("\n" + "=" * 60)
    print("Experiment completed!")
    print("Results saved to: results/")
    print("Paper figures saved to: paper/")
    print("=" * 60)


if __name__ == '__main__':
    main()
