# Metaheuristic Feature Selection: A Comparative Study

A comparative analysis of Genetic Algorithm (GA), Particle Swarm Optimization (PSO), and Simulated Annealing (SA) against a SHAP baseline for feature selection on the Breast Cancer Wisconsin dataset.

## Project Structure

```
term/
├── algorithms/
│   ├── __init__.py
│   ├── genetic_algorithm.py     # GA implementation
│   ├── particle_swarm.py        # Binary PSO implementation
│   └── simulated_annealing.py   # SA implementation
├── experiment_rigorous.py       # Main experiment script
├── shap_baseline.py             # SHAP baseline implementation
├── results/                     # Generated results
│   ├── summary_table.csv
│   ├── wilcoxon_test.csv
│   ├── convergence_comparison.png
│   └── accuracy_comparison.png
├── paper/
│   ├── metaheuristic_feature_selection.tex   # LaTeX paper
│   ├── metaheuristic_feature_selection.pdf   # Compiled PDF
│   └── *.png                                 # Figures
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Running the Experiment

```bash
python experiment_rigorous.py
```

This executes 30 independent runs per method (GA, PSO, SA, SHAP) and generates:
- Descriptive statistics (mean, std, min, max)
- Wilcoxon signed-rank tests for statistical significance
- Convergence and accuracy comparison plots

## Methodology

### Dataset
Breast Cancer Wisconsin (569 samples, 30 features, binary classification)

### Algorithms
- **GA**: Tournament selection, two-point crossover, bit-flip mutation
- **PSO**: Binary PSO with sigmoid transfer function
- **SA**: Exponential cooling with multi-bit flip neighborhood
- **SHAP**: TreeSHAP feature ranking with top-k selection

### Evaluation
- Fitness: 5-fold stratified cross-validation with k-NN (k=5)
- Fitness function: 0.99 × accuracy + 0.01 × (1 - feature_ratio)
- Statistical testing: Wilcoxon rank-sum test (Mann-Whitney U, α = 0.05)

## Results

Metaheuristics achieve fitness above 97\% while SHAP is lower. GA provides the most consistent results across runs. See `results/` for detailed outputs.
