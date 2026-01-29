"""
SHAP-based Feature Selection Baseline.

This module implements SHAP (SHapley Additive exPlanations) based feature
selection as the SOTA baseline for comparison with metaheuristic algorithms.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class SHAPFeatureSelection:
    """
    SHAP-based feature selection using TreeSHAP with Random Forest.
    
    This implements the baseline approach where features are ranked by
    mean absolute SHAP value, and the top-k features are selected.
    
    Attributes:
        n_features: Number of features in the dataset.
        n_estimators: Number of trees in the Random Forest.
        seed: Random seed for reproducibility.
    """
    
    def __init__(
        self,
        n_features: int,
        X: np.ndarray,
        y: np.ndarray,
        n_estimators: int = 100,
        k_neighbors: int = 5,
        seed: int = None
    ):
        """Initialize SHAP feature selection."""
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP library not installed. Run: pip install shap")
        
        self.n_features = n_features
        self.X = X
        self.y = y
        self.n_estimators = n_estimators
        self.k_neighbors = k_neighbors
        self.seed = seed
        
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)
        
        self.feature_importance = None
        self.feature_ranking = None
        self.best_solution = None
        self.best_fitness = -np.inf
        self.convergence_curve = []
    
    def _compute_shap_importance(self) -> np.ndarray:
        """Compute SHAP feature importance using TreeSHAP."""
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.seed,
            n_jobs=-1
        )
        rf.fit(self.X_scaled, self.y)
        
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(self.X_scaled)
        
        # Handle both binary and multi-class cases
        if isinstance(shap_values, list):
            # For binary classification, use class 1
            importance = np.abs(shap_values[1]).mean(axis=0)
        else:
            importance = np.abs(shap_values).mean(axis=0)
        
        return importance
    
    def _evaluate_subset(self, selected_indices: np.ndarray) -> float:
        """Evaluate a feature subset using k-NN with 5-fold CV."""
        if len(selected_indices) == 0:
            return 0.0
        
        X_selected = self.X_scaled[:, selected_indices]
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in kf.split(X_selected, self.y):
            X_train, X_val = X_selected[train_idx], X_selected[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]
            
            clf = KNeighborsClassifier(n_neighbors=self.k_neighbors)
            clf.fit(X_train, y_train)
            scores.append(clf.score(X_val, y_val))
        
        accuracy = np.mean(scores)
        n_features_ratio = len(selected_indices) / self.n_features
        
        # Same fitness function as metaheuristics
        return 0.99 * accuracy + 0.01 * (1 - n_features_ratio)
    
    def optimize(self, max_iter: int = 50) -> Tuple[np.ndarray, float, List[float]]:
        """
        Run SHAP-based feature selection.
        
        Evaluates all possible top-k subsets (k=1 to n_features) and selects
        the one with maximum fitness. Generates a pseudo-convergence curve
        by incrementally adding features.
        
        Args:
            max_iter: Not used directly, included for API compatibility.
            
        Returns:
            best_solution: Binary mask of selected features.
            best_fitness: Fitness value of best solution.
            convergence_curve: List of fitness values as features are added.
        """
        # Compute SHAP importance
        self.feature_importance = self._compute_shap_importance()
        self.feature_ranking = np.argsort(self.feature_importance)[::-1]
        
        # Evaluate all top-k subsets
        self.convergence_curve = []
        best_k = 1
        
        for k in range(1, self.n_features + 1):
            selected_indices = self.feature_ranking[:k]
            fitness = self._evaluate_subset(selected_indices)
            self.convergence_curve.append(fitness)
            
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                best_k = k
        
        # Create binary mask for best subset
        self.best_solution = np.zeros(self.n_features, dtype=int)
        self.best_solution[self.feature_ranking[:best_k]] = 1
        
        # Extend convergence curve to match max_iter if needed
        if len(self.convergence_curve) < max_iter + 1:
            last_value = self.convergence_curve[-1]
            self.convergence_curve.extend([last_value] * (max_iter + 1 - len(self.convergence_curve)))
        elif len(self.convergence_curve) > max_iter + 1:
            self.convergence_curve = self.convergence_curve[:max_iter + 1]
        
        return self.best_solution, self.best_fitness, self.convergence_curve
    
    def get_selected_features(self) -> np.ndarray:
        """Return indices of selected features."""
        if self.best_solution is None:
            raise ValueError("Run optimize() first")
        return np.where(self.best_solution == 1)[0]
    
    def get_feature_ranking(self) -> np.ndarray:
        """Return feature indices sorted by SHAP importance (descending)."""
        if self.feature_ranking is None:
            raise ValueError("Run optimize() first")
        return self.feature_ranking
    
    def get_shap_values(self) -> np.ndarray:
        """Return the mean absolute SHAP values for each feature."""
        if self.feature_importance is None:
            raise ValueError("Run optimize() first")
        return self.feature_importance
