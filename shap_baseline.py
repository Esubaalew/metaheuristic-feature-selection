"""
SHAP Baseline for Feature Selection.

Implements SHAP-based feature selection using TreeSHAP for comparison
with metaheuristic approaches.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class SHAPFeatureSelector:
    """
    SHAP-based feature selection using TreeSHAP importance ranking.
    
    Attributes:
        n_features: Number of features in the dataset.
        n_estimators: Number of trees in Random Forest.
        random_state: Random seed for reproducibility.
    """
    
    def __init__(
        self,
        n_features: int,
        n_estimators: int = 100,
        random_state: int = 42
    ):
        """Initialize SHAP Feature Selector."""
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP package required. Install with: pip install shap")
        
        self.n_features = n_features
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.feature_importance = None
        self.best_k = None
        self.best_accuracy = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SHAPFeatureSelector':
        """
        Compute SHAP values and rank features by importance.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target vector of shape (n_samples,).
            
        Returns:
            self
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=8,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X_scaled, y)
        
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_scaled)
        
        if isinstance(shap_values, list):
            shap_values = np.abs(shap_values[1])
        elif len(shap_values.shape) == 3:
            shap_values = np.abs(shap_values[:, :, 1])
        else:
            shap_values = np.abs(shap_values)
        
        importance = np.mean(shap_values, axis=0)
        if len(importance.shape) > 1:
            importance = importance.flatten()
        self.feature_importance = importance
        return self
    
    def select_features(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        cv_folds: int = 5
    ) -> Tuple[np.ndarray, float, int]:
        """
        Find optimal number of top-k features via cross-validation.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            cv_folds: Number of cross-validation folds.
            
        Returns:
            selected_mask: Binary mask of selected features.
            best_accuracy: CV accuracy with selected features.
            best_k: Number of features selected.
        """
        if self.feature_importance is None:
            self.fit(X, y)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        ranked_indices = np.argsort(self.feature_importance)[::-1]
        ranked_indices = np.asarray(ranked_indices).flatten()
        
        best_accuracy = 0
        best_k = 1
        
        for k in range(1, self.n_features + 1):
            selected_indices = ranked_indices[:k].astype(int)
            X_selected = X_scaled[:, selected_indices]
            
            if len(X_selected.shape) != 2:
                X_selected = X_selected.reshape(X_scaled.shape[0], -1)
            
            rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=8,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scores = cross_val_score(rf, X_selected, y, cv=cv, scoring='accuracy')
            accuracy = np.mean(scores)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k
        
        selected_mask = np.zeros(self.n_features, dtype=int)
        selected_mask[ranked_indices[:best_k].astype(int)] = 1
        
        self.best_k = best_k
        self.best_accuracy = best_accuracy
        
        return selected_mask, best_accuracy, best_k
    
    def get_selected_features(self) -> np.ndarray:
        """Return indices of selected features."""
        if self.feature_importance is None:
            raise ValueError("Run fit() and select_features() first")
        ranked_indices = np.argsort(self.feature_importance)[::-1]
        return ranked_indices[:self.best_k]


def run_shap_baseline(
    X: np.ndarray,
    y: np.ndarray,
    n_runs: int = 30,
    cv_folds: int = 5
) -> dict:
    """
    Run SHAP baseline multiple times for comparison.
    
    Args:
        X: Feature matrix.
        y: Target vector.
        n_runs: Number of independent runs.
        cv_folds: Cross-validation folds.
        
    Returns:
        Dictionary with fitness scores and feature counts.
    """
    results = {
        'fitness': [],
        'n_selected': [],
        'accuracies': []
    }
    
    for run in range(n_runs):
        selector = SHAPFeatureSelector(
            n_features=X.shape[1],
            random_state=run * 100 + 42
        )
        
        mask, accuracy, k = selector.select_features(X, y, cv_folds)
        
        n_features_ratio = k / X.shape[1]
        fitness = 0.99 * accuracy + 0.01 * (1 - n_features_ratio)
        
        results['fitness'].append(fitness)
        results['n_selected'].append(k)
        results['accuracies'].append(accuracy)
    
    return results
