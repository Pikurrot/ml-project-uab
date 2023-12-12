from sklearn.naive_bayes import GaussianNB
from src.base import Base
import numpy as np
import pandas as pd

class GNaiveBayes(GaussianNB, Base):
    def __init__(self, 
                random_state: int = 42,
                priors: list = None,
                var_smoothing: float = 1e-9):
        """
        Gaussian Naive Bayes model.
        """
        self.priors = priors
        self.var_smoothing = var_smoothing

        GaussianNB.__init__(
            self,
            priors = self.priors,
            var_smoothing = self.var_smoothing)

        Base.__init__(
            self,
            name = "Gaussian Naive Bayes",
            random_state = random_state)
        
    def calculate_permutation_importance(self, X_train, y_train, random_state=None):
        """
        Calculate feature importance using permutation importance with Gaussian Naive Bayes model.

        Parameters:
        X_train : array-like or sparse matrix, shape (n_samples, n_features)
            The input training samples.
        y_train : array-like, shape (n_samples,)
            The target training values.
        random_state : int or None, optional (default=None)
            Seed for the random number generator.

        Returns:
        feature_importance : array-like, shape (n_features,)
            Feature importances based on permutation importance.
        """
        self.fit(X_train, y_train)
        base_score = self.score(X_train, y_train)
        feature_importance = np.zeros(X_train.shape[1])

        rng = np.random.default_rng(random_state)

        for col in X_train.columns:
            X_permuted = X_train.copy().to_numpy()
            col_index = X_train.columns.get_loc(col)
            permuted_col = rng.permutation(X_permuted[:, col_index])
            X_permuted[:, col_index] = permuted_col
            permuted_score = self.score(X_permuted, y_train)
            feature_importance[col_index] = base_score - permuted_score

        return feature_importance
    
