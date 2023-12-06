from sklearn.neighbors import KNeighborsClassifier
from src.base import Base
import numpy as np

class KNN_model(KNeighborsClassifier, Base):
    def __init__(self,
                n_neighbors: int = 5, # hyperparameter to tune
                weights: str = "uniform",
                algorithm: str = "auto",
                leaf_size: int = 30, # TODO: check if changing it can improve computation time
                p: int = 2, # tipically use euclidean metric, try manhattan since we have somewhat high dimensional data
                metric: str = "minkowski",
                metric_params: dict = None,
                n_jobs: int = -1): # set to -1 to use all processors and speed up computation
        """
        KNN model
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        # KNN
        KNeighborsClassifier.__init__(
            self,
            n_neighbors = self.n_neighbors,
            weights = self.weights,
            algorithm = self.algorithm,
            leaf_size = self.leaf_size,
            p = self.p,
            metric = self.metric,
            metric_params = self.metric_params,
            n_jobs = self.n_jobs)
        # Base
        Base.__init__(
            self,
            name = "K-Nearest Neighbors",
            random_state = None) # No random state for KNN