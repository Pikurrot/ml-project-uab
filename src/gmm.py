from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from multiprocessing import Pool
import numpy as np
from src.base import Base

class GMM(Base):
	def __init__(self,
				random_state: int = 42,
				n_models: int = 5,
				n_components: int = 1,
				covariance_type: str = "full",
				tol: float = 1e-3,
				max_iter: int = 100,
				n_init: int = 1,
				init_params: str = "kmeans"):
		"""
		Gaussian Mixture Model.
		"""
		
		self.n_models = n_models
		self.n_components = n_components
		self.covariance_type = covariance_type
		self.tol = tol
		self.max_iter = max_iter
		self.n_init = n_init
		self.init_params = init_params

		self.models = []
		
		Base.__init__(
			self,
			name = "Gaussian Mixture Model",
			random_state = random_state)

	def predict(self, X: np.ndarray) -> np.ndarray:
		"""
		Predict class labels for samples in X.

		## Parameters
		X: np.ndarray of shape (n_samples, n_features)

		## Returns
		y_pred: np.ndarray of shape (n_samples,)
		"""
		return np.argmax(np.stack([model.score_samples(X) for model in self.models]), axis=0)
	
	def predict_proba(self, X: np.ndarray) -> np.ndarray:
		"""
		Predict posterior probability of each component given the data.

		## Parameters
		X: np.ndarray of shape (n_samples, n_features)

		## Returns
		y_pred: np.ndarray of shape (n_samples, n_components)
		"""
		probabilities = np.zeros((X.shape[0], self.n_models))

		for idx, model in enumerate(self.models):
			probabilities[:, idx] = np.exp(model.score_samples(X))

		# Normalize probabilities so they sum to 1 for each sample
		probabilities /= probabilities.sum(axis=1, keepdims=True)

		return probabilities

	def fit(self, X: np.ndarray, y: np.ndarray = None):
		"""
		Estimate models parameters with the EM algorithm.

		Asumes target classes are 0, 1, ..., n_models - 1.

		## Parameters
		X: np.ndarray of shape (n_samples, n_features)
		y: np.ndarray of shape (n_samples,)
		"""
		self.models = []  # Clear self.models before creating and training new models
		for target in range(self.n_models):
			model = GaussianMixture(
				n_components=self.n_components,
				covariance_type=self.covariance_type,
				tol=self.tol,
				max_iter=self.max_iter,
				n_init=self.n_init,
				init_params=self.init_params,
				random_state=self.random_state
			)
			model.fit(X[y == target])
			self.models.append(model)

	def score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
		"""
		Compute the accuracy of the model.

		## Parameters
		y_true: np.ndarray of shape (n_samples,). True target values.
		y_pred: np.ndarray of shape (n_samples,). Predicted target values.

		## Returns
		accuracy: float
		"""
		return accuracy_score(y_true, y_pred)
	
	def _cross_validate_fold(self, X, y, model_copy, train_index, test_index):
		X_train, X_test = X.iloc[train_index], X.iloc[test_index]
		y_train, y_test = y.iloc[train_index], y.iloc[test_index]
		model_copy.fit(X_train, y_train)
		y_pred = model_copy.predict(X_test)
		return accuracy_score(y_test, y_pred)

	def cross_validation(self, X: np.ndarray, y: np.ndarray, n_splits: int, val_size: float):
		"""
		Compute the cross validation accuracy of the model.

		## Parameters
		X: np.ndarray of shape (n_samples, n_features)
		y: np.ndarray of shape (n_samples,)
		n_splits: int. Number of folds.
		val_size: float. Proportion of the dataset to include in the validation set.
		"""
		sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=val_size, random_state=self.random_state)

		model_copy = GMM(
			n_models = self.n_models,
			n_components = self.n_components,
			covariance_type = self.covariance_type,
			tol = self.tol,
			max_iter = self.max_iter,
			n_init = self.n_init,
			init_params = self.init_params,
			random_state = self.random_state)

		# Create a pool of worker processes
		with Pool() as pool:
			# Perform cross-validation on each fold in parallel
			scores = pool.starmap(self._cross_validate_fold, [(X, y, model_copy, train_index, test_index)
												 for train_index, test_index in sss.split(X, y)])
		
		# Print the mean and standard deviation of the accuracy scores
		print("Cross validation (accuracy) scores:")
		print("\tmean:", np.mean(scores), "std:", np.std(scores))

	def get_params(self, deep: bool = False) -> dict:
		"""
		Get parameters for this estimator.

		## Parameters
		deep: bool. If True, will return the parameters for this estimator and contained subobjects that are estimators.

		## Returns
		params: dict. Parameter names mapped to their values.
		"""
		return {
			"n_models": self.n_models,
			"n_components": self.n_components,
			"covariance_type": self.covariance_type,
			"tol": self.tol,
			"max_iter": self.max_iter,
			"n_init": self.n_init,
			"init_params": self.init_params,
			"random_state": self.random_state
		}

	def set_params(self, **params):
		"""
		Set the parameters of this estimator.

		## Parameters
		**params: dict. Estimator parameters.
		"""
		for param, value in params.items():
			setattr(self, param, value)
		return self
