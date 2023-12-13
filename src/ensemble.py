import numpy as np
from src.base import Base

class Ensemble(Base):
	def __init__(self,
			  	models: list,
				voting: str = "soft",
				weights: list = None):
		"""
		Ensemble model.
		"""
		self.models = models
		self.voting = voting
		self.weights = weights

		super().__init__(
			name="Ensemble",
			random_state=42)

	def predict_probas(self, X: np.ndarray) -> np.ndarray:
		"""
		Compute probabilities of each model of each class for samples in X.

		## Parameters
		X: np.ndarray of shape (n_samples, n_features)

		## Returns
		y_pred: np.ndarray of shape (n_samples, n_classes, n_models)
		"""
		return np.stack([model.predict_proba(X) for model in self.models], axis=2)

	def predict_proba(self, X: np.ndarray) -> np.ndarray:
		"""
		Compute weighted average of probabilities of each class for samples in X.

		## Parameters
		X: np.ndarray of shape (n_samples, n_features)

		## Returns
		y_pred: np.ndarray of shape (n_samples, n_classes)
		"""
		probas = self.predict_probas(X)
		return np.average(probas, axis=2, weights=self.weights)

	def predict(self, X: np.ndarray) -> np.ndarray:
		"""
		Predict class labels for samples in X.

		## Parameters
		X: np.ndarray of shape (n_samples, n_features)

		## Returns
		y_pred: np.ndarray of shape (n_samples,)
		"""
		if self.voting == "soft":
			return np.argmax(self.predict_proba(X), axis=1)
		elif self.voting == "hard":
			probas = self.predict_probas(X) # (n_samples, n_classes, n_models)
			models_pred = np.argmax(probas, axis=1) # (n_samples, n_models)
			# one-hot encode
			hardmax = np.zeros_like(probas, dtype=int) # (n_samples, n_classes, n_models)
			for sample in range(probas.shape[0]):
				for model in range(probas.shape[2]):
					hardmax[sample, models_pred[sample, model], model] = 1
			sum = np.sum(hardmax, axis=2) # (n_samples, n_classes)
			majority = np.argmax(sum, axis=1) # (n_samples,)
			return majority
