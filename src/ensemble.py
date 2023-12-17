import numpy as np
import pandas as pd
from typing import Union
from src.base import Base
import src.utils as utils

class Ensemble(Base):
	def __init__(self,
			  	models: list,
				voting: str = "soft",
				weights: list = None,
				scaler=None):
		"""
		Ensemble model.
		"""
		self.models = models
		self.voting = voting
		self.weights = weights
		self.scaler = scaler

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
		
	def __call__(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.Series, np.ndarray]:
		"""
		Predict class names for samples in X. Similar to predict,
		but it takes raw data and preprocesses it.

		## Parameters
		X: shape (n_samples, n_features)

		## Returns
		y_pred: shape (n_samples,)
		"""
		if isinstance(X, pd.DataFrame):
			if "Unnamed: 0" not in X.columns: # the data has already been processed
				return self.predict(X)
			X_copy = X.copy().reset_index(drop=True)
			Xy = pd.concat([X_copy, pd.DataFrame(np.zeros((X_copy.shape[0], 1)), columns=["cut"])], axis=1)
			X_new, _ = utils.preprocessing_LS_simple(Xy, scaler=self.scaler)
		elif isinstance(X, np.ndarray):
			Xy = np.concatenate([X, np.zeros((X.shape[0], 1))], axis=1)
			X_df = pd.DataFrame(Xy,
				columns=["Unnamed: 0", "carat", "color", "clarity", "depth", "table", "price", "x", "y", "z", "cut"])
			X_new, _ = utils.preprocessing_LS_simple(X_df, scaler=self.scaler)
		pred_label = self.predict(X_new)
		cut_mapping = {4: "Ideal", 3: "Premium", 2: "Very Good", 1: "Good", 0: "Fair"}
		pred = pd.Series(pred_label).map(cut_mapping)
		if isinstance(X, pd.DataFrame):
			return pred
		elif isinstance(X, np.ndarray):
			return pred.to_numpy()
