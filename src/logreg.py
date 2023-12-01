import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

class LogReg(LogisticRegression):
	def __init__(self,
				scaler: StandardScaler,
				epsilon: float = 1e-4,
				random_state: int = 42,
				penalty: str = "l2",
				max_iter: int = 10000):
		"""
		Logistic regression model.
		"""

		self.scaler = scaler
		self.penalty = penalty
		self.epsilon = epsilon
		self.max_iter = max_iter
		self.random_state = random_state

		# Logistic regression
		super().__init__(
			penalty = self.penalty,
			tol = self.epsilon,
			max_iter = self.max_iter,
			random_state = self.random_state)

	def __call__(self, X_pred: np.ndarray) -> np.ndarray:
		"""
		Predict class labels for samples in X_pred.

		## Parameters
		X_pred: np.ndarray of shape (n_samples, n_features)

		## Returns
		y_pred: np.ndarray of shape (n_samples,)
		"""
		X_pred_scaled = self.scaler.transform(X_pred)
		return self.predict(X_pred_scaled)

	def conf_matrix(self, X_test: np.ndarray, y_test: np.ndarray) -> np.ndarray:
		"""
		Compute the confusion matrix of the model.

		## Returns
		confusion_matrix: np.ndarray of shape (n_classes, n_classes)
		"""
		return confusion_matrix(y_test, self(X_test))
	
	def compute_metrics(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
		"""
		Compute the metrics of the model.

		## Returns
		metrics: dict
			{accuracy: float,
			precision: float,
			recall: float,
			f1: float}
		"""
		y_pred = self(X_test)
		report = classification_report(y_test, y_pred, output_dict=True)
		return {
			"accuracy": accuracy_score(y_test, y_pred),
			"precision": report["1"]["precision"],
			"recall": report["1"]["recall"],
			"f1": report["1"]["f1-score"]
		}

	def compare(self, model2, X_test: np.ndarray, y_test: np.ndarray, print_diff: bool = True):
		"""
		Compare the metrics of two models (metrics1 - metrics2).

		## Parameters
		model2: LogReg. Second model to compare.
		X_test: np.ndarray of shape (n_samples, n_features). Test data.
		y_test: np.ndarray of shape (n_samples,). Test target values.
		print_diff: bool. Whether to print the difference between the metrics of the two models.

		## Returns
		metrics_diff: dict
			{accuracy: float,
			precision: float,
			recall: float,
			f1: float,
			confusion matrix: np.ndarray of shape (n_classes, n_classes)}
		"""
		metrics1 = self.compute_metrics(X_test, y_test)
		metrics2 = model2.compute_metrics(X_test, y_test)
		
		metrics_diff = {key: metrics1[key] - metrics2[key] for key in metrics1.keys()}

		cm_diff = self.conf_matrix(X_test, y_test) - model2.conf_matrix(X_test, y_test)
		metrics_diff["confusion matrix"] = cm_diff

		if print_diff:
			for key in metrics_diff.keys():
				print(key, metrics_diff[key])

		return metrics_diff
		

	def cross_validation(self, X: np.ndarray, y: np.ndarray, n_splits: int, val_size:float):
		"""
		Perform cross validation on the model and prints the resulting mean and standard deviation of the accuracy.

		## Parameters
		X: np.ndarray of shape (n_samples, n_features). Training data.
		y: np.ndarray of shape (n_samples,). Target values.
		n_splits: int. Number of folds.
		val_size: float. Validation set size, fraction of the training set.
		"""
		cv = ShuffleSplit(n_splits = n_splits, test_size = val_size, random_state = self.random_state)
		print("Performing cross validation")
		scores_log_reg = cross_val_score(self, X, y, cv = cv, verbose=10, n_jobs=-1)
		print("Cross validation (accuracy) scores:")
		print("    Logistic Regression -> mean:", scores_log_reg.mean(), "std:", scores_log_reg.std())
