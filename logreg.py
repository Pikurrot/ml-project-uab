from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

class LogReg(LogisticRegression):
	def __init__(self,
				scaler: StandardScaler,
				epsilon: float = 1e-4,
				random_state: int = 42,
				penalty: str = "l2",
				max_iter: int = 10000):

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

	def __call__(self, X_pred):
		X_pred_scaled = self.scaler.transform(X_pred)
		return self.predict(X_pred_scaled)

	def conf_matrix(self):
		pass

	def compare(self, model2):
		pass

	def cross_validation(self, X, y, n_splits: int, test_size:float):
		cv = ShuffleSplit(n_splits = n_splits, test_size = test_size, random_state = self.random_state)
		X = self.scaler.transform(X)
		print("Performing cv")
		scores_log_reg = cross_val_score(self, X, y, cv = cv, verbose=3)
		print("Cross validation (accuracy) scores:")
		print("    Logistic Regression -> mean:", scores_log_reg.mean(), "std:", scores_log_reg.std())
