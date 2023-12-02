from sklearn.linear_model import LogisticRegression
from src.base import Base

class LogReg(LogisticRegression, Base):
	def __init__(self,
				random_state: int = 42,
				epsilon: float = 1e-4,
				penalty: str = "l2",
				max_iter: int = 10000):
		"""
		Logistic regression model.
		"""

		self.penalty = penalty
		self.epsilon = epsilon
		self.max_iter = max_iter
		LogisticRegression.__init__(
			self,
			penalty = self.penalty,
			tol = self.epsilon,
			max_iter = self.max_iter,
			random_state = random_state)
		
		Base.__init__(
			self,
			name = "Logistic Regression",
			random_state = random_state)
