from sklearn.linear_model import LogisticRegression
from src.base import Base

class LogReg(LogisticRegression, Base):
	def __init__(self,
				random_state: int = 42,
				epsilon: float = 1e-4,
				penalty: str = "l2",
				lam: float = 1.0,
				max_iter: int = 10000,
				class_weight: str = None,
				solver: str = "lbfgs"):
		"""
		Logistic regression model.
		"""

		self.penalty = penalty
		self.epsilon = epsilon
		self.lam = lam
		self.max_iter = max_iter
		self.class_weight = class_weight
		self.solver = solver

		LogisticRegression.__init__(
			self,
			penalty = self.penalty,
			C = 1/lam,
			tol = self.epsilon,
			max_iter = self.max_iter,
			class_weight = class_weight,
			solver = self.solver,
			random_state = random_state)
		
		Base.__init__(
			self,
			name = "Logistic Regression",
			random_state = random_state)
