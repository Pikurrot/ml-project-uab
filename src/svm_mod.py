from sklearn import svm
import pandas as pd
import numpy as np
from src.base import Base

class SVM_model(svm.SVC, Base):
	def __init__(self,
				epsilon: float = 1e-4, 
				random_state: int = 42,
				kernel: str = "rbf",
				gamma: str = "scale",
				C: float = 1.0,
				degree: int = 3,
				coef0: float = 0.0,
				max_iter: int = 10000,
				class_weight: str = None):
		"""
		SVM model
		"""
		self.epsilon = epsilon 
		self.random_state = random_state
		self.kernel = kernel
		self.gamma = gamma 
		self.C = C
		self.degree = degree
		self.coef0 = coef0
		self.max_iter = max_iter
		self.class_weight = class_weight
		# SVM
		svm.SVC.__init__(
			self,
			kernel = self.kernel,
			gamma = self.gamma,
			C = self.C,
			degree = self.degree,
			coef0 = self.coef0,
			max_iter = self.max_iter,
			random_state = self.random_state,
			class_weight = self.class_weight)
		# Base
		Base.__init__(
			self,
			name = "Support Vector Machine",
			random_state = random_state)
		
	def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
		"""
		## Returns
		Probability of each class.
		"""
		# voting
		v = self.decision_function(X)
		# softmax
		return (np.exp(v) / np.sum(np.exp(v), axis=1, keepdims=True))
