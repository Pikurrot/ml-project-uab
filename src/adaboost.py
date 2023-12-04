import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from src.base import Base

class AdaBoost(AdaBoostClassifier, Base):
	def __init__(self,
				random_state: int = 42,
				n_estimators: int = 100,
				criterion: str = "entropy",
				max_depth: int = 3,
				min_samples_split: int = 2,
				min_samples_leaf: int = 1,
				learning_rate: float = 1.0,
				algorithm: str = "SAMME.R"):
		"""
		AdaBoost model.
		"""

		self.n_estimators = n_estimators
		self.criterion = criterion
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.min_samples_leaf = min_samples_leaf
		self.learning_rate = learning_rate
		self.algorithm = algorithm

		AdaBoostClassifier.__init__(
			self,
			estimator = DecisionTreeClassifier(
				random_state=random_state,
				criterion=criterion,
				max_depth=max_depth,
				min_samples_split=min_samples_split,
				min_samples_leaf=min_samples_leaf
			),
			random_state=random_state,
			n_estimators=n_estimators,
			learning_rate=learning_rate,
			algorithm=algorithm)

		Base.__init__(
			self,
			name="AdaBoost",
			random_state=random_state)

	def importances(self, features: list = None, show: bool = False):
		"""
		Compute the feature importances.

		## Parameters
		features: list. Feature names.
		show: bool. Whether to show the feature importances plot.

		## Returns
		importances: if features is provided, a dict with the feature names and their\
			importances sorted from highest to lowest. Otherwise, a list with the importances.
		"""
		if features is None:
			importances = self.feature_importances_
		else:
			importances = dict(zip(features, self.feature_importances_))
			importances = {k: v for k, v in sorted(importances.items(), key=lambda item: item[1], reverse=True)}

		if show:
			plt.figure(figsize=(10, 5))
			plt.bar(importances.keys(), importances.values())
			plt.xticks(rotation=90)
			plt.title("Feature importances")
			plt.show()

		return importances
