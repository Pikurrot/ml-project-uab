import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from src.base import Base

class GradBoost(GradientBoostingClassifier, Base):
	def __init__(self,
				random_state: int = 42,
				loss: str = "log_loss",
				learning_rate: float = 0.1,
				n_estimators: int = 100,
				subsample: float = 1.0,
				criterion: str = "friedman_mse",
				min_samples_split: int = 2,
				min_samples_leaf: int = 1,
				max_depth: int = 3):
		"""
		Gradient Boosting model.
		"""

		self.loss = loss
		self.learning_rate = learning_rate
		self.n_estimators = n_estimators
		self.subsample = subsample
		self.criterion = criterion
		self.min_samples_split = min_samples_split
		self.min_samples_leaf = min_samples_leaf
		self.max_depth = max_depth

		GradientBoostingClassifier.__init__(
			self,
			random_state=random_state,
			loss=loss,
			learning_rate=learning_rate,
			n_estimators=n_estimators,
			subsample=subsample,
			criterion=criterion,
			min_samples_split=min_samples_split,
			min_samples_leaf=min_samples_leaf,
			max_depth=max_depth)

		Base.__init__(
			self,
			name="Gradient Boosting",
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
			plt.bar(importances.keys(), importances.values())
			plt.xticks(rotation=90)
			plt.show()

		return importances
