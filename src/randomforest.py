import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from src.base import Base

class RandomForest(RandomForestClassifier, Base):
	def __init__(self,
				random_state: int = 42,
				n_estimators: int = 100,
				criterion: str = "entropy",
				max_depth: int = 3,
				min_samples_split: int = 2,
				min_samples_leaf: int = 1,
				bootstrap: bool = True,
				class_weight: dict = None):
		"""
		Random Forest model.
		"""

		self.n_estimators = n_estimators
		self.criterion = criterion
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.min_samples_leaf = min_samples_leaf

		RandomForestClassifier.__init__(
			self,
			random_state=random_state,
			n_estimators=n_estimators,
			criterion=criterion,
			max_depth=max_depth,
			min_samples_split=min_samples_split,
			min_samples_leaf=min_samples_leaf,
			bootstrap=bootstrap,
			class_weight=class_weight,
			n_jobs=-1)

		Base.__init__(
			self,
			name="Random Forest",
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
			# bar plot, with feature names is available
			plt.bar(range(len(importances)), list(importances.values()), align='center')
			plt.xticks(range(len(importances)), list(importances.keys()), rotation=90)
			plt.show()
