import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from src.base import Base

class DecisionTree(DecisionTreeClassifier, Base):
	def __init__(self,
				random_state: int = 42,
				criterion: str = "entropy",
				max_depth: int = 3,
				min_samples_split: int = 2,
				min_samples_leaf: int = 1):
		"""
		Decision Tree model.
		"""

		self.criterion = criterion
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.min_samples_leaf = min_samples_leaf

		DecisionTreeClassifier.__init__(
			self,
			random_state=random_state,
			criterion=criterion,
			max_depth=max_depth,
			min_samples_split=min_samples_split,
			min_samples_leaf=min_samples_leaf)

		Base.__init__(
			self,
			name="Decision Tree",
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
