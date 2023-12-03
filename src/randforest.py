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
