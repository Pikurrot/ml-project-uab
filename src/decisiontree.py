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
