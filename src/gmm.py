from sklearn.mixture import GaussianMixture
from src.base import Base

class GMM(GaussianMixture, Base):
	def __init__(self,
				random_state: int = 42,
				n_components: int = 2,
				covariance_type: str = "full",
				tol: float = 1e-3,
				max_iter: int = 100,
				n_init: int = 1,
				init_params: str = "kmeans"):
		"""
		Gaussian Mixture Model.
		"""
		
		self.n_components = n_components
		self.covariance_type = covariance_type
		self.tol = tol
		self.max_iter = max_iter
		self.n_init = n_init
		self.init_params = init_params

		GaussianMixture.__init__(
			self,
			n_components = self.n_components,
			covariance_type = self.covariance_type,
			tol = self.tol,
			max_iter = self.max_iter,
			n_init = self.n_init,
			init_params = self.init_params,
			random_state = random_state)
		
		Base.__init__(
			self,
			name = "Gaussian Mixture Model",
			random_state = random_state)
