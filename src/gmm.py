from sklearn.mixture import GaussianMixture
from src.base import Base

class GMM(GaussianMixture, Base):
	def __init__(self,
				random_state: int = 42,
				n_models: int = 5,
				n_components: int = 1,
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

		self.models = []
		for _ in range(n_models):
			self.models.append(GaussianMixture(
				n_components = n_components,
				covariance_type = covariance_type,
				tol = tol,
				max_iter = max_iter,
				n_init = n_init,
				init_params = init_params,
				random_state = random_state))
		
		Base.__init__(
			self,
			name = "Gaussian Mixture Model",
			random_state = random_state)
