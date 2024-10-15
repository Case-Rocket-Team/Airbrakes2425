
import numpy as np

class SigmaGenerator:
  def __init__(self, num_states, alpha, beta, kappa):
    self.num_states = num_states
    self.alpha = alpha
    self.beta = beta
    self.kappa = kappa

  def sigma_points(self, state, covariance):
    sigma_points = np.zeros((self.num_states, 2 * self.num_states + 1))
    mean_weights = np.zeros(2 * self.num_states + 1)
    cov_weights = np.zeros(2 * self.num_states + 1)

    sigma_points[:,0] = state
    mean_weights[0] = (self.alpha**2 * self.kappa - self.num_states) / (self.alpha**2 * self.kappa)
    cov_weights[0] = mean_weights[0] - self.alpha**2 + self.beta

    A = np.linalg.cholesky(covariance)

    remaining_weights = 1 / (2 * self.alpha**2 * self.kappa)

    for i in range(self.num_states):
      sigma_points[:,1 + i] = state + self.alpha * np.sqrt(self.kappa) * A[:,i]
      sigma_points[:,1 + self.num_states + i] = state - self.alpha * np.sqrt(self.kappa) * A[:,i]

      mean_weights[1 + i:: self.num_states] = remaining_weights
      cov_weights[1 + i:: self.num_states] = remaining_weights

    return sigma_points, mean_weights, cov_weights
