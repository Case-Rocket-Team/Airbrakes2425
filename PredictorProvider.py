from abc import ABC, abstractmethod
import numpy as np

import src.SigmaGenerator as SigmaGenerator

class PredictorProvider(ABC):
    def __init__(self, process_noise, noise_lerp):
        self.process_noise = process_noise
        self.noise_lerp = noise_lerp

    @abstractmethod
    def point_prop(self, state, dt):
        pass

    @abstractmethod
    def predict(self, state, covariance, dt):
        pass

    def update_process_noise(self, innovation, kalman_gain):
        # A(U|E)?KF: Exponential moving average of process noise
        vec = kalman_gain @ innovation

        self.process_noise = (1 - self.noise_lerp) * self.process_noise + self.noise_lerp * (vec.T @ vec)


class LinearPredictorProvider(PredictorProvider):
    @abstractmethod
    def linear_prop_mat(self, state, dt):
        # Implement linear propagation logic here
        pass

    # Implement non-linear point_prop if desired
    def point_prop(self, state, dt):
        return self.linear_prop_mat(state, dt) @ state

    def predict(self, state, covariance, dt):
        return self.point_prop(state, dt), self.linear_prop_mat(state, dt) @ covariance @ self.linear_prop_mat(state, dt).T + self.process_noise

class UnscentedPredictorProvider(PredictorProvider):
    def __init__(self, process_noise, noise_lerp, num_states, alpha=1e-3, beta=2, kappa=1):
        super().__init__(process_noise, noise_lerp)
        self.num_states = num_states

        self.sigma_generator = SigmaGenerator.SigmaGenerator(num_states, alpha, beta, kappa)

    @abstractmethod
    def point_prop(self, state, dt):
        # Implement unscented point propagation
        pass

    def predict(self, state, covariance, dt):
        sigma_points, mean_weights, cov_weights = self.sigma_generator.sigma_points(state, covariance)

        x_prime = self.point_prop(sigma_points, dt)

        mean = mean_weights @ x_prime.T

        dev = x_prime - mean[:, np.newaxis]

        # could also be calculated with appropriately configured weighted covariance calc
        covariance = np.einsum('w,iw,jw->ij', cov_weights, dev, dev) + self.process_noise


        return mean, covariance
