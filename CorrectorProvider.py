
from abc import ABC, abstractmethod
import numpy as np

import src.SigmaGenerator as SigmaGenerator
import src.PredictorProvider as PredictorProvider

# Each corrector provider handles a single, atomic sensor update
class CorrectorProvider(ABC):
    def __init__(self, measurement_noise, noise_lerp):
        self.measurement_noise = measurement_noise
        self.noise_lerp = noise_lerp

    @abstractmethod
    def obs_pred(self, state):
        pass

    @abstractmethod
    def correct(self, state, covariance, measurement, predictor_provider: PredictorProvider):
        pass

    def update_measurement_noise(self, residual, pred_measure_cov):
        # A(U|E)?KF: Exponential moving average of measurement noise
        self.measurement_noise = (1 - self.noise_lerp) * self.measurement_noise + self.noise_lerp * (residual @ residual.T + pred_measure_cov)

class LinearCorrectorProvider(CorrectorProvider):
    @abstractmethod
    def linear_obs_mat(self, state):
        pass

    def obs_pred(self, state):
        return self.linear_obs_mat(state) @ state

    def correct(self, state, covariance, measurement, predictor_provider: PredictorProvider):
        obs_mat = self.linear_obs_mat(state)
        pred_measure_cov = obs_mat @ covariance @ obs_mat.T

        kalman_gain = covariance @ obs_mat.T @ np.linalg.inv(obs_mat @ covariance @ obs_mat.T + self.measurement_noise)

        posterior_x = state + kalman_gain @ (measurement - self.obs_pred(state))
        posterior_covariance = (np.eye(len(state)) - kalman_gain @ obs_mat) @ covariance

        # AEKF: Exponential moving average of measurement noise estimate
        innovation = measurement - self.obs_pred(state)
        residual = measurement - self.obs_pred(posterior_x)

        self.update_measurement_noise(residual, pred_measure_cov)
        predictor_provider.update_process_noise(innovation, kalman_gain)

        return posterior_x, posterior_covariance


class UnscentedCorrectorProvider(CorrectorProvider):
    def __init__(self, measurement_noise, noise_lerp, num_states, alpha=1e-3, beta=2, kappa=1):
        super().__init__(measurement_noise, noise_lerp)
        self.num_states = num_states

        self.sigma_generator = SigmaGenerator.SigmaGenerator(num_states, alpha, beta, kappa)

    def correct(self, state, covariance, measurement, predictor_provider: PredictorProvider):
        sigma_points, mean_weights, cov_weights = self.sigma_generator.sigma_points(state, covariance)

        pred_measurements = self.obs_pred(sigma_points)

        mean_measure = mean_weights @ pred_measurements.T
        dev = pred_measurements - mean_measure[:, np.newaxis]

        raw_cov_measure = np.einsum('w,iw,jw->ij', cov_weights, dev, dev)
        cov_measure = raw_cov_measure + self.measurement_noise

        sigma_dev = sigma_points - state[:, np.newaxis]

        cross_cov = np.einsum('w,iw,jw->ij', cov_weights, sigma_dev, dev)

        kalman_gain = cross_cov @ np.linalg.inv(cov_measure)

        posterior_x = state + kalman_gain @ (measurement - mean_measure)
        posterior_cov = covariance - kalman_gain @ cov_measure @ kalman_gain.T

        # AUKF: Exponential moving average of measurement noise estimate
        innovation = measurement - mean_measure
        residual = measurement - self.obs_pred(posterior_x[:, np.newaxis]) # This could be replaced with a sigma point weighted calc

        self.update_measurement_noise(residual, raw_cov_measure)
        predictor_provider.update_process_noise(innovation, kalman_gain)

        posterior_cov = (posterior_cov + posterior_cov.T) / 2

        return posterior_x, posterior_cov
