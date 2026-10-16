from __future__ import annotations

import numpy as np

from .rabi import (
    EPSILON,
    RabiExperimentConfig,
    build_rabi_particle_grid,
    detuning_sinc_factor,
    longitudinal_relaxation_factor,
    _known_or_grid_values,
    _normalize_log_bayes_checkpoints,
    _prior_masses,
    transverse_relaxation_factor,
)

try:
    from numba import njit, prange
except Exception:  # pragma: no cover - exercised only on environments without numba.
    njit = None
    prange = None


def is_numba_available() -> bool:
    return njit is not None


if njit is not None:

    @njit
    def _clip_probability(value: float) -> float:
        if value < EPSILON:
            return EPSILON
        if value > 1.0 - EPSILON:
            return 1.0 - EPSILON
        return value


    @njit
    def _binary_entropy_scalar(probability_one: float) -> float:
        p1 = _clip_probability(probability_one)
        p0 = 1.0 - p1
        return -(p1 * np.log(p1) + p0 * np.log(p0))


    @njit
    def _binary_kl_scalar(probability_one: float, reference_one: float) -> float:
        p1 = _clip_probability(probability_one)
        p0 = 1.0 - p1
        q1 = _clip_probability(reference_one)
        q0 = 1.0 - q1
        return p1 * np.log(p1 / q1) + p0 * np.log(p0 / q0)


    @njit
    def _binary_log_likelihood_moments(
        probability_one: float,
        reference_one: float,
    ) -> tuple[float, float]:
        p1 = _clip_probability(probability_one)
        p0 = 1.0 - p1
        q1 = _clip_probability(reference_one)
        q0 = 1.0 - q1
        log_ratio_one = np.log(p1 / q1)
        log_ratio_zero = np.log(p0 / q0)
        mean = p1 * log_ratio_one + p0 * log_ratio_zero
        variance = p1 * (log_ratio_one - mean) ** 2 + p0 * (log_ratio_zero - mean) ** 2
        return mean, variance


    @njit
    def _noisy_probability_from_projection(projection: float, contrast: float, readout_error: float) -> float:
        ideal_probability = 0.5 * (1.0 + contrast * projection)
        noisy_probability = readout_error + (1.0 - 2.0 * readout_error) * ideal_probability
        return _clip_probability(noisy_probability)


    @njit
    def _transverse_probability(
        transverse_amplitude: float,
        phase_cosine: float,
        phase_sine: float,
        angle_cosine: float,
        angle_sine: float,
        contrast: float,
        readout_error: float,
    ) -> float:
        projection = transverse_amplitude * (angle_sine * phase_cosine - angle_cosine * phase_sine)
        return _noisy_probability_from_projection(projection, contrast, readout_error)


    @njit
    def _canonical_sphere_action(readout_polar: float, readout_angle: float) -> tuple[float, float]:
        x = np.sin(readout_polar) * np.cos(readout_angle)
        y = np.sin(readout_polar) * np.sin(readout_angle)
        z = np.cos(readout_polar)
        if z < 0.0:
            x = -x
            y = -y
            z = -z
        norm = np.sqrt(x * x + y * y + z * z)
        if norm <= 0.0:
            return 0.0, 0.0
        x = x / norm
        y = y / norm
        z = z / norm
        if z < -1.0:
            z = -1.0
        elif z > 1.0:
            z = 1.0
        polar = np.arccos(z)
        azimuth = np.arctan2(y, x)
        two_pi = 2.0 * np.pi
        azimuth = azimuth - np.floor(azimuth / two_pi) * two_pi
        if azimuth >= np.pi:
            polar = np.pi - polar
            azimuth -= np.pi
        return polar, azimuth


    @njit
    def _sphere_probability_from_components(
        transverse_amplitude: float,
        z_projection: float,
        phase_cosine: float,
        phase_sine: float,
        angle_cosine: float,
        angle_sine: float,
        polar_cosine: float,
        polar_sine: float,
        contrast: float,
        readout_error: float,
    ) -> float:
        transverse_projection = transverse_amplitude * (angle_sine * phase_cosine - angle_cosine * phase_sine)
        projection = polar_cosine * z_projection + polar_sine * transverse_projection
        return _noisy_probability_from_projection(projection, contrast, readout_error)


    @njit
    def _sphere_h0_probability(
        polar_cosine: float,
        contrast: float,
        readout_error: float,
    ) -> float:
        return _noisy_probability_from_projection(-polar_cosine, contrast, readout_error)


    @njit
    def _sphere_particle_probability(
        particle_index: int,
        particle_transverse_amplitudes: np.ndarray,
        particle_z_projections: np.ndarray,
        phase_cosines: np.ndarray,
        phase_sines: np.ndarray,
        angle_cosine: float,
        angle_sine: float,
        polar_cosine: float,
        polar_sine: float,
        contrast: float,
        readout_error: float,
    ) -> float:
        return _sphere_probability_from_components(
            particle_transverse_amplitudes[particle_index],
            particle_z_projections[particle_index],
            phase_cosines[particle_index],
            phase_sines[particle_index],
            angle_cosine,
            angle_sine,
            polar_cosine,
            polar_sine,
            contrast,
            readout_error,
        )


    @njit
    def _fill_future_phases(
        phase_cosines: np.ndarray,
        phase_sines: np.ndarray,
        step_cosines: np.ndarray,
        step_sines: np.ndarray,
        future_cosines: np.ndarray,
        future_sines: np.ndarray,
    ) -> None:
        for particle_index in range(phase_cosines.size):
            future_cosines[particle_index] = (
                phase_cosines[particle_index] * step_cosines[particle_index]
                - phase_sines[particle_index] * step_sines[particle_index]
            )
            future_sines[particle_index] = (
                phase_sines[particle_index] * step_cosines[particle_index]
                + phase_cosines[particle_index] * step_sines[particle_index]
            )


    @njit
    def _fill_transverse_candidate_tables(
        phase_cosines: np.ndarray,
        phase_sines: np.ndarray,
        particle_transverse_amplitudes: np.ndarray,
        candidate_cosines: np.ndarray,
        candidate_sines: np.ndarray,
        contrast: float,
        readout_error: float,
        particle_one_candidates: np.ndarray,
        particle_entropy_candidates: np.ndarray,
    ) -> None:
        for candidate_index in range(candidate_cosines.size):
            candidate_cosine = candidate_cosines[candidate_index]
            candidate_sine = candidate_sines[candidate_index]
            for particle_index in range(particle_transverse_amplitudes.size):
                probability_one = _transverse_probability(
                    particle_transverse_amplitudes[particle_index],
                    phase_cosines[particle_index],
                    phase_sines[particle_index],
                    candidate_cosine,
                    candidate_sine,
                    contrast,
                    readout_error,
                )
                particle_one_candidates[candidate_index, particle_index] = probability_one
                particle_entropy_candidates[candidate_index, particle_index] = _binary_entropy_scalar(
                    probability_one
                )


    @njit
    def _candidate_utility_and_mixture(
        weights: np.ndarray,
        h1_probability: float,
        candidate_index: int,
        particle_one_candidates: np.ndarray,
        particle_entropy_candidates: np.ndarray,
        h0_transverse_one: float,
        h0_transverse_entropy: float,
    ) -> tuple[float, float, float]:
        mixture_one = 0.0
        h1_conditional_entropy = 0.0
        for particle_index in range(weights.size):
            weight = weights[particle_index]
            mixture_one += weight * particle_one_candidates[candidate_index, particle_index]
            h1_conditional_entropy += weight * particle_entropy_candidates[candidate_index, particle_index]

        one_minus_h1 = 1.0 - h1_probability
        predictive_one = one_minus_h1 * h0_transverse_one + h1_probability * mixture_one
        conditional_entropy = (
            one_minus_h1 * h0_transverse_entropy
            + h1_probability * h1_conditional_entropy
        )
        utility = _binary_entropy_scalar(predictive_one) - conditional_entropy
        return utility, mixture_one, predictive_one


    @njit
    def _candidate_bayes_drift_and_mixture(
        weights: np.ndarray,
        candidate_index: int,
        particle_one_candidates: np.ndarray,
        h0_transverse_one: float,
    ) -> tuple[float, float]:
        mixture_one = 0.0
        for particle_index in range(weights.size):
            mixture_one += weights[particle_index] * particle_one_candidates[candidate_index, particle_index]
        utility = _binary_kl_scalar(mixture_one, h0_transverse_one)
        return utility, mixture_one


    @njit
    def _phase_concentration(
        weights: np.ndarray,
        phase_cosines: np.ndarray,
        phase_sines: np.ndarray,
    ) -> float:
        mean_cosine = 0.0
        mean_sine = 0.0
        for particle_index in range(weights.size):
            mean_cosine += weights[particle_index] * phase_cosines[particle_index]
            mean_sine += weights[particle_index] * phase_sines[particle_index]
        return np.sqrt(mean_cosine * mean_cosine + mean_sine * mean_sine)


    @njit
    def _blend_lambda(
        concentration: float,
        lambda_max: float,
        r_start: float,
        r_stop: float,
    ) -> float:
        if concentration <= r_start:
            return lambda_max
        if concentration >= r_stop:
            return 0.0
        return lambda_max * (r_stop - concentration) / (r_stop - r_start)


    @njit
    def _candidate_blended_value_and_mixture(
        weights: np.ndarray,
        h1_probability: float,
        candidate_index: int,
        particle_one_candidates: np.ndarray,
        particle_entropy_candidates: np.ndarray,
        h0_transverse_one: float,
        h0_transverse_entropy: float,
        lambda_information: float,
    ) -> tuple[float, float]:
        information_gain, mixture_one, _ = _candidate_utility_and_mixture(
            weights,
            h1_probability,
            candidate_index,
            particle_one_candidates,
            particle_entropy_candidates,
            h0_transverse_one,
            h0_transverse_entropy,
        )
        bayes_drift = _binary_kl_scalar(mixture_one, h0_transverse_one)
        return bayes_drift + lambda_information * information_gain, mixture_one


    @njit
    def _sphere_information_gain_and_mixture(
        weights: np.ndarray,
        h1_probability: float,
        particle_transverse_amplitudes: np.ndarray,
        particle_z_projections: np.ndarray,
        phase_cosines: np.ndarray,
        phase_sines: np.ndarray,
        angle_cosine: float,
        angle_sine: float,
        polar_cosine: float,
        polar_sine: float,
        contrast: float,
        readout_error: float,
    ) -> tuple[float, float]:
        h0_one = _sphere_h0_probability(polar_cosine, contrast, readout_error)
        h0_entropy = _binary_entropy_scalar(h0_one)
        mixture_one = 0.0
        h1_conditional_entropy = 0.0
        for particle_index in range(weights.size):
            probability_one = _sphere_particle_probability(
                particle_index,
                particle_transverse_amplitudes,
                particle_z_projections,
                phase_cosines,
                phase_sines,
                angle_cosine,
                angle_sine,
                polar_cosine,
                polar_sine,
                contrast,
                readout_error,
            )
            mixture_one += weights[particle_index] * probability_one
            h1_conditional_entropy += weights[particle_index] * _binary_entropy_scalar(probability_one)
        one_minus_h1 = 1.0 - h1_probability
        predictive_one = one_minus_h1 * h0_one + h1_probability * mixture_one
        conditional_entropy = one_minus_h1 * h0_entropy + h1_probability * h1_conditional_entropy
        return _binary_entropy_scalar(predictive_one) - conditional_entropy, mixture_one


    @njit
    def _posterior_bloch_moments(
        weights: np.ndarray,
        particle_transverse_amplitudes: np.ndarray,
        particle_z_projections: np.ndarray,
        phase_cosines: np.ndarray,
        phase_sines: np.ndarray,
    ) -> tuple[float, float, float]:
        mean_x = 0.0
        mean_y = 0.0
        mean_z = 0.0
        for particle_index in range(weights.size):
            weight = weights[particle_index]
            transverse_amplitude = particle_transverse_amplitudes[particle_index]
            mean_x += weight * (-transverse_amplitude * phase_sines[particle_index])
            mean_y += weight * (transverse_amplitude * phase_cosines[particle_index])
            mean_z += weight * particle_z_projections[particle_index]
        return mean_x, mean_y, mean_z


    @njit
    def _sphere_detection_value_and_mixture(
        mean_x: float,
        mean_y: float,
        mean_z: float,
        angle_cosine: float,
        angle_sine: float,
        polar_cosine: float,
        polar_sine: float,
        contrast: float,
        readout_error: float,
        bayes_objective: bool,
        stein_objective: bool,
    ) -> tuple[float, float]:
        h1_projection = polar_sine * angle_cosine * mean_x + polar_sine * angle_sine * mean_y + polar_cosine * mean_z
        h0_projection = -polar_cosine
        mixture_one = _noisy_probability_from_projection(h1_projection, contrast, readout_error)
        h0_one = _noisy_probability_from_projection(h0_projection, contrast, readout_error)
        if bayes_objective:
            return _binary_kl_scalar(mixture_one, h0_one), mixture_one
        if stein_objective:
            return _binary_kl_scalar(h0_one, mixture_one), mixture_one
        return abs(mixture_one - h0_one), mixture_one


    @njit
    def _continuous_angle_utility_and_mixture(
        weights: np.ndarray,
        h1_probability: float,
        phase_cosines: np.ndarray,
        phase_sines: np.ndarray,
        particle_transverse_amplitudes: np.ndarray,
        angle_cosine: float,
        angle_sine: float,
        contrast: float,
        readout_error: float,
        h0_transverse_one: float,
        h0_transverse_entropy: float,
    ) -> tuple[float, float]:
        mixture_one = 0.0
        h1_conditional_entropy = 0.0
        for particle_index in range(weights.size):
            probability_one = _transverse_probability(
                particle_transverse_amplitudes[particle_index],
                phase_cosines[particle_index],
                phase_sines[particle_index],
                angle_cosine,
                angle_sine,
                contrast,
                readout_error,
            )
            mixture_one += weights[particle_index] * probability_one
            h1_conditional_entropy += weights[particle_index] * _binary_entropy_scalar(probability_one)

        one_minus_h1 = 1.0 - h1_probability
        predictive_one = one_minus_h1 * h0_transverse_one + h1_probability * mixture_one
        conditional_entropy = one_minus_h1 * h0_transverse_entropy + h1_probability * h1_conditional_entropy
        utility = _binary_entropy_scalar(predictive_one) - conditional_entropy
        return utility, mixture_one


    @njit
    def _continuous_angle_bayes_drift_and_mixture(
        weights: np.ndarray,
        phase_cosines: np.ndarray,
        phase_sines: np.ndarray,
        particle_transverse_amplitudes: np.ndarray,
        angle_cosine: float,
        angle_sine: float,
        contrast: float,
        readout_error: float,
        h0_transverse_one: float,
    ) -> tuple[float, float]:
        mixture_one = 0.0
        for particle_index in range(weights.size):
            probability_one = _transverse_probability(
                particle_transverse_amplitudes[particle_index],
                phase_cosines[particle_index],
                phase_sines[particle_index],
                angle_cosine,
                angle_sine,
                contrast,
                readout_error,
            )
            mixture_one += weights[particle_index] * probability_one
        utility = _binary_kl_scalar(mixture_one, h0_transverse_one)
        return utility, mixture_one


    @njit
    def _continuous_angle_blended_value_and_mixture(
        weights: np.ndarray,
        h1_probability: float,
        phase_cosines: np.ndarray,
        phase_sines: np.ndarray,
        particle_transverse_amplitudes: np.ndarray,
        angle_cosine: float,
        angle_sine: float,
        contrast: float,
        readout_error: float,
        h0_transverse_one: float,
        h0_transverse_entropy: float,
        lambda_information: float,
    ) -> tuple[float, float]:
        information_gain, mixture_one = _continuous_angle_utility_and_mixture(
            weights,
            h1_probability,
            phase_cosines,
            phase_sines,
            particle_transverse_amplitudes,
            angle_cosine,
            angle_sine,
            contrast,
            readout_error,
            h0_transverse_one,
            h0_transverse_entropy,
        )
        bayes_drift = _binary_kl_scalar(mixture_one, h0_transverse_one)
        return bayes_drift + lambda_information * information_gain, mixture_one


    @njit
    def _refine_continuous_angle(
        weights: np.ndarray,
        h1_probability: float,
        phase_cosines: np.ndarray,
        phase_sines: np.ndarray,
        particle_transverse_amplitudes: np.ndarray,
        center_angle: float,
        bracket_width: float,
        contrast: float,
        readout_error: float,
        h0_transverse_one: float,
        h0_transverse_entropy: float,
    ) -> tuple[float, float, float]:
        left = center_angle - bracket_width
        right = center_angle + bracket_width
        best_angle = center_angle
        best_utility, best_mixture = _continuous_angle_utility_and_mixture(
            weights,
            h1_probability,
            phase_cosines,
            phase_sines,
            particle_transverse_amplitudes,
            np.cos(center_angle),
            np.sin(center_angle),
            contrast,
            readout_error,
            h0_transverse_one,
            h0_transverse_entropy,
        )

        # A short ternary search is enough after the coarse scan brackets the best lobe.
        for _ in range(8):
            mid_left = left + (right - left) / 3.0
            mid_right = right - (right - left) / 3.0
            utility_left, mixture_left = _continuous_angle_utility_and_mixture(
                weights,
                h1_probability,
                phase_cosines,
                phase_sines,
                particle_transverse_amplitudes,
                np.cos(mid_left),
                np.sin(mid_left),
                contrast,
                readout_error,
                h0_transverse_one,
                h0_transverse_entropy,
            )
            utility_right, mixture_right = _continuous_angle_utility_and_mixture(
                weights,
                h1_probability,
                phase_cosines,
                phase_sines,
                particle_transverse_amplitudes,
                np.cos(mid_right),
                np.sin(mid_right),
                contrast,
                readout_error,
                h0_transverse_one,
                h0_transverse_entropy,
            )
            if utility_left > best_utility + 1.0e-14:
                best_utility = utility_left
                best_mixture = mixture_left
                best_angle = mid_left
            if utility_right > best_utility + 1.0e-14:
                best_utility = utility_right
                best_mixture = mixture_right
                best_angle = mid_right
            if utility_left < utility_right:
                left = mid_left
            else:
                right = mid_right

        return best_angle, best_utility, best_mixture


    @njit
    def _refine_continuous_angle_bayes_drift(
        weights: np.ndarray,
        phase_cosines: np.ndarray,
        phase_sines: np.ndarray,
        particle_transverse_amplitudes: np.ndarray,
        center_angle: float,
        bracket_width: float,
        contrast: float,
        readout_error: float,
        h0_transverse_one: float,
    ) -> tuple[float, float, float]:
        left = center_angle - bracket_width
        right = center_angle + bracket_width
        best_angle = center_angle
        best_utility, best_mixture = _continuous_angle_bayes_drift_and_mixture(
            weights,
            phase_cosines,
            phase_sines,
            particle_transverse_amplitudes,
            np.cos(center_angle),
            np.sin(center_angle),
            contrast,
            readout_error,
            h0_transverse_one,
        )

        for _ in range(8):
            mid_left = left + (right - left) / 3.0
            mid_right = right - (right - left) / 3.0
            utility_left, mixture_left = _continuous_angle_bayes_drift_and_mixture(
                weights,
                phase_cosines,
                phase_sines,
                particle_transverse_amplitudes,
                np.cos(mid_left),
                np.sin(mid_left),
                contrast,
                readout_error,
                h0_transverse_one,
            )
            utility_right, mixture_right = _continuous_angle_bayes_drift_and_mixture(
                weights,
                phase_cosines,
                phase_sines,
                particle_transverse_amplitudes,
                np.cos(mid_right),
                np.sin(mid_right),
                contrast,
                readout_error,
                h0_transverse_one,
            )
            if utility_left > best_utility + 1.0e-14:
                best_utility = utility_left
                best_mixture = mixture_left
                best_angle = mid_left
            if utility_right > best_utility + 1.0e-14:
                best_utility = utility_right
                best_mixture = mixture_right
                best_angle = mid_right
            if utility_left < utility_right:
                left = mid_left
            else:
                right = mid_right

        angle_period = 2.0 * np.pi
        best_angle = best_angle - np.floor(best_angle / angle_period) * angle_period
        return best_angle, best_utility, best_mixture


    @njit
    def _refine_continuous_angle_blended(
        weights: np.ndarray,
        h1_probability: float,
        phase_cosines: np.ndarray,
        phase_sines: np.ndarray,
        particle_transverse_amplitudes: np.ndarray,
        center_angle: float,
        bracket_width: float,
        contrast: float,
        readout_error: float,
        h0_transverse_one: float,
        h0_transverse_entropy: float,
        lambda_information: float,
    ) -> tuple[float, float, float]:
        left = center_angle - bracket_width
        right = center_angle + bracket_width
        best_angle = center_angle
        best_utility, best_mixture = _continuous_angle_blended_value_and_mixture(
            weights,
            h1_probability,
            phase_cosines,
            phase_sines,
            particle_transverse_amplitudes,
            np.cos(center_angle),
            np.sin(center_angle),
            contrast,
            readout_error,
            h0_transverse_one,
            h0_transverse_entropy,
            lambda_information,
        )

        for _ in range(8):
            mid_left = left + (right - left) / 3.0
            mid_right = right - (right - left) / 3.0
            utility_left, mixture_left = _continuous_angle_blended_value_and_mixture(
                weights,
                h1_probability,
                phase_cosines,
                phase_sines,
                particle_transverse_amplitudes,
                np.cos(mid_left),
                np.sin(mid_left),
                contrast,
                readout_error,
                h0_transverse_one,
                h0_transverse_entropy,
                lambda_information,
            )
            utility_right, mixture_right = _continuous_angle_blended_value_and_mixture(
                weights,
                h1_probability,
                phase_cosines,
                phase_sines,
                particle_transverse_amplitudes,
                np.cos(mid_right),
                np.sin(mid_right),
                contrast,
                readout_error,
                h0_transverse_one,
                h0_transverse_entropy,
                lambda_information,
            )
            if utility_left > best_utility + 1.0e-14:
                best_utility = utility_left
                best_mixture = mixture_left
                best_angle = mid_left
            if utility_right > best_utility + 1.0e-14:
                best_utility = utility_right
                best_mixture = mixture_right
                best_angle = mid_right
            if utility_left < utility_right:
                left = mid_left
            else:
                right = mid_right

        angle_period = 2.0 * np.pi
        best_angle = best_angle - np.floor(best_angle / angle_period) * angle_period
        return best_angle, best_utility, best_mixture


    @njit
    def _sphere_action_value_and_mixture(
        weights: np.ndarray,
        h1_probability: float,
        particle_transverse_amplitudes: np.ndarray,
        particle_z_projections: np.ndarray,
        phase_cosines: np.ndarray,
        phase_sines: np.ndarray,
        readout_polar: float,
        readout_angle: float,
        contrast: float,
        readout_error: float,
        information_objective: bool,
        bayes_detection_objective: bool,
        stein_detection_objective: bool,
        mean_x: float,
        mean_y: float,
        mean_z: float,
    ) -> tuple[float, float, float, float]:
        canonical_polar, canonical_angle = _canonical_sphere_action(readout_polar, readout_angle)
        polar_cosine = np.cos(canonical_polar)
        polar_sine = np.sin(canonical_polar)
        angle_cosine = np.cos(canonical_angle)
        angle_sine = np.sin(canonical_angle)
        if information_objective:
            utility, mixture_one = _sphere_information_gain_and_mixture(
                weights,
                h1_probability,
                particle_transverse_amplitudes,
                particle_z_projections,
                phase_cosines,
                phase_sines,
                angle_cosine,
                angle_sine,
                polar_cosine,
                polar_sine,
                contrast,
                readout_error,
            )
        else:
            utility, mixture_one = _sphere_detection_value_and_mixture(
                mean_x,
                mean_y,
                mean_z,
                angle_cosine,
                angle_sine,
                polar_cosine,
                polar_sine,
                contrast,
                readout_error,
                bayes_detection_objective,
                stein_detection_objective,
            )
        return utility, mixture_one, canonical_polar, canonical_angle


    @njit
    def _refine_continuous_sphere_action(
        weights: np.ndarray,
        h1_probability: float,
        particle_transverse_amplitudes: np.ndarray,
        particle_z_projections: np.ndarray,
        phase_cosines: np.ndarray,
        phase_sines: np.ndarray,
        center_polar: float,
        center_angle: float,
        initial_polar_step: float,
        initial_angle_step: float,
        contrast: float,
        readout_error: float,
        information_objective: bool,
        bayes_detection_objective: bool,
        stein_detection_objective: bool,
        mean_x: float,
        mean_y: float,
        mean_z: float,
    ) -> tuple[float, float, float, float]:
        best_utility, best_mixture, best_polar, best_angle = _sphere_action_value_and_mixture(
            weights,
            h1_probability,
            particle_transverse_amplitudes,
            particle_z_projections,
            phase_cosines,
            phase_sines,
            center_polar,
            center_angle,
            contrast,
            readout_error,
            information_objective,
            bayes_detection_objective,
            stein_detection_objective,
            mean_x,
            mean_y,
            mean_z,
        )
        polar_step = initial_polar_step
        angle_step = initial_angle_step
        for _ in range(8):
            for polar_direction in range(3):
                delta_polar = (polar_direction - 1) * polar_step
                for angle_direction in range(3):
                    delta_angle = (angle_direction - 1) * angle_step
                    if delta_polar == 0.0 and delta_angle == 0.0:
                        continue
                    utility, mixture_one, trial_polar, trial_angle = _sphere_action_value_and_mixture(
                        weights,
                        h1_probability,
                        particle_transverse_amplitudes,
                        particle_z_projections,
                        phase_cosines,
                        phase_sines,
                        best_polar + delta_polar,
                        best_angle + delta_angle,
                        contrast,
                        readout_error,
                        information_objective,
                        bayes_detection_objective,
                        stein_detection_objective,
                        mean_x,
                        mean_y,
                        mean_z,
                    )
                    if utility > best_utility + 1.0e-14:
                        best_utility = utility
                        best_mixture = mixture_one
                        best_polar = trial_polar
                        best_angle = trial_angle
            polar_step *= 0.5
            angle_step *= 0.5
        return best_polar, best_angle, best_utility, best_mixture


    @njit
    def _update_hypothetical_branch(
        weights: np.ndarray,
        h1_probability: float,
        candidate_index: int,
        outcome_is_one: bool,
        mixture_one: float,
        particle_one_candidates: np.ndarray,
        h0_transverse_one: float,
        branch_weights: np.ndarray,
    ) -> float:
        if outcome_is_one:
            mixture_likelihood = _clip_probability(mixture_one)
            h0_likelihood = h0_transverse_one
        else:
            mixture_likelihood = _clip_probability(1.0 - mixture_one)
            h0_likelihood = _clip_probability(1.0 - h0_transverse_one)

        for particle_index in range(weights.size):
            particle_likelihood = particle_one_candidates[candidate_index, particle_index]
            if not outcome_is_one:
                particle_likelihood = 1.0 - particle_likelihood
            branch_weights[particle_index] = weights[particle_index] * particle_likelihood / mixture_likelihood

        evidence = h1_probability * mixture_likelihood + (1.0 - h1_probability) * h0_likelihood
        if evidence < EPSILON:
            evidence = EPSILON
        return h1_probability * mixture_likelihood / evidence


    @njit
    def _best_depth1_value(
        weights: np.ndarray,
        h1_probability: float,
        particle_one_candidates: np.ndarray,
        particle_entropy_candidates: np.ndarray,
        h0_transverse_one: float,
        h0_transverse_entropy: float,
    ) -> float:
        best_value = -1.0e300
        for candidate_index in range(particle_one_candidates.shape[0]):
            utility, _, _ = _candidate_utility_and_mixture(
                weights,
                h1_probability,
                candidate_index,
                particle_one_candidates,
                particle_entropy_candidates,
                h0_transverse_one,
                h0_transverse_entropy,
            )
            if utility > best_value + 1.0e-14:
                best_value = utility
        return best_value


    @njit
    def _candidate_depth2_value(
        weights: np.ndarray,
        h1_probability: float,
        candidate_index: int,
        particle_one_candidates_now: np.ndarray,
        particle_entropy_candidates_now: np.ndarray,
        particle_one_candidates_next: np.ndarray,
        particle_entropy_candidates_next: np.ndarray,
        h0_transverse_one: float,
        h0_transverse_entropy: float,
        branch_weights: np.ndarray,
    ) -> float:
        utility, mixture_one, predictive_one = _candidate_utility_and_mixture(
            weights,
            h1_probability,
            candidate_index,
            particle_one_candidates_now,
            particle_entropy_candidates_now,
            h0_transverse_one,
            h0_transverse_entropy,
        )

        next_h1 = _update_hypothetical_branch(
            weights,
            h1_probability,
            candidate_index,
            True,
            mixture_one,
            particle_one_candidates_now,
            h0_transverse_one,
            branch_weights,
        )
        future_one = _best_depth1_value(
            branch_weights,
            next_h1,
            particle_one_candidates_next,
            particle_entropy_candidates_next,
            h0_transverse_one,
            h0_transverse_entropy,
        )

        next_h1 = _update_hypothetical_branch(
            weights,
            h1_probability,
            candidate_index,
            False,
            mixture_one,
            particle_one_candidates_now,
            h0_transverse_one,
            branch_weights,
        )
        future_zero = _best_depth1_value(
            branch_weights,
            next_h1,
            particle_one_candidates_next,
            particle_entropy_candidates_next,
            h0_transverse_one,
            h0_transverse_entropy,
        )

        return utility + predictive_one * future_one + (1.0 - predictive_one) * future_zero


    @njit
    def _best_depth2_value(
        weights: np.ndarray,
        h1_probability: float,
        particle_one_candidates_now: np.ndarray,
        particle_entropy_candidates_now: np.ndarray,
        particle_one_candidates_next: np.ndarray,
        particle_entropy_candidates_next: np.ndarray,
        h0_transverse_one: float,
        h0_transverse_entropy: float,
        branch_weights: np.ndarray,
    ) -> float:
        best_value = -1.0e300
        for candidate_index in range(particle_one_candidates_now.shape[0]):
            value = _candidate_depth2_value(
                weights,
                h1_probability,
                candidate_index,
                particle_one_candidates_now,
                particle_entropy_candidates_now,
                particle_one_candidates_next,
                particle_entropy_candidates_next,
                h0_transverse_one,
                h0_transverse_entropy,
                branch_weights,
            )
            if value > best_value + 1.0e-14:
                best_value = value
        return best_value


    @njit
    def _candidate_depth3_value(
        weights: np.ndarray,
        h1_probability: float,
        candidate_index: int,
        particle_one_candidates_now: np.ndarray,
        particle_entropy_candidates_now: np.ndarray,
        particle_one_candidates_next: np.ndarray,
        particle_entropy_candidates_next: np.ndarray,
        particle_one_candidates_next2: np.ndarray,
        particle_entropy_candidates_next2: np.ndarray,
        h0_transverse_one: float,
        h0_transverse_entropy: float,
        branch_weights: np.ndarray,
        nested_branch_weights: np.ndarray,
    ) -> float:
        utility, mixture_one, predictive_one = _candidate_utility_and_mixture(
            weights,
            h1_probability,
            candidate_index,
            particle_one_candidates_now,
            particle_entropy_candidates_now,
            h0_transverse_one,
            h0_transverse_entropy,
        )

        next_h1 = _update_hypothetical_branch(
            weights,
            h1_probability,
            candidate_index,
            True,
            mixture_one,
            particle_one_candidates_now,
            h0_transverse_one,
            branch_weights,
        )
        future_one = _best_depth2_value(
            branch_weights,
            next_h1,
            particle_one_candidates_next,
            particle_entropy_candidates_next,
            particle_one_candidates_next2,
            particle_entropy_candidates_next2,
            h0_transverse_one,
            h0_transverse_entropy,
            nested_branch_weights,
        )

        next_h1 = _update_hypothetical_branch(
            weights,
            h1_probability,
            candidate_index,
            False,
            mixture_one,
            particle_one_candidates_now,
            h0_transverse_one,
            branch_weights,
        )
        future_zero = _best_depth2_value(
            branch_weights,
            next_h1,
            particle_one_candidates_next,
            particle_entropy_candidates_next,
            particle_one_candidates_next2,
            particle_entropy_candidates_next2,
            h0_transverse_one,
            h0_transverse_entropy,
            nested_branch_weights,
        )

        return utility + predictive_one * future_one + (1.0 - predictive_one) * future_zero


    @njit
    def _best_depth3_value(
        weights: np.ndarray,
        h1_probability: float,
        particle_one_candidates_now: np.ndarray,
        particle_entropy_candidates_now: np.ndarray,
        particle_one_candidates_next: np.ndarray,
        particle_entropy_candidates_next: np.ndarray,
        particle_one_candidates_next2: np.ndarray,
        particle_entropy_candidates_next2: np.ndarray,
        h0_transverse_one: float,
        h0_transverse_entropy: float,
        branch_weights: np.ndarray,
        nested_branch_weights: np.ndarray,
    ) -> float:
        best_value = -1.0e300
        for candidate_index in range(particle_one_candidates_now.shape[0]):
            value = _candidate_depth3_value(
                weights,
                h1_probability,
                candidate_index,
                particle_one_candidates_now,
                particle_entropy_candidates_now,
                particle_one_candidates_next,
                particle_entropy_candidates_next,
                particle_one_candidates_next2,
                particle_entropy_candidates_next2,
                h0_transverse_one,
                h0_transverse_entropy,
                branch_weights,
                nested_branch_weights,
            )
            if value > best_value + 1.0e-14:
                best_value = value
        return best_value


    @njit
    def _candidate_depth4_value(
        weights: np.ndarray,
        h1_probability: float,
        candidate_index: int,
        particle_one_candidates_now: np.ndarray,
        particle_entropy_candidates_now: np.ndarray,
        particle_one_candidates_next: np.ndarray,
        particle_entropy_candidates_next: np.ndarray,
        particle_one_candidates_next2: np.ndarray,
        particle_entropy_candidates_next2: np.ndarray,
        particle_one_candidates_next3: np.ndarray,
        particle_entropy_candidates_next3: np.ndarray,
        h0_transverse_one: float,
        h0_transverse_entropy: float,
        branch_weights: np.ndarray,
        nested_branch_weights: np.ndarray,
        deep_branch_weights: np.ndarray,
    ) -> float:
        utility, mixture_one, predictive_one = _candidate_utility_and_mixture(
            weights,
            h1_probability,
            candidate_index,
            particle_one_candidates_now,
            particle_entropy_candidates_now,
            h0_transverse_one,
            h0_transverse_entropy,
        )

        next_h1 = _update_hypothetical_branch(
            weights,
            h1_probability,
            candidate_index,
            True,
            mixture_one,
            particle_one_candidates_now,
            h0_transverse_one,
            branch_weights,
        )
        future_one = _best_depth3_value(
            branch_weights,
            next_h1,
            particle_one_candidates_next,
            particle_entropy_candidates_next,
            particle_one_candidates_next2,
            particle_entropy_candidates_next2,
            particle_one_candidates_next3,
            particle_entropy_candidates_next3,
            h0_transverse_one,
            h0_transverse_entropy,
            nested_branch_weights,
            deep_branch_weights,
        )

        next_h1 = _update_hypothetical_branch(
            weights,
            h1_probability,
            candidate_index,
            False,
            mixture_one,
            particle_one_candidates_now,
            h0_transverse_one,
            branch_weights,
        )
        future_zero = _best_depth3_value(
            branch_weights,
            next_h1,
            particle_one_candidates_next,
            particle_entropy_candidates_next,
            particle_one_candidates_next2,
            particle_entropy_candidates_next2,
            particle_one_candidates_next3,
            particle_entropy_candidates_next3,
            h0_transverse_one,
            h0_transverse_entropy,
            nested_branch_weights,
            deep_branch_weights,
        )

        return utility + predictive_one * future_one + (1.0 - predictive_one) * future_zero


    @njit
    def _myopic_choice_for_tables(
        weights: np.ndarray,
        h1_probability: float,
        particle_one_candidates: np.ndarray,
        particle_entropy_candidates: np.ndarray,
        h0_transverse_one: float,
        h0_transverse_entropy: float,
    ) -> tuple[int, float, float, float]:
        best_candidate = 0
        best_utility = -1.0e300
        best_mixture_one = 0.5
        best_predictive_one = 0.5
        for candidate_index in range(particle_one_candidates.shape[0]):
            utility, mixture_one, predictive_one = _candidate_utility_and_mixture(
                weights,
                h1_probability,
                candidate_index,
                particle_one_candidates,
                particle_entropy_candidates,
                h0_transverse_one,
                h0_transverse_entropy,
            )
            if utility > best_utility + 1.0e-14:
                best_candidate = candidate_index
                best_utility = utility
                best_mixture_one = mixture_one
                best_predictive_one = predictive_one
        return best_candidate, best_utility, best_mixture_one, best_predictive_one


    @njit
    def _candidate_rollout_value_and_mixture(
        weights: np.ndarray,
        h1_probability: float,
        first_candidate: int,
        rollout_particle_one_candidates: np.ndarray,
        rollout_particle_entropy_candidates: np.ndarray,
        h0_transverse_one: float,
        h0_transverse_entropy: float,
        depth: int,
        path_weights: np.ndarray,
    ) -> tuple[float, float]:
        utility, mixture_one, predictive_one = _candidate_utility_and_mixture(
            weights,
            h1_probability,
            first_candidate,
            rollout_particle_one_candidates[0],
            rollout_particle_entropy_candidates[0],
            h0_transverse_one,
            h0_transverse_entropy,
        )
        if depth <= 1:
            return utility, mixture_one

        expected_future = 0.0
        branch_count = 1 << (depth - 1)
        for branch_mask in range(branch_count):
            branch_probability = 1.0
            future_value = 0.0

            outcome_is_one = (branch_mask & 1) != 0
            if outcome_is_one:
                branch_probability *= _clip_probability(predictive_one)
            else:
                branch_probability *= _clip_probability(1.0 - predictive_one)
            next_h1 = _update_hypothetical_branch(
                weights,
                h1_probability,
                first_candidate,
                outcome_is_one,
                mixture_one,
                rollout_particle_one_candidates[0],
                h0_transverse_one,
                path_weights,
            )

            for offset in range(1, depth):
                (
                    best_candidate,
                    best_utility,
                    best_mixture_one,
                    best_predictive_one,
                ) = _myopic_choice_for_tables(
                    path_weights,
                    next_h1,
                    rollout_particle_one_candidates[offset],
                    rollout_particle_entropy_candidates[offset],
                    h0_transverse_one,
                    h0_transverse_entropy,
                )
                future_value += best_utility

                if offset < depth - 1:
                    outcome_is_one = ((branch_mask >> offset) & 1) != 0
                    if outcome_is_one:
                        branch_probability *= _clip_probability(best_predictive_one)
                    else:
                        branch_probability *= _clip_probability(1.0 - best_predictive_one)
                    next_h1 = _update_hypothetical_branch(
                        path_weights,
                        next_h1,
                        best_candidate,
                        outcome_is_one,
                        best_mixture_one,
                        rollout_particle_one_candidates[offset],
                        h0_transverse_one,
                        path_weights,
                    )

            expected_future += branch_probability * future_value

        return utility + expected_future, mixture_one


    @njit(parallel=True)
    def _run_rabi_chunk(
        uniforms: np.ndarray,
        weights: np.ndarray,
        h1_probabilities: np.ndarray,
        log_bayes_factors: np.ndarray,
        current_time: float,
        shot_offset: int,
        strategy_code: int,
        prior_weights: np.ndarray,
        particle_detunings: np.ndarray,
        particle_drive_phases: np.ndarray,
        particle_transverse_amplitudes: np.ndarray,
        particle_z_projections: np.ndarray,
        particle_z_one: np.ndarray,
        step_cosines: np.ndarray,
        step_sines: np.ndarray,
        candidate_polars: np.ndarray,
        candidate_angles: np.ndarray,
        candidate_cosines: np.ndarray,
        candidate_sines: np.ndarray,
        fixed_angle: float,
        fixed_angle_cosine: float,
        fixed_angle_sine: float,
        sensing_time: float,
        dead_time: float,
        contrast: float,
        readout_error: float,
        h0_z_one: float,
        h0_transverse_one: float,
        h0_transverse_entropy: float,
        true_is_h1: bool,
        true_detuning: float,
        true_drive_phase: float,
        true_drive_phases_batch: np.ndarray,
        use_true_drive_phases_batch: bool,
        true_transverse_amplitude: float,
        true_z_projection: float,
        true_z_one: float,
        true_step_cosine: float,
        true_step_sine: float,
        total_shots: int,
        lookahead_depth: int,
        two_stage_switch_shot: int,
        adaptive_hybrid_switch_shot: int,
        periodic_hybrid_period: int,
        blend_lambda_max: float,
        blend_r_start: float,
        blend_r_stop: float,
        checkpoint_shots: np.ndarray,
        checkpoint_log_bayes_factors: np.ndarray,
        diagnostics_enabled: bool,
        cumulative_predictable_kl: np.ndarray,
        cumulative_predictable_variance: np.ndarray,
        cumulative_design_xx: np.ndarray,
        cumulative_design_xy: np.ndarray,
        cumulative_design_yy: np.ndarray,
        checkpoint_predictable_kl: np.ndarray,
        checkpoint_predictable_variance: np.ndarray,
        checkpoint_design: np.ndarray,
        cumulative_signal_null_kl: np.ndarray,
        cumulative_signal_predictive_kl: np.ndarray,
        checkpoint_signal_null_kl: np.ndarray,
        checkpoint_signal_predictive_kl: np.ndarray,
    ) -> float:
        shot_count = uniforms.shape[0]
        batch_count = uniforms.shape[1]
        particle_count = prior_weights.size
        candidate_polar_count = candidate_polars.size
        candidate_count = candidate_angles.size

        particle_phase_cosines = np.empty(particle_count, dtype=np.float64)
        particle_phase_sines = np.empty(particle_count, dtype=np.float64)
        particle_phase_cosines_next = np.empty(particle_count, dtype=np.float64)
        particle_phase_sines_next = np.empty(particle_count, dtype=np.float64)
        particle_phase_cosines_next2 = np.empty(particle_count, dtype=np.float64)
        particle_phase_sines_next2 = np.empty(particle_count, dtype=np.float64)
        particle_phase_cosines_next3 = np.empty(particle_count, dtype=np.float64)
        particle_phase_sines_next3 = np.empty(particle_count, dtype=np.float64)
        rollout_phase_cosines = np.empty(particle_count, dtype=np.float64)
        rollout_phase_sines = np.empty(particle_count, dtype=np.float64)
        rollout_phase_cosines_next = np.empty(particle_count, dtype=np.float64)
        rollout_phase_sines_next = np.empty(particle_count, dtype=np.float64)
        particle_one = np.empty(particle_count, dtype=np.float64)
        particle_one_candidates = np.empty((candidate_count, particle_count), dtype=np.float64)
        particle_entropy_candidates = np.empty((candidate_count, particle_count), dtype=np.float64)
        particle_one_candidates_next = np.empty((candidate_count, particle_count), dtype=np.float64)
        particle_entropy_candidates_next = np.empty((candidate_count, particle_count), dtype=np.float64)
        particle_one_candidates_next2 = np.empty((candidate_count, particle_count), dtype=np.float64)
        particle_entropy_candidates_next2 = np.empty((candidate_count, particle_count), dtype=np.float64)
        particle_one_candidates_next3 = np.empty((candidate_count, particle_count), dtype=np.float64)
        particle_entropy_candidates_next3 = np.empty((candidate_count, particle_count), dtype=np.float64)
        rollout_depth_capacity = lookahead_depth
        if rollout_depth_capacity < 1:
            rollout_depth_capacity = 1
        rollout_particle_one_candidates = np.empty(
            (rollout_depth_capacity, candidate_count, particle_count), dtype=np.float64
        )
        rollout_particle_entropy_candidates = np.empty(
            (rollout_depth_capacity, candidate_count, particle_count), dtype=np.float64
        )
        true_phase_cosines = np.empty(batch_count, dtype=np.float64)
        true_phase_sines = np.empty(batch_count, dtype=np.float64)
        true_one_fixed_batch = np.empty(batch_count, dtype=np.float64)
        true_one_candidates = np.empty((candidate_count, batch_count), dtype=np.float64)
        branch_weights = np.empty((batch_count, particle_count), dtype=np.float64)
        nested_branch_weights = np.empty((batch_count, particle_count), dtype=np.float64)
        deep_branch_weights = np.empty((batch_count, particle_count), dtype=np.float64)

        for particle_index in range(particle_count):
            phase = particle_drive_phases[particle_index] + particle_detunings[particle_index] * (
                current_time + 0.5 * sensing_time
            )
            particle_phase_cosines[particle_index] = np.cos(phase)
            particle_phase_sines[particle_index] = np.sin(phase)

        for batch_index in range(batch_count):
            true_phase = true_drive_phase
            if use_true_drive_phases_batch:
                true_phase = true_drive_phases_batch[batch_index]
            true_phase = true_phase + true_detuning * (current_time + 0.5 * sensing_time)
            true_phase_cosines[batch_index] = np.cos(true_phase)
            true_phase_sines[batch_index] = np.sin(true_phase)

        for local_shot in range(shot_count):
            global_shot = shot_offset + local_shot
            active_strategy_code = strategy_code
            if strategy_code == 5:
                if global_shot < two_stage_switch_shot:
                    active_strategy_code = 2
                else:
                    active_strategy_code = 3
            elif strategy_code == 9:
                if global_shot < adaptive_hybrid_switch_shot:
                    active_strategy_code = 3
                else:
                    active_strategy_code = 7
            elif strategy_code == 10:
                if global_shot < adaptive_hybrid_switch_shot:
                    active_strategy_code = 6
                else:
                    active_strategy_code = 8
            axis_code = 1
            angle_cosine = fixed_angle_cosine
            angle_sine = fixed_angle_sine
            h0_one = h0_transverse_one
            effective_depth = 1

            if active_strategy_code == 0:
                axis_code = 0
                h0_one = h0_z_one
                for batch_index in range(batch_count):
                    true_one_fixed_batch[batch_index] = h0_z_one
                    if true_is_h1:
                        true_one_fixed_batch[batch_index] = true_z_one
                for particle_index in range(particle_count):
                    particle_one[particle_index] = particle_z_one[particle_index]
            elif active_strategy_code == 1:
                for batch_index in range(batch_count):
                    true_one_fixed_batch[batch_index] = h0_transverse_one
                    if true_is_h1:
                        true_one_fixed_batch[batch_index] = _transverse_probability(
                            true_transverse_amplitude,
                            true_phase_cosines[batch_index],
                            true_phase_sines[batch_index],
                            angle_cosine,
                            angle_sine,
                            contrast,
                            readout_error,
                        )
                for particle_index in range(particle_count):
                    particle_one[particle_index] = _transverse_probability(
                        particle_transverse_amplitudes[particle_index],
                        particle_phase_cosines[particle_index],
                        particle_phase_sines[particle_index],
                        angle_cosine,
                        angle_sine,
                        contrast,
                        readout_error,
                    )
            elif active_strategy_code == 2:
                angle_index = global_shot % 2
                if angle_index == 0:
                    angle_cosine = 1.0
                    angle_sine = 0.0
                else:
                    angle_cosine = 0.0
                    angle_sine = 1.0
                for batch_index in range(batch_count):
                    true_one_fixed_batch[batch_index] = h0_transverse_one
                    if true_is_h1:
                        true_one_fixed_batch[batch_index] = _transverse_probability(
                            true_transverse_amplitude,
                            true_phase_cosines[batch_index],
                            true_phase_sines[batch_index],
                            angle_cosine,
                            angle_sine,
                            contrast,
                            readout_error,
                        )
                for particle_index in range(particle_count):
                    particle_one[particle_index] = _transverse_probability(
                        particle_transverse_amplitudes[particle_index],
                        particle_phase_cosines[particle_index],
                        particle_phase_sines[particle_index],
                        angle_cosine,
                        angle_sine,
                        contrast,
                        readout_error,
                    )
            elif (
                active_strategy_code == 13
                or active_strategy_code == 14
                or active_strategy_code == 15
                or active_strategy_code == 16
                or active_strategy_code == 17
            ):
                pass
            else:
                _fill_transverse_candidate_tables(
                    particle_phase_cosines,
                    particle_phase_sines,
                    particle_transverse_amplitudes,
                    candidate_cosines,
                    candidate_sines,
                    contrast,
                    readout_error,
                    particle_one_candidates,
                    particle_entropy_candidates,
                )
                for candidate_index in range(candidate_count):
                    candidate_cosine = candidate_cosines[candidate_index]
                    candidate_sine = candidate_sines[candidate_index]
                    for batch_index in range(batch_count):
                        true_one_candidates[candidate_index, batch_index] = h0_transverse_one
                        if true_is_h1:
                            true_one_candidates[candidate_index, batch_index] = _transverse_probability(
                                true_transverse_amplitude,
                                true_phase_cosines[batch_index],
                                true_phase_sines[batch_index],
                                candidate_cosine,
                                candidate_sine,
                                contrast,
                                readout_error,
                            )

                remaining_shots = total_shots - global_shot
                effective_depth = lookahead_depth
                if effective_depth > remaining_shots:
                    effective_depth = remaining_shots
                if active_strategy_code == 4:
                    for candidate_index in range(candidate_count):
                        for particle_index in range(particle_count):
                            rollout_particle_one_candidates[0, candidate_index, particle_index] = (
                                particle_one_candidates[candidate_index, particle_index]
                            )
                            rollout_particle_entropy_candidates[0, candidate_index, particle_index] = (
                                particle_entropy_candidates[candidate_index, particle_index]
                            )
                    for particle_index in range(particle_count):
                        rollout_phase_cosines[particle_index] = particle_phase_cosines[particle_index]
                        rollout_phase_sines[particle_index] = particle_phase_sines[particle_index]
                    for rollout_offset in range(1, effective_depth):
                        _fill_future_phases(
                            rollout_phase_cosines,
                            rollout_phase_sines,
                            step_cosines,
                            step_sines,
                            rollout_phase_cosines_next,
                            rollout_phase_sines_next,
                        )
                        for particle_index in range(particle_count):
                            rollout_phase_cosines[particle_index] = rollout_phase_cosines_next[particle_index]
                            rollout_phase_sines[particle_index] = rollout_phase_sines_next[particle_index]
                        _fill_transverse_candidate_tables(
                            rollout_phase_cosines,
                            rollout_phase_sines,
                            particle_transverse_amplitudes,
                            candidate_cosines,
                            candidate_sines,
                            contrast,
                            readout_error,
                            rollout_particle_one_candidates[rollout_offset],
                            rollout_particle_entropy_candidates[rollout_offset],
                        )
                if active_strategy_code == 3 and effective_depth >= 2:
                    _fill_future_phases(
                        particle_phase_cosines,
                        particle_phase_sines,
                        step_cosines,
                        step_sines,
                        particle_phase_cosines_next,
                        particle_phase_sines_next,
                    )
                    _fill_transverse_candidate_tables(
                        particle_phase_cosines_next,
                        particle_phase_sines_next,
                        particle_transverse_amplitudes,
                        candidate_cosines,
                        candidate_sines,
                        contrast,
                        readout_error,
                        particle_one_candidates_next,
                        particle_entropy_candidates_next,
                    )
                if active_strategy_code == 3 and effective_depth >= 3:
                    _fill_future_phases(
                        particle_phase_cosines_next,
                        particle_phase_sines_next,
                        step_cosines,
                        step_sines,
                        particle_phase_cosines_next2,
                        particle_phase_sines_next2,
                    )
                    _fill_transverse_candidate_tables(
                        particle_phase_cosines_next2,
                        particle_phase_sines_next2,
                        particle_transverse_amplitudes,
                        candidate_cosines,
                        candidate_sines,
                        contrast,
                        readout_error,
                        particle_one_candidates_next2,
                        particle_entropy_candidates_next2,
                    )
                if active_strategy_code == 3 and effective_depth >= 4:
                    _fill_future_phases(
                        particle_phase_cosines_next2,
                        particle_phase_sines_next2,
                        step_cosines,
                        step_sines,
                        particle_phase_cosines_next3,
                        particle_phase_sines_next3,
                    )
                    _fill_transverse_candidate_tables(
                        particle_phase_cosines_next3,
                        particle_phase_sines_next3,
                        particle_transverse_amplitudes,
                        candidate_cosines,
                        candidate_sines,
                        contrast,
                        readout_error,
                        particle_one_candidates_next3,
                        particle_entropy_candidates_next3,
                    )

            for batch_index in prange(batch_count):
                diagnostic_h0_one = h0_transverse_one
                diagnostic_mixture_one = 0.5
                diagnostic_qx = 0.0
                diagnostic_qy = 0.0
                if (
                    active_strategy_code == 13
                    or active_strategy_code == 14
                    or active_strategy_code == 15
                    or active_strategy_code == 16
                    or active_strategy_code == 17
                ):
                    h1_probability = h1_probabilities[batch_index]
                    information_objective = active_strategy_code == 13
                    if active_strategy_code == 15 or active_strategy_code == 16 or active_strategy_code == 17:
                        information_objective = global_shot % periodic_hybrid_period == 0
                    bayes_detection_objective = active_strategy_code == 14 or active_strategy_code == 16
                    stein_detection_objective = active_strategy_code == 17

                    mean_x = 0.0
                    mean_y = 0.0
                    mean_z = 0.0
                    if not information_objective:
                        mean_x, mean_y, mean_z = _posterior_bloch_moments(
                            weights[batch_index],
                            particle_transverse_amplitudes,
                            particle_z_projections,
                            particle_phase_cosines,
                            particle_phase_sines,
                        )

                    best_polar = 0.5 * np.pi
                    best_angle = candidate_angles[0]
                    best_utility = -1.0e300
                    best_mixture_one = 0.5
                    for polar_index in range(candidate_polar_count):
                        polar = candidate_polars[polar_index]
                        for candidate_index in range(candidate_count):
                            utility, mixture_one, trial_polar, trial_angle = _sphere_action_value_and_mixture(
                                weights[batch_index],
                                h1_probability,
                                particle_transverse_amplitudes,
                                particle_z_projections,
                                particle_phase_cosines,
                                particle_phase_sines,
                                polar,
                                candidate_angles[candidate_index],
                                contrast,
                                readout_error,
                                information_objective,
                                bayes_detection_objective,
                                stein_detection_objective,
                                mean_x,
                                mean_y,
                                mean_z,
                            )
                            if utility > best_utility + 1.0e-14:
                                best_utility = utility
                                best_mixture_one = mixture_one
                                best_polar = trial_polar
                                best_angle = trial_angle

                    polar_step = np.pi
                    if candidate_polar_count > 1:
                        polar_step = np.pi / (candidate_polar_count - 1)
                    angle_step = np.pi
                    if candidate_count > 1:
                        angle_step = np.pi / candidate_count
                    best_polar, best_angle, best_utility, best_mixture_one = _refine_continuous_sphere_action(
                        weights[batch_index],
                        h1_probability,
                        particle_transverse_amplitudes,
                        particle_z_projections,
                        particle_phase_cosines,
                        particle_phase_sines,
                        best_polar,
                        best_angle,
                        polar_step,
                        angle_step,
                        contrast,
                        readout_error,
                        information_objective,
                        bayes_detection_objective,
                        stein_detection_objective,
                        mean_x,
                        mean_y,
                        mean_z,
                    )

                    best_polar_cosine = np.cos(best_polar)
                    best_polar_sine = np.sin(best_polar)
                    best_angle_cosine = np.cos(best_angle)
                    best_angle_sine = np.sin(best_angle)
                    # Keep the trajectory-dependent null separate from the shared fixed-axis null.
                    sphere_h0_one = _sphere_h0_probability(best_polar_cosine, contrast, readout_error)
                    diagnostic_h0_one = sphere_h0_one
                    diagnostic_mixture_one = best_mixture_one
                    diagnostic_qx = best_polar_sine * best_angle_sine
                    diagnostic_qy = -best_polar_sine * best_angle_cosine

                    true_one = sphere_h0_one
                    if true_is_h1:
                        true_one = _sphere_probability_from_components(
                            true_transverse_amplitude,
                            true_z_projection,
                            true_phase_cosines[batch_index],
                            true_phase_sines[batch_index],
                            best_angle_cosine,
                            best_angle_sine,
                            best_polar_cosine,
                            best_polar_sine,
                            contrast,
                            readout_error,
                        )

                    diagnostic_true_one = true_one
                    outcome_is_one = uniforms[local_shot, batch_index] < true_one
                    if outcome_is_one:
                        mixture_likelihood = _clip_probability(best_mixture_one)
                        h0_likelihood = sphere_h0_one
                    else:
                        mixture_likelihood = _clip_probability(1.0 - best_mixture_one)
                        h0_likelihood = _clip_probability(1.0 - sphere_h0_one)

                    for particle_index in range(particle_count):
                        particle_likelihood = _sphere_particle_probability(
                            particle_index,
                            particle_transverse_amplitudes,
                            particle_z_projections,
                            particle_phase_cosines,
                            particle_phase_sines,
                            best_angle_cosine,
                            best_angle_sine,
                            best_polar_cosine,
                            best_polar_sine,
                            contrast,
                            readout_error,
                        )
                        if not outcome_is_one:
                            particle_likelihood = 1.0 - particle_likelihood
                        weights[batch_index, particle_index] *= particle_likelihood / mixture_likelihood
                elif active_strategy_code == 6 or active_strategy_code == 8 or active_strategy_code == 12:
                    h1_probability = h1_probabilities[batch_index]
                    best_candidate = 0
                    best_utility = -1.0e300
                    best_mixture_one = 0.5
                    lambda_information = 0.0
                    if active_strategy_code == 12:
                        lambda_information = _blend_lambda(
                            _phase_concentration(
                                weights[batch_index],
                                particle_phase_cosines,
                                particle_phase_sines,
                            ),
                            blend_lambda_max,
                            blend_r_start,
                            blend_r_stop,
                        )

                    for candidate_index in range(candidate_count):
                        if active_strategy_code == 12:
                            utility, mixture_one = _candidate_blended_value_and_mixture(
                                weights[batch_index],
                                h1_probability,
                                candidate_index,
                                particle_one_candidates,
                                particle_entropy_candidates,
                                h0_transverse_one,
                                h0_transverse_entropy,
                                lambda_information,
                            )
                        elif active_strategy_code == 8:
                            utility, mixture_one = _candidate_bayes_drift_and_mixture(
                                weights[batch_index],
                                candidate_index,
                                particle_one_candidates,
                                h0_transverse_one,
                            )
                        else:
                            utility, mixture_one, _ = _candidate_utility_and_mixture(
                                weights[batch_index],
                                h1_probability,
                                candidate_index,
                                particle_one_candidates,
                                particle_entropy_candidates,
                                h0_transverse_one,
                                h0_transverse_entropy,
                            )
                        if utility > best_utility + 1.0e-14:
                            best_utility = utility
                            best_candidate = candidate_index
                            best_mixture_one = mixture_one

                    bracket_width = np.pi
                    if candidate_count > 1:
                        bracket_width = np.pi / candidate_count

                    if active_strategy_code == 12:
                        best_angle, best_utility, best_mixture_one = _refine_continuous_angle_blended(
                            weights[batch_index],
                            h1_probability,
                            particle_phase_cosines,
                            particle_phase_sines,
                            particle_transverse_amplitudes,
                            candidate_angles[best_candidate],
                            bracket_width,
                            contrast,
                            readout_error,
                            h0_transverse_one,
                            h0_transverse_entropy,
                            lambda_information,
                        )
                    elif active_strategy_code == 8:
                        best_angle, best_utility, best_mixture_one = _refine_continuous_angle_bayes_drift(
                            weights[batch_index],
                            particle_phase_cosines,
                            particle_phase_sines,
                            particle_transverse_amplitudes,
                            candidate_angles[best_candidate],
                            bracket_width,
                            contrast,
                            readout_error,
                            h0_transverse_one,
                        )
                    else:
                        best_angle, best_utility, best_mixture_one = _refine_continuous_angle(
                            weights[batch_index],
                            h1_probability,
                            particle_phase_cosines,
                            particle_phase_sines,
                            particle_transverse_amplitudes,
                            candidate_angles[best_candidate],
                            bracket_width,
                            contrast,
                            readout_error,
                            h0_transverse_one,
                            h0_transverse_entropy,
                        )
                    best_cosine = np.cos(best_angle)
                    best_sine = np.sin(best_angle)
                    diagnostic_h0_one = h0_transverse_one
                    diagnostic_mixture_one = best_mixture_one
                    diagnostic_qx = best_sine
                    diagnostic_qy = -best_cosine

                    true_one = h0_transverse_one
                    if true_is_h1:
                        true_one = _transverse_probability(
                            true_transverse_amplitude,
                            true_phase_cosines[batch_index],
                            true_phase_sines[batch_index],
                            best_cosine,
                            best_sine,
                            contrast,
                            readout_error,
                        )

                    diagnostic_true_one = true_one
                    outcome_is_one = uniforms[local_shot, batch_index] < true_one
                    if outcome_is_one:
                        mixture_likelihood = _clip_probability(best_mixture_one)
                        h0_likelihood = h0_transverse_one
                    else:
                        mixture_likelihood = _clip_probability(1.0 - best_mixture_one)
                        h0_likelihood = _clip_probability(1.0 - h0_transverse_one)

                    for particle_index in range(particle_count):
                        particle_likelihood = _transverse_probability(
                            particle_transverse_amplitudes[particle_index],
                            particle_phase_cosines[particle_index],
                            particle_phase_sines[particle_index],
                            best_cosine,
                            best_sine,
                            contrast,
                            readout_error,
                        )
                        if not outcome_is_one:
                            particle_likelihood = 1.0 - particle_likelihood
                        weights[batch_index, particle_index] *= particle_likelihood / mixture_likelihood
                elif (
                    active_strategy_code == 3
                    or active_strategy_code == 4
                    or active_strategy_code == 7
                    or active_strategy_code == 11
                ):
                    h1_probability = h1_probabilities[batch_index]
                    one_minus_h1 = 1.0 - h1_probability
                    best_candidate = 0
                    best_utility = -1.0e300
                    best_mixture_one = 0.5
                    lambda_information = 0.0
                    if active_strategy_code == 11:
                        lambda_information = _blend_lambda(
                            _phase_concentration(
                                weights[batch_index],
                                particle_phase_cosines,
                                particle_phase_sines,
                            ),
                            blend_lambda_max,
                            blend_r_start,
                            blend_r_stop,
                        )

                    for candidate_index in range(candidate_count):
                        if active_strategy_code == 11:
                            utility, mixture_one = _candidate_blended_value_and_mixture(
                                weights[batch_index],
                                h1_probability,
                                candidate_index,
                                particle_one_candidates,
                                particle_entropy_candidates,
                                h0_transverse_one,
                                h0_transverse_entropy,
                                lambda_information,
                            )
                        elif active_strategy_code == 7:
                            utility, mixture_one = _candidate_bayes_drift_and_mixture(
                                weights[batch_index],
                                candidate_index,
                                particle_one_candidates,
                                h0_transverse_one,
                            )
                        elif active_strategy_code == 4:
                            utility, mixture_one = _candidate_rollout_value_and_mixture(
                                weights[batch_index],
                                h1_probability,
                                candidate_index,
                                rollout_particle_one_candidates,
                                rollout_particle_entropy_candidates,
                                h0_transverse_one,
                                h0_transverse_entropy,
                                effective_depth,
                                branch_weights[batch_index],
                            )
                        else:
                            mixture_one = 0.0
                            h1_conditional_entropy = 0.0
                            for particle_index in range(particle_count):
                                weight = weights[batch_index, particle_index]
                                mixture_one += weight * particle_one_candidates[candidate_index, particle_index]
                                h1_conditional_entropy += weight * particle_entropy_candidates[
                                    candidate_index, particle_index
                                ]

                            predictive_one = one_minus_h1 * h0_transverse_one + h1_probability * mixture_one
                            conditional_entropy = (
                                one_minus_h1 * h0_transverse_entropy
                                + h1_probability * h1_conditional_entropy
                            )
                            utility = _binary_entropy_scalar(predictive_one) - conditional_entropy

                        if active_strategy_code == 3 and lookahead_depth >= 4 and effective_depth >= 4:
                            utility = _candidate_depth4_value(
                                weights[batch_index],
                                h1_probability,
                                candidate_index,
                                particle_one_candidates,
                                particle_entropy_candidates,
                                particle_one_candidates_next,
                                particle_entropy_candidates_next,
                                particle_one_candidates_next2,
                                particle_entropy_candidates_next2,
                                particle_one_candidates_next3,
                                particle_entropy_candidates_next3,
                                h0_transverse_one,
                                h0_transverse_entropy,
                                branch_weights[batch_index],
                                nested_branch_weights[batch_index],
                                deep_branch_weights[batch_index],
                            )
                        elif active_strategy_code == 3 and lookahead_depth >= 3 and effective_depth >= 3:
                            utility = _candidate_depth3_value(
                                weights[batch_index],
                                h1_probability,
                                candidate_index,
                                particle_one_candidates,
                                particle_entropy_candidates,
                                particle_one_candidates_next,
                                particle_entropy_candidates_next,
                                particle_one_candidates_next2,
                                particle_entropy_candidates_next2,
                                h0_transverse_one,
                                h0_transverse_entropy,
                                branch_weights[batch_index],
                                nested_branch_weights[batch_index],
                            )
                        elif active_strategy_code == 3 and lookahead_depth >= 2 and effective_depth >= 2:
                            utility = _candidate_depth2_value(
                                weights[batch_index],
                                h1_probability,
                                candidate_index,
                                particle_one_candidates,
                                particle_entropy_candidates,
                                particle_one_candidates_next,
                                particle_entropy_candidates_next,
                                h0_transverse_one,
                                h0_transverse_entropy,
                                branch_weights[batch_index],
                            )
                        if utility > best_utility + 1.0e-14:
                            best_utility = utility
                            best_candidate = candidate_index
                            best_mixture_one = mixture_one

                    diagnostic_h0_one = h0_transverse_one
                    diagnostic_mixture_one = best_mixture_one
                    diagnostic_qx = candidate_sines[best_candidate]
                    diagnostic_qy = -candidate_cosines[best_candidate]
                    diagnostic_true_one = true_one_candidates[best_candidate, batch_index]
                    outcome_is_one = uniforms[local_shot, batch_index] < diagnostic_true_one
                    if outcome_is_one:
                        mixture_likelihood = _clip_probability(best_mixture_one)
                        h0_likelihood = h0_transverse_one
                    else:
                        mixture_likelihood = _clip_probability(1.0 - best_mixture_one)
                        h0_likelihood = _clip_probability(1.0 - h0_transverse_one)

                    for particle_index in range(particle_count):
                        particle_likelihood = particle_one_candidates[best_candidate, particle_index]
                        if not outcome_is_one:
                            particle_likelihood = 1.0 - particle_likelihood
                        weights[batch_index, particle_index] *= particle_likelihood / mixture_likelihood
                else:
                    mixture_one = 0.0
                    for particle_index in range(particle_count):
                        mixture_one += weights[batch_index, particle_index] * particle_one[particle_index]

                    diagnostic_h0_one = h0_one
                    diagnostic_mixture_one = mixture_one
                    if axis_code != 0:
                        diagnostic_qx = angle_sine
                        diagnostic_qy = -angle_cosine
                    diagnostic_true_one = true_one_fixed_batch[batch_index]
                    outcome_is_one = uniforms[local_shot, batch_index] < diagnostic_true_one
                    if outcome_is_one:
                        mixture_likelihood = _clip_probability(mixture_one)
                        h0_likelihood = h0_one
                    else:
                        mixture_likelihood = _clip_probability(1.0 - mixture_one)
                        h0_likelihood = _clip_probability(1.0 - h0_one)

                    for particle_index in range(particle_count):
                        particle_likelihood = particle_one[particle_index]
                        if not outcome_is_one:
                            particle_likelihood = 1.0 - particle_likelihood
                        weights[batch_index, particle_index] *= particle_likelihood / mixture_likelihood

                h1_probability = h1_probabilities[batch_index]
                evidence = h1_probability * mixture_likelihood + (1.0 - h1_probability) * h0_likelihood
                if evidence < EPSILON:
                    evidence = EPSILON
                h1_probabilities[batch_index] = h1_probability * mixture_likelihood / evidence
                log_bayes_factors[batch_index] += np.log(mixture_likelihood) - np.log(h0_likelihood)
                if diagnostics_enabled:
                    diagnostic_kl, diagnostic_variance = _binary_log_likelihood_moments(
                        diagnostic_h0_one,
                        diagnostic_mixture_one,
                    )
                    if true_is_h1:
                        # All three probabilities describe the chosen action before its outcome.
                        cumulative_signal_null_kl[batch_index] += _binary_kl_scalar(
                            diagnostic_true_one, diagnostic_h0_one,
                        )
                        cumulative_signal_predictive_kl[batch_index] += _binary_kl_scalar(
                            diagnostic_true_one, diagnostic_mixture_one,
                        )
                    cumulative_predictable_kl[batch_index] += diagnostic_kl
                    cumulative_predictable_variance[batch_index] += diagnostic_variance
                    cumulative_design_xx[batch_index] += diagnostic_qx * diagnostic_qx
                    cumulative_design_xy[batch_index] += diagnostic_qx * diagnostic_qy
                    cumulative_design_yy[batch_index] += diagnostic_qy * diagnostic_qy

            for checkpoint_index in range(checkpoint_shots.size):
                if checkpoint_shots[checkpoint_index] == global_shot + 1:
                    for batch_index in range(batch_count):
                        checkpoint_log_bayes_factors[checkpoint_index, batch_index] = log_bayes_factors[batch_index]
                        if diagnostics_enabled:
                            checkpoint_predictable_kl[checkpoint_index, batch_index] = (
                                cumulative_predictable_kl[batch_index]
                            )
                            checkpoint_predictable_variance[checkpoint_index, batch_index] = (
                                cumulative_predictable_variance[batch_index]
                            )
                            checkpoint_design[checkpoint_index, batch_index, 0] = cumulative_design_xx[batch_index]
                            checkpoint_design[checkpoint_index, batch_index, 1] = cumulative_design_xy[batch_index]
                            checkpoint_design[checkpoint_index, batch_index, 2] = cumulative_design_yy[batch_index]
                            if true_is_h1:
                                checkpoint_signal_null_kl[checkpoint_index, batch_index] = (
                                    cumulative_signal_null_kl[batch_index]
                                )
                                checkpoint_signal_predictive_kl[checkpoint_index, batch_index] = (
                                    cumulative_signal_predictive_kl[batch_index]
                                )

            for particle_index in range(particle_count):
                next_cosine = (
                    particle_phase_cosines[particle_index] * step_cosines[particle_index]
                    - particle_phase_sines[particle_index] * step_sines[particle_index]
                )
                next_sine = (
                    particle_phase_sines[particle_index] * step_cosines[particle_index]
                    + particle_phase_cosines[particle_index] * step_sines[particle_index]
                )
                particle_phase_cosines[particle_index] = next_cosine
                particle_phase_sines[particle_index] = next_sine

            for batch_index in range(batch_count):
                next_true_cosine = (
                    true_phase_cosines[batch_index] * true_step_cosine
                    - true_phase_sines[batch_index] * true_step_sine
                )
                next_true_sine = (
                    true_phase_sines[batch_index] * true_step_cosine
                    + true_phase_cosines[batch_index] * true_step_sine
                )
                true_phase_cosines[batch_index] = next_true_cosine
                true_phase_sines[batch_index] = next_true_sine
            current_time += sensing_time + dead_time

        return current_time


    @njit
    def _run_rabi_sphere_learning_trajectory(
        uniforms: np.ndarray,
        prior_weights: np.ndarray,
        particle_detunings: np.ndarray,
        particle_drive_phases: np.ndarray,
        particle_transverse_amplitudes: np.ndarray,
        particle_z_projections: np.ndarray,
        step_cosines: np.ndarray,
        step_sines: np.ndarray,
        candidate_polars: np.ndarray,
        candidate_angles: np.ndarray,
        sensing_time: float,
        dead_time: float,
        contrast: float,
        readout_error: float,
        true_is_h1: bool,
        true_detuning: float,
        true_drive_phase: float,
        true_transverse_amplitude: float,
        true_z_projection: float,
        true_step_cosine: float,
        true_step_sine: float,
        initial_h1_probability: float,
        strategy_code: int,
        periodic_hybrid_period: int,
        snapshot_shots: np.ndarray,
        records: np.ndarray,
        snapshot_weights: np.ndarray,
        snapshot_h1_probabilities: np.ndarray,
        snapshot_log_bayes_factors: np.ndarray,
    ) -> None:
        n_shots = uniforms.size
        particle_count = prior_weights.size
        candidate_polar_count = candidate_polars.size
        candidate_count = candidate_angles.size
        weights = prior_weights.copy()
        particle_phase_cosines = np.empty(particle_count, dtype=np.float64)
        particle_phase_sines = np.empty(particle_count, dtype=np.float64)

        for particle_index in range(particle_count):
            phase = particle_drive_phases[particle_index] + particle_detunings[particle_index] * (0.5 * sensing_time)
            particle_phase_cosines[particle_index] = np.cos(phase)
            particle_phase_sines[particle_index] = np.sin(phase)

        true_phase = true_drive_phase + true_detuning * (0.5 * sensing_time)
        true_phase_cosine = np.cos(true_phase)
        true_phase_sine = np.sin(true_phase)
        h1_probability = initial_h1_probability
        log_bayes_factor = 0.0

        snapshot_index = 0
        if snapshot_shots.size > 0 and snapshot_shots[0] == 0:
            for particle_index in range(particle_count):
                snapshot_weights[0, particle_index] = weights[particle_index]
            snapshot_h1_probabilities[0] = h1_probability
            snapshot_log_bayes_factors[0] = log_bayes_factor
            snapshot_index = 1

        for shot_index in range(n_shots):
            concentration = _phase_concentration(weights, particle_phase_cosines, particle_phase_sines)
            active_objective_code = 0
            information_objective = True
            bayes_detection_objective = False
            stein_detection_objective = False
            if strategy_code == 14:
                active_objective_code = 2
                information_objective = False
                bayes_detection_objective = True
            elif strategy_code == 15:
                if shot_index % periodic_hybrid_period != 0:
                    active_objective_code = 1
                    information_objective = False
            elif strategy_code == 16:
                if shot_index % periodic_hybrid_period != 0:
                    active_objective_code = 2
                    information_objective = False
                    bayes_detection_objective = True
            elif strategy_code == 17:
                if shot_index % periodic_hybrid_period != 0:
                    active_objective_code = 3
                    information_objective = False
                    stein_detection_objective = True

            mean_x = 0.0
            mean_y = 0.0
            mean_z = 0.0
            if not information_objective:
                mean_x, mean_y, mean_z = _posterior_bloch_moments(
                    weights,
                    particle_transverse_amplitudes,
                    particle_z_projections,
                    particle_phase_cosines,
                    particle_phase_sines,
                )

            best_polar = 0.5 * np.pi
            best_angle = candidate_angles[0]
            best_utility = -1.0e300
            best_mixture_one = 0.5
            for polar_index in range(candidate_polar_count):
                polar = candidate_polars[polar_index]
                for candidate_index in range(candidate_count):
                    utility, mixture_one, trial_polar, trial_angle = _sphere_action_value_and_mixture(
                        weights,
                        h1_probability,
                        particle_transverse_amplitudes,
                        particle_z_projections,
                        particle_phase_cosines,
                        particle_phase_sines,
                        polar,
                        candidate_angles[candidate_index],
                        contrast,
                        readout_error,
                        information_objective,
                        bayes_detection_objective,
                        stein_detection_objective,
                        mean_x,
                        mean_y,
                        mean_z,
                    )
                    if utility > best_utility + 1.0e-14:
                        best_utility = utility
                        best_mixture_one = mixture_one
                        best_polar = trial_polar
                        best_angle = trial_angle

            polar_step = np.pi
            if candidate_polar_count > 1:
                polar_step = np.pi / (candidate_polar_count - 1)
            angle_step = np.pi
            if candidate_count > 1:
                angle_step = np.pi / candidate_count
            best_polar, best_angle, best_utility, best_mixture_one = _refine_continuous_sphere_action(
                weights,
                h1_probability,
                particle_transverse_amplitudes,
                particle_z_projections,
                particle_phase_cosines,
                particle_phase_sines,
                best_polar,
                best_angle,
                polar_step,
                angle_step,
                contrast,
                readout_error,
                information_objective,
                bayes_detection_objective,
                stein_detection_objective,
                mean_x,
                mean_y,
                mean_z,
            )

            best_polar_cosine = np.cos(best_polar)
            best_polar_sine = np.sin(best_polar)
            best_angle_cosine = np.cos(best_angle)
            best_angle_sine = np.sin(best_angle)
            h0_one = _sphere_h0_probability(best_polar_cosine, contrast, readout_error)

            true_one = h0_one
            if true_is_h1:
                true_one = _sphere_probability_from_components(
                    true_transverse_amplitude,
                    true_z_projection,
                    true_phase_cosine,
                    true_phase_sine,
                    best_angle_cosine,
                    best_angle_sine,
                    best_polar_cosine,
                    best_polar_sine,
                    contrast,
                    readout_error,
                )

            outcome_is_one = uniforms[shot_index] < true_one
            if outcome_is_one:
                mixture_likelihood = _clip_probability(best_mixture_one)
                h0_likelihood = h0_one
            else:
                mixture_likelihood = _clip_probability(1.0 - best_mixture_one)
                h0_likelihood = _clip_probability(1.0 - h0_one)

            for particle_index in range(particle_count):
                particle_likelihood = _sphere_particle_probability(
                    particle_index,
                    particle_transverse_amplitudes,
                    particle_z_projections,
                    particle_phase_cosines,
                    particle_phase_sines,
                    best_angle_cosine,
                    best_angle_sine,
                    best_polar_cosine,
                    best_polar_sine,
                    contrast,
                    readout_error,
                )
                if not outcome_is_one:
                    particle_likelihood = 1.0 - particle_likelihood
                weights[particle_index] *= particle_likelihood / mixture_likelihood

            weight_sum = 0.0
            for particle_index in range(particle_count):
                weight_sum += weights[particle_index]
            if weight_sum < EPSILON:
                weight_sum = EPSILON
            for particle_index in range(particle_count):
                weights[particle_index] /= weight_sum

            evidence = h1_probability * mixture_likelihood + (1.0 - h1_probability) * h0_likelihood
            if evidence < EPSILON:
                evidence = EPSILON
            h1_probability = h1_probability * mixture_likelihood / evidence
            log_bayes_factor += np.log(mixture_likelihood) - np.log(h0_likelihood)

            records[shot_index, 0] = shot_index + 1
            records[shot_index, 1] = h1_probability
            records[shot_index, 2] = log_bayes_factor
            records[shot_index, 3] = best_angle
            records[shot_index, 4] = best_polar
            records[shot_index, 5] = active_objective_code
            records[shot_index, 6] = concentration
            records[shot_index, 7] = best_utility
            records[shot_index, 8] = best_utility if active_objective_code == 0 else np.nan
            records[shot_index, 9] = best_utility if active_objective_code == 1 else np.nan
            records[shot_index, 10] = best_utility if active_objective_code == 2 else np.nan
            records[shot_index, 11] = best_utility if active_objective_code == 3 else np.nan
            records[shot_index, 12] = 1.0 if outcome_is_one else 0.0
            records[shot_index, 13] = best_mixture_one
            records[shot_index, 14] = true_one

            while snapshot_index < snapshot_shots.size and snapshot_shots[snapshot_index] == shot_index + 1:
                for particle_index in range(particle_count):
                    snapshot_weights[snapshot_index, particle_index] = weights[particle_index]
                snapshot_h1_probabilities[snapshot_index] = h1_probability
                snapshot_log_bayes_factors[snapshot_index] = log_bayes_factor
                snapshot_index += 1

            for particle_index in range(particle_count):
                next_cosine = (
                    particle_phase_cosines[particle_index] * step_cosines[particle_index]
                    - particle_phase_sines[particle_index] * step_sines[particle_index]
                )
                next_sine = (
                    particle_phase_sines[particle_index] * step_cosines[particle_index]
                    + particle_phase_cosines[particle_index] * step_sines[particle_index]
                )
                particle_phase_cosines[particle_index] = next_cosine
                particle_phase_sines[particle_index] = next_sine

            next_true_cosine = true_phase_cosine * true_step_cosine - true_phase_sine * true_step_sine
            next_true_sine = true_phase_sine * true_step_cosine + true_phase_cosine * true_step_sine
            true_phase_cosine = next_true_cosine
            true_phase_sine = next_true_sine


    @njit
    def _smc_detuning_sinc_factor(detuning: float, sensing_time: float) -> float:
        x = 0.5 * detuning * sensing_time
        if abs(x) < 1.0e-12:
            return 1.0
        return np.sin(x) / x


    @njit
    def _smc_particle_one_probability(
        phase_amplitude: float,
        detuning: float,
        drive_phase: float,
        current_time: float,
        sensing_time: float,
        axis_code: int,
        angle_cosine: float,
        angle_sine: float,
        eta2: float,
        eta1: float,
        contrast: float,
        readout_error: float,
    ) -> float:
        effective_area = phase_amplitude * _smc_detuning_sinc_factor(detuning, sensing_time)
        if axis_code == 0:
            projection = -1.0 + eta1 * (1.0 - np.cos(effective_area))
            return _noisy_probability_from_projection(projection, contrast, readout_error)

        effective_phase = drive_phase + detuning * (current_time + 0.5 * sensing_time)
        return _transverse_probability(
            eta2 * np.sin(effective_area),
            np.cos(effective_phase),
            np.sin(effective_phase),
            angle_cosine,
            angle_sine,
            contrast,
            readout_error,
        )


    @njit
    def _smc_transformed_log_prior_scalar(
        phase_amplitude: float,
        phase_amplitude_min: float,
        phase_amplitude_max: float,
        phase_prior_code: int,
    ) -> float:
        if phase_amplitude < phase_amplitude_min or phase_amplitude > phase_amplitude_max:
            return -1.0e300
        if phase_amplitude_min == phase_amplitude_max:
            return 0.0
        if phase_prior_code == 0:
            return 0.0
        if phase_amplitude < EPSILON:
            return np.log(EPSILON)
        return np.log(phase_amplitude)


    @njit
    def _smc_history_log_likelihood_scalar(
        phase_amplitude: float,
        detuning: float,
        drive_phase: float,
        history_count: int,
        history_axis_codes: np.ndarray,
        history_angle_cosines: np.ndarray,
        history_angle_sines: np.ndarray,
        history_outcomes: np.ndarray,
        sensing_time: float,
        dead_time: float,
        eta2: float,
        eta1: float,
        contrast: float,
        readout_error: float,
    ) -> float:
        log_likelihood = 0.0
        time_step = sensing_time + dead_time
        for history_index in range(history_count):
            probability_one = _smc_particle_one_probability(
                phase_amplitude,
                detuning,
                drive_phase,
                history_index * time_step,
                sensing_time,
                history_axis_codes[history_index],
                history_angle_cosines[history_index],
                history_angle_sines[history_index],
                eta2,
                eta1,
                contrast,
                readout_error,
            )
            if history_outcomes[history_index]:
                log_likelihood += np.log(probability_one)
            else:
                log_likelihood += np.log(1.0 - probability_one)
        return log_likelihood


    @njit
    def _smc_rejuvenate_numba(
        phase_amplitudes: np.ndarray,
        detunings: np.ndarray,
        drive_phases: np.ndarray,
        log_likelihoods: np.ndarray,
        history_count: int,
        history_axis_codes: np.ndarray,
        history_angle_cosines: np.ndarray,
        history_angle_sines: np.ndarray,
        history_outcomes: np.ndarray,
        phase_amplitude_min: float,
        phase_amplitude_max: float,
        detuning_min: float,
        detuning_max: float,
        phase_prior_code: int,
        sensing_time: float,
        dead_time: float,
        eta2: float,
        eta1: float,
        contrast: float,
        readout_error: float,
        move_steps: int,
        log_phase_amplitude_step: float,
        detuning_step: float,
        drive_phase_step: float,
    ) -> None:
        particle_count = phase_amplitudes.size
        amplitude_known = phase_amplitude_min == phase_amplitude_max
        detuning_known = detuning_min == detuning_max

        for _ in range(move_steps):
            for particle_index in range(particle_count):
                old_phase_amplitude = phase_amplitudes[particle_index]
                old_detuning = detunings[particle_index]
                old_drive_phase = drive_phases[particle_index]

                proposed_phase_amplitude = old_phase_amplitude
                if (not amplitude_known) and log_phase_amplitude_step > 0.0:
                    proposed_phase_amplitude = np.exp(
                        np.log(old_phase_amplitude) + np.random.normal(0.0, log_phase_amplitude_step)
                    )
                    if proposed_phase_amplitude < phase_amplitude_min or proposed_phase_amplitude > phase_amplitude_max:
                        continue

                proposed_detuning = old_detuning
                if (not detuning_known) and detuning_step > 0.0:
                    proposed_detuning = old_detuning + np.random.normal(0.0, detuning_step)
                    if proposed_detuning < detuning_min or proposed_detuning > detuning_max:
                        continue

                proposed_drive_phase = old_drive_phase + np.random.normal(0.0, drive_phase_step)
                proposed_drive_phase = proposed_drive_phase % (2.0 * np.pi)

                proposed_log_likelihood = _smc_history_log_likelihood_scalar(
                    proposed_phase_amplitude,
                    proposed_detuning,
                    proposed_drive_phase,
                    history_count,
                    history_axis_codes,
                    history_angle_cosines,
                    history_angle_sines,
                    history_outcomes,
                    sensing_time,
                    dead_time,
                    eta2,
                    eta1,
                    contrast,
                    readout_error,
                )
                old_log_target = log_likelihoods[particle_index] + _smc_transformed_log_prior_scalar(
                    old_phase_amplitude,
                    phase_amplitude_min,
                    phase_amplitude_max,
                    phase_prior_code,
                )
                proposed_log_target = proposed_log_likelihood + _smc_transformed_log_prior_scalar(
                    proposed_phase_amplitude,
                    phase_amplitude_min,
                    phase_amplitude_max,
                    phase_prior_code,
                )
                if np.log(_clip_probability(np.random.random())) < proposed_log_target - old_log_target:
                    phase_amplitudes[particle_index] = proposed_phase_amplitude
                    detunings[particle_index] = proposed_detuning
                    drive_phases[particle_index] = proposed_drive_phase
                    log_likelihoods[particle_index] = proposed_log_likelihood


    @njit
    def _smc_systematic_resample_numba(
        weights: np.ndarray,
        phase_amplitudes: np.ndarray,
        detunings: np.ndarray,
        drive_phases: np.ndarray,
        log_likelihoods: np.ndarray,
        tmp_phase_amplitudes: np.ndarray,
        tmp_detunings: np.ndarray,
        tmp_drive_phases: np.ndarray,
        tmp_log_likelihoods: np.ndarray,
    ) -> None:
        particle_count = weights.size
        start = np.random.random() / particle_count
        cumulative = weights[0]
        source_index = 0
        for output_index in range(particle_count):
            position = start + output_index / particle_count
            while position > cumulative and source_index < particle_count - 1:
                source_index += 1
                cumulative += weights[source_index]
            tmp_phase_amplitudes[output_index] = phase_amplitudes[source_index]
            tmp_detunings[output_index] = detunings[source_index]
            tmp_drive_phases[output_index] = drive_phases[source_index]
            tmp_log_likelihoods[output_index] = log_likelihoods[source_index]

        uniform_weight = 1.0 / particle_count
        for particle_index in range(particle_count):
            phase_amplitudes[particle_index] = tmp_phase_amplitudes[particle_index]
            detunings[particle_index] = tmp_detunings[particle_index]
            drive_phases[particle_index] = tmp_drive_phases[particle_index]
            log_likelihoods[particle_index] = tmp_log_likelihoods[particle_index]
            weights[particle_index] = uniform_weight


    @njit
    def _smc_build_refined_detuning_grid(
        particle_detunings: np.ndarray,
        log_likelihoods: np.ndarray,
        detuning_min: float,
        detuning_max: float,
        n_shots: int,
        time_step: float,
        detuning_step: float,
        requested_width: float,
        center_count: int,
        local_count: int,
        coarse_count: int,
        detuning_values_out: np.ndarray,
        detuning_weights_out: np.ndarray,
    ) -> int:
        if detuning_min == detuning_max:
            detuning_values_out[0] = detuning_min
            detuning_weights_out[0] = 1.0
            return 1

        total_width = detuning_max - detuning_min
        value_count = 0

        effective_coarse_count = coarse_count
        if effective_coarse_count < 2:
            effective_coarse_count = 2
        for coarse_index in range(effective_coarse_count):
            value = detuning_min + total_width * coarse_index / (effective_coarse_count - 1)
            detuning_values_out[value_count] = value
            value_count += 1

        effective_center_count = center_count
        if effective_center_count < 0:
            effective_center_count = 0
        effective_local_count = local_count
        if effective_local_count < 1:
            effective_local_count = 1
        local_half_width = requested_width
        if local_half_width <= 0.0:
            local_half_width = 3.0 / (max(1, n_shots) * time_step)
            if detuning_step > 0.0 and 2.0 * detuning_step > local_half_width:
                local_half_width = 2.0 * detuning_step
            if local_half_width > 0.5 * total_width:
                local_half_width = 0.5 * total_width

        selected_centers = np.empty(effective_center_count, dtype=np.float64)
        selected_count = 0
        separation = 0.25 * local_half_width
        if separation <= 0.0:
            separation = 1.0e-15

        for _ in range(effective_center_count):
            best_index = -1
            best_value = -1.0e300
            for particle_index in range(particle_detunings.size):
                candidate = particle_detunings[particle_index]
                separated = True
                for center_index in range(selected_count):
                    if abs(candidate - selected_centers[center_index]) < separation:
                        separated = False
                        break
                if separated and log_likelihoods[particle_index] > best_value:
                    best_value = log_likelihoods[particle_index]
                    best_index = particle_index
            if best_index < 0:
                break
            center = particle_detunings[best_index]
            selected_centers[selected_count] = center
            selected_count += 1

            if effective_local_count == 1:
                detuning_values_out[value_count] = min(detuning_max, max(detuning_min, center))
                value_count += 1
            else:
                low = center - local_half_width
                high = center + local_half_width
                if low < detuning_min:
                    low = detuning_min
                if high > detuning_max:
                    high = detuning_max
                for local_index in range(effective_local_count):
                    value = low + (high - low) * local_index / (effective_local_count - 1)
                    detuning_values_out[value_count] = value
                    value_count += 1

        sorted_values = np.sort(detuning_values_out[:value_count])
        unique_count = 0
        last_value = 1.0e300
        for value_index in range(value_count):
            value = sorted_values[value_index]
            scale = abs(value)
            if abs(last_value) > scale:
                scale = abs(last_value)
            if scale < 1.0:
                scale = 1.0
            if unique_count == 0 or abs(value - last_value) > 1.0e-13 * scale:
                detuning_values_out[unique_count] = value
                last_value = value
                unique_count += 1

        for value_index in range(unique_count):
            left_edge = detuning_min
            right_edge = detuning_max
            if value_index > 0:
                left_edge = 0.5 * (detuning_values_out[value_index - 1] + detuning_values_out[value_index])
            if value_index + 1 < unique_count:
                right_edge = 0.5 * (detuning_values_out[value_index] + detuning_values_out[value_index + 1])
            if left_edge < detuning_min:
                left_edge = detuning_min
            if right_edge > detuning_max:
                right_edge = detuning_max
            detuning_weights_out[value_index] = (right_edge - left_edge) / total_width

        weight_sum = 0.0
        for value_index in range(unique_count):
            weight_sum += detuning_weights_out[value_index]
        if weight_sum > 0.0:
            for value_index in range(unique_count):
                detuning_weights_out[value_index] /= weight_sum
        return unique_count


    @njit
    def _smc_replay_refined_log_bf(
        phase_amplitude_values: np.ndarray,
        phase_amplitude_weights: np.ndarray,
        detuning_values: np.ndarray,
        detuning_weights: np.ndarray,
        detuning_count: int,
        drive_phase_values: np.ndarray,
        history_count: int,
        history_axis_codes: np.ndarray,
        history_angle_cosines: np.ndarray,
        history_angle_sines: np.ndarray,
        history_outcomes: np.ndarray,
        sensing_time: float,
        dead_time: float,
        eta2: float,
        eta1: float,
        contrast: float,
        readout_error: float,
    ) -> float:
        phase_count = phase_amplitude_values.size
        drive_phase_count = drive_phase_values.size
        particle_count = phase_count * detuning_count * drive_phase_count
        weights = np.empty(particle_count, dtype=np.float64)
        index = 0
        for phase_index in range(phase_count):
            for detuning_index in range(detuning_count):
                for drive_phase_index in range(drive_phase_count):
                    weights[index] = (
                        phase_amplitude_weights[phase_index]
                        * detuning_weights[detuning_index]
                        / drive_phase_count
                    )
                    index += 1

        log_bf = 0.0
        time_step = sensing_time + dead_time
        for shot_index in range(history_count):
            axis_code = history_axis_codes[shot_index]
            angle_cosine = history_angle_cosines[shot_index]
            angle_sine = history_angle_sines[shot_index]
            current_time = shot_index * time_step
            h0_one = _noisy_probability_from_projection(0.0, contrast, readout_error)
            if axis_code == 0:
                h0_one = _noisy_probability_from_projection(-1.0, contrast, readout_error)

            mixture_one = 0.0
            index = 0
            for phase_index in range(phase_count):
                phase_amplitude = phase_amplitude_values[phase_index]
                for detuning_index in range(detuning_count):
                    detuning = detuning_values[detuning_index]
                    for drive_phase_index in range(drive_phase_count):
                        probability_one = _smc_particle_one_probability(
                            phase_amplitude,
                            detuning,
                            drive_phase_values[drive_phase_index],
                            current_time,
                            sensing_time,
                            axis_code,
                            angle_cosine,
                            angle_sine,
                            eta2,
                            eta1,
                            contrast,
                            readout_error,
                        )
                        mixture_one += weights[index] * probability_one
                        index += 1

            outcome_is_one = history_outcomes[shot_index]
            if outcome_is_one:
                mixture_likelihood = _clip_probability(mixture_one)
                h0_likelihood = _clip_probability(h0_one)
            else:
                mixture_likelihood = _clip_probability(1.0 - mixture_one)
                h0_likelihood = _clip_probability(1.0 - h0_one)

            index = 0
            for phase_index in range(phase_count):
                phase_amplitude = phase_amplitude_values[phase_index]
                for detuning_index in range(detuning_count):
                    detuning = detuning_values[detuning_index]
                    for drive_phase_index in range(drive_phase_count):
                        probability_one = _smc_particle_one_probability(
                            phase_amplitude,
                            detuning,
                            drive_phase_values[drive_phase_index],
                            current_time,
                            sensing_time,
                            axis_code,
                            angle_cosine,
                            angle_sine,
                            eta2,
                            eta1,
                            contrast,
                            readout_error,
                        )
                        particle_likelihood = probability_one
                        if not outcome_is_one:
                            particle_likelihood = 1.0 - probability_one
                        weights[index] *= _clip_probability(particle_likelihood) / mixture_likelihood
                        index += 1

            log_bf += np.log(mixture_likelihood) - np.log(h0_likelihood)

        return log_bf


    @njit(parallel=True)
    def _sample_rabi_smc_numba_core(
        initial_phase_amplitudes: np.ndarray,
        initial_detunings: np.ndarray,
        initial_drive_phases: np.ndarray,
        prior_weights: np.ndarray,
        strategy_code: int,
        true_is_h1: bool,
        n_experiments: int,
        n_shots: int,
        seed: int,
        initial_h1_probability: float,
        sensing_time: float,
        dead_time: float,
        contrast: float,
        readout_error: float,
        eta2: float,
        eta1: float,
        h0_z_one: float,
        h0_transverse_one: float,
        h0_transverse_entropy: float,
        true_phase_amplitude: float,
        true_detuning: float,
        true_drive_phase: float,
        true_drive_phases: np.ndarray,
        use_true_drive_phases: bool,
        candidate_cosines: np.ndarray,
        candidate_sines: np.ndarray,
        fixed_angle_cosine: float,
        fixed_angle_sine: float,
        phase_amplitude_min: float,
        phase_amplitude_max: float,
        detuning_min: float,
        detuning_max: float,
        phase_prior_code: int,
        ess_fraction: float,
        move_steps: int,
        log_phase_amplitude_step: float,
        detuning_step: float,
        drive_phase_step: float,
        refined_evidence: bool,
        refined_phase_amplitude_values: np.ndarray,
        refined_phase_amplitude_weights: np.ndarray,
        refined_drive_phase_values: np.ndarray,
        refine_detuning_center_count: int,
        refine_detuning_local_count: int,
        refine_detuning_coarse_count: int,
        refine_detuning_width: float,
    ) -> np.ndarray:
        particle_count = prior_weights.size
        candidate_count = candidate_cosines.size
        output = np.empty(n_experiments, dtype=np.float64)
        time_step = sensing_time + dead_time
        max_refined_detuning_count = 1
        if detuning_min != detuning_max:
            effective_coarse_count = refine_detuning_coarse_count
            if effective_coarse_count < 2:
                effective_coarse_count = 2
            effective_center_count = refine_detuning_center_count
            if effective_center_count < 0:
                effective_center_count = 0
            effective_local_count = refine_detuning_local_count
            if effective_local_count < 1:
                effective_local_count = 1
            max_refined_detuning_count = effective_coarse_count + effective_center_count * effective_local_count

        true_effective_area = true_phase_amplitude * _smc_detuning_sinc_factor(true_detuning, sensing_time)
        true_transverse_amplitude = eta2 * np.sin(true_effective_area)
        true_z_projection = -1.0 + eta1 * (1.0 - np.cos(true_effective_area))
        true_z_one = _noisy_probability_from_projection(true_z_projection, contrast, readout_error)

        for experiment_index in prange(n_experiments):
            np.random.seed(seed + 104729 * (experiment_index + 1))

            phase_amplitudes = initial_phase_amplitudes.copy()
            detunings = initial_detunings.copy()
            drive_phases = initial_drive_phases.copy()
            weights = prior_weights.copy()
            log_likelihoods = np.zeros(particle_count, dtype=np.float64)
            particle_one = np.empty(particle_count, dtype=np.float64)
            tmp_phase_amplitudes = np.empty(particle_count, dtype=np.float64)
            tmp_detunings = np.empty(particle_count, dtype=np.float64)
            tmp_drive_phases = np.empty(particle_count, dtype=np.float64)
            tmp_log_likelihoods = np.empty(particle_count, dtype=np.float64)
            history_axis_codes = np.empty(n_shots, dtype=np.int64)
            history_angle_cosines = np.empty(n_shots, dtype=np.float64)
            history_angle_sines = np.empty(n_shots, dtype=np.float64)
            history_outcomes = np.empty(n_shots, dtype=np.bool_)
            refined_detuning_values = np.empty(max_refined_detuning_count, dtype=np.float64)
            refined_detuning_weights = np.empty(max_refined_detuning_count, dtype=np.float64)

            experiment_true_drive_phase = true_drive_phase
            if use_true_drive_phases:
                experiment_true_drive_phase = true_drive_phases[experiment_index]

            h1_probability = initial_h1_probability
            log_bf = 0.0

            for shot_index in range(n_shots):
                current_time = shot_index * time_step
                axis_code = 1
                angle_cosine = fixed_angle_cosine
                angle_sine = fixed_angle_sine
                h0_one = h0_transverse_one
                mixture_one = 0.0

                if strategy_code == 0:
                    axis_code = 0
                    h0_one = h0_z_one
                    angle_cosine = 1.0
                    angle_sine = 0.0
                    for particle_index in range(particle_count):
                        probability_one = _smc_particle_one_probability(
                            phase_amplitudes[particle_index],
                            detunings[particle_index],
                            drive_phases[particle_index],
                            current_time,
                            sensing_time,
                            axis_code,
                            angle_cosine,
                            angle_sine,
                            eta2,
                            eta1,
                            contrast,
                            readout_error,
                        )
                        particle_one[particle_index] = probability_one
                        mixture_one += weights[particle_index] * probability_one
                elif strategy_code == 1 or strategy_code == 2:
                    if strategy_code == 2:
                        if shot_index % 2 == 0:
                            angle_cosine = 1.0
                            angle_sine = 0.0
                        else:
                            angle_cosine = 0.0
                            angle_sine = 1.0
                    for particle_index in range(particle_count):
                        probability_one = _smc_particle_one_probability(
                            phase_amplitudes[particle_index],
                            detunings[particle_index],
                            drive_phases[particle_index],
                            current_time,
                            sensing_time,
                            axis_code,
                            angle_cosine,
                            angle_sine,
                            eta2,
                            eta1,
                            contrast,
                            readout_error,
                        )
                        particle_one[particle_index] = probability_one
                        mixture_one += weights[particle_index] * probability_one
                else:
                    best_candidate = 0
                    best_utility = -1.0e300
                    best_mixture_one = 0.5
                    one_minus_h1 = 1.0 - h1_probability
                    for candidate_index in range(candidate_count):
                        candidate_mixture_one = 0.0
                        conditional_entropy_h1 = 0.0
                        candidate_cosine = candidate_cosines[candidate_index]
                        candidate_sine = candidate_sines[candidate_index]
                        for particle_index in range(particle_count):
                            probability_one = _smc_particle_one_probability(
                                phase_amplitudes[particle_index],
                                detunings[particle_index],
                                drive_phases[particle_index],
                                current_time,
                                sensing_time,
                                1,
                                candidate_cosine,
                                candidate_sine,
                                eta2,
                                eta1,
                                contrast,
                                readout_error,
                            )
                            candidate_mixture_one += weights[particle_index] * probability_one
                            conditional_entropy_h1 += weights[particle_index] * _binary_entropy_scalar(probability_one)
                        predictive_one = one_minus_h1 * h0_transverse_one + h1_probability * candidate_mixture_one
                        conditional_entropy = (
                            one_minus_h1 * h0_transverse_entropy
                            + h1_probability * conditional_entropy_h1
                        )
                        utility = _binary_entropy_scalar(predictive_one) - conditional_entropy
                        if utility > best_utility + 1.0e-14:
                            best_utility = utility
                            best_candidate = candidate_index
                            best_mixture_one = candidate_mixture_one

                    angle_cosine = candidate_cosines[best_candidate]
                    angle_sine = candidate_sines[best_candidate]
                    mixture_one = best_mixture_one
                    for particle_index in range(particle_count):
                        particle_one[particle_index] = _smc_particle_one_probability(
                            phase_amplitudes[particle_index],
                            detunings[particle_index],
                            drive_phases[particle_index],
                            current_time,
                            sensing_time,
                            1,
                            angle_cosine,
                            angle_sine,
                            eta2,
                            eta1,
                            contrast,
                            readout_error,
                        )

                true_one = h0_one
                if true_is_h1:
                    if axis_code == 0:
                        true_one = true_z_one
                    else:
                        true_phase = experiment_true_drive_phase + true_detuning * (current_time + 0.5 * sensing_time)
                        true_one = _transverse_probability(
                            true_transverse_amplitude,
                            np.cos(true_phase),
                            np.sin(true_phase),
                            angle_cosine,
                            angle_sine,
                            contrast,
                            readout_error,
                        )

                outcome_is_one = np.random.random() < true_one
                if outcome_is_one:
                    mixture_likelihood = _clip_probability(mixture_one)
                    h0_likelihood = _clip_probability(h0_one)
                else:
                    mixture_likelihood = _clip_probability(1.0 - mixture_one)
                    h0_likelihood = _clip_probability(1.0 - h0_one)

                ess_denominator = 0.0
                for particle_index in range(particle_count):
                    particle_likelihood = particle_one[particle_index]
                    if not outcome_is_one:
                        particle_likelihood = 1.0 - particle_likelihood
                    particle_likelihood = _clip_probability(particle_likelihood)
                    weights[particle_index] *= particle_likelihood / mixture_likelihood
                    log_likelihoods[particle_index] += np.log(particle_likelihood)
                    ess_denominator += weights[particle_index] * weights[particle_index]

                evidence = h1_probability * mixture_likelihood + (1.0 - h1_probability) * h0_likelihood
                if evidence < EPSILON:
                    evidence = EPSILON
                h1_probability = h1_probability * mixture_likelihood / evidence
                log_bf += np.log(mixture_likelihood) - np.log(h0_likelihood)

                history_axis_codes[shot_index] = axis_code
                history_angle_cosines[shot_index] = angle_cosine
                history_angle_sines[shot_index] = angle_sine
                history_outcomes[shot_index] = outcome_is_one

                effective_sample_size = 1.0 / ess_denominator
                if effective_sample_size < ess_fraction * particle_count:
                    _smc_systematic_resample_numba(
                        weights,
                        phase_amplitudes,
                        detunings,
                        drive_phases,
                        log_likelihoods,
                        tmp_phase_amplitudes,
                        tmp_detunings,
                        tmp_drive_phases,
                        tmp_log_likelihoods,
                    )
                    _smc_rejuvenate_numba(
                        phase_amplitudes,
                        detunings,
                        drive_phases,
                        log_likelihoods,
                        shot_index + 1,
                        history_axis_codes,
                        history_angle_cosines,
                        history_angle_sines,
                        history_outcomes,
                        phase_amplitude_min,
                        phase_amplitude_max,
                        detuning_min,
                        detuning_max,
                        phase_prior_code,
                        sensing_time,
                        dead_time,
                        eta2,
                        eta1,
                        contrast,
                        readout_error,
                        move_steps,
                        log_phase_amplitude_step,
                        detuning_step,
                        drive_phase_step,
                    )

            if refined_evidence:
                refined_detuning_count = _smc_build_refined_detuning_grid(
                    detunings,
                    log_likelihoods,
                    detuning_min,
                    detuning_max,
                    n_shots,
                    time_step,
                    detuning_step,
                    refine_detuning_width,
                    refine_detuning_center_count,
                    refine_detuning_local_count,
                    refine_detuning_coarse_count,
                    refined_detuning_values,
                    refined_detuning_weights,
                )
                output[experiment_index] = _smc_replay_refined_log_bf(
                    refined_phase_amplitude_values,
                    refined_phase_amplitude_weights,
                    refined_detuning_values,
                    refined_detuning_weights,
                    refined_detuning_count,
                    refined_drive_phase_values,
                    n_shots,
                    history_axis_codes,
                    history_angle_cosines,
                    history_angle_sines,
                    history_outcomes,
                    sensing_time,
                    dead_time,
                    eta2,
                    eta1,
                    contrast,
                    readout_error,
                )
            else:
                output[experiment_index] = log_bf

        return output


    @njit
    def _multires_detuning_value(detuning_min: float, detuning_max: float, detuning_count: int, index: int) -> float:
        if detuning_count <= 1:
            return detuning_min
        return detuning_min + (detuning_max - detuning_min) * index / (detuning_count - 1)


    @njit
    def _multires_target_detuning_count(
        detuning_min: float,
        detuning_max: float,
        maximum_count: int,
        current_count: int,
        observed_shots: int,
        time_step: float,
        initial_count: int,
        initial_shots: int,
        phase_tolerance: float,
    ) -> int:
        if detuning_min == detuning_max:
            return 1
        if maximum_count < 2:
            maximum_count = 2
        if initial_count < 2:
            initial_count = 2
        if initial_count > maximum_count:
            initial_count = maximum_count
        if current_count < initial_count:
            current_count = initial_count
        if observed_shots < initial_shots:
            return current_count
        if phase_tolerance <= 0.0:
            phase_tolerance = 1.0

        elapsed = observed_shots * time_step
        if elapsed < initial_shots * time_step:
            elapsed = initial_shots * time_step
        required = int(np.ceil((detuning_max - detuning_min) * elapsed / phase_tolerance)) + 1
        if required < initial_count:
            required = initial_count
        if required > maximum_count:
            required = maximum_count

        target = current_count
        while target < required:
            next_target = 2 * target - 1
            if next_target <= target:
                next_target = target + 1
            target = next_target
            if target > maximum_count:
                target = maximum_count
        return target


    @njit
    def _multires_initialize_weights(
        phase_amplitude_weights: np.ndarray,
        detuning_count: int,
        drive_phase_count: int,
        weights: np.ndarray,
    ) -> int:
        particle_count = phase_amplitude_weights.size * detuning_count * drive_phase_count
        index = 0
        detuning_drive_weight = 1.0 / (detuning_count * drive_phase_count)
        for phase_index in range(phase_amplitude_weights.size):
            for _detuning_index in range(detuning_count):
                for _drive_phase_index in range(drive_phase_count):
                    weights[index] = phase_amplitude_weights[phase_index] * detuning_drive_weight
                    index += 1
        return particle_count


    @njit
    def _multires_replay_history(
        phase_amplitude_values: np.ndarray,
        phase_amplitude_weights: np.ndarray,
        detuning_min: float,
        detuning_max: float,
        detuning_count: int,
        drive_phase_values: np.ndarray,
        history_count: int,
        history_axis_codes: np.ndarray,
        history_angle_cosines: np.ndarray,
        history_angle_sines: np.ndarray,
        history_outcomes: np.ndarray,
        initial_h1_probability: float,
        sensing_time: float,
        dead_time: float,
        eta2: float,
        eta1: float,
        contrast: float,
        readout_error: float,
        weights: np.ndarray,
    ) -> tuple[float, float, int]:
        drive_phase_count = drive_phase_values.size
        particle_count = _multires_initialize_weights(
            phase_amplitude_weights,
            detuning_count,
            drive_phase_count,
            weights,
        )
        h1_probability = initial_h1_probability
        log_bf = 0.0
        time_step = sensing_time + dead_time

        for shot_index in range(history_count):
            axis_code = history_axis_codes[shot_index]
            angle_cosine = history_angle_cosines[shot_index]
            angle_sine = history_angle_sines[shot_index]
            current_time = shot_index * time_step
            h0_one = _noisy_probability_from_projection(0.0, contrast, readout_error)
            if axis_code == 0:
                h0_one = _noisy_probability_from_projection(-1.0, contrast, readout_error)

            mixture_one = 0.0
            index = 0
            for phase_index in range(phase_amplitude_values.size):
                phase_amplitude = phase_amplitude_values[phase_index]
                for detuning_index in range(detuning_count):
                    detuning = _multires_detuning_value(detuning_min, detuning_max, detuning_count, detuning_index)
                    for drive_phase_index in range(drive_phase_count):
                        probability_one = _smc_particle_one_probability(
                            phase_amplitude,
                            detuning,
                            drive_phase_values[drive_phase_index],
                            current_time,
                            sensing_time,
                            axis_code,
                            angle_cosine,
                            angle_sine,
                            eta2,
                            eta1,
                            contrast,
                            readout_error,
                        )
                        mixture_one += weights[index] * probability_one
                        index += 1

            outcome_is_one = history_outcomes[shot_index]
            if outcome_is_one:
                mixture_likelihood = _clip_probability(mixture_one)
                h0_likelihood = _clip_probability(h0_one)
            else:
                mixture_likelihood = _clip_probability(1.0 - mixture_one)
                h0_likelihood = _clip_probability(1.0 - h0_one)

            index = 0
            for phase_index in range(phase_amplitude_values.size):
                phase_amplitude = phase_amplitude_values[phase_index]
                for detuning_index in range(detuning_count):
                    detuning = _multires_detuning_value(detuning_min, detuning_max, detuning_count, detuning_index)
                    for drive_phase_index in range(drive_phase_count):
                        probability_one = _smc_particle_one_probability(
                            phase_amplitude,
                            detuning,
                            drive_phase_values[drive_phase_index],
                            current_time,
                            sensing_time,
                            axis_code,
                            angle_cosine,
                            angle_sine,
                            eta2,
                            eta1,
                            contrast,
                            readout_error,
                        )
                        particle_likelihood = probability_one
                        if not outcome_is_one:
                            particle_likelihood = 1.0 - probability_one
                        weights[index] *= _clip_probability(particle_likelihood) / mixture_likelihood
                        index += 1

            evidence = h1_probability * mixture_likelihood + (1.0 - h1_probability) * h0_likelihood
            if evidence < EPSILON:
                evidence = EPSILON
            h1_probability = h1_probability * mixture_likelihood / evidence
            log_bf += np.log(mixture_likelihood) - np.log(h0_likelihood)

        return h1_probability, log_bf, particle_count


    @njit(parallel=True)
    def _sample_rabi_multires_numba_core(
        phase_amplitude_values: np.ndarray,
        phase_amplitude_weights: np.ndarray,
        detuning_min: float,
        detuning_max: float,
        maximum_detuning_count: int,
        drive_phase_values: np.ndarray,
        strategy_code: int,
        true_is_h1: bool,
        n_experiments: int,
        n_shots: int,
        seed: int,
        initial_h1_probability: float,
        sensing_time: float,
        dead_time: float,
        contrast: float,
        readout_error: float,
        eta2: float,
        eta1: float,
        h0_z_one: float,
        h0_transverse_one: float,
        h0_transverse_entropy: float,
        true_phase_amplitude: float,
        true_detuning: float,
        true_drive_phase: float,
        true_drive_phases: np.ndarray,
        use_true_drive_phases: bool,
        candidate_cosines: np.ndarray,
        candidate_sines: np.ndarray,
        fixed_angle_cosine: float,
        fixed_angle_sine: float,
        initial_detuning_count: int,
        initial_shots: int,
        phase_tolerance: float,
        final_replay: bool,
    ) -> np.ndarray:
        output = np.empty(n_experiments, dtype=np.float64)
        time_step = sensing_time + dead_time
        drive_phase_count = drive_phase_values.size
        max_particle_count = phase_amplitude_values.size * maximum_detuning_count * drive_phase_count

        true_effective_area = true_phase_amplitude * _smc_detuning_sinc_factor(true_detuning, sensing_time)
        true_transverse_amplitude = eta2 * np.sin(true_effective_area)
        true_z_projection = -1.0 + eta1 * (1.0 - np.cos(true_effective_area))
        true_z_one = _noisy_probability_from_projection(true_z_projection, contrast, readout_error)

        for experiment_index in prange(n_experiments):
            np.random.seed(seed + 104729 * (experiment_index + 1))
            weights = np.empty(max_particle_count, dtype=np.float64)
            particle_one = np.empty(max_particle_count, dtype=np.float64)
            history_axis_codes = np.empty(n_shots, dtype=np.int64)
            history_angle_cosines = np.empty(n_shots, dtype=np.float64)
            history_angle_sines = np.empty(n_shots, dtype=np.float64)
            history_outcomes = np.empty(n_shots, dtype=np.bool_)

            active_detuning_count = initial_detuning_count
            if detuning_min == detuning_max:
                active_detuning_count = 1
            elif active_detuning_count < 2:
                active_detuning_count = 2
            if active_detuning_count > maximum_detuning_count:
                active_detuning_count = maximum_detuning_count
            active_particle_count = _multires_initialize_weights(
                phase_amplitude_weights,
                active_detuning_count,
                drive_phase_count,
                weights,
            )
            h1_probability = initial_h1_probability
            log_bf = 0.0

            experiment_true_drive_phase = true_drive_phase
            if use_true_drive_phases:
                experiment_true_drive_phase = true_drive_phases[experiment_index]

            for shot_index in range(n_shots):
                target_detuning_count = _multires_target_detuning_count(
                    detuning_min,
                    detuning_max,
                    maximum_detuning_count,
                    active_detuning_count,
                    shot_index,
                    time_step,
                    initial_detuning_count,
                    initial_shots,
                    phase_tolerance,
                )
                if target_detuning_count != active_detuning_count:
                    active_detuning_count = target_detuning_count
                    h1_probability, log_bf, active_particle_count = _multires_replay_history(
                        phase_amplitude_values,
                        phase_amplitude_weights,
                        detuning_min,
                        detuning_max,
                        active_detuning_count,
                        drive_phase_values,
                        shot_index,
                        history_axis_codes,
                        history_angle_cosines,
                        history_angle_sines,
                        history_outcomes,
                        initial_h1_probability,
                        sensing_time,
                        dead_time,
                        eta2,
                        eta1,
                        contrast,
                        readout_error,
                        weights,
                    )

                current_time = shot_index * time_step
                axis_code = 1
                angle_cosine = fixed_angle_cosine
                angle_sine = fixed_angle_sine
                h0_one = h0_transverse_one
                mixture_one = 0.0

                if strategy_code == 0:
                    axis_code = 0
                    angle_cosine = 1.0
                    angle_sine = 0.0
                    h0_one = h0_z_one
                    index = 0
                    for phase_index in range(phase_amplitude_values.size):
                        phase_amplitude = phase_amplitude_values[phase_index]
                        for detuning_index in range(active_detuning_count):
                            detuning = _multires_detuning_value(
                                detuning_min,
                                detuning_max,
                                active_detuning_count,
                                detuning_index,
                            )
                            for drive_phase_index in range(drive_phase_count):
                                probability_one = _smc_particle_one_probability(
                                    phase_amplitude,
                                    detuning,
                                    drive_phase_values[drive_phase_index],
                                    current_time,
                                    sensing_time,
                                    axis_code,
                                    angle_cosine,
                                    angle_sine,
                                    eta2,
                                    eta1,
                                    contrast,
                                    readout_error,
                                )
                                particle_one[index] = probability_one
                                mixture_one += weights[index] * probability_one
                                index += 1
                elif strategy_code == 1 or strategy_code == 2:
                    if strategy_code == 2:
                        if shot_index % 2 == 0:
                            angle_cosine = 1.0
                            angle_sine = 0.0
                        else:
                            angle_cosine = 0.0
                            angle_sine = 1.0
                    index = 0
                    for phase_index in range(phase_amplitude_values.size):
                        phase_amplitude = phase_amplitude_values[phase_index]
                        for detuning_index in range(active_detuning_count):
                            detuning = _multires_detuning_value(
                                detuning_min,
                                detuning_max,
                                active_detuning_count,
                                detuning_index,
                            )
                            for drive_phase_index in range(drive_phase_count):
                                probability_one = _smc_particle_one_probability(
                                    phase_amplitude,
                                    detuning,
                                    drive_phase_values[drive_phase_index],
                                    current_time,
                                    sensing_time,
                                    axis_code,
                                    angle_cosine,
                                    angle_sine,
                                    eta2,
                                    eta1,
                                    contrast,
                                    readout_error,
                                )
                                particle_one[index] = probability_one
                                mixture_one += weights[index] * probability_one
                                index += 1
                else:
                    best_candidate = 0
                    best_utility = -1.0e300
                    best_mixture_one = 0.5
                    one_minus_h1 = 1.0 - h1_probability
                    for candidate_index in range(candidate_cosines.size):
                        candidate_mixture_one = 0.0
                        conditional_entropy_h1 = 0.0
                        candidate_cosine = candidate_cosines[candidate_index]
                        candidate_sine = candidate_sines[candidate_index]
                        index = 0
                        for phase_index in range(phase_amplitude_values.size):
                            phase_amplitude = phase_amplitude_values[phase_index]
                            for detuning_index in range(active_detuning_count):
                                detuning = _multires_detuning_value(
                                    detuning_min,
                                    detuning_max,
                                    active_detuning_count,
                                    detuning_index,
                                )
                                for drive_phase_index in range(drive_phase_count):
                                    probability_one = _smc_particle_one_probability(
                                        phase_amplitude,
                                        detuning,
                                        drive_phase_values[drive_phase_index],
                                        current_time,
                                        sensing_time,
                                        1,
                                        candidate_cosine,
                                        candidate_sine,
                                        eta2,
                                        eta1,
                                        contrast,
                                        readout_error,
                                    )
                                    candidate_mixture_one += weights[index] * probability_one
                                    conditional_entropy_h1 += weights[index] * _binary_entropy_scalar(probability_one)
                                    index += 1
                        predictive_one = one_minus_h1 * h0_transverse_one + h1_probability * candidate_mixture_one
                        conditional_entropy = (
                            one_minus_h1 * h0_transverse_entropy
                            + h1_probability * conditional_entropy_h1
                        )
                        utility = _binary_entropy_scalar(predictive_one) - conditional_entropy
                        if utility > best_utility + 1.0e-14:
                            best_utility = utility
                            best_candidate = candidate_index
                            best_mixture_one = candidate_mixture_one

                    angle_cosine = candidate_cosines[best_candidate]
                    angle_sine = candidate_sines[best_candidate]
                    mixture_one = best_mixture_one
                    index = 0
                    for phase_index in range(phase_amplitude_values.size):
                        phase_amplitude = phase_amplitude_values[phase_index]
                        for detuning_index in range(active_detuning_count):
                            detuning = _multires_detuning_value(
                                detuning_min,
                                detuning_max,
                                active_detuning_count,
                                detuning_index,
                            )
                            for drive_phase_index in range(drive_phase_count):
                                particle_one[index] = _smc_particle_one_probability(
                                    phase_amplitude,
                                    detuning,
                                    drive_phase_values[drive_phase_index],
                                    current_time,
                                    sensing_time,
                                    1,
                                    angle_cosine,
                                    angle_sine,
                                    eta2,
                                    eta1,
                                    contrast,
                                    readout_error,
                                )
                                index += 1

                true_one = h0_one
                if true_is_h1:
                    if axis_code == 0:
                        true_one = true_z_one
                    else:
                        true_phase = experiment_true_drive_phase + true_detuning * (current_time + 0.5 * sensing_time)
                        true_one = _transverse_probability(
                            true_transverse_amplitude,
                            np.cos(true_phase),
                            np.sin(true_phase),
                            angle_cosine,
                            angle_sine,
                            contrast,
                            readout_error,
                        )

                outcome_is_one = np.random.random() < true_one
                if outcome_is_one:
                    mixture_likelihood = _clip_probability(mixture_one)
                    h0_likelihood = _clip_probability(h0_one)
                else:
                    mixture_likelihood = _clip_probability(1.0 - mixture_one)
                    h0_likelihood = _clip_probability(1.0 - h0_one)

                for particle_index in range(active_particle_count):
                    particle_likelihood = particle_one[particle_index]
                    if not outcome_is_one:
                        particle_likelihood = 1.0 - particle_likelihood
                    weights[particle_index] *= _clip_probability(particle_likelihood) / mixture_likelihood

                evidence = h1_probability * mixture_likelihood + (1.0 - h1_probability) * h0_likelihood
                if evidence < EPSILON:
                    evidence = EPSILON
                h1_probability = h1_probability * mixture_likelihood / evidence
                log_bf += np.log(mixture_likelihood) - np.log(h0_likelihood)

                history_axis_codes[shot_index] = axis_code
                history_angle_cosines[shot_index] = angle_cosine
                history_angle_sines[shot_index] = angle_sine
                history_outcomes[shot_index] = outcome_is_one

            if final_replay and active_detuning_count != maximum_detuning_count:
                _, log_bf, _ = _multires_replay_history(
                    phase_amplitude_values,
                    phase_amplitude_weights,
                    detuning_min,
                    detuning_max,
                    maximum_detuning_count,
                    drive_phase_values,
                    n_shots,
                    history_axis_codes,
                    history_angle_cosines,
                    history_angle_sines,
                    history_outcomes,
                    initial_h1_probability,
                    sensing_time,
                    dead_time,
                    eta2,
                    eta1,
                    contrast,
                    readout_error,
                    weights,
                )

            output[experiment_index] = log_bf

        return output


def _strategy_code(strategy: str) -> int:
    if strategy == "fixed_z":
        return 0
    if strategy == "fixed_x":
        return 1
    if strategy == "two_axis":
        return 2
    if strategy == "adaptive" or strategy.startswith("lookahead_"):
        return 3
    if strategy.startswith("rollout_"):
        return 4
    if strategy.startswith("two_stage_"):
        _two_stage_switch_shot(strategy)
        return 5
    if strategy == "adaptive_continuous":
        return 6
    if strategy == "adaptive_bayes":
        return 7
    if strategy == "adaptive_bayes_continuous":
        return 8
    if strategy.startswith("adaptive_hybrid_continuous_"):
        _adaptive_hybrid_switch_shot(strategy)
        return 10
    if strategy.startswith("adaptive_hybrid_"):
        _adaptive_hybrid_switch_shot(strategy)
        return 9
    if strategy == "adaptive_blended_continuous" or strategy.startswith("adaptive_blended_continuous_"):
        _adaptive_blended_parameters(strategy)
        return 12
    if strategy == "adaptive_blended" or strategy.startswith("adaptive_blended_"):
        _adaptive_blended_parameters(strategy)
        return 11
    if strategy == "adaptive_sphere_continuous":
        return 13
    if strategy == "adaptive_bayes_sphere_continuous":
        return 14
    if strategy.startswith("periodic_hybrid_sphere_continuous_"):
        _periodic_hybrid_period(strategy)
        return 15
    if strategy.startswith("periodic_bayes_hybrid_sphere_continuous_"):
        _periodic_hybrid_period(strategy)
        return 16
    if strategy.startswith("periodic_stein_hybrid_sphere_continuous_"):
        _periodic_hybrid_period(strategy)
        return 17
    raise ValueError(
        "strategy must be 'fixed_z', 'fixed_x', 'two_axis', 'adaptive', 'lookahead_<depth>', "
        "'rollout_<depth>', 'two_stage_<switch_shot>', 'adaptive_continuous', "
        "'adaptive_bayes', 'adaptive_bayes_continuous', 'adaptive_hybrid_<switch_shot>', "
        "'adaptive_hybrid_continuous_<switch_shot>', 'adaptive_blended', "
        "'adaptive_blended_continuous', 'adaptive_sphere_continuous', "
        "'adaptive_bayes_sphere_continuous', 'periodic_hybrid_sphere_continuous_<period>', "
        "'periodic_bayes_hybrid_sphere_continuous_<period>', or "
        "'periodic_stein_hybrid_sphere_continuous_<period>'"
    )


def _two_stage_switch_shot(strategy: str) -> int:
    try:
        switch_shot = int(strategy.rsplit("_", maxsplit=1)[1])
    except ValueError as exc:
        raise ValueError("two-stage strategies must be named like 'two_stage_1024'") from exc
    if switch_shot < 0:
        raise ValueError("two-stage switch shot must be non-negative")
    return switch_shot


def _adaptive_hybrid_switch_shot(strategy: str) -> int:
    try:
        switch_shot = int(strategy.rsplit("_", maxsplit=1)[1])
    except ValueError as exc:
        raise ValueError(
            "adaptive hybrid strategies must be named like "
            "'adaptive_hybrid_1024' or 'adaptive_hybrid_continuous_1024'"
        ) from exc
    if switch_shot < 0:
        raise ValueError("adaptive hybrid switch shot must be non-negative")
    return switch_shot


def _periodic_hybrid_period(strategy: str) -> int:
    try:
        period = int(strategy.rsplit("_", maxsplit=1)[1])
    except ValueError as exc:
        raise ValueError(
            "periodic sphere hybrid strategies must be named like "
            "'periodic_hybrid_sphere_continuous_8' or "
            "'periodic_bayes_hybrid_sphere_continuous_8' or "
            "'periodic_stein_hybrid_sphere_continuous_8'"
        ) from exc
    if period <= 0:
        raise ValueError("periodic sphere hybrid period must be positive")
    return period


def _adaptive_blended_parameters(strategy: str) -> tuple[float, float, float]:
    if strategy in {"adaptive_blended", "adaptive_blended_continuous"}:
        return 1.0, 0.2, 0.9
    if strategy.startswith("adaptive_blended_continuous_"):
        suffix = strategy[len("adaptive_blended_continuous_") :]
    elif strategy.startswith("adaptive_blended_"):
        suffix = strategy[len("adaptive_blended_") :]
    else:
        raise ValueError("adaptive blended strategies must start with 'adaptive_blended'")
    pieces = suffix.split("_")
    if len(pieces) != 3:
        raise ValueError(
            "adaptive blended strategies must be named like "
            "'adaptive_blended_1.0_0.2_0.9' or 'adaptive_blended_continuous_1.0_0.2_0.9'"
        )
    try:
        lambda_max = float(pieces[0])
        r_start = float(pieces[1])
        r_stop = float(pieces[2])
    except ValueError as exc:
        raise ValueError("adaptive blended parameters must be numeric") from exc
    if lambda_max < 0.0:
        raise ValueError("adaptive blended lambda_max must be non-negative")
    if r_start < 0.0 or r_stop > 1.0 or r_start >= r_stop:
        raise ValueError("adaptive blended concentrations must satisfy 0 <= r_start < r_stop <= 1")
    return lambda_max, r_start, r_stop


def _lookahead_depth(strategy: str) -> int:
    if (
        strategy in {"adaptive", "adaptive_continuous", "adaptive_bayes", "adaptive_bayes_continuous"}
        or strategy.startswith("adaptive_hybrid_")
        or strategy.startswith("adaptive_blended")
        or strategy in {"adaptive_sphere_continuous", "adaptive_bayes_sphere_continuous"}
        or strategy.startswith("periodic_hybrid_sphere_continuous_")
        or strategy.startswith("periodic_bayes_hybrid_sphere_continuous_")
        or strategy.startswith("periodic_stein_hybrid_sphere_continuous_")
    ):
        return 1
    if strategy.startswith("lookahead_"):
        try:
            depth = int(strategy.rsplit("_", maxsplit=1)[1])
        except ValueError as exc:
            raise ValueError("look-ahead strategies must be named like 'lookahead_2'") from exc
        if depth <= 0:
            raise ValueError("look-ahead depth must be positive")
        if depth > 4:
            raise ValueError("numba Rabi look-ahead currently supports depths 1, 2, 3, and 4")
        return depth
    if strategy.startswith("rollout_"):
        try:
            depth = int(strategy.rsplit("_", maxsplit=1)[1])
        except ValueError as exc:
            raise ValueError("rollout strategies must be named like 'rollout_4'") from exc
        if depth <= 0:
            raise ValueError("rollout depth must be positive")
        if depth > 10:
            raise ValueError("numba Rabi rollout currently supports depths up to 10")
        return depth
    if strategy.startswith("two_stage_"):
        _two_stage_switch_shot(strategy)
        return 1
    return 0


def sample_rabi_log_bayes_factors_numba(
    config: RabiExperimentConfig,
    strategy: str,
    hypothesis: str,
    n_experiments: int,
    batch_size: int = 256,
    seed: int | None = None,
    fixed_angle: float = 0.0,
    chunk_size: int = 8192,
    true_drive_phases: np.ndarray | None = None,
) -> np.ndarray:
    if njit is None:
        raise ImportError("method='numba' requires numba to be installed")
    return sample_rabi_log_bayes_factors_at_checkpoints_numba(
        config,
        strategy,
        hypothesis,
        n_experiments,
        (config.n_shots,),
        batch_size=batch_size,
        seed=seed,
        fixed_angle=fixed_angle,
        chunk_size=chunk_size,
        true_drive_phases=true_drive_phases,
    )[:, 0]


def sample_rabi_log_bayes_factors_at_checkpoints_numba(
    config: RabiExperimentConfig,
    strategy: str,
    hypothesis: str,
    n_experiments: int,
    checkpoints: np.ndarray | list[int] | tuple[int, ...],
    batch_size: int = 256,
    seed: int | None = None,
    fixed_angle: float = 0.0,
    chunk_size: int = 8192,
    true_drive_phases: np.ndarray | None = None,
    return_diagnostics: bool = False,
) -> np.ndarray | dict[str, np.ndarray]:
    if njit is None:
        raise ImportError("checkpoint sampling with method='numba' requires numba to be installed")
    if hypothesis not in {"H0", "H1"}:
        raise ValueError("hypothesis must be 'H0' or 'H1'")

    count = int(n_experiments)
    if count <= 0:
        raise ValueError("n_experiments must be positive")

    checkpoint_values = _normalize_log_bayes_checkpoints(checkpoints, int(config.n_shots))
    selected_seed = config.seed if seed is None else int(seed)
    generator = np.random.default_rng(selected_seed)
    effective_batch_size = max(1, min(int(batch_size), count))
    effective_chunk_size = max(1, min(int(chunk_size), int(config.n_shots)))
    strategy_code = _strategy_code(strategy)
    lookahead_depth = _lookahead_depth(strategy)
    two_stage_switch_shot = _two_stage_switch_shot(strategy) if strategy_code == 5 else 0
    adaptive_hybrid_switch_shot = _adaptive_hybrid_switch_shot(strategy) if strategy_code in {9, 10} else 0
    periodic_hybrid_period = _periodic_hybrid_period(strategy) if strategy_code in {15, 16, 17} else 1
    blend_lambda_max, blend_r_start, blend_r_stop = (
        _adaptive_blended_parameters(strategy) if strategy_code in {11, 12} else (0.0, 0.0, 1.0)
    )
    use_true_drive_phases = true_drive_phases is not None
    if use_true_drive_phases:
        true_drive_phases_array = np.asarray(true_drive_phases, dtype=float)
        if true_drive_phases_array.shape != (count,):
            raise ValueError("true_drive_phases must have shape (n_experiments,)")
    else:
        true_drive_phases_array = np.empty(0, dtype=float)

    grid = build_rabi_particle_grid(config.particles)
    sensing_time = float(config.sensing_time)
    dead_time = float(config.dead_time)
    readout = config.readout
    contrast = float(readout.contrast)
    readout_error = float(readout.readout_error)
    eta2 = transverse_relaxation_factor(sensing_time, readout.t2)
    eta1 = longitudinal_relaxation_factor(sensing_time, readout.t1, readout.t2)

    particle_effective_areas = grid.phase_amplitudes * detuning_sinc_factor(grid.detunings, sensing_time)
    particle_transverse_amplitudes = eta2 * np.sin(particle_effective_areas)
    particle_z_projections = -1.0 + eta1 * (1.0 - np.cos(particle_effective_areas))
    particle_z_one = np.asarray(
        readout_error + (1.0 - 2.0 * readout_error) * 0.5 * (1.0 + contrast * particle_z_projections),
        dtype=float,
    )
    particle_z_one = np.clip(particle_z_one, EPSILON, 1.0 - EPSILON)

    h0_z_one = float(np.clip(readout_error + (1.0 - 2.0 * readout_error) * 0.5 * (1.0 - contrast), EPSILON, 1.0 - EPSILON))
    h0_transverse_one = float(np.clip(readout_error + (1.0 - 2.0 * readout_error) * 0.5, EPSILON, 1.0 - EPSILON))
    h0_transverse_entropy = float(
        -(h0_transverse_one * np.log(h0_transverse_one) + (1.0 - h0_transverse_one) * np.log(1.0 - h0_transverse_one))
    )

    true_effective_area = float(config.true_phase_amplitude) * float(
        detuning_sinc_factor(float(config.true_detuning), sensing_time)
    )
    true_transverse_amplitude = float(eta2 * np.sin(true_effective_area))
    true_z_projection = float(-1.0 + eta1 * (1.0 - np.cos(true_effective_area)))
    true_z_one = float(
        np.clip(
            readout_error + (1.0 - 2.0 * readout_error) * 0.5 * (1.0 + contrast * true_z_projection),
            EPSILON,
            1.0 - EPSILON,
        )
    )

    step_arguments = grid.detunings * (sensing_time + dead_time)
    step_cosines = np.asarray(np.cos(step_arguments), dtype=float)
    step_sines = np.asarray(np.sin(step_arguments), dtype=float)
    true_step_argument = float(config.true_detuning) * (sensing_time + dead_time)
    true_step_cosine = float(np.cos(true_step_argument))
    true_step_sine = float(np.sin(true_step_argument))

    candidate_angles = np.asarray(config.candidate_angles, dtype=float)
    candidate_polars = np.asarray(config.candidate_polars, dtype=float)
    if candidate_polars.size <= 0:
        raise ValueError("config.candidate_polars must contain at least one polar angle")
    candidate_cosines = np.asarray(np.cos(candidate_angles), dtype=float)
    candidate_sines = np.asarray(np.sin(candidate_angles), dtype=float)
    fixed_angle = float(fixed_angle)
    fixed_angle_cosine = float(np.cos(fixed_angle))
    fixed_angle_sine = float(np.sin(fixed_angle))

    samples = np.empty((count, checkpoint_values.size), dtype=float)
    if return_diagnostics:
        predictable_kl_samples = np.zeros((count, checkpoint_values.size), dtype=float)
        predictable_variance_samples = np.zeros((count, checkpoint_values.size), dtype=float)
        design_samples = np.zeros((count, checkpoint_values.size, 3), dtype=float)
    else:
        predictable_kl_samples = np.empty((0, 0), dtype=float)
        predictable_variance_samples = np.empty((0, 0), dtype=float)
        design_samples = np.empty((0, 0, 3), dtype=float)
    signal_diagnostics = return_diagnostics and hypothesis == "H1"
    if signal_diagnostics:
        signal_null_kl_samples = np.zeros((count, checkpoint_values.size), dtype=float)
        signal_predictive_kl_samples = np.zeros((count, checkpoint_values.size), dtype=float)
    positive_checkpoints = np.asarray(checkpoint_values[checkpoint_values > 0], dtype=np.int64)
    zero_checkpoint_indices = np.flatnonzero(checkpoint_values == 0)

    for batch_start in range(0, count, effective_batch_size):
        batch_stop = min(batch_start + effective_batch_size, count)
        batch_count = batch_stop - batch_start
        weights = np.repeat(grid.prior_weights[None, :], batch_count, axis=0)
        h1_probabilities = np.full(batch_count, float(config.initial_h1_probability), dtype=float)
        batch_log_bayes_factors = np.zeros(batch_count, dtype=float)
        checkpoint_log_bayes_factors = np.empty((positive_checkpoints.size, batch_count), dtype=float)
        if return_diagnostics:
            cumulative_predictable_kl = np.zeros(batch_count, dtype=float)
            cumulative_predictable_variance = np.zeros(batch_count, dtype=float)
            cumulative_design_xx = np.zeros(batch_count, dtype=float)
            cumulative_design_xy = np.zeros(batch_count, dtype=float)
            cumulative_design_yy = np.zeros(batch_count, dtype=float)
            checkpoint_predictable_kl = np.empty((positive_checkpoints.size, batch_count), dtype=float)
            checkpoint_predictable_variance = np.empty((positive_checkpoints.size, batch_count), dtype=float)
            checkpoint_design = np.empty((positive_checkpoints.size, batch_count, 3), dtype=float)
        else:
            cumulative_predictable_kl = np.empty(0, dtype=float)
            cumulative_predictable_variance = np.empty(0, dtype=float)
            cumulative_design_xx = np.empty(0, dtype=float)
            cumulative_design_xy = np.empty(0, dtype=float)
            cumulative_design_yy = np.empty(0, dtype=float)
            checkpoint_predictable_kl = np.empty((0, batch_count), dtype=float)
            checkpoint_predictable_variance = np.empty((0, batch_count), dtype=float)
            checkpoint_design = np.empty((0, batch_count, 3), dtype=float)
        if signal_diagnostics:
            cumulative_signal_null_kl = np.zeros(batch_count, dtype=float)
            cumulative_signal_predictive_kl = np.zeros(batch_count, dtype=float)
            checkpoint_signal_null_kl = np.empty((positive_checkpoints.size, batch_count), dtype=float)
            checkpoint_signal_predictive_kl = np.empty((positive_checkpoints.size, batch_count), dtype=float)
        else:
            cumulative_signal_null_kl = np.empty(0, dtype=float)
            cumulative_signal_predictive_kl = np.empty(0, dtype=float)
            checkpoint_signal_null_kl = np.empty((0, batch_count), dtype=float)
            checkpoint_signal_predictive_kl = np.empty((0, batch_count), dtype=float)
        if use_true_drive_phases:
            true_drive_phases_batch = true_drive_phases_array[batch_start:batch_stop]
        else:
            true_drive_phases_batch = true_drive_phases_array
        current_time = 0.0

        for checkpoint_index in zero_checkpoint_indices:
            samples[batch_start:batch_stop, int(checkpoint_index)] = 0.0

        for shot_start in range(0, int(config.n_shots), effective_chunk_size):
            shot_stop = min(shot_start + effective_chunk_size, int(config.n_shots))
            uniforms = generator.random((shot_stop - shot_start, batch_count))
            current_time = _run_rabi_chunk(
                uniforms,
                weights,
                h1_probabilities,
                batch_log_bayes_factors,
                current_time,
                shot_start,
                strategy_code,
                grid.prior_weights,
                grid.detunings,
                grid.drive_phases,
                np.asarray(particle_transverse_amplitudes, dtype=float),
                np.asarray(particle_z_projections, dtype=float),
                particle_z_one,
                step_cosines,
                step_sines,
                candidate_polars,
                candidate_angles,
                candidate_cosines,
                candidate_sines,
                fixed_angle,
                fixed_angle_cosine,
                fixed_angle_sine,
                sensing_time,
                dead_time,
                contrast,
                readout_error,
                h0_z_one,
                h0_transverse_one,
                h0_transverse_entropy,
                hypothesis == "H1",
                float(config.true_detuning),
                float(config.true_drive_phase),
                true_drive_phases_batch,
                use_true_drive_phases,
                true_transverse_amplitude,
                true_z_projection,
                true_z_one,
                true_step_cosine,
                true_step_sine,
                int(config.n_shots),
                lookahead_depth,
                two_stage_switch_shot,
                adaptive_hybrid_switch_shot,
                periodic_hybrid_period,
                blend_lambda_max,
                blend_r_start,
                blend_r_stop,
                positive_checkpoints,
                checkpoint_log_bayes_factors,
                return_diagnostics,
                cumulative_predictable_kl,
                cumulative_predictable_variance,
                cumulative_design_xx,
                cumulative_design_xy,
                cumulative_design_yy,
                checkpoint_predictable_kl,
                checkpoint_predictable_variance,
                checkpoint_design,
                cumulative_signal_null_kl,
                cumulative_signal_predictive_kl,
                checkpoint_signal_null_kl,
                checkpoint_signal_predictive_kl,
            )

        for positive_index, checkpoint in enumerate(positive_checkpoints):
            output_indices = np.flatnonzero(checkpoint_values == int(checkpoint))
            for output_index in output_indices:
                samples[batch_start:batch_stop, int(output_index)] = checkpoint_log_bayes_factors[positive_index]
                if return_diagnostics:
                    predictable_kl_samples[batch_start:batch_stop, int(output_index)] = (
                        checkpoint_predictable_kl[positive_index]
                    )
                    predictable_variance_samples[batch_start:batch_stop, int(output_index)] = (
                        checkpoint_predictable_variance[positive_index]
                    )
                    design_samples[batch_start:batch_stop, int(output_index)] = checkpoint_design[positive_index]
                    if signal_diagnostics:
                        signal_null_kl_samples[batch_start:batch_stop, int(output_index)] = (
                            checkpoint_signal_null_kl[positive_index]
                        )
                        signal_predictive_kl_samples[batch_start:batch_stop, int(output_index)] = (
                            checkpoint_signal_predictive_kl[positive_index]
                        )

    if return_diagnostics:
        result = {
            "log_bayes_factors": samples,
            "predictable_kl": predictable_kl_samples,
            "predictable_variance": predictable_variance_samples,
            "polarization_design": design_samples,
        }
        if signal_diagnostics:
            result["signal_null_kl"] = signal_null_kl_samples
            result["signal_predictive_kl"] = signal_predictive_kl_samples
        return result
    return samples


def run_rabi_sphere_learning_trajectory_numba(
    config: RabiExperimentConfig,
    strategy: str,
    hypothesis: str,
    snapshot_shots: tuple[int, ...] | list[int] | np.ndarray,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    if njit is None:
        raise ImportError("run_rabi_sphere_learning_trajectory_numba requires numba to be installed")
    if hypothesis not in {"H0", "H1"}:
        raise ValueError("hypothesis must be 'H0' or 'H1'")
    strategy_code = _strategy_code(strategy)
    if strategy_code not in {13, 14, 15, 16, 17}:
        raise ValueError(
            "run_rabi_sphere_learning_trajectory_numba currently supports "
            "adaptive_sphere_continuous, adaptive_bayes_sphere_continuous, "
            "periodic_hybrid_sphere_continuous_<period>, "
            "periodic_bayes_hybrid_sphere_continuous_<period>, and "
            "periodic_stein_hybrid_sphere_continuous_<period>"
        )

    selected_seed = config.seed if seed is None else int(seed)
    generator = np.random.default_rng(selected_seed)
    n_shots = int(config.n_shots)
    if n_shots <= 0:
        raise ValueError("config.n_shots must be positive")
    snapshots = np.asarray(snapshot_shots, dtype=np.int64)
    snapshots = np.unique(snapshots[(snapshots >= 0) & (snapshots <= n_shots)])
    if snapshots.size == 0 or snapshots[0] != 0:
        snapshots = np.concatenate([np.asarray([0], dtype=np.int64), snapshots])
    if snapshots[-1] != n_shots:
        snapshots = np.concatenate([snapshots, np.asarray([n_shots], dtype=np.int64)])

    grid = build_rabi_particle_grid(config.particles)
    sensing_time = float(config.sensing_time)
    dead_time = float(config.dead_time)
    readout = config.readout
    contrast = float(readout.contrast)
    readout_error = float(readout.readout_error)
    eta2 = transverse_relaxation_factor(sensing_time, readout.t2)
    eta1 = longitudinal_relaxation_factor(sensing_time, readout.t1, readout.t2)

    particle_effective_areas = grid.phase_amplitudes * detuning_sinc_factor(grid.detunings, sensing_time)
    particle_transverse_amplitudes = np.asarray(eta2 * np.sin(particle_effective_areas), dtype=float)
    particle_z_projections = np.asarray(-1.0 + eta1 * (1.0 - np.cos(particle_effective_areas)), dtype=float)

    step_arguments = grid.detunings * (sensing_time + dead_time)
    step_cosines = np.asarray(np.cos(step_arguments), dtype=float)
    step_sines = np.asarray(np.sin(step_arguments), dtype=float)

    true_effective_area = float(config.true_phase_amplitude) * float(
        detuning_sinc_factor(float(config.true_detuning), sensing_time)
    )
    true_transverse_amplitude = float(eta2 * np.sin(true_effective_area))
    true_z_projection = float(-1.0 + eta1 * (1.0 - np.cos(true_effective_area)))
    true_step_argument = float(config.true_detuning) * (sensing_time + dead_time)
    true_step_cosine = float(np.cos(true_step_argument))
    true_step_sine = float(np.sin(true_step_argument))

    candidate_polars = np.asarray(config.candidate_polars, dtype=float)
    candidate_angles = np.asarray(config.candidate_angles, dtype=float)
    if candidate_polars.size <= 0 or candidate_angles.size <= 0:
        raise ValueError("config.candidate_polars and config.candidate_angles must be non-empty")

    records = np.empty((n_shots, 15), dtype=float)
    snapshot_weights = np.empty((snapshots.size, grid.prior_weights.size), dtype=float)
    snapshot_h1_probabilities = np.empty(snapshots.size, dtype=float)
    snapshot_log_bayes_factors = np.empty(snapshots.size, dtype=float)
    uniforms = generator.random(n_shots)

    _run_rabi_sphere_learning_trajectory(
        uniforms,
        np.asarray(grid.prior_weights, dtype=float),
        np.asarray(grid.detunings, dtype=float),
        np.asarray(grid.drive_phases, dtype=float),
        particle_transverse_amplitudes,
        particle_z_projections,
        step_cosines,
        step_sines,
        candidate_polars,
        candidate_angles,
        sensing_time,
        dead_time,
        contrast,
        readout_error,
        hypothesis == "H1",
        float(config.true_detuning),
        float(config.true_drive_phase),
        true_transverse_amplitude,
        true_z_projection,
        true_step_cosine,
        true_step_sine,
        float(config.initial_h1_probability),
        strategy_code,
        _periodic_hybrid_period(strategy) if strategy_code in {15, 16, 17} else 1,
        snapshots,
        records,
        snapshot_weights,
        snapshot_h1_probabilities,
        snapshot_log_bayes_factors,
    )
    return {
        "records": records,
        "snapshot_shots": snapshots,
        "snapshot_weights": snapshot_weights,
        "snapshot_h1_probabilities": snapshot_h1_probabilities,
        "snapshot_log_bayes_factors": snapshot_log_bayes_factors,
    }


def sample_rabi_log_bayes_factors_smc_numba(
    config: RabiExperimentConfig,
    strategy: str,
    hypothesis: str,
    n_experiments: int,
    seed: int | None = None,
    fixed_angle: float = 0.0,
    true_drive_phases: np.ndarray | None = None,
) -> np.ndarray:
    if njit is None:
        raise ImportError("method='numba' requires numba to be installed")
    if hypothesis not in {"H0", "H1"}:
        raise ValueError("hypothesis must be 'H0' or 'H1'")
    if strategy not in {"fixed_z", "fixed_x", "two_axis", "adaptive"}:
        raise ValueError("numba SMC currently supports fixed_z, fixed_x, two_axis, and adaptive")
    count = int(n_experiments)
    if count <= 0:
        raise ValueError("n_experiments must be positive")

    use_true_drive_phases = true_drive_phases is not None
    if use_true_drive_phases:
        true_drive_phases_array = np.asarray(true_drive_phases, dtype=float)
        if true_drive_phases_array.shape != (count,):
            raise ValueError("true_drive_phases must have shape (n_experiments,)")
    else:
        true_drive_phases_array = np.empty(0, dtype=float)

    selected_seed = config.seed if seed is None else int(seed)
    strategy_code = _strategy_code(strategy)
    if strategy_code > 3:
        raise ValueError("numba SMC currently supports fixed_z, fixed_x, two_axis, and adaptive")

    grid = build_rabi_particle_grid(config.particles)
    readout = config.readout
    sensing_time = float(config.sensing_time)
    dead_time = float(config.dead_time)
    contrast = float(readout.contrast)
    readout_error = float(readout.readout_error)
    eta2 = transverse_relaxation_factor(sensing_time, readout.t2)
    eta1 = longitudinal_relaxation_factor(sensing_time, readout.t1, readout.t2)

    h0_z_one = float(
        np.clip(
            readout_error + (1.0 - 2.0 * readout_error) * 0.5 * (1.0 - contrast),
            EPSILON,
            1.0 - EPSILON,
        )
    )
    h0_transverse_one = float(
        np.clip(readout_error + (1.0 - 2.0 * readout_error) * 0.5, EPSILON, 1.0 - EPSILON)
    )
    h0_transverse_entropy = float(
        -(
            h0_transverse_one * np.log(h0_transverse_one)
            + (1.0 - h0_transverse_one) * np.log(1.0 - h0_transverse_one)
        )
    )

    candidate_angles = np.asarray(config.candidate_angles, dtype=float)
    candidate_cosines = np.asarray(np.cos(candidate_angles), dtype=float)
    candidate_sines = np.asarray(np.sin(candidate_angles), dtype=float)
    fixed_angle = float(fixed_angle)
    fixed_angle_cosine = float(np.cos(fixed_angle))
    fixed_angle_sine = float(np.sin(fixed_angle))
    phase_prior_code = 0 if config.particles.phase_amplitude_prior == "log_uniform" else 1
    refined_evidence = config.particles.particle_mode == "smc_refined"

    refined_phase_amplitude_values = _known_or_grid_values(
        float(config.particles.phase_amplitude_min),
        float(config.particles.phase_amplitude_max),
        int(config.particles.phase_amplitude_count),
        parameter_name="phase_amplitude",
        log_spacing=True,
    )
    if config.particles.phase_amplitude_min == config.particles.phase_amplitude_max:
        refined_phase_amplitude_weights = np.ones(1, dtype=float)
    else:
        refined_phase_amplitude_weights = _prior_masses(
            refined_phase_amplitude_values,
            float(config.particles.phase_amplitude_min),
            float(config.particles.phase_amplitude_max),
            config.particles.phase_amplitude_prior,
            log_spacing=True,
        )
    refined_drive_phase_values = np.linspace(
        0.0,
        2.0 * np.pi,
        int(config.particles.drive_phase_count),
        endpoint=False,
        dtype=float,
    )

    return _sample_rabi_smc_numba_core(
        np.asarray(grid.phase_amplitudes, dtype=float),
        np.asarray(grid.detunings, dtype=float),
        np.asarray(grid.drive_phases, dtype=float),
        np.asarray(grid.prior_weights, dtype=float),
        strategy_code,
        hypothesis == "H1",
        count,
        int(config.n_shots),
        selected_seed,
        float(config.initial_h1_probability),
        sensing_time,
        dead_time,
        contrast,
        readout_error,
        float(eta2),
        float(eta1),
        h0_z_one,
        h0_transverse_one,
        h0_transverse_entropy,
        float(config.true_phase_amplitude),
        float(config.true_detuning),
        float(config.true_drive_phase),
        true_drive_phases_array,
        use_true_drive_phases,
        candidate_cosines,
        candidate_sines,
        fixed_angle_cosine,
        fixed_angle_sine,
        float(config.particles.phase_amplitude_min),
        float(config.particles.phase_amplitude_max),
        float(config.particles.detuning_min),
        float(config.particles.detuning_max),
        phase_prior_code,
        float(config.particles.smc_resample_ess_fraction),
        int(config.particles.smc_move_steps),
        float(config.particles.smc_log_phase_amplitude_step),
        float(config.particles.smc_detuning_step),
        float(config.particles.smc_drive_phase_step),
        refined_evidence,
        np.asarray(refined_phase_amplitude_values, dtype=float),
        np.asarray(refined_phase_amplitude_weights, dtype=float),
        np.asarray(refined_drive_phase_values, dtype=float),
        int(config.particles.smc_refine_detuning_center_count),
        int(config.particles.smc_refine_detuning_local_count),
        int(config.particles.smc_refine_detuning_coarse_count),
        float(config.particles.smc_refine_detuning_width),
    )


def sample_rabi_log_bayes_factors_multires_numba(
    config: RabiExperimentConfig,
    strategy: str,
    hypothesis: str,
    n_experiments: int,
    seed: int | None = None,
    fixed_angle: float = 0.0,
    true_drive_phases: np.ndarray | None = None,
) -> np.ndarray:
    if njit is None:
        raise ImportError("method='numba' requires numba to be installed")
    if hypothesis not in {"H0", "H1"}:
        raise ValueError("hypothesis must be 'H0' or 'H1'")
    if strategy not in {"fixed_z", "fixed_x", "two_axis", "adaptive"}:
        raise ValueError("numba multires_grid currently supports fixed_z, fixed_x, two_axis, and adaptive")

    count = int(n_experiments)
    if count <= 0:
        raise ValueError("n_experiments must be positive")
    use_true_drive_phases = true_drive_phases is not None
    if use_true_drive_phases:
        true_drive_phases_array = np.asarray(true_drive_phases, dtype=float)
        if true_drive_phases_array.shape != (count,):
            raise ValueError("true_drive_phases must have shape (n_experiments,)")
    else:
        true_drive_phases_array = np.empty(0, dtype=float)

    spec = config.particles
    if spec.detuning_min == spec.detuning_max:
        maximum_detuning_count = 1
    else:
        maximum_detuning_count = int(spec.detuning_count)
        if maximum_detuning_count <= 1:
            raise ValueError("detuning_count must be at least 2 for multires_grid unless detuning_min=detuning_max")

    phase_amplitude_values = _known_or_grid_values(
        float(spec.phase_amplitude_min),
        float(spec.phase_amplitude_max),
        int(spec.phase_amplitude_count),
        parameter_name="phase_amplitude",
        log_spacing=True,
    )
    if spec.phase_amplitude_min == spec.phase_amplitude_max:
        phase_amplitude_weights = np.ones(1, dtype=float)
    else:
        phase_amplitude_weights = _prior_masses(
            phase_amplitude_values,
            float(spec.phase_amplitude_min),
            float(spec.phase_amplitude_max),
            spec.phase_amplitude_prior,
            log_spacing=True,
        )
    drive_phase_values = np.linspace(0.0, 2.0 * np.pi, int(spec.drive_phase_count), endpoint=False, dtype=float)

    readout = config.readout
    sensing_time = float(config.sensing_time)
    dead_time = float(config.dead_time)
    contrast = float(readout.contrast)
    readout_error = float(readout.readout_error)
    eta2 = transverse_relaxation_factor(sensing_time, readout.t2)
    eta1 = longitudinal_relaxation_factor(sensing_time, readout.t1, readout.t2)
    h0_z_one = float(
        np.clip(
            readout_error + (1.0 - 2.0 * readout_error) * 0.5 * (1.0 - contrast),
            EPSILON,
            1.0 - EPSILON,
        )
    )
    h0_transverse_one = float(
        np.clip(readout_error + (1.0 - 2.0 * readout_error) * 0.5, EPSILON, 1.0 - EPSILON)
    )
    h0_transverse_entropy = float(
        -(
            h0_transverse_one * np.log(h0_transverse_one)
            + (1.0 - h0_transverse_one) * np.log(1.0 - h0_transverse_one)
        )
    )

    candidate_angles = np.asarray(config.candidate_angles, dtype=float)
    candidate_cosines = np.asarray(np.cos(candidate_angles), dtype=float)
    candidate_sines = np.asarray(np.sin(candidate_angles), dtype=float)
    fixed_angle = float(fixed_angle)
    selected_seed = config.seed if seed is None else int(seed)

    return _sample_rabi_multires_numba_core(
        np.asarray(phase_amplitude_values, dtype=float),
        np.asarray(phase_amplitude_weights, dtype=float),
        float(spec.detuning_min),
        float(spec.detuning_max),
        maximum_detuning_count,
        np.asarray(drive_phase_values, dtype=float),
        _strategy_code(strategy),
        hypothesis == "H1",
        count,
        int(config.n_shots),
        selected_seed,
        float(config.initial_h1_probability),
        sensing_time,
        dead_time,
        contrast,
        readout_error,
        float(eta2),
        float(eta1),
        h0_z_one,
        h0_transverse_one,
        h0_transverse_entropy,
        float(config.true_phase_amplitude),
        float(config.true_detuning),
        float(config.true_drive_phase),
        true_drive_phases_array,
        use_true_drive_phases,
        candidate_cosines,
        candidate_sines,
        float(np.cos(fixed_angle)),
        float(np.sin(fixed_angle)),
        int(spec.multires_initial_detuning_count),
        int(spec.multires_initial_shots),
        float(spec.multires_refine_phase_tolerance),
        bool(spec.multires_final_replay),
    )


__all__ = [
    "is_numba_available",
    "run_rabi_sphere_learning_trajectory_numba",
    "sample_rabi_log_bayes_factors_at_checkpoints_numba",
    "sample_rabi_log_bayes_factors_numba",
    "sample_rabi_log_bayes_factors_smc_numba",
    "sample_rabi_log_bayes_factors_multires_numba",
]
