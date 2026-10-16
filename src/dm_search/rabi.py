from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

EPSILON = 1.0e-12


@dataclass(frozen=True)
class RabiReadoutConfig:
    contrast: float = 0.99
    readout_error: float = 0.0
    t1: float | None = None
    t2: float | None = None


@dataclass(frozen=True)
class RabiParticleSpec:
    particle_mode: str = "grid"
    phase_amplitude_min: float = 1.0e-2
    phase_amplitude_max: float = 2.0
    phase_amplitude_count: int = 7
    phase_amplitude_prior: str = "log_uniform"
    detuning_min: float = -0.05
    detuning_max: float = 0.05
    detuning_count: int = 5
    drive_phase_count: int = 12
    random_particle_count: int = 0
    random_seed: int = 12345
    scheduled_initial_detuning_count: int = 1
    scheduled_refine_phase_tolerance: float = 1.0
    scheduled_prune_relative_threshold: float = 1.0e-8
    scheduled_prune_cumulative_mass: float = 0.999999
    smc_resample_ess_fraction: float = 0.5
    smc_move_steps: int = 1
    smc_log_phase_amplitude_step: float = 0.15
    smc_detuning_step: float = 1.0e-3
    smc_drive_phase_step: float = 0.35
    smc_refine_detuning_center_count: int = 3
    smc_refine_detuning_local_count: int = 21
    smc_refine_detuning_coarse_count: int = 9
    smc_refine_detuning_width: float = 0.0
    multires_initial_detuning_count: int = 5
    multires_initial_shots: int = 128
    multires_refine_phase_tolerance: float = 1.0
    multires_final_replay: bool = True


@dataclass(frozen=True)
class RabiExperimentConfig:
    n_shots: int = 512
    sensing_time: float = 1.0
    dead_time: float = 0.0
    true_phase_amplitude: float = 0.5
    true_detuning: float = 0.0
    true_drive_phase: float = 0.7
    initial_h1_probability: float = 0.5
    readout: RabiReadoutConfig = RabiReadoutConfig()
    particles: RabiParticleSpec = RabiParticleSpec()
    candidate_angles: tuple[float, ...] = tuple(np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False))
    candidate_polars: tuple[float, ...] = tuple(np.linspace(0.0, math.pi, 5))
    seed: int = 12345


@dataclass(frozen=True)
class RabiParticleGrid:
    phase_amplitudes: np.ndarray
    detunings: np.ndarray
    drive_phases: np.ndarray
    prior_weights: np.ndarray
    shape: tuple[int, ...]
    phase_amplitude_values: np.ndarray
    detuning_values: np.ndarray
    drive_phase_values: np.ndarray


@dataclass(frozen=True)
class _RabiTerms:
    prior_weights: np.ndarray
    initial_h1_probability: float
    particle_one_probabilities: np.ndarray
    h0_one_probabilities: np.ndarray
    true_one_probabilities: np.ndarray


def _adaptive_depth_for_strategy(strategy: str) -> int | None:
    if strategy == "adaptive":
        return 1
    if strategy.startswith("lookahead_"):
        label = "look-ahead"
    elif strategy.startswith("rollout_"):
        label = "rollout"
    else:
        return None
    try:
        depth = int(strategy.rsplit("_", maxsplit=1)[1])
    except ValueError as exc:
        raise ValueError(f"{label} strategies must be named like '{strategy.rsplit('_', maxsplit=1)[0]}_2'") from exc
    if depth <= 0:
        raise ValueError(f"{label} depth must be positive")
    return depth


def _is_rollout_strategy(strategy: str) -> bool:
    return strategy.startswith("rollout_")


def _cell_edges(points: np.ndarray, lower: float, upper: float, log_spacing: bool) -> np.ndarray:
    if points.size == 1:
        return np.asarray([lower, upper], dtype=float)
    edges = np.empty(points.size + 1, dtype=float)
    edges[0] = lower
    edges[-1] = upper
    if log_spacing:
        edges[1:-1] = np.sqrt(points[:-1] * points[1:])
    else:
        edges[1:-1] = 0.5 * (points[:-1] + points[1:])
    return edges


def _prior_masses(points: np.ndarray, lower: float, upper: float, prior_type: str, log_spacing: bool) -> np.ndarray:
    edges = _cell_edges(points, lower, upper, log_spacing)
    if prior_type == "uniform":
        masses = edges[1:] - edges[:-1]
    elif prior_type == "log_uniform":
        masses = np.log(edges[1:] / edges[:-1])
    else:
        raise ValueError(f"unsupported prior type: {prior_type}")
    masses = np.asarray(masses, dtype=float)
    return masses / masses.sum()


def _known_or_grid_values(
    lower: float,
    upper: float,
    count: int,
    *,
    parameter_name: str,
    log_spacing: bool,
) -> np.ndarray:
    if lower > upper:
        raise ValueError(f"{parameter_name}_min must not be larger than {parameter_name}_max")
    if lower == upper:
        return np.asarray([lower], dtype=float)
    if count <= 0:
        raise ValueError(f"{parameter_name}_count must be positive unless min=max")
    if log_spacing:
        return np.geomspace(lower, upper, count)
    return np.linspace(lower, upper, count)


def build_rabi_particle_grid(spec: RabiParticleSpec) -> RabiParticleGrid:
    if spec.particle_mode not in {"grid", "random", "scheduled", "smc", "smc_refined", "multires_grid"}:
        raise ValueError(
            "particle_mode must be 'grid', 'random', 'scheduled', 'smc', 'smc_refined', or 'multires_grid'"
        )
    if spec.phase_amplitude_min <= 0.0 or spec.phase_amplitude_max <= 0.0:
        raise ValueError("phase-amplitude range must be strictly positive")
    if spec.phase_amplitude_min > spec.phase_amplitude_max:
        raise ValueError("phase_amplitude_min must not be larger than phase_amplitude_max")
    if spec.detuning_min > spec.detuning_max:
        raise ValueError("detuning_min must not be larger than detuning_max")
    if spec.drive_phase_count <= 0:
        raise ValueError("drive_phase_count must be positive")
    if spec.phase_amplitude_prior not in {"uniform", "log_uniform"}:
        raise ValueError("phase_amplitude_prior must be 'uniform' or 'log_uniform'")

    if spec.particle_mode in {"random", "smc", "smc_refined"}:
        if spec.random_particle_count <= 0:
            raise ValueError(
                "random_particle_count must be positive when particle_mode is 'random', 'smc', or 'smc_refined'"
            )

        rng = np.random.default_rng(int(spec.random_seed))
        particle_count = int(spec.random_particle_count)
        if spec.phase_amplitude_min == spec.phase_amplitude_max:
            phase_amplitudes = np.full(particle_count, spec.phase_amplitude_min, dtype=float)
        elif spec.phase_amplitude_prior == "log_uniform":
            log_min = math.log(spec.phase_amplitude_min)
            log_max = math.log(spec.phase_amplitude_max)
            phase_amplitudes = np.exp(rng.uniform(log_min, log_max, particle_count))
        else:
            phase_amplitudes = rng.uniform(
                spec.phase_amplitude_min,
                spec.phase_amplitude_max,
                particle_count,
            )
        if spec.detuning_min == spec.detuning_max:
            detunings = np.full(particle_count, spec.detuning_min, dtype=float)
        else:
            detunings = rng.uniform(spec.detuning_min, spec.detuning_max, particle_count)
        drive_phases = rng.uniform(0.0, 2.0 * math.pi, particle_count)
        prior_weights = np.full(particle_count, 1.0 / particle_count, dtype=float)

        return RabiParticleGrid(
            phase_amplitudes=np.asarray(phase_amplitudes, dtype=float),
            detunings=np.asarray(detunings, dtype=float),
            drive_phases=np.asarray(drive_phases, dtype=float),
            prior_weights=prior_weights,
            shape=(particle_count,),
            phase_amplitude_values=np.asarray(phase_amplitudes, dtype=float),
            detuning_values=np.asarray(detunings, dtype=float),
            drive_phase_values=np.asarray(drive_phases, dtype=float),
        )

    phase_amplitude_values = _known_or_grid_values(
        spec.phase_amplitude_min,
        spec.phase_amplitude_max,
        spec.phase_amplitude_count,
        parameter_name="phase_amplitude",
        log_spacing=True,
    )
    detuning_values = _known_or_grid_values(
        spec.detuning_min,
        spec.detuning_max,
        spec.detuning_count,
        parameter_name="detuning",
        log_spacing=False,
    )
    drive_phase_values = np.linspace(0.0, 2.0 * math.pi, spec.drive_phase_count, endpoint=False)

    if spec.phase_amplitude_min == spec.phase_amplitude_max:
        phase_amplitude_weights = np.ones(1, dtype=float)
    else:
        phase_amplitude_weights = _prior_masses(
            phase_amplitude_values,
            spec.phase_amplitude_min,
            spec.phase_amplitude_max,
            spec.phase_amplitude_prior,
            log_spacing=True,
        )
    detuning_weights = np.full(detuning_values.size, 1.0 / detuning_values.size, dtype=float)
    drive_phase_weights = np.full(spec.drive_phase_count, 1.0 / spec.drive_phase_count, dtype=float)

    phase_mesh, detuning_mesh, drive_phase_mesh = np.meshgrid(
        phase_amplitude_values,
        detuning_values,
        drive_phase_values,
        indexing="ij",
    )
    prior_weights = (
        phase_amplitude_weights[:, None, None]
        * detuning_weights[None, :, None]
        * drive_phase_weights[None, None, :]
    ).reshape(-1)

    return RabiParticleGrid(
        phase_amplitudes=phase_mesh.reshape(-1),
        detunings=detuning_mesh.reshape(-1),
        drive_phases=drive_phase_mesh.reshape(-1),
        prior_weights=prior_weights,
        shape=(phase_amplitude_values.size, detuning_values.size, drive_phase_values.size),
        phase_amplitude_values=phase_amplitude_values,
        detuning_values=detuning_values,
        drive_phase_values=drive_phase_values,
    )


def detuning_sinc_factor(detuning: np.ndarray | float, sensing_time: float) -> np.ndarray | float:
    half_detuning_phase = 0.5 * np.asarray(detuning, dtype=float) * float(sensing_time)
    return np.sinc(half_detuning_phase / math.pi)


def _relaxation_rate(time_constant: float | None) -> float:
    if time_constant is None or math.isinf(float(time_constant)):
        return 0.0
    if time_constant <= 0.0:
        raise ValueError("relaxation times must be positive or None")
    return 1.0 / float(time_constant)


def transverse_relaxation_factor(sensing_time: float, t2: float | None) -> float:
    """Average T2 damping of a weak resonant transverse Rabi response."""
    tau = float(sensing_time)
    if tau < 0.0:
        raise ValueError("sensing_time must be non-negative")
    if tau == 0.0:
        return 1.0
    rate2 = _relaxation_rate(t2)
    if rate2 == 0.0:
        return 1.0
    x = rate2 * tau
    return float(np.clip(-math.expm1(-x) / x, 0.0, 1.0))


def longitudinal_relaxation_factor(sensing_time: float, t1: float | None, t2: float | None) -> float:
    """Weak-drive T1/T2 damping factor for the Rabi excitation probability.

    The factor multiplies the ideal coherent response ``1 - cos(Phi)``.  It is
    derived by solving the weak-drive Bloch equations: coherence is built up
    during the pulse with T2 damping and the population created from that
    coherence subsequently relaxes with T1.
    """
    tau = float(sensing_time)
    if tau < 0.0:
        raise ValueError("sensing_time must be non-negative")
    if tau == 0.0:
        return 1.0

    rate1 = _relaxation_rate(t1)
    rate2 = _relaxation_rate(t2)

    if rate1 == 0.0 and rate2 == 0.0:
        return 1.0

    if rate2 == 0.0:
        if rate1 == 0.0:
            integral = 0.5 * tau**2
        else:
            x = rate1 * tau
            first_moment = (1.0 - math.exp(-x) * (1.0 + x)) / rate1**2
            integral = tau * (-math.expm1(-x)) / rate1 - first_moment
    else:
        if rate1 == 0.0:
            survival_integral = tau
        else:
            survival_integral = -math.expm1(-rate1 * tau) / rate1

        if abs(rate1 - rate2) <= 1.0e-12 * max(1.0, abs(rate1), abs(rate2)):
            coherence_survival_integral = tau * math.exp(-rate1 * tau)
        else:
            coherence_survival_integral = (math.exp(-rate2 * tau) - math.exp(-rate1 * tau)) / (rate1 - rate2)

        integral = (survival_integral - coherence_survival_integral) / rate2

    return float(np.clip(2.0 * integral / tau**2, 0.0, 1.0))


def rabi_probability_one(
    phase_amplitude: np.ndarray | float,
    detuning: np.ndarray | float,
    drive_phase: np.ndarray | float,
    start_time: float,
    sensing_time: float,
    axis: str,
    readout_angle: float,
    readout: RabiReadoutConfig,
) -> np.ndarray | float:
    effective_area = np.asarray(phase_amplitude, dtype=float) * detuning_sinc_factor(detuning, sensing_time)
    effective_phase = np.asarray(drive_phase, dtype=float) + np.asarray(detuning, dtype=float) * (
        float(start_time) + 0.5 * float(sensing_time)
    )
    transverse_factor = transverse_relaxation_factor(sensing_time, readout.t2)
    longitudinal_factor = longitudinal_relaxation_factor(sensing_time, readout.t1, readout.t2)

    if axis == "z":
        bloch_projection = -1.0 + longitudinal_factor * (1.0 - np.cos(effective_area))
    elif axis == "transverse":
        bloch_projection = transverse_factor * np.sin(effective_area) * np.sin(float(readout_angle) - effective_phase)
    else:
        raise ValueError("axis must be 'z' or 'transverse'")

    ideal_probability = 0.5 * (1.0 + readout.contrast * bloch_projection)
    noisy_probability = readout.readout_error + (1.0 - 2.0 * readout.readout_error) * ideal_probability
    return noisy_probability


def _clipped_probability_one(*args, **kwargs) -> np.ndarray | float:
    return np.clip(rabi_probability_one(*args, **kwargs), EPSILON, 1.0 - EPSILON)


def _h0_probability_one(axis: str, readout_angle: float, readout: RabiReadoutConfig) -> float:
    if axis == "z":
        projection = -1.0
    elif axis == "transverse":
        projection = 0.0
    else:
        raise ValueError("axis must be 'z' or 'transverse'")
    ideal_probability = 0.5 * (1.0 + readout.contrast * projection)
    noisy_probability = readout.readout_error + (1.0 - 2.0 * readout.readout_error) * ideal_probability
    return float(np.clip(noisy_probability, EPSILON, 1.0 - EPSILON))


def _fixed_axis_for_strategy(strategy: str, shot_index: int, fixed_angle: float) -> tuple[str, float]:
    if strategy == "fixed_z":
        return "z", 0.0
    if strategy == "fixed_x":
        return "transverse", float(fixed_angle)
    if strategy == "two_axis":
        return "transverse", 0.5 * math.pi * (shot_index % 2)
    raise ValueError("fixed strategy must be 'fixed_z', 'fixed_x', or 'two_axis'")


def _fixed_terms(
    config: RabiExperimentConfig,
    hypothesis: str,
    strategy: str,
    fixed_angle: float,
) -> _RabiTerms:
    if hypothesis not in {"H0", "H1"}:
        raise ValueError("hypothesis must be 'H0' or 'H1'")

    grid = build_rabi_particle_grid(config.particles)
    particle_one_probabilities = np.empty((config.n_shots, grid.prior_weights.size), dtype=float)
    h0_one_probabilities = np.empty(config.n_shots, dtype=float)
    true_one_probabilities = np.empty(config.n_shots, dtype=float)

    current_time = 0.0
    for shot_index in range(config.n_shots):
        axis, readout_angle = _fixed_axis_for_strategy(strategy, shot_index, fixed_angle)
        particle_one_probabilities[shot_index] = _clipped_probability_one(
            phase_amplitude=grid.phase_amplitudes,
            detuning=grid.detunings,
            drive_phase=grid.drive_phases,
            start_time=current_time,
            sensing_time=config.sensing_time,
            axis=axis,
            readout_angle=readout_angle,
            readout=config.readout,
        )
        h0_one = _h0_probability_one(axis, readout_angle, config.readout)
        h0_one_probabilities[shot_index] = h0_one
        if hypothesis == "H1":
            true_one_probabilities[shot_index] = float(
                _clipped_probability_one(
                    phase_amplitude=config.true_phase_amplitude,
                    detuning=config.true_detuning,
                    drive_phase=config.true_drive_phase,
                    start_time=current_time,
                    sensing_time=config.sensing_time,
                    axis=axis,
                    readout_angle=readout_angle,
                    readout=config.readout,
                )
            )
        else:
            true_one_probabilities[shot_index] = h0_one
        current_time += config.sensing_time + config.dead_time

    return _RabiTerms(
        prior_weights=grid.prior_weights.copy(),
        initial_h1_probability=float(config.initial_h1_probability),
        particle_one_probabilities=particle_one_probabilities,
        h0_one_probabilities=h0_one_probabilities,
        true_one_probabilities=true_one_probabilities,
    )


def _adaptive_terms(config: RabiExperimentConfig, hypothesis: str) -> _RabiTerms:
    if hypothesis not in {"H0", "H1"}:
        raise ValueError("hypothesis must be 'H0' or 'H1'")

    grid = build_rabi_particle_grid(config.particles)
    candidate_angles = np.asarray(config.candidate_angles, dtype=float)
    particle_one_probabilities = np.empty((config.n_shots, candidate_angles.size, grid.prior_weights.size), dtype=float)
    h0_one_probabilities = np.empty((config.n_shots, candidate_angles.size), dtype=float)
    true_one_probabilities = np.empty((config.n_shots, candidate_angles.size), dtype=float)

    current_time = 0.0
    for shot_index in range(config.n_shots):
        for angle_index, readout_angle in enumerate(candidate_angles):
            particle_one_probabilities[shot_index, angle_index] = _clipped_probability_one(
                phase_amplitude=grid.phase_amplitudes,
                detuning=grid.detunings,
                drive_phase=grid.drive_phases,
                start_time=current_time,
                sensing_time=config.sensing_time,
                axis="transverse",
                readout_angle=float(readout_angle),
                readout=config.readout,
            )
            h0_one = _h0_probability_one("transverse", float(readout_angle), config.readout)
            h0_one_probabilities[shot_index, angle_index] = h0_one
            if hypothesis == "H1":
                true_one_probabilities[shot_index, angle_index] = float(
                    _clipped_probability_one(
                        phase_amplitude=config.true_phase_amplitude,
                        detuning=config.true_detuning,
                        drive_phase=config.true_drive_phase,
                        start_time=current_time,
                        sensing_time=config.sensing_time,
                        axis="transverse",
                        readout_angle=float(readout_angle),
                        readout=config.readout,
                    )
                )
            else:
                true_one_probabilities[shot_index, angle_index] = h0_one
        current_time += config.sensing_time + config.dead_time

    return _RabiTerms(
        prior_weights=grid.prior_weights.copy(),
        initial_h1_probability=float(config.initial_h1_probability),
        particle_one_probabilities=particle_one_probabilities,
        h0_one_probabilities=h0_one_probabilities,
        true_one_probabilities=true_one_probabilities,
    )


def _binary_entropy(probability_one: np.ndarray | float) -> np.ndarray | float:
    p1 = np.clip(probability_one, EPSILON, 1.0 - EPSILON)
    p0 = 1.0 - p1
    return -(p1 * np.log(p1) + p0 * np.log(p0))


def _argmax_first_with_tolerance(values: np.ndarray, tolerance: float = 1.0e-14) -> np.ndarray:
    maxima = np.max(values, axis=1)
    return np.argmax(values >= (maxima[:, None] - tolerance), axis=1)


def binomial_neyman_pearson_power(
    n_shots: int,
    probability_h0: np.ndarray | float,
    probability_h1: np.ndarray | float,
    alpha: float,
) -> np.ndarray | float:
    """Exact randomized Neyman-Pearson power for repeated Bernoulli trials.

    The test uses the count of outcome ``1`` as the sufficient statistic.  If
    ``probability_h1 < probability_h0`` the rejection region is in the lower
    tail instead.  Randomization at the boundary is included so that the test
    has size exactly ``alpha`` rather than a conservative lattice-rounded size.
    """
    if n_shots <= 0:
        raise ValueError("n_shots must be positive")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")

    try:
        from scipy.stats import binom
    except ImportError as exc:  # pragma: no cover - exercised only without scipy
        raise ImportError("binomial_neyman_pearson_power requires scipy") from exc

    p0, p1 = np.broadcast_arrays(
        np.clip(np.asarray(probability_h0, dtype=float), EPSILON, 1.0 - EPSILON),
        np.clip(np.asarray(probability_h1, dtype=float), EPSILON, 1.0 - EPSILON),
    )
    result = np.empty(p0.shape, dtype=float)

    same = np.abs(p1 - p0) <= 10.0 * EPSILON
    result[same] = float(alpha)

    upper = (p1 > p0) & ~same
    if np.any(upper):
        upper_p0 = p0[upper]
        upper_p1 = p1[upper]
        boundary = np.asarray(binom.ppf(1.0 - alpha, n_shots, upper_p0), dtype=np.int64)
        boundary = np.clip(boundary, 0, n_shots)
        strict_tail_h0 = binom.sf(boundary, n_shots, upper_p0)
        boundary_mass_h0 = binom.pmf(boundary, n_shots, upper_p0)
        randomization = np.divide(
            alpha - strict_tail_h0,
            boundary_mass_h0,
            out=np.zeros_like(strict_tail_h0, dtype=float),
            where=boundary_mass_h0 > 0.0,
        )
        randomization = np.clip(randomization, 0.0, 1.0)
        result[upper] = binom.sf(boundary, n_shots, upper_p1) + randomization * binom.pmf(
            boundary, n_shots, upper_p1
        )

    lower = (p1 < p0) & ~same
    if np.any(lower):
        lower_p0 = p0[lower]
        lower_p1 = p1[lower]
        boundary = np.asarray(binom.ppf(alpha, n_shots, lower_p0), dtype=np.int64)
        boundary = np.clip(boundary, 0, n_shots)
        strict_tail_h0 = binom.cdf(boundary - 1, n_shots, lower_p0)
        boundary_mass_h0 = binom.pmf(boundary, n_shots, lower_p0)
        randomization = np.divide(
            alpha - strict_tail_h0,
            boundary_mass_h0,
            out=np.zeros_like(strict_tail_h0, dtype=float),
            where=boundary_mass_h0 > 0.0,
        )
        randomization = np.clip(randomization, 0.0, 1.0)
        result[lower] = binom.cdf(boundary - 1, n_shots, lower_p1) + randomization * binom.pmf(
            boundary, n_shots, lower_p1
        )

    result = np.clip(result, 0.0, 1.0)
    if result.shape == ():
        return float(result)
    return result


def rabi_oracle_known_signal_power(
    config: RabiExperimentConfig,
    phase_amplitudes: np.ndarray | float,
    alpha: float,
    polar_angle_count: int = 2001,
    return_best_angles: bool = False,
) -> np.ndarray | float | tuple[np.ndarray | float, np.ndarray | float]:
    """Power of a known-signal oracle within single-shot projective readout.

    The oracle knows the signal amplitude, detuning, and transverse phase.  For
    each shot it can align the azimuthal readout direction with the known Rabi
    drive and choose the best polar angle in the plane spanned by that direction
    and ``z``.  The resulting repeated experiment is an i.i.d. Bernoulli test,
    so the power is computed by the exact binomial Neyman-Pearson formula rather
    than by Monte Carlo.
    """
    if polar_angle_count < 2:
        raise ValueError("polar_angle_count must be at least 2")

    scalar_input = np.asarray(phase_amplitudes).shape == ()
    phases = np.atleast_1d(np.asarray(phase_amplitudes, dtype=float))
    polar_angles = np.linspace(0.0, math.pi, int(polar_angle_count))
    sin_polar = np.sin(polar_angles)
    cos_polar = np.cos(polar_angles)

    contrast = float(config.readout.contrast)
    readout_error = float(config.readout.readout_error)
    readout_scale = 1.0 - 2.0 * readout_error
    eta2 = transverse_relaxation_factor(config.sensing_time, config.readout.t2)
    eta1 = longitudinal_relaxation_factor(config.sensing_time, config.readout.t1, config.readout.t2)
    sinc_factor = float(detuning_sinc_factor(config.true_detuning, config.sensing_time))

    projection_h0 = -cos_polar
    probability_h0 = readout_error + readout_scale * 0.5 * (1.0 + contrast * projection_h0)
    probability_h0 = np.clip(probability_h0, EPSILON, 1.0 - EPSILON)

    best_powers = np.empty(phases.size, dtype=float)
    best_angles = np.empty(phases.size, dtype=float)
    for index, phase_amplitude in enumerate(phases):
        effective_area = float(phase_amplitude) * sinc_factor
        transverse_projection = eta2 * math.sin(effective_area)
        longitudinal_projection = -1.0 + eta1 * (1.0 - math.cos(effective_area))
        projection_h1 = transverse_projection * sin_polar + longitudinal_projection * cos_polar
        probability_h1 = readout_error + readout_scale * 0.5 * (1.0 + contrast * projection_h1)
        probability_h1 = np.clip(probability_h1, EPSILON, 1.0 - EPSILON)
        powers = binomial_neyman_pearson_power(config.n_shots, probability_h0, probability_h1, alpha)
        best_index = int(np.nanargmax(powers))
        best_powers[index] = float(powers[best_index])
        best_angles[index] = float(polar_angles[best_index])

    if scalar_input:
        power: np.ndarray | float = float(best_powers[0])
        angle: np.ndarray | float = float(best_angles[0])
    else:
        power = best_powers
        angle = best_angles
    if return_best_angles:
        return power, angle
    return power


def _sample_from_fixed_terms(
    terms: _RabiTerms,
    n_experiments: int,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    return _sample_from_fixed_terms_at_checkpoints(
        terms,
        n_experiments,
        batch_size,
        seed,
        np.asarray([terms.particle_one_probabilities.shape[0]], dtype=np.int64),
    )[:, 0]


def _normalize_log_bayes_checkpoints(checkpoints: np.ndarray | list[int] | tuple[int, ...], n_shots: int) -> np.ndarray:
    checkpoint_array = np.asarray(checkpoints, dtype=np.int64)
    if checkpoint_array.ndim != 1:
        raise ValueError("checkpoints must be a one-dimensional sequence")
    if checkpoint_array.size == 0:
        raise ValueError("checkpoints must contain at least one shot count")
    if np.any(checkpoint_array < 0):
        raise ValueError("checkpoints must be non-negative")
    if np.any(checkpoint_array > int(n_shots)):
        raise ValueError("checkpoints must not exceed config.n_shots")
    return np.asarray(sorted(set(int(x) for x in checkpoint_array)), dtype=np.int64)


def _sample_from_fixed_terms_at_checkpoints(
    terms: _RabiTerms,
    n_experiments: int,
    batch_size: int,
    seed: int,
    checkpoints: np.ndarray,
) -> np.ndarray:
    count = int(n_experiments)
    effective_batch_size = max(1, min(int(batch_size), count))
    checkpoint_values = _normalize_log_bayes_checkpoints(checkpoints, terms.particle_one_probabilities.shape[0])
    samples = np.empty((count, checkpoint_values.size), dtype=float)
    generator = np.random.default_rng(seed)

    zero_checkpoint_indices = np.flatnonzero(checkpoint_values == 0)
    shot_to_checkpoint = {int(shot): index for index, shot in enumerate(checkpoint_values) if int(shot) > 0}

    for batch_start in range(0, count, effective_batch_size):
        batch_stop = min(batch_start + effective_batch_size, count)
        batch_count = batch_stop - batch_start
        weights = np.repeat(terms.prior_weights[None, :], batch_count, axis=0)
        h1_probabilities = np.full(batch_count, terms.initial_h1_probability, dtype=float)
        log_bf = np.zeros(batch_count, dtype=float)

        for checkpoint_index in zero_checkpoint_indices:
            samples[batch_start:batch_stop, int(checkpoint_index)] = log_bf

        for shot_index in range(terms.particle_one_probabilities.shape[0]):
            particle_one = terms.particle_one_probabilities[shot_index]
            particle_zero = 1.0 - particle_one
            h0_one = terms.h0_one_probabilities[shot_index]
            h0_zero = 1.0 - h0_one
            true_one = terms.true_one_probabilities[shot_index]
            mixture_one = weights @ particle_one
            mixture_zero = 1.0 - mixture_one
            outcomes = generator.random(batch_count) < true_one

            particle_likelihood = np.where(outcomes[:, None], particle_one[None, :], particle_zero[None, :])
            mixture_likelihood = np.where(outcomes, mixture_one, mixture_zero)
            h0_likelihood = np.where(outcomes, h0_one, h0_zero)
            mixture_likelihood = np.clip(mixture_likelihood, EPSILON, 1.0)
            h0_likelihood = np.clip(h0_likelihood, EPSILON, 1.0)

            weights *= particle_likelihood
            weights /= mixture_likelihood[:, None]

            evidence = h1_probabilities * mixture_likelihood + (1.0 - h1_probabilities) * h0_likelihood
            evidence = np.clip(evidence, EPSILON, None)
            h1_probabilities = h1_probabilities * mixture_likelihood / evidence
            log_bf += np.log(mixture_likelihood) - np.log(h0_likelihood)

            checkpoint_index = shot_to_checkpoint.get(shot_index + 1)
            if checkpoint_index is not None:
                samples[batch_start:batch_stop, checkpoint_index] = log_bf

    return samples


def _sample_from_adaptive_terms(
    terms: _RabiTerms,
    n_experiments: int,
    batch_size: int,
    seed: int,
    lookahead_depth: int = 1,
    rollout: bool = False,
) -> np.ndarray:
    return _sample_from_adaptive_terms_at_checkpoints(
        terms,
        n_experiments,
        batch_size,
        seed,
        np.asarray([terms.particle_one_probabilities.shape[0]], dtype=np.int64),
        lookahead_depth,
        rollout,
    )[:, 0]


def _sample_from_adaptive_terms_at_checkpoints(
    terms: _RabiTerms,
    n_experiments: int,
    batch_size: int,
    seed: int,
    checkpoints: np.ndarray,
    lookahead_depth: int = 1,
    rollout: bool = False,
) -> np.ndarray:
    count = int(n_experiments)
    effective_batch_size = max(1, min(int(batch_size), count))
    checkpoint_values = _normalize_log_bayes_checkpoints(checkpoints, terms.particle_one_probabilities.shape[0])
    samples = np.empty((count, checkpoint_values.size), dtype=float)
    generator = np.random.default_rng(seed)
    depth = max(1, int(lookahead_depth))

    zero_checkpoint_indices = np.flatnonzero(checkpoint_values == 0)
    shot_to_checkpoint = {int(shot): index for index, shot in enumerate(checkpoint_values) if int(shot) > 0}

    for batch_start in range(0, count, effective_batch_size):
        batch_stop = min(batch_start + effective_batch_size, count)
        batch_count = batch_stop - batch_start
        weights = np.repeat(terms.prior_weights[None, :], batch_count, axis=0)
        h1_probabilities = np.full(batch_count, terms.initial_h1_probability, dtype=float)
        log_bf = np.zeros(batch_count, dtype=float)

        for checkpoint_index in zero_checkpoint_indices:
            samples[batch_start:batch_stop, int(checkpoint_index)] = log_bf

        for shot_index in range(terms.particle_one_probabilities.shape[0]):
            candidate_particle_one = terms.particle_one_probabilities[shot_index]
            candidate_particle_zero = 1.0 - candidate_particle_one
            candidate_h0_one = terms.h0_one_probabilities[shot_index]
            candidate_h0_zero = 1.0 - candidate_h0_one
            candidate_true_one = terms.true_one_probabilities[shot_index]

            utilities, mixture_one_matrix, _ = _adaptive_immediate_values(
                weights,
                h1_probabilities,
                candidate_particle_one,
                candidate_h0_one,
            )
            if depth == 1:
                pass
            elif rollout:
                utilities = _adaptive_rollout_values(
                    terms,
                    shot_index=shot_index,
                    weights=weights,
                    h1_probabilities=h1_probabilities,
                    depth=min(depth, terms.particle_one_probabilities.shape[0] - shot_index),
                )
            else:
                utilities = _adaptive_lookahead_values(
                    terms,
                    shot_index=shot_index,
                    weights=weights,
                    h1_probabilities=h1_probabilities,
                    depth=min(depth, terms.particle_one_probabilities.shape[0] - shot_index),
                )

            chosen_indices = _argmax_first_with_tolerance(utilities)
            chosen_particle_one = candidate_particle_one[chosen_indices]
            chosen_particle_zero = candidate_particle_zero[chosen_indices]
            chosen_h0_one = candidate_h0_one[chosen_indices]
            chosen_h0_zero = candidate_h0_zero[chosen_indices]
            chosen_true_one = candidate_true_one[chosen_indices]
            chosen_mixture_one = mixture_one_matrix[np.arange(batch_count), chosen_indices]
            chosen_mixture_zero = 1.0 - chosen_mixture_one

            outcomes = generator.random(batch_count) < chosen_true_one
            particle_likelihood = np.where(outcomes[:, None], chosen_particle_one, chosen_particle_zero)
            mixture_likelihood = np.where(outcomes, chosen_mixture_one, chosen_mixture_zero)
            h0_likelihood = np.where(outcomes, chosen_h0_one, chosen_h0_zero)
            mixture_likelihood = np.clip(mixture_likelihood, EPSILON, 1.0)
            h0_likelihood = np.clip(h0_likelihood, EPSILON, 1.0)

            weights *= particle_likelihood
            weights /= mixture_likelihood[:, None]

            evidence = h1_probabilities * mixture_likelihood + (1.0 - h1_probabilities) * h0_likelihood
            evidence = np.clip(evidence, EPSILON, None)
            h1_probabilities = h1_probabilities * mixture_likelihood / evidence
            log_bf += np.log(mixture_likelihood) - np.log(h0_likelihood)

            checkpoint_index = shot_to_checkpoint.get(shot_index + 1)
            if checkpoint_index is not None:
                samples[batch_start:batch_stop, checkpoint_index] = log_bf

    return samples


def _adaptive_immediate_values(
    weights: np.ndarray,
    h1_probabilities: np.ndarray,
    candidate_particle_one: np.ndarray,
    candidate_h0_one: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    one_minus_h1 = 1.0 - h1_probabilities
    mixture_one_matrix = weights @ candidate_particle_one.T
    predictive_one_matrix = (
        one_minus_h1[:, None] * candidate_h0_one[None, :]
        + h1_probabilities[:, None] * mixture_one_matrix
    )
    conditional_entropy_matrix = (
        one_minus_h1[:, None] * _binary_entropy(candidate_h0_one)[None, :]
        + h1_probabilities[:, None] * (weights @ _binary_entropy(candidate_particle_one).T)
    )
    utilities = _binary_entropy(predictive_one_matrix) - conditional_entropy_matrix
    return utilities, mixture_one_matrix, predictive_one_matrix


def _adaptive_lookahead_values(
    terms: _RabiTerms,
    shot_index: int,
    weights: np.ndarray,
    h1_probabilities: np.ndarray,
    depth: int,
) -> np.ndarray:
    candidate_particle_one = terms.particle_one_probabilities[shot_index]
    candidate_h0_one = terms.h0_one_probabilities[shot_index]
    values, mixture_one_matrix, predictive_one_matrix = _adaptive_immediate_values(
        weights,
        h1_probabilities,
        candidate_particle_one,
        candidate_h0_one,
    )

    if depth <= 1 or shot_index + 1 >= terms.particle_one_probabilities.shape[0]:
        return values

    candidate_particle_zero = 1.0 - candidate_particle_one
    candidate_h0_zero = 1.0 - candidate_h0_one

    for candidate_index in range(candidate_particle_one.shape[0]):
        mixture_one = np.clip(mixture_one_matrix[:, candidate_index], EPSILON, 1.0)
        mixture_zero = np.clip(1.0 - mixture_one_matrix[:, candidate_index], EPSILON, 1.0)
        predictive_one = np.clip(predictive_one_matrix[:, candidate_index], EPSILON, 1.0)
        predictive_zero = np.clip(1.0 - predictive_one_matrix[:, candidate_index], EPSILON, 1.0)

        for branch_probability, mixture_likelihood, h0_likelihood, particle_likelihood in (
            (
                predictive_one,
                mixture_one,
                candidate_h0_one[candidate_index],
                candidate_particle_one[candidate_index],
            ),
            (
                predictive_zero,
                mixture_zero,
                candidate_h0_zero[candidate_index],
                candidate_particle_zero[candidate_index],
            ),
        ):
            next_weights = weights * particle_likelihood[None, :] / mixture_likelihood[:, None]
            evidence = h1_probabilities * mixture_likelihood + (1.0 - h1_probabilities) * h0_likelihood
            evidence = np.clip(evidence, EPSILON, None)
            next_h1 = h1_probabilities * mixture_likelihood / evidence
            future_values = _adaptive_lookahead_values(
                terms,
                shot_index=shot_index + 1,
                weights=next_weights,
                h1_probabilities=next_h1,
                depth=depth - 1,
            )
            values[:, candidate_index] += branch_probability * np.max(future_values, axis=1)

    return values


def _adaptive_rollout_values(
    terms: _RabiTerms,
    shot_index: int,
    weights: np.ndarray,
    h1_probabilities: np.ndarray,
    depth: int,
) -> np.ndarray:
    """Evaluate first actions with a myopic rollout for the remaining horizon.

    Exact look-ahead maximizes over every future action at every branch.  The
    rollout approximation only maximizes the first action exactly; after each
    hypothetical outcome it follows the depth-1 myopic base policy.  This keeps
    the useful value-of-information signal while replacing the future action
    tree by a binary outcome tree.
    """
    candidate_particle_one = terms.particle_one_probabilities[shot_index]
    candidate_h0_one = terms.h0_one_probabilities[shot_index]
    values, mixture_one_matrix, predictive_one_matrix = _adaptive_immediate_values(
        weights,
        h1_probabilities,
        candidate_particle_one,
        candidate_h0_one,
    )

    if depth <= 1 or shot_index + 1 >= terms.particle_one_probabilities.shape[0]:
        return values

    candidate_particle_zero = 1.0 - candidate_particle_one
    candidate_h0_zero = 1.0 - candidate_h0_one

    for candidate_index in range(candidate_particle_one.shape[0]):
        mixture_one = np.clip(mixture_one_matrix[:, candidate_index], EPSILON, 1.0)
        mixture_zero = np.clip(1.0 - mixture_one_matrix[:, candidate_index], EPSILON, 1.0)
        predictive_one = np.clip(predictive_one_matrix[:, candidate_index], EPSILON, 1.0)
        predictive_zero = np.clip(1.0 - predictive_one_matrix[:, candidate_index], EPSILON, 1.0)

        for branch_probability, mixture_likelihood, h0_likelihood, particle_likelihood in (
            (
                predictive_one,
                mixture_one,
                candidate_h0_one[candidate_index],
                candidate_particle_one[candidate_index],
            ),
            (
                predictive_zero,
                mixture_zero,
                candidate_h0_zero[candidate_index],
                candidate_particle_zero[candidate_index],
            ),
        ):
            next_weights = weights * particle_likelihood[None, :] / mixture_likelihood[:, None]
            evidence = h1_probabilities * mixture_likelihood + (1.0 - h1_probabilities) * h0_likelihood
            evidence = np.clip(evidence, EPSILON, None)
            next_h1 = h1_probabilities * mixture_likelihood / evidence
            future_value = _myopic_rollout_base_value(
                terms,
                shot_index=shot_index + 1,
                weights=next_weights,
                h1_probabilities=next_h1,
                depth=depth - 1,
            )
            values[:, candidate_index] += branch_probability * future_value

    return values


def _myopic_rollout_base_value(
    terms: _RabiTerms,
    shot_index: int,
    weights: np.ndarray,
    h1_probabilities: np.ndarray,
    depth: int,
) -> np.ndarray:
    candidate_particle_one = terms.particle_one_probabilities[shot_index]
    candidate_particle_zero = 1.0 - candidate_particle_one
    candidate_h0_one = terms.h0_one_probabilities[shot_index]
    candidate_h0_zero = 1.0 - candidate_h0_one
    values, mixture_one_matrix, predictive_one_matrix = _adaptive_immediate_values(
        weights,
        h1_probabilities,
        candidate_particle_one,
        candidate_h0_one,
    )

    batch_indices = np.arange(weights.shape[0])
    chosen_indices = _argmax_first_with_tolerance(values)
    total_value = values[batch_indices, chosen_indices].copy()

    if depth <= 1 or shot_index + 1 >= terms.particle_one_probabilities.shape[0]:
        return total_value

    chosen_mixture_one = np.clip(mixture_one_matrix[batch_indices, chosen_indices], EPSILON, 1.0)
    chosen_mixture_zero = np.clip(1.0 - mixture_one_matrix[batch_indices, chosen_indices], EPSILON, 1.0)
    chosen_predictive_one = np.clip(predictive_one_matrix[batch_indices, chosen_indices], EPSILON, 1.0)
    chosen_predictive_zero = np.clip(1.0 - predictive_one_matrix[batch_indices, chosen_indices], EPSILON, 1.0)
    chosen_particle_one = candidate_particle_one[chosen_indices]
    chosen_particle_zero = candidate_particle_zero[chosen_indices]
    chosen_h0_one = candidate_h0_one[chosen_indices]
    chosen_h0_zero = candidate_h0_zero[chosen_indices]

    for branch_probability, mixture_likelihood, h0_likelihood, particle_likelihood in (
        (chosen_predictive_one, chosen_mixture_one, chosen_h0_one, chosen_particle_one),
        (chosen_predictive_zero, chosen_mixture_zero, chosen_h0_zero, chosen_particle_zero),
    ):
        next_weights = weights * particle_likelihood / mixture_likelihood[:, None]
        evidence = h1_probabilities * mixture_likelihood + (1.0 - h1_probabilities) * h0_likelihood
        evidence = np.clip(evidence, EPSILON, None)
        next_h1 = h1_probabilities * mixture_likelihood / evidence
        total_value += branch_probability * _myopic_rollout_base_value(
            terms,
            shot_index=shot_index + 1,
            weights=next_weights,
            h1_probabilities=next_h1,
            depth=depth - 1,
        )

    return total_value


def _scheduled_detuning_count(spec: RabiParticleSpec, observed_shots: int, time_step: float, current_count: int) -> int:
    if spec.detuning_min == spec.detuning_max:
        return 1
    maximum_count = int(spec.detuning_count)
    if maximum_count <= 0:
        raise ValueError("detuning_count must be positive unless detuning_min=detuning_max")
    current_count = max(1, int(current_count))
    if current_count >= maximum_count:
        return maximum_count

    elapsed = max(1.0, float(observed_shots) * float(time_step))
    tolerance = float(spec.scheduled_refine_phase_tolerance)
    if tolerance <= 0.0:
        raise ValueError("scheduled_refine_phase_tolerance must be positive")

    next_count = current_count
    total_width = float(spec.detuning_max - spec.detuning_min)
    while next_count < maximum_count and total_width * elapsed / next_count > tolerance:
        next_count = min(maximum_count, 2 * next_count + 1)
    return next_count


def _scheduled_particle_arrays(
    config: RabiExperimentConfig,
    detuning_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    spec = config.particles
    phase_amplitude_values = _known_or_grid_values(
        spec.phase_amplitude_min,
        spec.phase_amplitude_max,
        spec.phase_amplitude_count,
        parameter_name="phase_amplitude",
        log_spacing=True,
    )
    if spec.phase_amplitude_min == spec.phase_amplitude_max:
        phase_amplitude_weights = np.ones(1, dtype=float)
    else:
        phase_amplitude_weights = _prior_masses(
            phase_amplitude_values,
            spec.phase_amplitude_min,
            spec.phase_amplitude_max,
            spec.phase_amplitude_prior,
            log_spacing=True,
        )

    detuning_values = _known_or_grid_values(
        spec.detuning_min,
        spec.detuning_max,
        detuning_count,
        parameter_name="detuning",
        log_spacing=False,
    )
    detuning_weights = np.full(detuning_values.size, 1.0 / detuning_values.size, dtype=float)
    drive_phase_values = np.linspace(0.0, 2.0 * math.pi, spec.drive_phase_count, endpoint=False)
    drive_phase_weights = np.full(drive_phase_values.size, 1.0 / drive_phase_values.size, dtype=float)

    phase_mesh, detuning_mesh, drive_phase_mesh = np.meshgrid(
        phase_amplitude_values,
        detuning_values,
        drive_phase_values,
        indexing="ij",
    )
    detuning_index_mesh = np.broadcast_to(
        np.arange(detuning_values.size, dtype=int)[None, :, None],
        phase_mesh.shape,
    )
    prior_weights = (
        phase_amplitude_weights[:, None, None]
        * detuning_weights[None, :, None]
        * drive_phase_weights[None, None, :]
    ).reshape(-1)
    return (
        phase_mesh.reshape(-1),
        detuning_mesh.reshape(-1),
        drive_phase_mesh.reshape(-1),
        prior_weights,
        detuning_index_mesh.reshape(-1),
    )


def _scheduled_replay_history(
    config: RabiExperimentConfig,
    detuning_count: int,
    history_axes: list[str],
    history_angles: list[float],
    history_outcomes: list[bool],
    *,
    prune: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    phase_amplitudes, detunings, drive_phases, prior_weights, detuning_indices = _scheduled_particle_arrays(
        config,
        detuning_count,
    )
    weights = prior_weights.copy()
    h1_probability = float(config.initial_h1_probability)
    log_bf = 0.0
    current_time = 0.0

    for axis, readout_angle, outcome_is_one in zip(history_axes, history_angles, history_outcomes, strict=True):
        particle_one = _clipped_probability_one(
            phase_amplitude=phase_amplitudes,
            detuning=detunings,
            drive_phase=drive_phases,
            start_time=current_time,
            sensing_time=config.sensing_time,
            axis=axis,
            readout_angle=readout_angle,
            readout=config.readout,
        )
        mixture_one = float(weights @ particle_one)
        if outcome_is_one:
            mixture_likelihood = np.clip(mixture_one, EPSILON, 1.0)
            h0_likelihood = _h0_probability_one(axis, readout_angle, config.readout)
            particle_likelihood = particle_one
        else:
            mixture_likelihood = np.clip(1.0 - mixture_one, EPSILON, 1.0)
            h0_likelihood = 1.0 - _h0_probability_one(axis, readout_angle, config.readout)
            particle_likelihood = 1.0 - particle_one

        weights *= particle_likelihood
        weights /= mixture_likelihood
        evidence = h1_probability * mixture_likelihood + (1.0 - h1_probability) * h0_likelihood
        evidence = max(float(evidence), EPSILON)
        h1_probability = h1_probability * mixture_likelihood / evidence
        log_bf += math.log(mixture_likelihood) - math.log(max(float(h0_likelihood), EPSILON))
        current_time += config.sensing_time + config.dead_time

    if prune and detuning_count > 1:
        detuning_mass = np.bincount(detuning_indices, weights=weights, minlength=detuning_count)
        maximum_mass = float(np.max(detuning_mass))
        keep_detuning = detuning_mass >= maximum_mass * float(config.particles.scheduled_prune_relative_threshold)
        target_mass = float(config.particles.scheduled_prune_cumulative_mass)
        if 0.0 < target_mass < 1.0:
            order = np.argsort(detuning_mass)[::-1]
            cumulative = np.cumsum(detuning_mass[order])
            cutoff = int(np.searchsorted(cumulative, target_mass, side="left")) + 1
            keep_detuning[order[:cutoff]] = True
        particle_mask = keep_detuning[detuning_indices]
        if np.any(particle_mask) and not np.all(particle_mask):
            phase_amplitudes = phase_amplitudes[particle_mask]
            detunings = detunings[particle_mask]
            drive_phases = drive_phases[particle_mask]
            detuning_indices = detuning_indices[particle_mask]
            weights = weights[particle_mask]
            weights /= np.sum(weights)

    return phase_amplitudes, detunings, drive_phases, weights, detuning_indices, log_bf, h1_probability


def _sample_rabi_log_bayes_factors_scheduled(
    config: RabiExperimentConfig,
    strategy: str,
    hypothesis: str,
    n_experiments: int,
    seed: int,
    fixed_angle: float,
    true_drive_phases: np.ndarray | None,
) -> np.ndarray:
    if hypothesis not in {"H0", "H1"}:
        raise ValueError("hypothesis must be 'H0' or 'H1'")
    if strategy not in {"fixed_z", "fixed_x", "two_axis", "adaptive"}:
        raise ValueError("scheduled particle mode currently supports fixed_z, fixed_x, two_axis, and adaptive")
    if true_drive_phases is not None:
        true_drive_phases = np.asarray(true_drive_phases, dtype=float)
        if true_drive_phases.shape != (int(n_experiments),):
            raise ValueError("true_drive_phases must have shape (n_experiments,)")

    count = int(n_experiments)
    generator = np.random.default_rng(seed)
    samples = np.empty(count, dtype=float)
    time_step = float(config.sensing_time + config.dead_time)
    initial_count = int(config.particles.scheduled_initial_detuning_count)
    if config.particles.detuning_min == config.particles.detuning_max:
        initial_count = 1
    initial_count = max(1, initial_count)
    if config.particles.detuning_min != config.particles.detuning_max:
        initial_count = min(initial_count, int(config.particles.detuning_count))

    candidate_angles = np.asarray(config.candidate_angles, dtype=float)

    for experiment_index in range(count):
        true_drive_phase = float(config.true_drive_phase)
        if true_drive_phases is not None:
            true_drive_phase = float(true_drive_phases[experiment_index])

        detuning_count = initial_count
        history_axes: list[str] = []
        history_angles: list[float] = []
        history_outcomes: list[bool] = []
        (
            phase_amplitudes,
            detunings,
            drive_phases,
            weights,
            detuning_indices,
            log_bf,
            h1_probability,
        ) = _scheduled_replay_history(config, detuning_count, history_axes, history_angles, history_outcomes, prune=False)

        current_time = 0.0
        for shot_index in range(config.n_shots):
            next_count = _scheduled_detuning_count(config.particles, shot_index, time_step, detuning_count)
            if next_count != detuning_count:
                detuning_count = next_count
                (
                    phase_amplitudes,
                    detunings,
                    drive_phases,
                    weights,
                    detuning_indices,
                    log_bf,
                    h1_probability,
                ) = _scheduled_replay_history(
                    config,
                    detuning_count,
                    history_axes,
                    history_angles,
                    history_outcomes,
                    prune=True,
                )

            if strategy == "adaptive":
                utilities = np.empty(candidate_angles.size, dtype=float)
                mixtures = np.empty(candidate_angles.size, dtype=float)
                h0_ones = np.empty(candidate_angles.size, dtype=float)
                one_minus_h1 = 1.0 - h1_probability
                for angle_index, readout_angle in enumerate(candidate_angles):
                    particle_one_for_angle = _clipped_probability_one(
                        phase_amplitude=phase_amplitudes,
                        detuning=detunings,
                        drive_phase=drive_phases,
                        start_time=current_time,
                        sensing_time=config.sensing_time,
                        axis="transverse",
                        readout_angle=float(readout_angle),
                        readout=config.readout,
                    )
                    mixture_one_for_angle = float(weights @ particle_one_for_angle)
                    h0_one_for_angle = _h0_probability_one("transverse", float(readout_angle), config.readout)
                    predictive_one = one_minus_h1 * h0_one_for_angle + h1_probability * mixture_one_for_angle
                    conditional_entropy = (
                        one_minus_h1 * _binary_entropy(h0_one_for_angle)
                        + h1_probability * float(weights @ _binary_entropy(particle_one_for_angle))
                    )
                    utilities[angle_index] = float(_binary_entropy(predictive_one) - conditional_entropy)
                    mixtures[angle_index] = mixture_one_for_angle
                    h0_ones[angle_index] = h0_one_for_angle
                chosen_index = int(np.argmax(utilities >= (np.max(utilities) - 1.0e-14)))
                axis = "transverse"
                readout_angle = float(candidate_angles[chosen_index])
                particle_one = _clipped_probability_one(
                    phase_amplitude=phase_amplitudes,
                    detuning=detunings,
                    drive_phase=drive_phases,
                    start_time=current_time,
                    sensing_time=config.sensing_time,
                    axis=axis,
                    readout_angle=readout_angle,
                    readout=config.readout,
                )
                mixture_one = float(mixtures[chosen_index])
                h0_one = float(h0_ones[chosen_index])
            else:
                axis, readout_angle = _fixed_axis_for_strategy(strategy, shot_index, fixed_angle)
                particle_one = _clipped_probability_one(
                    phase_amplitude=phase_amplitudes,
                    detuning=detunings,
                    drive_phase=drive_phases,
                    start_time=current_time,
                    sensing_time=config.sensing_time,
                    axis=axis,
                    readout_angle=readout_angle,
                    readout=config.readout,
                )
                mixture_one = float(weights @ particle_one)
                h0_one = _h0_probability_one(axis, readout_angle, config.readout)

            if hypothesis == "H1":
                true_one = float(
                    _clipped_probability_one(
                        phase_amplitude=config.true_phase_amplitude,
                        detuning=config.true_detuning,
                        drive_phase=true_drive_phase,
                        start_time=current_time,
                        sensing_time=config.sensing_time,
                        axis=axis,
                        readout_angle=readout_angle,
                        readout=config.readout,
                    )
                )
            else:
                true_one = h0_one

            outcome_is_one = bool(generator.random() < true_one)
            if outcome_is_one:
                mixture_likelihood = np.clip(mixture_one, EPSILON, 1.0)
                h0_likelihood = h0_one
                particle_likelihood = particle_one
            else:
                mixture_likelihood = np.clip(1.0 - mixture_one, EPSILON, 1.0)
                h0_likelihood = 1.0 - h0_one
                particle_likelihood = 1.0 - particle_one
            weights *= particle_likelihood
            weights /= mixture_likelihood
            evidence = h1_probability * mixture_likelihood + (1.0 - h1_probability) * h0_likelihood
            evidence = max(float(evidence), EPSILON)
            h1_probability = h1_probability * mixture_likelihood / evidence
            log_bf += math.log(float(mixture_likelihood)) - math.log(max(float(h0_likelihood), EPSILON))

            history_axes.append(axis)
            history_angles.append(float(readout_angle))
            history_outcomes.append(outcome_is_one)
            current_time += time_step

        samples[experiment_index] = log_bf

    return samples


def _systematic_resample_indices(weights: np.ndarray, generator: np.random.Generator) -> np.ndarray:
    count = weights.size
    positions = (generator.random() + np.arange(count, dtype=float)) / count
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right")


def _smc_transformed_log_prior_array(phase_amplitudes: np.ndarray, spec: RabiParticleSpec) -> np.ndarray:
    values = np.asarray(phase_amplitudes, dtype=float)
    log_prior = np.full(values.shape, -math.inf, dtype=float)
    valid = (values >= spec.phase_amplitude_min) & (values <= spec.phase_amplitude_max)
    if spec.phase_amplitude_min == spec.phase_amplitude_max:
        log_prior[valid] = 0.0
    elif spec.phase_amplitude_prior == "log_uniform":
        log_prior[valid] = 0.0
    elif spec.phase_amplitude_prior == "uniform":
        log_prior[valid] = np.log(np.clip(values[valid], EPSILON, None))
    else:
        raise ValueError(f"unsupported prior type: {spec.phase_amplitude_prior}")
    return log_prior


def _particles_history_log_likelihood(
    config: RabiExperimentConfig,
    phase_amplitudes: np.ndarray,
    detunings: np.ndarray,
    drive_phases: np.ndarray,
    history_axes: list[str],
    history_angles: list[float],
    history_outcomes: list[bool],
) -> np.ndarray:
    log_likelihoods = np.zeros(np.asarray(phase_amplitudes).shape, dtype=float)
    current_time = 0.0
    for axis, readout_angle, outcome_is_one in zip(history_axes, history_angles, history_outcomes, strict=True):
        probability_one = _clipped_probability_one(
            phase_amplitude=phase_amplitudes,
            detuning=detunings,
            drive_phase=drive_phases,
            start_time=current_time,
            sensing_time=config.sensing_time,
            axis=axis,
            readout_angle=readout_angle,
            readout=config.readout,
        )
        if outcome_is_one:
            log_likelihoods += np.log(np.clip(probability_one, EPSILON, 1.0))
        else:
            log_likelihoods += np.log(np.clip(1.0 - probability_one, EPSILON, 1.0))
        current_time += config.sensing_time + config.dead_time
    return log_likelihoods


def _smc_rejuvenate_particles(
    config: RabiExperimentConfig,
    phase_amplitudes: np.ndarray,
    detunings: np.ndarray,
    drive_phases: np.ndarray,
    log_likelihoods: np.ndarray,
    history_axes: list[str],
    history_angles: list[float],
    history_outcomes: list[bool],
    generator: np.random.Generator,
) -> None:
    spec = config.particles
    move_steps = int(spec.smc_move_steps)
    if move_steps <= 0 or not history_outcomes:
        return

    amplitude_known = spec.phase_amplitude_min == spec.phase_amplitude_max
    detuning_known = spec.detuning_min == spec.detuning_max
    log_amplitude_step = float(spec.smc_log_phase_amplitude_step)
    detuning_step = float(spec.smc_detuning_step)
    drive_phase_step = float(spec.smc_drive_phase_step)

    for _ in range(move_steps):
        proposed_phase_amplitudes = phase_amplitudes.copy()
        proposed_detunings = detunings.copy()
        proposed_drive_phases = (drive_phases + generator.normal(0.0, drive_phase_step, drive_phases.size)) % (
            2.0 * math.pi
        )
        valid = np.ones(phase_amplitudes.size, dtype=bool)

        if not amplitude_known and log_amplitude_step > 0.0:
            proposed_phase_amplitudes = np.exp(
                np.log(phase_amplitudes) + generator.normal(0.0, log_amplitude_step, phase_amplitudes.size)
            )
            valid &= (proposed_phase_amplitudes >= spec.phase_amplitude_min) & (
                proposed_phase_amplitudes <= spec.phase_amplitude_max
            )

        if not detuning_known and detuning_step > 0.0:
            proposed_detunings = detunings + generator.normal(0.0, detuning_step, detunings.size)
            valid &= (proposed_detunings >= spec.detuning_min) & (proposed_detunings <= spec.detuning_max)

        if not np.any(valid):
            continue

        old_log_target = log_likelihoods + _smc_transformed_log_prior_array(phase_amplitudes, spec)
        proposed_log_likelihoods = np.empty_like(log_likelihoods)
        proposed_log_likelihoods[valid] = _particles_history_log_likelihood(
            config,
            proposed_phase_amplitudes[valid],
            proposed_detunings[valid],
            proposed_drive_phases[valid],
            history_axes,
            history_angles,
            history_outcomes,
        )
        proposed_log_likelihoods[~valid] = -math.inf
        proposed_log_target = proposed_log_likelihoods + _smc_transformed_log_prior_array(
            proposed_phase_amplitudes,
            spec,
        )
        log_uniforms = np.log(np.clip(generator.random(phase_amplitudes.size), EPSILON, 1.0))
        accept = valid & (log_uniforms < proposed_log_target - old_log_target)
        if not np.any(accept):
            continue
        phase_amplitudes[accept] = proposed_phase_amplitudes[accept]
        detunings[accept] = proposed_detunings[accept]
        drive_phases[accept] = proposed_drive_phases[accept]
        log_likelihoods[accept] = proposed_log_likelihoods[accept]


def _sample_rabi_log_bayes_factors_smc(
    config: RabiExperimentConfig,
    strategy: str,
    hypothesis: str,
    n_experiments: int,
    seed: int,
    fixed_angle: float,
    true_drive_phases: np.ndarray | None,
) -> np.ndarray:
    if hypothesis not in {"H0", "H1"}:
        raise ValueError("hypothesis must be 'H0' or 'H1'")
    if strategy not in {"fixed_z", "fixed_x", "two_axis", "adaptive"}:
        raise ValueError("SMC particle mode currently supports fixed_z, fixed_x, two_axis, and adaptive")
    if true_drive_phases is not None:
        true_drive_phases = np.asarray(true_drive_phases, dtype=float)
        if true_drive_phases.shape != (int(n_experiments),):
            raise ValueError("true_drive_phases must have shape (n_experiments,)")

    count = int(n_experiments)
    generator = np.random.default_rng(seed)
    samples = np.empty(count, dtype=float)
    candidate_angles = np.asarray(config.candidate_angles, dtype=float)
    ess_fraction = float(config.particles.smc_resample_ess_fraction)
    if not (0.0 < ess_fraction <= 1.0):
        raise ValueError("smc_resample_ess_fraction must lie in (0, 1]")

    for experiment_index in range(count):
        true_drive_phase = float(config.true_drive_phase)
        if true_drive_phases is not None:
            true_drive_phase = float(true_drive_phases[experiment_index])

        grid = build_rabi_particle_grid(config.particles)
        phase_amplitudes = grid.phase_amplitudes.copy()
        detunings = grid.detunings.copy()
        drive_phases = grid.drive_phases.copy()
        weights = grid.prior_weights.copy()
        log_likelihoods = np.zeros(weights.size, dtype=float)
        h1_probability = float(config.initial_h1_probability)
        log_bf = 0.0
        current_time = 0.0
        history_axes: list[str] = []
        history_angles: list[float] = []
        history_outcomes: list[bool] = []

        for shot_index in range(config.n_shots):
            if strategy == "adaptive":
                utilities = np.empty(candidate_angles.size, dtype=float)
                mixtures = np.empty(candidate_angles.size, dtype=float)
                h0_ones = np.empty(candidate_angles.size, dtype=float)
                one_minus_h1 = 1.0 - h1_probability
                for angle_index, readout_angle in enumerate(candidate_angles):
                    particle_one_for_angle = _clipped_probability_one(
                        phase_amplitude=phase_amplitudes,
                        detuning=detunings,
                        drive_phase=drive_phases,
                        start_time=current_time,
                        sensing_time=config.sensing_time,
                        axis="transverse",
                        readout_angle=float(readout_angle),
                        readout=config.readout,
                    )
                    mixture_one_for_angle = float(weights @ particle_one_for_angle)
                    h0_one_for_angle = _h0_probability_one("transverse", float(readout_angle), config.readout)
                    predictive_one = one_minus_h1 * h0_one_for_angle + h1_probability * mixture_one_for_angle
                    conditional_entropy = (
                        one_minus_h1 * _binary_entropy(h0_one_for_angle)
                        + h1_probability * float(weights @ _binary_entropy(particle_one_for_angle))
                    )
                    utilities[angle_index] = float(_binary_entropy(predictive_one) - conditional_entropy)
                    mixtures[angle_index] = mixture_one_for_angle
                    h0_ones[angle_index] = h0_one_for_angle
                chosen_index = int(np.argmax(utilities >= (np.max(utilities) - 1.0e-14)))
                axis = "transverse"
                readout_angle = float(candidate_angles[chosen_index])
                particle_one = _clipped_probability_one(
                    phase_amplitude=phase_amplitudes,
                    detuning=detunings,
                    drive_phase=drive_phases,
                    start_time=current_time,
                    sensing_time=config.sensing_time,
                    axis=axis,
                    readout_angle=readout_angle,
                    readout=config.readout,
                )
                mixture_one = float(mixtures[chosen_index])
                h0_one = float(h0_ones[chosen_index])
            else:
                axis, readout_angle = _fixed_axis_for_strategy(strategy, shot_index, fixed_angle)
                particle_one = _clipped_probability_one(
                    phase_amplitude=phase_amplitudes,
                    detuning=detunings,
                    drive_phase=drive_phases,
                    start_time=current_time,
                    sensing_time=config.sensing_time,
                    axis=axis,
                    readout_angle=readout_angle,
                    readout=config.readout,
                )
                mixture_one = float(weights @ particle_one)
                h0_one = _h0_probability_one(axis, readout_angle, config.readout)

            if hypothesis == "H1":
                true_one = float(
                    _clipped_probability_one(
                        phase_amplitude=config.true_phase_amplitude,
                        detuning=config.true_detuning,
                        drive_phase=true_drive_phase,
                        start_time=current_time,
                        sensing_time=config.sensing_time,
                        axis=axis,
                        readout_angle=readout_angle,
                        readout=config.readout,
                    )
                )
            else:
                true_one = h0_one

            outcome_is_one = bool(generator.random() < true_one)
            if outcome_is_one:
                mixture_likelihood = np.clip(mixture_one, EPSILON, 1.0)
                h0_likelihood = h0_one
                particle_likelihood = particle_one
            else:
                mixture_likelihood = np.clip(1.0 - mixture_one, EPSILON, 1.0)
                h0_likelihood = 1.0 - h0_one
                particle_likelihood = 1.0 - particle_one

            weights *= particle_likelihood
            weights /= mixture_likelihood
            log_likelihoods += np.log(np.clip(particle_likelihood, EPSILON, 1.0))
            evidence = h1_probability * mixture_likelihood + (1.0 - h1_probability) * h0_likelihood
            evidence = max(float(evidence), EPSILON)
            h1_probability = h1_probability * mixture_likelihood / evidence
            log_bf += math.log(float(mixture_likelihood)) - math.log(max(float(h0_likelihood), EPSILON))

            history_axes.append(axis)
            history_angles.append(float(readout_angle))
            history_outcomes.append(outcome_is_one)

            effective_sample_size = 1.0 / float(np.sum(weights * weights))
            if effective_sample_size < ess_fraction * weights.size:
                indices = _systematic_resample_indices(weights, generator)
                phase_amplitudes = phase_amplitudes[indices].copy()
                detunings = detunings[indices].copy()
                drive_phases = drive_phases[indices].copy()
                log_likelihoods = log_likelihoods[indices].copy()
                weights = np.full(weights.size, 1.0 / weights.size, dtype=float)
                _smc_rejuvenate_particles(
                    config,
                    phase_amplitudes,
                    detunings,
                    drive_phases,
                    log_likelihoods,
                    history_axes,
                    history_angles,
                    history_outcomes,
                    generator,
                )

            current_time += config.sensing_time + config.dead_time

        samples[experiment_index] = log_bf

    return samples


def sample_rabi_log_bayes_factors(
    config: RabiExperimentConfig,
    strategy: str,
    hypothesis: str,
    n_experiments: int,
    batch_size: int = 512,
    seed: int | None = None,
    fixed_angle: float = 0.0,
    method: str = "vectorized",
    true_drive_phases: np.ndarray | None = None,
) -> np.ndarray:
    if n_experiments <= 0:
        raise ValueError("n_experiments must be positive")
    if method not in {"vectorized", "numba"}:
        raise ValueError("method must be 'vectorized' or 'numba'")
    if (
        true_drive_phases is not None
        and method != "numba"
        and config.particles.particle_mode not in {"scheduled", "smc", "smc_refined", "multires_grid"}
    ):
        raise ValueError("true_drive_phases is currently supported only with method='numba'")
    selected_seed = config.seed if seed is None else int(seed)
    adaptive_depth = _adaptive_depth_for_strategy(strategy)

    if config.particles.particle_mode == "multires_grid":
        if method != "numba":
            raise ValueError("particle_mode='multires_grid' currently requires method='numba'")
        if strategy in {"fixed_z", "fixed_x", "two_axis"}:
            deterministic_config = replace(config, particles=replace(config.particles, particle_mode="grid"))
            from .numba_rabi import sample_rabi_log_bayes_factors_numba

            return sample_rabi_log_bayes_factors_numba(
                deterministic_config,
                strategy,
                hypothesis,
                n_experiments,
                batch_size=batch_size,
                seed=selected_seed,
                fixed_angle=fixed_angle,
                true_drive_phases=true_drive_phases,
            )
        from .numba_rabi import sample_rabi_log_bayes_factors_multires_numba

        return sample_rabi_log_bayes_factors_multires_numba(
            config,
            strategy=strategy,
            hypothesis=hypothesis,
            n_experiments=n_experiments,
            seed=selected_seed,
            fixed_angle=fixed_angle,
            true_drive_phases=true_drive_phases,
        )

    if config.particles.particle_mode == "scheduled":
        return _sample_rabi_log_bayes_factors_scheduled(
            config,
            strategy=strategy,
            hypothesis=hypothesis,
            n_experiments=n_experiments,
            seed=selected_seed,
            fixed_angle=fixed_angle,
            true_drive_phases=true_drive_phases,
        )

    if config.particles.particle_mode in {"smc", "smc_refined"}:
        if method == "numba":
            if config.particles.particle_mode == "smc_refined" and strategy in {"fixed_z", "fixed_x", "two_axis"}:
                deterministic_particles = replace(
                    config.particles,
                    particle_mode="grid",
                    detuning_count=max(
                        int(config.particles.detuning_count),
                        int(config.particles.smc_refine_detuning_coarse_count),
                        1,
                    ),
                )
                deterministic_config = replace(config, particles=deterministic_particles)
                from .numba_rabi import sample_rabi_log_bayes_factors_numba

                return sample_rabi_log_bayes_factors_numba(
                    deterministic_config,
                    strategy,
                    hypothesis,
                    n_experiments,
                    batch_size=batch_size,
                    seed=selected_seed,
                    fixed_angle=fixed_angle,
                    true_drive_phases=true_drive_phases,
                )

            from .numba_rabi import sample_rabi_log_bayes_factors_smc_numba

            return sample_rabi_log_bayes_factors_smc_numba(
                config,
                strategy=strategy,
                hypothesis=hypothesis,
                n_experiments=n_experiments,
                seed=selected_seed,
                fixed_angle=fixed_angle,
                true_drive_phases=true_drive_phases,
            )
        if config.particles.particle_mode == "smc_refined":
            raise ValueError("particle_mode='smc_refined' currently requires method='numba'")
        return _sample_rabi_log_bayes_factors_smc(
            config,
            strategy=strategy,
            hypothesis=hypothesis,
            n_experiments=n_experiments,
            seed=selected_seed,
            fixed_angle=fixed_angle,
            true_drive_phases=true_drive_phases,
        )

    if method == "numba":
        from .numba_rabi import sample_rabi_log_bayes_factors_numba

        return sample_rabi_log_bayes_factors_numba(
            config,
            strategy,
            hypothesis,
            n_experiments,
            batch_size=batch_size,
            seed=selected_seed,
            fixed_angle=fixed_angle,
            true_drive_phases=true_drive_phases,
        )

    if strategy in {"fixed_z", "fixed_x", "two_axis"}:
        terms = _fixed_terms(config, hypothesis=hypothesis, strategy=strategy, fixed_angle=fixed_angle)
        return _sample_from_fixed_terms(terms, n_experiments, batch_size=batch_size, seed=selected_seed)
    if adaptive_depth is not None:
        terms = _adaptive_terms(config, hypothesis=hypothesis)
        return _sample_from_adaptive_terms(
            terms,
            n_experiments,
            batch_size=batch_size,
            seed=selected_seed,
            lookahead_depth=adaptive_depth,
            rollout=_is_rollout_strategy(strategy),
        )

    raise ValueError(
        "strategy must be 'fixed_z', 'fixed_x', 'two_axis', 'adaptive', 'lookahead_<depth>', "
        "'rollout_<depth>', or 'two_stage_<switch_shot>'"
    )


def sample_rabi_log_bayes_factors_at_checkpoints(
    config: RabiExperimentConfig,
    strategy: str,
    hypothesis: str,
    n_experiments: int,
    checkpoints: np.ndarray | list[int] | tuple[int, ...],
    batch_size: int = 512,
    seed: int | None = None,
    fixed_angle: float = 0.0,
    method: str = "vectorized",
    true_drive_phases: np.ndarray | None = None,
) -> np.ndarray:
    """Sample log Bayes factors along trajectory prefixes.

    The returned array has shape ``(n_experiments, n_checkpoints)``.  Column
    ``j`` contains the log Bayes factor after ``checkpoints[j]`` shots from the
    same pseudo experiments, so an ``n_shot`` scan can reuse long trajectories
    rather than rerunning each shorter experiment independently.
    """
    if n_experiments <= 0:
        raise ValueError("n_experiments must be positive")
    if method not in {"vectorized", "numba"}:
        raise ValueError("method must be 'vectorized' or 'numba'")
    checkpoint_values = _normalize_log_bayes_checkpoints(checkpoints, config.n_shots)
    selected_seed = config.seed if seed is None else int(seed)
    adaptive_depth = _adaptive_depth_for_strategy(strategy)

    if config.particles.particle_mode != "grid":
        raise ValueError("checkpoint sampling currently supports particle_mode='grid'")
    if method == "numba":
        from .numba_rabi import sample_rabi_log_bayes_factors_at_checkpoints_numba

        return sample_rabi_log_bayes_factors_at_checkpoints_numba(
            config,
            strategy,
            hypothesis,
            n_experiments,
            checkpoint_values,
            batch_size=batch_size,
            seed=selected_seed,
            fixed_angle=fixed_angle,
            true_drive_phases=true_drive_phases,
        )
    if true_drive_phases is not None:
        raise ValueError("true_drive_phases is supported only with method='numba'")
    if strategy in {"fixed_z", "fixed_x", "two_axis"}:
        terms = _fixed_terms(config, hypothesis=hypothesis, strategy=strategy, fixed_angle=fixed_angle)
        return _sample_from_fixed_terms_at_checkpoints(
            terms,
            n_experiments,
            batch_size=batch_size,
            seed=selected_seed,
            checkpoints=checkpoint_values,
        )
    if adaptive_depth is not None:
        terms = _adaptive_terms(config, hypothesis=hypothesis)
        return _sample_from_adaptive_terms_at_checkpoints(
            terms,
            n_experiments,
            batch_size=batch_size,
            seed=selected_seed,
            checkpoints=checkpoint_values,
            lookahead_depth=adaptive_depth,
            rollout=_is_rollout_strategy(strategy),
        )

    raise ValueError(
        "checkpoint sampling currently supports 'fixed_z', 'fixed_x', 'two_axis', "
        "'adaptive', 'lookahead_<depth>', and 'rollout_<depth>'"
    )


def sample_rabi_diagnostics_at_checkpoints(
    config: RabiExperimentConfig,
    strategy: str,
    hypothesis: str,
    n_experiments: int,
    checkpoints: np.ndarray | list[int] | tuple[int, ...],
    batch_size: int = 512,
    seed: int | None = None,
    fixed_angle: float = 0.0,
    true_drive_phases: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Sample prefix log Bayes factors and sufficient trajectory diagnostics.

    Besides ``log_bayes_factors``, the result stores ``predictable_kl`` =
    sum D_KL(p0 || m), its null conditional variance, and the cumulative
    transverse design matrix ``(Q_xx, Q_xy, Q_yy)`` at each checkpoint.
    For H1 histories it additionally returns ``signal_null_kl`` =
    sum D_KL(p_true || p0) and ``signal_predictive_kl`` =
    sum D_KL(p_true || m). Their difference is the H1 log-Bayes-factor
    compensator; each probability is evaluated before observing that shot.
    Here m is the H1 posterior predictive, conditional on the same history.
    Diagnostics currently use the compiled grid-particle implementation.
    """
    if n_experiments <= 0:
        raise ValueError("n_experiments must be positive")
    if config.particles.particle_mode != "grid":
        raise ValueError("diagnostic checkpoint sampling currently supports particle_mode='grid'")

    checkpoint_values = _normalize_log_bayes_checkpoints(checkpoints, config.n_shots)
    selected_seed = config.seed if seed is None else int(seed)
    from .numba_rabi import sample_rabi_log_bayes_factors_at_checkpoints_numba

    result = sample_rabi_log_bayes_factors_at_checkpoints_numba(
        config,
        strategy,
        hypothesis,
        n_experiments,
        checkpoint_values,
        batch_size=batch_size,
        seed=selected_seed,
        fixed_angle=fixed_angle,
        true_drive_phases=true_drive_phases,
        return_diagnostics=True,
    )
    if not isinstance(result, dict):  # pragma: no cover - guards the compiled API contract.
        raise RuntimeError("compiled diagnostic sampler returned an unexpected result")
    return result


def with_true_phase_amplitude(config: RabiExperimentConfig, phase_amplitude: float) -> RabiExperimentConfig:
    return replace(config, true_phase_amplitude=float(phase_amplitude))
