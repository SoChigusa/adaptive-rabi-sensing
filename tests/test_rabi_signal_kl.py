"""A49 diagnostics checked against explicit probabilities on the sampled histories."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dm_search.numba_rabi import is_numba_available, run_rabi_sphere_learning_trajectory_numba
from dm_search.rabi import (
    RabiExperimentConfig, RabiParticleSpec, RabiReadoutConfig,
    build_rabi_particle_grid, rabi_probability_one,
    sample_rabi_diagnostics_at_checkpoints,
    sample_rabi_log_bayes_factors_at_checkpoints,
)


def binary_kl(p, q):
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


@unittest.skipUnless(is_numba_available(), "numba is not installed")
class SignalKLTests(unittest.TestCase):
    def setUp(self):
        self.config = RabiExperimentConfig(
            n_shots=6, sensing_time=0.7, dead_time=0.2,
            true_phase_amplitude=0.4, true_detuning=0.15, true_drive_phase=0.4,
            readout=RabiReadoutConfig(contrast=0.98, readout_error=0.01, t1=1.0, t2=1.0),
            particles=RabiParticleSpec(
                phase_amplitude_min=0.05, phase_amplitude_max=1.0, phase_amplitude_count=4,
                detuning_min=-0.2, detuning_max=0.2, detuning_count=3, drive_phase_count=8,
            ),
            candidate_angles=tuple(np.linspace(0.0, np.pi, 6, endpoint=False)),
            candidate_polars=tuple(np.linspace(0.0, np.pi, 4)),
        )

    def test_fixed_histories_match_explicit_posterior_replay(self):
        config = self.config
        checkpoints = np.arange(config.n_shots + 1)
        phases = np.array([0.4, 1.2, 2.5, 3.2, 5.8])
        seed, batch_size = 231, 3
        grid = build_rabi_particle_grid(config.particles)
        for strategy in ("fixed_z", "fixed_x", "two_axis"):
            with self.subTest(strategy=strategy):
                kwargs = dict(batch_size=batch_size, seed=seed, true_drive_phases=phases)
                diagnostics = sample_rabi_diagnostics_at_checkpoints(
                    config, strategy, "H1", len(phases), checkpoints, **kwargs,
                )
                plain = sample_rabi_log_bayes_factors_at_checkpoints(
                    config, strategy, "H1", len(phases), checkpoints, method="numba", **kwargs,
                )
                np.testing.assert_array_equal(diagnostics["log_bayes_factors"], plain)
                expected = {key: np.zeros_like(plain) for key in (
                    "log_bayes_factors", "signal_null_kl", "signal_predictive_kl",
                )}
                generator = np.random.default_rng(seed)
                for first in range(0, len(phases), batch_size):
                    stop = min(first + batch_size, len(phases))
                    uniforms = generator.random((config.n_shots, stop - first))
                    for column, phase in enumerate(phases[first:stop]):
                        weights = grid.prior_weights.copy()
                        for shot in range(config.n_shots):
                            axis = "z" if strategy == "fixed_z" else "transverse"
                            angle = np.pi / 2 if strategy == "two_axis" and shot % 2 else 0.0
                            def probability(amplitude, detuning, drive_phase):
                                return rabi_probability_one(
                                    amplitude, detuning, drive_phase,
                                    shot * (config.sensing_time + config.dead_time),
                                    config.sensing_time, axis, angle, config.readout,
                                )
                            p0 = probability(0.0, 0.0, 0.0)
                            ptrue = probability(config.true_phase_amplitude, config.true_detuning, phase)
                            particles = probability(grid.phase_amplitudes, grid.detunings, grid.drive_phases)
                            mixture = weights @ particles
                            i = first + column
                            expected["signal_null_kl"][i, shot + 1] = binary_kl(ptrue, p0)
                            expected["signal_predictive_kl"][i, shot + 1] = binary_kl(ptrue, mixture)
                            one = uniforms[shot, column] < ptrue
                            likelihood = particles if one else 1 - particles
                            denominator = mixture if one else 1 - mixture
                            expected["log_bayes_factors"][i, shot + 1] = np.log(
                                denominator / (p0 if one else 1 - p0)
                            )
                            weights *= likelihood / denominator
                for key, increments in expected.items():
                    np.testing.assert_allclose(diagnostics[key], np.cumsum(increments, axis=1), atol=1e-12)

    def test_sphere_compensator_matches_independent_trajectory_records(self):
        config = self.config
        checkpoints = np.arange(config.n_shots + 1)
        # A single history gives the same random stream to the separate record sampler.
        for strategy in ("adaptive_sphere_continuous", "periodic_hybrid_sphere_continuous_2"):
            with self.subTest(strategy=strategy):
                diagnostics = sample_rabi_diagnostics_at_checkpoints(
                    config, strategy, "H1", 1, checkpoints, seed=57,
                )
                plain = sample_rabi_log_bayes_factors_at_checkpoints(
                    config, strategy, "H1", 1, checkpoints, seed=57, method="numba",
                )
                np.testing.assert_array_equal(diagnostics["log_bayes_factors"], plain)
                trajectory = run_rabi_sphere_learning_trajectory_numba(
                    config, strategy, "H1", checkpoints, seed=57,
                )
                records = trajectory["records"]
                # The record sampler independently retains pre-outcome predictive/true probabilities.
                polar, mixture, ptrue = records[:, 4], records[:, 13], records[:, 14]
                readout = config.readout
                p0 = readout.readout_error + (1 - 2 * readout.readout_error) * (
                    1 - readout.contrast * np.cos(polar)
                ) / 2
                for key, reference in (("signal_null_kl", p0), ("signal_predictive_kl", mixture)):
                    expected = np.r_[0.0, np.cumsum(binary_kl(ptrue, reference))]
                    np.testing.assert_allclose(diagnostics[key][0], expected, atol=1e-12)
                np.testing.assert_allclose(plain[0], trajectory["snapshot_log_bayes_factors"], atol=1e-12)
                null = sample_rabi_diagnostics_at_checkpoints(
                    config, strategy, "H0", 1, (0, config.n_shots), seed=57,
                )
                self.assertNotIn("signal_null_kl", null)
                self.assertNotIn("signal_predictive_kl", null)


if __name__ == "__main__":
    unittest.main()
