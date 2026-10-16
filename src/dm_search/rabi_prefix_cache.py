"""Shared configuration and cache naming for Rabi prefix simulations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping

import numpy as np

from .rabi import (
    RabiExperimentConfig,
    RabiParticleGrid,
    RabiParticleSpec,
    RabiReadoutConfig,
    build_rabi_particle_grid,
)


RABI_PREFIX_PRESETS = {
    "sample": {
        "n_shot_max": 128,
        "n_experiments": 8,
        "batch_size": 4,
        "phi_r_list": "0.08,0.14,0.24",
        "target_powers": "0.5,0.8",
        "amplitude_count": 8,
        "drive_phase_count": 16,
        "candidate_angle_count": 8,
        "candidate_polar_count": 5,
    },
    "quick": {
        "n_shot_max": 256,
        "n_experiments": 24,
        "batch_size": 12,
        "phi_r_list": "0.06,0.10,0.16,0.25,0.38",
        "target_powers": "0.5,0.8",
        "amplitude_count": 12,
        "drive_phase_count": 24,
        "candidate_angle_count": 8,
        "candidate_polar_count": 5,
    },
    "draft": {
        "n_shot_max": 1024,
        "n_experiments": 100,
        "batch_size": 128,
        "phi_r_list": "0.025,0.035,0.05,0.065,0.08,0.10,0.12,0.15,0.19,0.24,0.32",
        "target_powers": "0.5,0.7,0.9",
        "amplitude_count": 24,
        "drive_phase_count": 48,
        "candidate_angle_count": 16,
        "candidate_polar_count": 5,
    },
    "final": {
        "n_shot_max": 16384,
        "n_experiments": 1000,
        "batch_size": 512,
        "phi_r_list": "0.025,0.035,0.05,0.065,0.08,0.10,0.12,0.15,0.19,0.24,0.32",
        "target_powers": "0.5,0.7,0.9",
        "amplitude_count": 48,
        "drive_phase_count": 96,
        "candidate_angle_count": 16,
        "candidate_polar_count": 7,
    },
}


RABI_PREFIX_DETECTOR_PROFILES = {
    "current": {
        "contrast": 0.99,
        "readout_error": 0.005,
    },
    "high_fidelity": {
        "contrast": 0.999,
        "readout_error": 0.0005,
    },
}


# The baseline publication scan trades redundant null-floor points for the
# larger injections needed to resolve the P=0.7 crossing down to n=64.
RABI_PREFIX_CURRENT_FINAL_PHI_R_LIST = (
    "0.05,0.08,0.10,0.12,0.15,0.19,0.24,0.32,0.40,0.50,0.60"
)


def _parse_int_list(text: str) -> np.ndarray:
    return np.asarray([int(piece.strip()) for piece in text.split(",") if piece.strip()], dtype=int)


def _parse_float_list(text: str) -> np.ndarray:
    return np.asarray([float(piece.strip()) for piece in text.split(",") if piece.strip()], dtype=float)


def _default_checkpoints(n_shot_max: int) -> np.ndarray:
    base = np.asarray(
        [
            32,
            48,
            64,
            96,
            128,
            192,
            256,
            384,
            512,
            768,
            1024,
            1536,
            2048,
            3072,
            4096,
            6144,
            8192,
            12288,
            16384,
            24576,
            32768,
            49152,
            65536,
        ],
        dtype=int,
    )
    checkpoints = base[base <= int(n_shot_max)]
    if checkpoints.size == 0 or checkpoints[-1] != int(n_shot_max):
        checkpoints = np.asarray(sorted(set([*checkpoints.tolist(), int(n_shot_max)])), dtype=int)
    return checkpoints


def stable_setup_hash(payload: Mapping[str, object]) -> str:
    blob = json.dumps(dict(payload), sort_keys=True, default=list).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def rabi_prefix_cache_path(
    cache_dir: Path,
    preset: str,
    raw_setup: Mapping[str, object],
    *,
    basename: str = "rabi_prefix_sensitivity_vs_shots",
) -> Path:
    return Path(cache_dir) / f"{basename}_{preset}_{stable_setup_hash(raw_setup)}.npz"


@dataclass(frozen=True)
class RabiPrefixSetup:
    preset: str
    alpha: float
    target_powers: tuple[float, ...]
    n_shot_max: int
    n_null: int
    n_signal: int
    batch_size: int
    phase_grid: np.ndarray
    checkpoints: np.ndarray
    phase_average_mode: str
    strategies: dict[str, dict[str, object]]
    base_template: RabiExperimentConfig
    particle_grid: RabiParticleGrid
    raw_setup: dict[str, object]
    analysis_setup: dict[str, object]


def build_rabi_prefix_setup(
    preset: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> RabiPrefixSetup:
    """Build the canonical raw setup from a preset and ``RABI_PREFIX_*`` values."""

    env = os.environ if environ is None else environ
    if preset not in RABI_PREFIX_PRESETS:
        raise ValueError(f"unknown RABI prefix preset={preset!r}")
    defaults = RABI_PREFIX_PRESETS[preset]

    detector_profile = env.get("RABI_PREFIX_DETECTOR_PROFILE", "current")
    if detector_profile not in RABI_PREFIX_DETECTOR_PROFILES:
        choices = ", ".join(sorted(RABI_PREFIX_DETECTOR_PROFILES))
        raise ValueError(
            f"unknown RABI_PREFIX_DETECTOR_PROFILE={detector_profile!r}; "
            f"choose one of {choices}"
        )
    detector_defaults = RABI_PREFIX_DETECTOR_PROFILES[detector_profile]

    alpha = float(env.get("RABI_PREFIX_ALPHA", "0.05"))
    default_n_shot_max = (
        4096
        if preset == "final" and detector_profile == "current"
        else int(defaults["n_shot_max"])
    )
    n_shot_max = int(env.get("RABI_PREFIX_N_SHOT_MAX", str(default_n_shot_max)))
    n_experiments = int(env.get("RABI_PREFIX_N_EXPERIMENTS", str(defaults["n_experiments"])))
    n_null = int(env.get("RABI_PREFIX_N_NULL", str(n_experiments)))
    n_signal = int(env.get("RABI_PREFIX_N_SIGNAL", str(n_experiments)))
    batch_size = int(env.get("RABI_PREFIX_BATCH_SIZE", str(defaults["batch_size"])))
    default_phi_r_list = (
        RABI_PREFIX_CURRENT_FINAL_PHI_R_LIST
        if preset == "final" and detector_profile == "current"
        else str(defaults["phi_r_list"])
    )
    phase_grid = _parse_float_list(
        env.get("RABI_PREFIX_PHI_R_LIST", default_phi_r_list)
    )
    target_powers = tuple(
        _parse_float_list(env.get("RABI_PREFIX_TARGET_POWERS", str(defaults["target_powers"]))).tolist()
    )

    checkpoint_text = env.get("RABI_PREFIX_CHECKPOINTS", "")
    checkpoints = _parse_int_list(checkpoint_text) if checkpoint_text.strip() else _default_checkpoints(n_shot_max)
    checkpoints = np.asarray(
        sorted(set(int(value) for value in checkpoints if 0 <= int(value) <= n_shot_max)),
        dtype=int,
    )
    if checkpoints.size == 0 or checkpoints[-1] != n_shot_max:
        checkpoints = np.asarray(sorted(set([*checkpoints.tolist(), n_shot_max])), dtype=int)

    sensing_time = float(env.get("RABI_PREFIX_SENSING_TIME", "1.0"))
    t1_time = float(env.get("RABI_PREFIX_T1", str(sensing_time)))
    t2_time = float(env.get("RABI_PREFIX_T2", str(sensing_time)))
    dead_time = float(env.get("RABI_PREFIX_DEAD_TIME", "0.0"))
    contrast = float(env.get("RABI_PREFIX_CONTRAST", str(detector_defaults["contrast"])))
    readout_error = float(
        env.get("RABI_PREFIX_READOUT_ERROR", str(detector_defaults["readout_error"]))
    )

    phase_amplitude_min = float(env.get("RABI_PREFIX_AMPLITUDE_MIN", "0.02"))
    phase_amplitude_max = float(env.get("RABI_PREFIX_AMPLITUDE_MAX", "1.2"))
    phase_amplitude_count = int(
        env.get("RABI_PREFIX_AMPLITUDE_COUNT", str(defaults["amplitude_count"]))
    )
    detuning_min = float(env.get("RABI_PREFIX_DETUNING_MIN", "0.0"))
    detuning_max = float(env.get("RABI_PREFIX_DETUNING_MAX", "0.0"))
    detuning_count = int(env.get("RABI_PREFIX_DETUNING_COUNT", "0"))
    drive_phase_count = int(
        env.get("RABI_PREFIX_DRIVE_PHASE_COUNT", str(defaults["drive_phase_count"]))
    )
    candidate_angle_count = int(
        env.get("RABI_PREFIX_CANDIDATE_ANGLE_COUNT", str(defaults["candidate_angle_count"]))
    )
    candidate_polar_count = int(
        env.get("RABI_PREFIX_CANDIDATE_POLAR_COUNT", str(defaults["candidate_polar_count"]))
    )
    adaptive_periods = tuple(
        dict.fromkeys(
            int(value)
            for value in _parse_int_list(env.get("RABI_PREFIX_ADAPTIVE_PERIODS", "1,2"))
        )
    )
    if not adaptive_periods or any(period < 1 for period in adaptive_periods):
        raise ValueError("RABI_PREFIX_ADAPTIVE_PERIODS must contain positive integers")
    include_stein = env.get("RABI_PREFIX_INCLUDE_STEIN", "0") == "1"
    phase_average_mode = env.get("RABI_PREFIX_PHASE_MODE", "random_per_experiment")

    strategies: dict[str, dict[str, object]] = {
        "fixed_z": {"label": r"Fixed $z$", "color": "#4C78A8", "marker": "o"},
        "fixed_x": {"label": r"Fixed $x$", "color": "#F58518", "marker": "D"},
        "two_axis": {"label": r"Fixed $x/y$", "color": "#B279A2", "marker": "v"},
    }
    adaptive_colors = ("#54A24B", "#E45756", "#72B7B2", "#FF9DA6")
    adaptive_markers = ("s", "P", "X", "^")
    for index, period in enumerate(adaptive_periods):
        strategies[f"adaptive_l{period}"] = {
            "label": rf"Adaptive $L={period}$",
            "color": adaptive_colors[index % len(adaptive_colors)],
            "marker": adaptive_markers[index % len(adaptive_markers)],
            "sampler_strategy": f"periodic_hybrid_sphere_continuous_{period}",
        }
        if include_stein:
            strategies[f"adaptive_stein_l{period}"] = {
                "label": rf"Adaptive KL/Stein $L={period}$",
                "color": "#2A6FBB",
                "marker": "D",
                "sampler_strategy": f"periodic_stein_hybrid_sphere_continuous_{period}",
            }

    base_template = RabiExperimentConfig(
        n_shots=n_shot_max,
        sensing_time=sensing_time,
        dead_time=dead_time,
        true_phase_amplitude=float(phase_grid[0]),
        true_detuning=0.0,
        true_drive_phase=0.0,
        readout=RabiReadoutConfig(
            contrast=contrast,
            readout_error=readout_error,
            t1=t1_time,
            t2=t2_time,
        ),
        particles=RabiParticleSpec(
            phase_amplitude_min=phase_amplitude_min,
            phase_amplitude_max=phase_amplitude_max,
            phase_amplitude_count=phase_amplitude_count,
            detuning_min=detuning_min,
            detuning_max=detuning_max,
            detuning_count=detuning_count,
            drive_phase_count=drive_phase_count,
        ),
        candidate_angles=tuple(np.linspace(0.0, math.pi, candidate_angle_count, endpoint=False)),
        candidate_polars=tuple(np.linspace(0.0, math.pi, candidate_polar_count)),
        seed=12345,
    )
    particle_grid = build_rabi_particle_grid(base_template.particles)

    raw_setup: dict[str, object] = {
        "preset": preset,
        "n_shot_max": n_shot_max,
        "checkpoints": checkpoints.tolist(),
        "n_null": n_null,
        "n_signal": n_signal,
        "batch_size": batch_size,
        "phase_grid": phase_grid.tolist(),
        "sensing_time": sensing_time,
        "t1_time": t1_time,
        "t2_time": t2_time,
        "dead_time": dead_time,
        "detector_profile": detector_profile,
        "contrast": contrast,
        "readout_error": readout_error,
        "particle_shape": particle_grid.shape,
        "particle_spec": base_template.particles.__dict__,
        "candidate_angle_count": len(base_template.candidate_angles),
        "candidate_polar_count": len(base_template.candidate_polars),
        "adaptive_periods": list(adaptive_periods),
        "adaptive_sampler_strategies": {
            key: value["sampler_strategy"]
            for key, value in strategies.items()
            if key.startswith("adaptive_l")
        },
        "phase_average_mode": phase_average_mode,
        "strategies": list(strategies),
        "diagnostic_schema": {
            "predictable_kl": "cumulative D_KL(P0 predictive || H1 posterior predictive)",
            "predictable_variance": "cumulative conditional variance of log(P0/H1 predictive)",
            "polarization_design": "cumulative (Q_xx,Q_xy,Q_yy) for transverse readout components",
        },
        "cache_version": "prefix-raw-logb-v4-detector-profiles-period-scan",
    }
    if include_stein:
        raw_setup["paired_adaptive_random_numbers"] = True
    analysis_setup: dict[str, object] = {
        "alpha": alpha,
        "target_powers": list(target_powers),
        "analysis_version": "prefix-analysis-v1",
    }
    return RabiPrefixSetup(
        preset=preset,
        alpha=alpha,
        target_powers=target_powers,
        n_shot_max=n_shot_max,
        n_null=n_null,
        n_signal=n_signal,
        batch_size=batch_size,
        phase_grid=phase_grid,
        checkpoints=checkpoints,
        phase_average_mode=phase_average_mode,
        strategies=strategies,
        base_template=base_template,
        particle_grid=particle_grid,
        raw_setup=raw_setup,
        analysis_setup=analysis_setup,
    )


def assert_matching_raw_setup(
    expected: Mapping[str, object],
    actual: Mapping[str, object],
) -> None:
    expected_blob = json.dumps(dict(expected), sort_keys=True, default=list)
    actual_blob = json.dumps(dict(actual), sort_keys=True, default=list)
    if expected_blob != actual_blob:
        raise ValueError(
            "The cache raw_setup does not match the requested raw_setup. "
            f"expected_hash={stable_setup_hash(expected)}, "
            f"actual_hash={stable_setup_hash(actual)}"
        )


__all__ = [
    "RABI_PREFIX_DETECTOR_PROFILES",
    "RABI_PREFIX_PRESETS",
    "RabiPrefixSetup",
    "assert_matching_raw_setup",
    "build_rabi_prefix_setup",
    "rabi_prefix_cache_path",
    "stable_setup_hash",
]
