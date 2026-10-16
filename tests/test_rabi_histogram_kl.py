"""Run with: python -m unittest discover -s tests -p test_rabi_histogram_kl.py"""

import ast
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np


def notebook_functions(*names):
    # Load only the requested helpers, without running plotting or sampling cells.
    path = Path(__file__).resolve().parents[1] / "plot_final/rabi_log_bayes_histograms.ipynb"
    notebook = json.loads(path.read_text())
    functions = [
        node
        for cell in notebook["cells"] if cell["cell_type"] == "code"
        for node in ast.parse("".join(cell["source"])).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = dict(np=np, Path=Path, json=json, hashlib=hashlib, time=time, os=os,
                     asdict=asdict, replace=replace)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class RabiHistogramKlTests(unittest.TestCase):
    def test_exact_size_count_tests_and_phase_averaged_power(self):
        from scipy.stats import binom

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from dm_search.rabi import RabiReadoutConfig, rabi_probability_one

        ns = notebook_functions(
            "randomized_threshold", "rejection_probabilities", "conventional_statistic",
            "conventional_null_support", "conventional_h0_center", "conventional_axis_probability",
            "conventional_xy_h0_moments", "conventional_xy_quadrature_statistic",
            "conventional_theory_phases", "conventional_exact_power", "conventional_detection_row",
        )
        ns.update(math=math, binom=binom, rabi_probability_one=rabi_probability_one,
                  CONVENTIONAL_N_SHOTS=1024, SENSING_TIME=1.0, TRUE_DETUNING=0.0,
                  CONVENTIONAL_HIST_PHASE=0.19, CONVENTIONAL_THEORY_PHASE_COUNT=2048,
                  CONVENTIONAL_READOUT=RabiReadoutConfig(contrast=.99, readout_error=.005, t1=1, t2=1),
                  ALPHA=.05, CONVENTIONAL_STRATEGIES={s: {"label": s} for s in ("fixed_z", "fixed_x", "two_axis")})
        # Independent count-sum benchmarks; a conservative integer cut fails these.
        for strategy, expected in (("fixed_z", .4090505320046242),
                                   ("fixed_x", .623787049520),
                                   ("two_axis", .657238241690)):
            with self.subTest(strategy=strategy):
                counts, statistic, mass = ns["conventional_null_support"](strategy)
                cut, gamma = ns["randomized_threshold"](statistic, mass, .05)
                reject = ns["rejection_probabilities"](statistic, cut, gamma)
                self.assertGreater(gamma, 0)
                self.assertLess(gamma, 1)
                self.assertAlmostEqual(float(mass @ reject), .05, places=13)
                self.assertAlmostEqual(ns["conventional_exact_power"](strategy, cut, gamma), expected, places=10)

        # At the boundary all records have the same rejection weight: the
        # integrated-coin estimator has zero MC SE, not a binomial SE.
        row = ns["conventional_detection_row"]("fixed_z", np.full(5, 16), np.full(5, 16))
        self.assertAlmostEqual(row["power"], .804541672609, places=10)
        self.assertAlmostEqual(row["power_se"], 0.0, places=14)

    def test_paired_null_and_signal_mean_checks(self):
        ns = notebook_functions("mean_evidence_check", "null_mean_kl_check", "signal_mean_kl_check")
        check = ns["null_mean_kl_check"]
        kl = np.array([1.0, 2.0, 3.0])
        result = check(-kl, kl)
        self.assertEqual(result["h0_mean_kl_prediction"], -2.0)
        self.assertGreater(result["h0_mean_kl_prediction_se"], 0.0)
        self.assertEqual(result["h0_mean_residual"], 0.0)
        self.assertEqual(result["h0_mean_residual_se"], 0.0)
        result = check(np.array([-1.0, 0.0, -3.0]), kl)
        self.assertAlmostEqual(result["h0_mean_residual"], 2.0 / 3.0)
        self.assertAlmostEqual(result["h0_mean_residual_se"], 2.0 / 3.0)
        self.assertAlmostEqual(result["h0_mean_residual_z"], 1.0)
        with self.assertRaises(ValueError):
            check(-kl, kl[:2])

        signal_check = ns["signal_mean_kl_check"]
        result = signal_check(kl, 2 * kl, kl)
        self.assertEqual(result["h1_mean_kl_prediction"], 2.0)
        self.assertAlmostEqual(result["h1_mean_kl_prediction_se"], 1.0 / np.sqrt(3))
        self.assertEqual(result["h1_mean_residual"], 0.0)
        self.assertEqual(result["h1_mean_residual_se"], 0.0)
        result = signal_check(kl + [0, 2, 0], 2 * kl, kl)
        self.assertAlmostEqual(result["h1_mean_residual"], 2.0 / 3.0)
        self.assertAlmostEqual(result["h1_mean_residual_se"], 2.0 / 3.0)
        with self.assertRaises(ValueError):
            signal_check(kl, 2 * kl, kl[:1])  # Broadcasting would silently corrupt pairing.

    @patch.dict(os.environ, {"RABI_ALLOW_SIMULATION": "1"})
    def test_h1_cache_reuse_and_model_identity(self):
        ns = notebook_functions("stable_hash", "random_true_drive_phases", "load_or_generate_h1_diagnostics")

        @dataclass
        class Config:
            n_shots: int = 6
            seed: int = 9
            contrast: float = 0.99

        calls = []
        def sample(config, **kwargs):
            calls.append(kwargs)
            self.assertEqual(kwargs["hypothesis"], "H1")
            self.assertEqual(kwargs["checkpoints"], [6])
            self.assertEqual(kwargs["true_drive_phases"].shape, (4,))
            return {"log_bayes_factors": np.arange(4.0)[:, None],
                    "signal_null_kl": (2 + np.arange(4.0))[:, None],
                    "signal_predictive_kl": np.full((4, 1), 2.0)}

        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            shared = cache_dir / "shared_prefix.npz"
            shared.write_bytes(b"existing shared data")
            ns.update(PARTICLE_MODE="grid", PHASE_AVERAGE_MODE="random_per_experiment",
                      N_SIGNAL=4, N_SHOTS=6, BATCH_SIZE=512, base_config=Config(),
                      sampler_strategy=lambda strategy: strategy, USE_CACHE=True, FORCE_RECOMPUTE=False,
                      CACHE_DIR=cache_dir, OUTPUT_BASENAME="rabi_log_bayes_histograms", RUN_PRESET="test",
                      H1_DIAGNOSTICS={}, PREFIX_TRUE_DRIVE_PHASES={}, SAMPLE_SOURCES={},
                      warmup_numba_once=lambda: None, sample_rabi_diagnostics_at_checkpoints=sample)
            load = ns["load_or_generate_h1_diagnostics"]
            with patch.dict(os.environ, {"RABI_ALLOW_SIMULATION": "0"}):
                with self.assertRaises(FileNotFoundError):
                    load("fixed_x", seed=77)
            self.assertEqual(calls, [])
            original = load("fixed_x", seed=77)
            self.assertEqual(len(calls), 1)
            ns["H1_DIAGNOSTICS"].clear()
            np.testing.assert_array_equal(load("fixed_x", seed=77), original)
            self.assertEqual(len(calls), 1)  # Cache hit must not invoke the sampler.
            np.testing.assert_array_equal(ns["H1_DIAGNOSTICS"]["fixed_x"]["signal_null_kl"], 2 + original)
            ns["base_config"] = replace(ns["base_config"], contrast=0.999)
            load("fixed_x", seed=77)
            self.assertEqual(len(calls), 2)  # A likelihood change requires a distinct cache.
            self.assertEqual(len(list(cache_dir.glob("*_h1_kl_*.npz"))), 2)
            self.assertEqual(shared.read_bytes(), b"existing shared data")


if __name__ == "__main__":
    unittest.main()
