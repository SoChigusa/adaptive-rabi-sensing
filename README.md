# Adaptive Rabi sensing: code and data

Numerical code and data for **So Chigusa, “Adaptive detection of Rabi signals under composite hypotheses,” [arXiv:2609.16119](https://arxiv.org/abs/2609.16119)**.

This repository contains the Monte Carlo records, posterior trajectories, analysis notebooks, and reference PDFs for all 15 figures in the paper. The included data reproduce the figures without repeating the production simulations. The release contains 20 input NPZ files (33.4 MB); Git LFS is not required.

## Reproduce the figures

Tested with Python 3.10.14 and the package versions in `requirements.txt`. From the repository root:

```sh
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py
```

`reproduce.py` verifies input SHA-256 hashes, then runs the analysis notebooks in separate processes with the paper settings. It uses a non-interactive Matplotlib backend. The cached analysis took about two minutes on the release machine; runtime depends on hardware. Missing or modified inputs cause an error before plotting. This command does not run the production Monte Carlo simulations.

The original paper PDFs are in [`paper_figures/`](paper_figures/). Newly generated plots and source tables are written to `plot_final/figure/`. [`manifest.json`](manifest.json) maps each paper figure to its generated filename and lists the input hashes. Notebook outputs are cleared in the repository.

To run one analysis, replace the final command with, for example, `python reproduce.py histograms`. Available jobs are:

| Paper figure | Content | Job | Notebook in `plot_final/` |
| --- | --- | --- | --- |
| 1, 2, 8–11 | Learning curves and posterior snapshots | `posterior` | `rabi_posterior_learning.ipynb` |
| 3 | Log Bayes factor histograms and mean checks | `histograms` | `rabi_log_bayes_histograms.ipynb` |
| 4, 5 | Detection power and sensitivity | `power` | `rabi_discovery_probability.ipynb` |
| 6, 7 | Type-II error and finite-window effective exponents | `exponents` | `rabi_beta_exponent.ipynb` |
| 12 | Readout axes for the higher fidelity detector | `posterior-high-fidelity` | `rabi_posterior_learning.ipynb` |
| 13, 14 | Polarization diagnostic and higher fidelity sensitivity | `power-high-fidelity` | `rabi_discovery_probability.ipynb` |
| 15 | Readout axes with nonzero detuning | `detuned` | `rabi_posterior_learning_detuned.ipynb` |

The runner sets the detector profile, cache selection, and trajectory seed where needed. Use these jobs to reproduce the paper configurations; running a notebook with its default settings alone selects only one of its configurations.

## Data and conventions

All data describe simulated experiments. Configuration values are retained in the NPZ metadata and the accompanying `*_setup.json` files. Hashes in cache filenames identify simulation configurations and are retained for compatibility with the original loaders.

The two ensemble caches in `plot_final/cache/` are:

| Filename suffix | Detector | Maximum shots | H0 records per policy | H1 records per amplitude and policy |
| --- | --- | --- | --- | --- |
| `63a56163f2d14a64.npz` | Baseline: contrast 0.99, readout error 0.005 | 4096 | 1000 | 1000 |
| `43fb9772170a2a1a.npz` | Higher fidelity: contrast 0.999, readout error 0.0005 | 16384 | 1000 | 1000 |

Both filenames begin with `rabi_prefix_sensitivity_vs_shots_final_`. Their principal arrays are:

- `raw_setup`: a scalar JSON string containing simulation settings and policy identifiers.
- `checkpoints`: retained shot counts. Records at different checkpoints share the same simulated history.
- `phase_grid`: injected signal **amplitudes** (the historical variable name does not denote the drive phase).
- `logB_h0__POLICY`: shape `(experiment, checkpoint)`.
- `logB_h1__POLICY`: shape `(amplitude, experiment, checkpoint)`.
- `true_drive_phases_h1__POLICY`: shape `(amplitude, experiment)`; the drive phase is drawn uniformly for each experiment.
- `predictable_kl_h0__POLICY` and `predictable_variance_h0__POLICY`: cumulative conditional KL and variance diagnostics at the checkpoints.
- `polarization_design_h0__POLICY` and `polarization_design_h1__POLICY`: cumulative transverse design components `(Q_xx, Q_xy, Q_yy)` in the last dimension.

Here `logB` is the natural logarithm of the alternative-to-null Bayes factor. The baseline cache labels pure information gain as `adaptive_l1`; `adaptive_l2` is the period-two hybrid. The higher fidelity cache uses `adaptive` for the period-eight information gain/Helstrom hybrid. Its `adaptive_stein` records provide the alternative hybrid diagnostic retained by the analysis. `two_axis` denotes fixed x/y readout.

For example:

```python
import json
import numpy as np

path = "plot_final/cache/rabi_prefix_sensitivity_vs_shots_final_63a56163f2d14a64.npz"
with np.load(path) as data:
    settings = json.loads(str(data["raw_setup"]))
    shots = data["checkpoints"]
    log_bayes_h0 = data["logB_h0__adaptive_l1"]
```

The four `rabi_log_bayes_histograms_final_h1_kl_*.npz` files contain separate, paired H1 records for Fig. 3 at 1024 shots and injected amplitude 0.19: `log_bayes_factors`, `signal_null_kl`, `signal_predictive_kl`, and the injected phases. The H1 mean prediction averages the difference of the two cumulative KL terms on the same histories. The H0 mean prediction uses the negative cumulative predictable KL from the baseline ensemble cache. These H1 histogram records are a separate ensemble from the H1 power-curve records.

The trajectory caches contain `records`, `snapshot_shots`, posterior probability arrays, and `conditional_stack` for the posterior snapshots. The structured `records` arrays include string-valued policy labels stored as NumPy object fields; the supplied trajectory loader therefore uses `allow_pickle=True`. Only load these fields from a trusted copy of the release. The accompanying setup JSON files specify the particle grids and trajectory seeds. These illustrative trajectories do not estimate detection power by themselves.

Derived CSV tables in `plot_final/figure/` retain calibrated thresholds, power estimates, sensitivity interpolations, fitted slopes, and mean diagnostics. The paper explains their statistical interpretation. In particular, the Fig. 7 error bars span the minimum and maximum secant slopes from the marginal 68% Jeffreys intervals at the two endpoints, with thresholds held fixed; they are not 95% confidence intervals for the slopes. The reported phase-averaged finite-window exponents are not worst-case composite Stein exponents.

## Simulation code and further calculations

`src/dm_search/` contains the Rabi likelihood, Bayesian updates, readout policies, accelerated samplers, cache configuration, and threshold calibration. The numerical modules are copied from the research implementation. The plotting notebooks have cleared outputs and added cache-miss checks for publication use.

`rabi_prefix_raw_data.ipynb` generates ensemble data with the current configuration builder. `rabi_prefix_sensitivity_vs_shots.ipynb` retains the earlier generator used for the higher fidelity ensemble. They are provided for further simulations and are **not** executed by `reproduce.py`. A full `final` simulation can take many hours; the original cache metadata report roughly 18 and 25 hours for the two ensembles on the original machine. Inspect the settings before running either generator. The saved configurations, policy names, seeds, batching, and software versions matter for reproducing a particular Monte Carlo realization.

The posterior and histogram notebooks also retain their simulation paths, which require explicitly setting `RABI_ALLOW_SIMULATION=1` when a selected cache is absent. The publication runner always uses the supplied records.

## Verification

```sh
python reproduce.py verify
python -m unittest discover -s tests
```

The release was checked by rebuilding all 15 figures from the included data and comparing the derived CSV tables with the research outputs. Thirteen figures matched the reference PDFs pixel for pixel at 100 dpi. Figs. 12 and 15 retain the same trajectories but have minor layout and stroke differences: their reference PDFs were exported with Matplotlib 3.11.0, whereas the tested environment uses 3.9.2. The original PDFs are included for comparison; exact PDF bytes are not expected to match because export metadata also change.

The tests check randomized threshold calibration, paired evidence/KL means, independent probability replay of short simulated histories, cache reuse, refusal of unintended resimulation, and corrupted-input detection. The production Monte Carlo ensembles were not regenerated for this release.

## Citation and licenses

Please cite the [paper](https://arxiv.org/abs/2609.16119) and identify the repository release or commit used. Machine-readable citation metadata are in [`CITATION.cff`](CITATION.cff).

Copyright © 2026 So Chigusa. Code and documentation are distributed under the [MIT License](LICENSE). Numerical data in `plot_final/cache/` and `plot_final/figure/`, figure PDFs in `paper_figures/`, and the data manifest are distributed under [Creative Commons Attribution 4.0 International](LICENSE-DATA). Attribute the data and figures to So Chigusa and the accompanying paper, link the license, and indicate modifications. This repository does not change the licenses of third-party Python dependencies or of the manuscript hosted on arXiv.
