"""Rebuild the paper plots from the included Monte Carlo records."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
JOBS = {
    "posterior": ("rabi_posterior_learning", {}),
    "posterior-high-fidelity": ("rabi_posterior_learning", {
        "RABI_LEARNING_DETECTOR_PROFILE": "high_fidelity",
        "RABI_LEARNING_SEED_BACKGROUND": "7096",
    }),
    "detuned": ("rabi_posterior_learning_detuned", {}),
    "histograms": ("rabi_log_bayes_histograms", {
        "RABI_HIST_WARMUP_NUMBA": "0",
        "RABI_HIST_PREFIX_RAW_CACHE": "plot_final/cache/rabi_prefix_sensitivity_vs_shots_final_63a56163f2d14a64.npz",
    }),
    "power": ("rabi_discovery_probability", {}),
    "power-high-fidelity": ("rabi_discovery_probability", {
        "RABI_DISCOVERY_DETECTOR_PROFILE": "high_fidelity",
    }),
    "exponents": ("rabi_beta_exponent", {}),
}


def verify_inputs():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for item in manifest["inputs"]:
        path = ROOT / item["file"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Missing or altered publication input: {path}")
    print(f"Verified {len(manifest['inputs'])} input files.", flush=True)


def run_job(name):
    # Do not inherit unrelated notebook settings from an interactive session.
    for key in list(os.environ):
        if key.startswith("RABI_"):
            del os.environ[key]
    notebook, overrides = JOBS[name]
    os.environ.update(overrides)
    os.environ["RABI_ALLOW_SIMULATION"] = "0"
    os.environ["MPLBACKEND"] = "Agg"
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime" / "matplotlib"))
    os.environ.setdefault("NUMBA_NUM_THREADS", "4")
    from IPython.display import display
    import matplotlib.pyplot as plt

    os.chdir(ROOT)
    path = ROOT / "plot_final" / (notebook + ".ipynb")
    document = json.loads(path.read_text())
    namespace = {"__name__": "__main__", "display": display}
    for index, cell in enumerate(document["cells"]):
        if cell["cell_type"] == "code":
            print(f"{name}: cell {index}", flush=True)
            exec(compile("".join(cell["source"]), f"{path.name}:cell{index}", "exec"), namespace)
            plt.close("all")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", nargs="?", default="all", choices=["all", "verify", *JOBS])
    args = parser.parse_args()
    verify_inputs()
    if args.job == "all":
        for name in JOBS:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), name], cwd=ROOT, check=True)
    elif args.job != "verify":
        run_job(args.job)


if __name__ == "__main__":
    main()
