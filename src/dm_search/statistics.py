from __future__ import annotations

import numpy as np


def null_threshold(null_statistics: np.ndarray, alpha: float) -> float:
    sorted_statistics = np.sort(np.asarray(null_statistics, dtype=float))
    index = int(np.ceil((1.0 - alpha) * len(sorted_statistics))) - 1
    index = min(max(index, 0), len(sorted_statistics) - 1)
    return float(sorted_statistics[index])
