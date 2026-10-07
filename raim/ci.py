"""Percentile interval over bootstrap resamples, in the project's convention.

The judging runners (`scripts/run_judge.py`, `scripts/run_ptjudge.py`, `scripts/run_minicheck.py`) each
report a cluster-bootstrap interval on the verdicts they have just written, so
the operator sees kappa and balanced accuracy with their uncertainty without
leaving the measurement box.

This is a five-line restatement of `bootstrap.py`'s `_ci`, which lives in the
`raim-analysis` repository: importing it here would make this repository depend
on the analysis one for one helper with no methodological content, so it is
duplicated instead. The two must be kept identical by hand.

Convention, frozen project-wide: `[point, lo, hi]` at the 95% percentile level,
the point estimate being the resample mean.
"""
from __future__ import annotations

import numpy as np


def ci(samples, lo=2.5, hi=97.5):
    """Return (mean, lo, hi) over `samples`, dropping nan; all-nan gives nans."""
    a = np.asarray([s for s in samples if s == s])  # drop nan
    if a.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    return float(a.mean()), float(np.percentile(a, lo)), float(np.percentile(a, hi))
