"""Burst-time prediction by exponential averaging.

A real CPU scheduler cannot know how long a job's next CPU burst will be.
The standard OS technique is to predict it from the job's past bursts:

    tau(n+1) = alpha * t(n) + (1 - alpha) * tau(n)

    t(n)    = actual length of the n-th burst
    tau(n)  = previous prediction (tau(0) is an initial guess)
    alpha   = weight of recent history (0 = ignore history, 1 = only last burst)
"""
from __future__ import annotations

from typing import Iterable, List, Optional

from .models import Job


def exponential_average(history: Iterable[float], alpha: float = 0.5,
                        tau0: float = 10.0) -> float:
    tau = float(tau0)
    for burst in history:
        tau = alpha * burst + (1 - alpha) * tau
    return tau


def predict_bursts(jobs: List[Job], alpha: float = 0.5,
                   tau0: Optional[float] = None) -> List[Job]:
    """Fill job.predicted for every job (in place) and return the list.

    If tau0 is None, the initial guess is the mean of all known history,
    or 5 if no job has any history.
    """
    if tau0 is None:
        all_hist = [b for j in jobs for b in j.history]
        tau0 = sum(all_hist) / len(all_hist) if all_hist else 5.0
    for j in jobs:
        j.predicted = round(exponential_average(j.history, alpha, tau0), 2)
    return jobs


def prediction_error(jobs: List[Job]) -> dict:
    """Mean absolute error and mean absolute percentage error of predictions."""
    pairs = [(j.predicted, j.burst) for j in jobs if j.predicted is not None]
    if not pairs:
        return {"mae": 0.0, "mape": 0.0}
    mae = sum(abs(p - a) for p, a in pairs) / len(pairs)
    mape = 100 * sum(abs(p - a) / a for p, a in pairs) / len(pairs)
    return {"mae": round(mae, 2), "mape": round(mape, 1)}
