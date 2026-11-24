"""Workload generation and CSV / DataFrame conversion."""
from __future__ import annotations

import random
import re
from typing import List

import pandas as pd

from .models import Job

COLUMNS = ["PID", "Arrival", "Burst", "Priority", "History"]


def generate_workload(n: int = 6, max_arrival: int = 10, burst_min: int = 1,
                      burst_max: int = 12, priority_max: int = 5,
                      history_len: int = 4, noise: float = 0.25,
                      seed: int | None = None) -> List[Job]:
    """Random job set. Each job has a 'typical' burst; its past bursts and its
    actual next burst vary around it, so predictions are close but not exact,
    as in a real system."""
    rng = random.Random(seed)
    jobs = []
    for i in range(n):
        typical = rng.randint(burst_min, burst_max)

        def vary():
            return max(1, round(typical * (1 + rng.gauss(0, noise))))

        jobs.append(Job(
            pid=f"P{i + 1}",
            arrival=0 if i == 0 else rng.randint(0, max_arrival),
            burst=vary(),
            priority=rng.randint(1, priority_max),
            history=[vary() for _ in range(history_len)],
        ))
    jobs.sort(key=lambda j: (j.arrival, int(j.pid[1:])))
    return jobs


def parse_history(value) -> List[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    parts = re.split(r"[;,\s|]+", str(value).strip())
    return [int(float(p)) for p in parts if p]


def jobs_to_df(jobs: List[Job]) -> pd.DataFrame:
    return pd.DataFrame([{
        "PID": j.pid, "Arrival": j.arrival, "Burst": j.burst,
        "Priority": j.priority, "History": ";".join(map(str, j.history)),
    } for j in jobs], columns=COLUMNS)


def df_to_jobs(df: pd.DataFrame) -> List[Job]:
    """Convert an edited/uploaded table to Job objects, with clear errors."""
    cols = {c.lower().strip(): c for c in df.columns}
    for need in ("pid", "arrival", "burst"):
        if need not in cols:
            raise ValueError(f"Missing column '{need}'. Expected: {', '.join(COLUMNS)}")
    jobs = []
    for idx, row in df.iterrows():
        pid = row[cols["pid"]]
        if pid is None or (isinstance(pid, float) and pd.isna(pid)) or str(pid).strip() == "":
            continue  # skip blank rows from the editor
        try:
            arrival = int(row[cols["arrival"]])
            burst = int(row[cols["burst"]])
            priority = int(row[cols["priority"]]) if "priority" in cols and not pd.isna(row[cols["priority"]]) else 0
            history = parse_history(row[cols["history"]]) if "history" in cols else []
        except (TypeError, ValueError):
            raise ValueError(f"Row {idx + 1} ({pid}): Arrival, Burst and Priority must be whole numbers.")
        jobs.append(Job(str(pid).strip(), arrival, burst, priority, history))
    return jobs
