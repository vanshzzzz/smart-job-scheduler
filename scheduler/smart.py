"""Smart layer: workload analysis and automatic algorithm recommendation.

How the recommendation works:
  1. Predict each job's burst from its history (exponential averaging).
  2. PLAN (Monte Carlo): a real OS only knows the predictions, and each job's
     history shows how much its bursts vary. We simulate every algorithm on
     many plausible futures (bursts drawn around the prediction with the job's
     own spread), where SJF/SRTF/SAS decide using predictions only.
     Averaging over futures favours algorithms that stay good even when the
     predictions are wrong.
  3. Score each algorithm on the metrics the user cares about (weighted,
     min-max normalised, lower = better) and recommend the best one.
  4. VERIFY: run every algorithm on the *actual* bursts and check whether the
     recommendation was really the best (or how close it came).
"""
from __future__ import annotations

import copy
import random
import statistics
from typing import Dict, List

import pandas as pd

from .algorithms import ALGORITHMS, run_algorithm
from .models import Job, ScheduleResult
from .predictor import prediction_error

METRICS = {
    "Avg Waiting": lambda r: r.avg_waiting,
    "Avg Turnaround": lambda r: r.avg_turnaround,
    "Avg Response": lambda r: r.avg_response,
    "Max Waiting": lambda r: r.max_waiting,
    "Context Switches": lambda r: r.context_switches,
}

PROFILES = {
    "Balanced": {"Avg Waiting": 3, "Avg Turnaround": 2, "Avg Response": 2,
                 "Max Waiting": 2, "Context Switches": 1},
    "Batch system (fast completion)": {"Avg Waiting": 4, "Avg Turnaround": 5, "Avg Response": 0,
                                       "Max Waiting": 1, "Context Switches": 2},
    "Interactive system (quick response)": {"Avg Waiting": 1, "Avg Turnaround": 1, "Avg Response": 5,
                                            "Max Waiting": 2, "Context Switches": 0},
    "Fair system (no starvation)": {"Avg Waiting": 2, "Avg Turnaround": 1, "Avg Response": 1,
                                    "Max Waiting": 5, "Context Switches": 0},
}


def analyze_workload(jobs: List[Job]) -> Dict[str, float]:
    bursts = [j.burst for j in jobs]
    arrivals = [j.arrival for j in jobs]
    mean_b = statistics.mean(bursts)
    cv = statistics.pstdev(bursts) / mean_b if mean_b else 0.0
    span = max(arrivals) - min(arrivals)
    load = sum(bursts) / (span + 1)
    err = prediction_error(jobs)
    return {
        "jobs": len(jobs),
        "mean_burst": round(mean_b, 2),
        "burst_cv": round(cv, 2),
        "arrival_span": span,
        "load": round(load, 2),
        "priority_levels": len({j.priority for j in jobs}),
        "pred_mae": err["mae"],
        "pred_mape": err["mape"],
    }


def metrics_table(results: Dict[str, ScheduleResult]) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        row = {"Algorithm": name}
        row.update({m: round(f(r), 2) for m, f in METRICS.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def score_table(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    """Weighted, min-max normalised score per algorithm (0 = best on every metric)."""
    df = df.copy()
    total_w = sum(weights.values()) or 1
    score = pd.Series(0.0, index=df.index)
    for m, w in weights.items():
        col = df[m].astype(float)
        lo, hi = col.min(), col.max()
        norm = (col - lo) / (hi - lo) if hi > lo else col * 0
        score += w * norm
    df["Score"] = (score / total_w).round(3)
    return df.sort_values(["Score", "Avg Waiting"]).reset_index(drop=True)


def score_results(results: Dict[str, ScheduleResult], weights: Dict[str, float]) -> pd.DataFrame:
    return score_table(metrics_table(results), weights)


def run_all(jobs: List[Job], cfg: dict) -> Dict[str, ScheduleResult]:
    return {name: run_algorithm(name, jobs, cfg) for name in ALGORITHMS}


def _burst_spread(job: Job) -> float:
    """Typical deviation of this job's bursts, from its own history."""
    if len(job.history) >= 2:
        return statistics.pstdev(job.history)
    base = job.predicted if job.predicted is not None else job.burst
    return 0.25 * base   # no history: assume +/-25 %


def plan(jobs: List[Job], cfg: dict, weights: Dict[str, float],
         scenarios: int = 30, seed: int = 7) -> pd.DataFrame:
    """Monte Carlo planning on predicted bursts. Returns the ranked table
    (metrics averaged over all simulated futures)."""
    rng = random.Random(seed)
    sim_cfg = dict(cfg, use_predicted=True)
    tables = []
    for _ in range(scenarios):
        future = copy.deepcopy(jobs)
        for j in future:
            mu = j.predicted if j.predicted is not None else j.burst
            j.burst = max(1, round(rng.gauss(mu, _burst_spread(j))))
        tables.append(metrics_table(run_all(future, sim_cfg)))
    avg = pd.concat(tables).groupby("Algorithm", sort=False).mean().round(2).reset_index()
    return score_table(avg, weights)


def recommend(jobs: List[Job], cfg: dict, weights: Dict[str, float],
              scenarios: int = 30) -> dict:
    """Plan on predictions, recommend, then verify on actual bursts.
    Jobs must already have .predicted filled."""
    plan_rank = plan(jobs, cfg, weights, scenarios)
    chosen = plan_rank.iloc[0]["Algorithm"]

    # Real world: actual bursts, SJF/SRTF/SAS can still only see predictions.
    actual = run_all(jobs, dict(cfg, use_predicted=True))
    actual_rank = score_results(actual, weights)
    actual_best = actual_rank.iloc[0]["Algorithm"]
    chosen_pos = int(actual_rank.index[actual_rank["Algorithm"] == chosen][0]) + 1
    wl = analyze_workload(jobs)

    return {
        "chosen": chosen,
        "plan_rank": plan_rank,
        "actual_rank": actual_rank,
        "actual_results": actual,
        "actual_best": actual_best,
        "chosen_actual_position": chosen_pos,
        "workload": wl,
        "scenarios": scenarios,
        "reasons": explain(chosen, wl, weights, actual),
    }


def explain(chosen: str, wl: dict, weights: Dict[str, float],
            actual: Dict[str, ScheduleResult]) -> List[str]:
    """Human-readable reasons behind the recommendation (for the UI / viva)."""
    reasons = []
    top = max(weights, key=weights.get)
    reasons.append(f"Your top priority is **{top}**, so it carries the most weight in the score.")

    if wl["burst_cv"] >= 0.6:
        reasons.append(f"Burst times vary a lot (CV = {wl['burst_cv']}). Running short jobs first "
                       "(SJF / SRTF / SAS) cuts average waiting time sharply.")
    elif wl["burst_cv"] <= 0.3:
        reasons.append(f"Jobs are similar in length (CV = {wl['burst_cv']}), so FCFS and Round Robin "
                       "perform close to SJF here.")
    else:
        reasons.append(f"Burst times vary moderately (CV = {wl['burst_cv']}).")

    if wl["load"] > 1.5:
        reasons.append(f"The CPU is heavily loaded ({wl['load']}x more work than arrival time span), "
                       "so jobs queue up and the order really matters.")

    reasons.append(f"Burst prediction error: MAE {wl['pred_mae']} time units "
                   f"({wl['pred_mape']}% average). The plan is only as good as these predictions.")

    srtf_r, sas_r = actual.get("SRTF"), actual.get("Smart Adaptive (SAS)")
    if srtf_r and sas_r and sas_r.max_waiting < srtf_r.max_waiting:
        reasons.append(f"Aging in SAS lowered the worst-case wait from {srtf_r.max_waiting} "
                       f"(SRTF) to {sas_r.max_waiting}: long jobs are not starved.")

    if chosen.startswith("Round Robin") or chosen.startswith("MLFQ"):
        reasons.append("Time-slicing gives every job the CPU quickly, which favours response time.")
    if chosen.startswith("Priority"):
        reasons.append("Job priorities differ enough that honouring them pays off.")
    return reasons
