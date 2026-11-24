"""CPU scheduling algorithms.

Every algorithm is simulated one time unit (tick) at a time. This keeps
preemptive and non-preemptive policies in one consistent engine and lets us
record a snapshot of the system at every tick for the step-by-step view.

Implemented:
    FCFS, SJF (non-preemptive), SRTF (preemptive SJF),
    Priority (non-preemptive / preemptive, optional aging),
    Round Robin, Multilevel Feedback Queue (MLFQ),
    Smart Adaptive Scheduler (SAS) - the proposed algorithm.
"""
from __future__ import annotations

from collections import deque
from typing import Callable, Dict, List, Optional

from .models import Job, JobStats, ScheduleResult, Segment

IDLE = "IDLE"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _validate(jobs: List[Job]) -> None:
    if not jobs:
        raise ValueError("No jobs to schedule.")
    seen = set()
    for j in jobs:
        if j.pid in seen:
            raise ValueError(f"Duplicate PID: {j.pid}")
        seen.add(j.pid)
        if j.burst < 1:
            raise ValueError(f"{j.pid}: burst time must be at least 1.")
        if j.arrival < 0:
            raise ValueError(f"{j.pid}: arrival time cannot be negative.")


def _build_result(name: str, jobs: List[Job], ticks: List[dict],
                  first_start: Dict[str, int], completion: Dict[str, int]) -> ScheduleResult:
    # Merge consecutive ticks with the same running job into Gantt segments.
    segments: List[Segment] = []
    for tk in ticks:
        pid = tk["running"] or IDLE
        if segments and segments[-1].pid == pid and segments[-1].end == tk["time"]:
            segments[-1].end = tk["time"] + 1
        else:
            segments.append(Segment(pid, tk["time"], tk["time"] + 1))

    # Drop idle time before the very first arrival (CPU was not "in use" yet).
    while segments and segments[0].pid == IDLE:
        segments.pop(0)

    stats = []
    for j in jobs:
        ct = completion[j.pid]
        tat = ct - j.arrival
        stats.append(JobStats(
            pid=j.pid, arrival=j.arrival, burst=j.burst, priority=j.priority,
            start=first_start[j.pid], completion=ct, turnaround=tat,
            waiting=tat - j.burst, response=first_start[j.pid] - j.arrival,
        ))

    busy = [s for s in segments if s.pid != IDLE]
    context_switches = max(0, len(busy) - 1)
    return ScheduleResult(name, segments, stats, ticks, context_switches)


def _simulate(name: str, jobs: List[Job], key: Callable, preemptive: bool) -> ScheduleResult:
    """Generic engine for 'pick the best ready job by a key' policies.

    key(job, ctx) -> sortable tuple; the smallest key runs next.
    ctx exposes: t, remaining, executed, waited, current, order.
    For non-preemptive policies a choice is only made when the CPU is free.
    """
    _validate(jobs)
    order = {j.pid: i for i, j in enumerate(jobs)}
    remaining = {j.pid: j.burst for j in jobs}
    executed = {j.pid: 0 for j in jobs}
    waited = {j.pid: 0 for j in jobs}          # time spent in the ready queue so far
    first_start: Dict[str, int] = {}
    completion: Dict[str, int] = {}
    ticks: List[dict] = []

    t = 0
    current: Optional[Job] = None
    while len(completion) < len(jobs):
        ready = [j for j in jobs if j.arrival <= t and j.pid not in completion]
        ctx = {"t": t, "remaining": remaining, "executed": executed,
               "waited": waited, "current": current.pid if current else None,
               "order": order}

        if ready and (current is None or preemptive):
            current = min(ready, key=lambda j: key(j, ctx))

        if current is None:  # nothing has arrived yet -> CPU idle
            ticks.append({"time": t, "running": None, "ready": [],
                          "done": sorted(completion, key=completion.get)})
            t += 1
            continue

        first_start.setdefault(current.pid, t)
        waiting_now = [j for j in ready if j.pid != current.pid]
        waiting_now.sort(key=lambda j: key(j, ctx))
        ticks.append({"time": t, "running": current.pid,
                      "ready": [j.pid for j in waiting_now],
                      "done": sorted(completion, key=completion.get)})

        remaining[current.pid] -= 1
        executed[current.pid] += 1
        for j in waiting_now:
            waited[j.pid] += 1
        t += 1

        if remaining[current.pid] == 0:
            completion[current.pid] = t
            current = None

    return _build_result(name, jobs, ticks, first_start, completion)


def _known_burst(job: Job, use_predicted: bool) -> float:
    if use_predicted and job.predicted is not None:
        return job.predicted
    return job.burst


# --------------------------------------------------------------------------- #
# Classic algorithms
# --------------------------------------------------------------------------- #
def fcfs(jobs: List[Job]) -> ScheduleResult:
    """First Come First Serve - non-preemptive, ordered by arrival."""
    return _simulate("FCFS", jobs,
                     lambda j, c: (j.arrival, c["order"][j.pid]), preemptive=False)


def sjf(jobs: List[Job], use_predicted: bool = False) -> ScheduleResult:
    """Shortest Job First - non-preemptive.

    use_predicted=False: textbook version, the scheduler knows the real burst.
    use_predicted=True : realistic version, it only knows the predicted burst.
    """
    return _simulate("SJF", jobs,
                     lambda j, c: (_known_burst(j, use_predicted), j.arrival, c["order"][j.pid]),
                     preemptive=False)


def srtf(jobs: List[Job], use_predicted: bool = False) -> ScheduleResult:
    """Shortest Remaining Time First - preemptive SJF."""
    def key(j, c):
        rem = max(_known_burst(j, use_predicted) - c["executed"][j.pid], 0)
        is_current = 0 if c["current"] == j.pid else 1   # don't preempt on a tie
        return (rem, is_current, j.arrival, c["order"][j.pid])
    return _simulate("SRTF", jobs, key, preemptive=True)


def priority_scheduling(jobs: List[Job], preemptive: bool = False,
                        aging_interval: int = 0) -> ScheduleResult:
    """Priority scheduling (lower number = higher priority).

    aging_interval > 0 enables aging: a job's effective priority improves by 1
    for every `aging_interval` time units it has spent waiting, so low-priority
    jobs cannot starve forever.
    """
    def key(j, c):
        eff = j.priority
        if aging_interval > 0:
            eff -= c["waited"][j.pid] // aging_interval
        is_current = 0 if c["current"] == j.pid else 1
        return (eff, is_current, j.arrival, c["order"][j.pid])

    name = "Priority (P)" if preemptive else "Priority (NP)"
    if aging_interval > 0:
        name += " + Aging"
    return _simulate(name, jobs, key, preemptive=preemptive)


def round_robin(jobs: List[Job], quantum: int = 2) -> ScheduleResult:
    """Round Robin with a fixed time quantum.

    Convention: jobs arriving at time t join the queue *before* a job whose
    quantum expired at time t is put back at the tail.
    """
    _validate(jobs)
    if quantum < 1:
        raise ValueError("Time quantum must be at least 1.")
    pending = sorted(jobs, key=lambda j: (j.arrival, jobs.index(j)))
    remaining = {j.pid: j.burst for j in jobs}
    first_start: Dict[str, int] = {}
    completion: Dict[str, int] = {}
    ticks: List[dict] = []
    queue: deque = deque()
    t, pi, used = 0, 0, 0
    current: Optional[Job] = None

    while len(completion) < len(jobs):
        while pi < len(pending) and pending[pi].arrival <= t:
            queue.append(pending[pi])
            pi += 1
        if current is not None and used == quantum:
            queue.append(current)
            current = None
        if current is None and queue:
            current = queue.popleft()
            used = 0

        if current is None:
            ticks.append({"time": t, "running": None, "ready": [],
                          "done": sorted(completion, key=completion.get)})
            t += 1
            continue

        first_start.setdefault(current.pid, t)
        ticks.append({"time": t, "running": current.pid,
                      "ready": [j.pid for j in queue],
                      "done": sorted(completion, key=completion.get)})
        remaining[current.pid] -= 1
        used += 1
        t += 1
        if remaining[current.pid] == 0:
            completion[current.pid] = t
            current = None

    return _build_result(f"Round Robin (q={quantum})", jobs, ticks, first_start, completion)


def mlfq(jobs: List[Job], q0: int = 4, q1: int = 8) -> ScheduleResult:
    """Multilevel Feedback Queue with three levels.

    Q0: Round Robin, quantum q0  (new jobs enter here - highest priority)
    Q1: Round Robin, quantum q1
    Q2: FCFS                     (CPU-heavy jobs sink here)

    A job that uses its full quantum is demoted one level. A job running in a
    lower queue is preempted when a job arrives in a higher queue; it goes back
    to the head of its own queue.
    """
    _validate(jobs)
    quanta = [q0, q1, None]
    pending = sorted(jobs, key=lambda j: (j.arrival, jobs.index(j)))
    remaining = {j.pid: j.burst for j in jobs}
    level = {j.pid: 0 for j in jobs}
    first_start: Dict[str, int] = {}
    completion: Dict[str, int] = {}
    ticks: List[dict] = []
    queues = [deque(), deque(), deque()]
    t, pi, used = 0, 0, 0
    current: Optional[Job] = None

    while len(completion) < len(jobs):
        while pi < len(pending) and pending[pi].arrival <= t:
            queues[0].append(pending[pi])
            pi += 1

        if current is not None:
            lvl = level[current.pid]
            if quanta[lvl] is not None and used == quanta[lvl]:
                level[current.pid] = min(lvl + 1, 2)
                queues[level[current.pid]].append(current)
                current = None
            elif any(queues[h] for h in range(lvl)):
                queues[lvl].appendleft(current)
                current = None

        if current is None:
            for q in queues:
                if q:
                    current = q.popleft()
                    used = 0
                    break

        if current is None:
            ticks.append({"time": t, "running": None, "ready": [],
                          "done": sorted(completion, key=completion.get)})
            t += 1
            continue

        first_start.setdefault(current.pid, t)
        ticks.append({"time": t, "running": current.pid,
                      "ready": [f"{j.pid}(Q{i})" for i, q in enumerate(queues) for j in q],
                      "done": sorted(completion, key=completion.get)})
        remaining[current.pid] -= 1
        used += 1
        t += 1
        if remaining[current.pid] == 0:
            completion[current.pid] = t
            current = None

    return _build_result(f"MLFQ (q={q0},{q1},FCFS)", jobs, ticks, first_start, completion)


# --------------------------------------------------------------------------- #
# Proposed algorithm
# --------------------------------------------------------------------------- #
def smart_adaptive(jobs: List[Job], aging_factor: float = 0.5) -> ScheduleResult:
    """Smart Adaptive Scheduler (SAS) - the project's proposed algorithm.

    Real schedulers do not know burst times in advance, so SAS works only on
    *predicted* bursts (exponential averaging of each job's history):

        score(job) = predicted_remaining(job) - aging_factor * time_waited(job)

    The ready job with the lowest score runs (preemptive).
      * predicted_remaining keeps the SRTF benefit: short jobs finish first,
        giving a low average waiting time.
      * the aging term grows while a job waits, so long jobs cannot starve
        (lower maximum waiting time than SRTF).
      * aging_factor < 1 guarantees no thrashing: a running job's score falls
        by 1 per tick, a waiting job's by only aging_factor per tick, so a
        running job is only preempted by a new arrival.
    """
    if not 0 <= aging_factor < 1:
        raise ValueError("aging_factor must be in [0, 1).")

    def key(j, c):
        pred = j.predicted if j.predicted is not None else j.burst
        pred_rem = max(pred - c["executed"][j.pid], 0)
        score = pred_rem - aging_factor * c["waited"][j.pid]
        is_current = 0 if c["current"] == j.pid else 1
        return (round(score, 6), is_current, j.arrival, c["order"][j.pid])

    return _simulate("Smart Adaptive (SAS)", jobs, key, preemptive=True)


# --------------------------------------------------------------------------- #
# Registry used by the UI and the recommender
# --------------------------------------------------------------------------- #
ALGORITHMS = [
    "FCFS", "SJF", "SRTF", "Priority (NP)", "Priority (P)",
    "Round Robin", "MLFQ", "Smart Adaptive (SAS)",
]


def run_algorithm(name: str, jobs: List[Job], cfg: Optional[dict] = None) -> ScheduleResult:
    """Run an algorithm by name. cfg keys (all optional):
    quantum, mlfq_q0, mlfq_q1, aging_interval, use_predicted, sas_aging.
    """
    cfg = cfg or {}
    use_pred = cfg.get("use_predicted", False)
    aging = cfg.get("aging_interval", 0)
    if name == "FCFS":
        return fcfs(jobs)
    if name == "SJF":
        return sjf(jobs, use_pred)
    if name == "SRTF":
        return srtf(jobs, use_pred)
    if name == "Priority (NP)":
        return priority_scheduling(jobs, False, aging)
    if name == "Priority (P)":
        return priority_scheduling(jobs, True, aging)
    if name == "Round Robin":
        return round_robin(jobs, cfg.get("quantum", 2))
    if name == "MLFQ":
        return mlfq(jobs, cfg.get("mlfq_q0", 4), cfg.get("mlfq_q1", 8))
    if name == "Smart Adaptive (SAS)":
        return smart_adaptive(jobs, cfg.get("sas_aging", 0.5))
    raise ValueError(f"Unknown algorithm: {name}")
