"""Core data models for the Smart Job Scheduling System."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Job:
    """A job (process) submitted to the CPU scheduler.

    priority: lower number = higher priority (textbook convention).
    history:  CPU bursts this job used in earlier runs, oldest first.
              Used by the predictor to estimate the next burst.
    predicted: estimated burst time (filled by scheduler.predictor).
    """
    pid: str
    arrival: int
    burst: int
    priority: int = 0
    history: List[int] = field(default_factory=list)
    predicted: Optional[float] = None


@dataclass
class Segment:
    """One continuous block on the Gantt chart. pid == 'IDLE' means CPU idle."""
    pid: str
    start: int
    end: int

    @property
    def duration(self) -> int:
        return self.end - self.start


@dataclass
class JobStats:
    pid: str
    arrival: int
    burst: int
    priority: int
    start: int          # first time the job got the CPU
    completion: int
    turnaround: int     # completion - arrival
    waiting: int        # turnaround - burst
    response: int       # start - arrival


@dataclass
class ScheduleResult:
    algorithm: str
    segments: List[Segment]
    stats: List[JobStats]
    timeline: List[dict]          # one snapshot per time unit (for step-by-step view)
    context_switches: int

    # ---- aggregate metrics -------------------------------------------------
    def _avg(self, attr: str) -> float:
        return sum(getattr(s, attr) for s in self.stats) / len(self.stats) if self.stats else 0.0

    @property
    def avg_waiting(self) -> float:
        return self._avg("waiting")

    @property
    def avg_turnaround(self) -> float:
        return self._avg("turnaround")

    @property
    def avg_response(self) -> float:
        return self._avg("response")

    @property
    def max_waiting(self) -> int:
        """Longest any single job waited - a starvation indicator."""
        return max((s.waiting for s in self.stats), default=0)

    @property
    def makespan(self) -> int:
        return self.segments[-1].end if self.segments else 0

    @property
    def start_time(self) -> int:
        return self.segments[0].start if self.segments else 0

    @property
    def busy_time(self) -> int:
        return sum(s.duration for s in self.segments if s.pid != "IDLE")

    @property
    def cpu_utilization(self) -> float:
        span = self.makespan - self.start_time
        return 100.0 * self.busy_time / span if span else 0.0

    @property
    def throughput(self) -> float:
        """Jobs completed per time unit."""
        span = self.makespan - self.start_time
        return len(self.stats) / span if span else 0.0

    def summary(self) -> dict:
        return {
            "Algorithm": self.algorithm,
            "Avg Waiting": round(self.avg_waiting, 2),
            "Avg Turnaround": round(self.avg_turnaround, 2),
            "Avg Response": round(self.avg_response, 2),
            "Max Waiting": self.max_waiting,
            "Context Switches": self.context_switches,
            "CPU Util %": round(self.cpu_utilization, 1),
            "Throughput": round(self.throughput, 3),
        }
