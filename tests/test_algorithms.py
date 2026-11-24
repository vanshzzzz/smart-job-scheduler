"""Checks against standard textbook examples (Silberschatz, Operating System Concepts)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scheduler.algorithms import (fcfs, mlfq, priority_scheduling, round_robin,  # noqa: E402
                                  sjf, smart_adaptive, srtf)
from scheduler.models import Job  # noqa: E402
from scheduler.predictor import exponential_average, predict_bursts  # noqa: E402


def three_jobs():
    return [Job("P1", 0, 24), Job("P2", 0, 3), Job("P3", 0, 3)]


def test_fcfs_textbook():
    r = fcfs(three_jobs())
    assert [s.pid for s in r.segments] == ["P1", "P2", "P3"]
    assert r.avg_waiting == pytest.approx(17.0)


def test_sjf_textbook():
    jobs = [Job("P1", 0, 6), Job("P2", 0, 8), Job("P3", 0, 7), Job("P4", 0, 3)]
    r = sjf(jobs)
    assert [s.pid for s in r.segments] == ["P4", "P1", "P3", "P2"]
    assert r.avg_waiting == pytest.approx(7.0)


def test_srtf_textbook():
    jobs = [Job("P1", 0, 8), Job("P2", 1, 4), Job("P3", 2, 9), Job("P4", 3, 5)]
    r = srtf(jobs)
    assert [(s.pid, s.start, s.end) for s in r.segments] == [
        ("P1", 0, 1), ("P2", 1, 5), ("P4", 5, 10), ("P1", 10, 17), ("P3", 17, 26)]
    assert r.avg_waiting == pytest.approx(6.5)


def test_priority_np_textbook():
    jobs = [Job("P1", 0, 10, 3), Job("P2", 0, 1, 1), Job("P3", 0, 2, 4),
            Job("P4", 0, 1, 5), Job("P5", 0, 5, 2)]
    r = priority_scheduling(jobs)
    assert [s.pid for s in r.segments] == ["P2", "P5", "P1", "P3", "P4"]
    assert r.avg_waiting == pytest.approx(8.2)


def test_round_robin_textbook():
    r = round_robin(three_jobs(), quantum=4)
    assert [(s.pid, s.start, s.end) for s in r.segments] == [
        ("P1", 0, 4), ("P2", 4, 7), ("P3", 7, 10), ("P1", 10, 30)]
    assert r.avg_waiting == pytest.approx(17 / 3)


def test_round_robin_arrival_before_requeue():
    # P2 arrives at t=2 exactly when P1's quantum ends -> P2 runs before P1 again.
    jobs = [Job("P1", 0, 4), Job("P2", 2, 2)]
    r = round_robin(jobs, quantum=2)
    assert [s.pid for s in r.segments] == ["P1", "P2", "P1"]


def test_idle_gap_and_metrics():
    jobs = [Job("A", 0, 2), Job("B", 5, 3)]
    r = fcfs(jobs)
    assert [(s.pid, s.start, s.end) for s in r.segments] == [
        ("A", 0, 2), ("IDLE", 2, 5), ("B", 5, 8)]
    assert r.cpu_utilization == pytest.approx(5 / 8 * 100)
    assert r.context_switches == 1


def test_priority_aging_prevents_starvation():
    jobs = [Job("LOW", 0, 3, 9)] + [Job(f"H{i}", i * 2, 2, 1) for i in range(10)]
    plain = priority_scheduling(jobs, preemptive=True)
    aged = priority_scheduling(jobs, preemptive=True, aging_interval=2)
    low_plain = next(s for s in plain.stats if s.pid == "LOW")
    low_aged = next(s for s in aged.stats if s.pid == "LOW")
    assert low_aged.waiting < low_plain.waiting


def test_mlfq_demotes_long_job():
    jobs = [Job("LONG", 0, 20), Job("SHORT", 5, 2)]
    r = mlfq(jobs, 4, 8)
    # SHORT arrives in Q0 while LONG is in Q1 -> SHORT preempts LONG.
    short = next(s for s in r.stats if s.pid == "SHORT")
    assert short.waiting == 0
    assert sum(s.burst for s in r.stats) == r.busy_time


def test_exponential_average():
    # Silberschatz example: alpha=0.5, tau0=10, bursts 6,4,6,4,13,13,13
    # predictions go 10 -> 8 -> 6 -> 6 -> 5 -> 9 -> 11 -> 12
    assert exponential_average([6, 4, 6, 4], 0.5, 10) == pytest.approx(5.0)
    assert exponential_average([6, 4, 6, 4, 13, 13, 13], 0.5, 10) == pytest.approx(12.0)


def test_sas_matches_srtf_with_perfect_prediction_and_no_aging():
    jobs = [Job("P1", 0, 8), Job("P2", 1, 4), Job("P3", 2, 9), Job("P4", 3, 5)]
    for j in jobs:
        j.predicted = j.burst
    assert smart_adaptive(jobs, 0.0).avg_waiting == pytest.approx(srtf(jobs).avg_waiting)


def test_sas_reduces_max_wait_vs_srtf():
    # A long job keeps getting overtaken by a stream of short jobs under SRTF.
    jobs = [Job("BIG", 0, 10)] + [Job(f"s{i}", i * 2 + 1, 2) for i in range(12)]
    for j in jobs:
        j.predicted = j.burst
    assert smart_adaptive(jobs, 0.5).max_waiting < srtf(jobs).max_waiting


def test_predict_bursts_sets_predicted():
    jobs = [Job("A", 0, 5, history=[4, 6]), Job("B", 0, 3)]
    predict_bursts(jobs, alpha=0.5, tau0=8)
    assert jobs[0].predicted == pytest.approx(0.5 * 6 + 0.5 * (0.5 * 4 + 0.5 * 8))
    assert jobs[1].predicted == pytest.approx(8)


def test_all_work_conserved():
    jobs = [Job("P1", 0, 5, 2), Job("P2", 1, 3, 1), Job("P3", 2, 8, 3), Job("P4", 9, 2, 1)]
    for j in jobs:
        j.predicted = j.burst + 1
    for r in [fcfs(jobs), sjf(jobs), srtf(jobs, True), priority_scheduling(jobs, True, 3),
              round_robin(jobs, 3), mlfq(jobs), smart_adaptive(jobs)]:
        assert r.busy_time == sum(j.burst for j in jobs)
        assert all(s.waiting >= 0 and s.response >= 0 for s in r.stats)
