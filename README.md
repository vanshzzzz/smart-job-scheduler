# Smart Job Scheduling System

An Operating Systems mini project: a CPU scheduling simulator that runs the classic
algorithms and adds three "smart" features: burst-time prediction, a proposed
**Smart Adaptive Scheduler (SAS)**, and automatic algorithm recommendation.

## Features

| Area | What it does |
|---|---|
| **Classic algorithms** | FCFS, SJF, SRTF, Priority (preemptive / non-preemptive, optional aging), Round Robin, MLFQ (3 levels) |
| **Gantt chart** | Interactive chart for every algorithm, with hover details |
| **Metrics** | Per-job completion / turnaround / waiting / response time, averages, CPU utilisation, throughput, context switches, max waiting (starvation) |
| **Comparison dashboard** | All algorithms on the same job set: a table with the best values highlighted, plus bar charts |
| **Burst prediction** | Exponential averaging τₙ₊₁ = α·tₙ + (1−α)·τₙ from each job's burst history |
| **Smart Adaptive Scheduler** | Proposed algorithm: runs the job with the lowest `predicted remaining − aging × time waited` |
| **Algorithm recommender** | Simulates 30 possible futures, scores the algorithms on your goal (batch / interactive / fair / custom), recommends one, then checks it against the actual bursts |
| **Step-by-step mode** | Time slider and Play button showing the running job, ready queue and completed jobs at every tick |
| **Workloads** | Editable job table, random generator, CSV upload/download, textbook presets |

## Run it

**Windows:** double-click `run.bat`.

**Any OS:**
```bash
pip install -r requirements.txt
streamlit run app.py
```
The app opens in your browser at http://localhost:8501.

Run the tests (checked against the Silberschatz textbook examples):
```bash
python -m pytest tests -q
```

## Project structure

```
smart_job_scheduler/
├── app.py                  Streamlit UI (6 tabs)
├── charts.py               Gantt chart + comparison bar charts (Plotly)
├── scheduler/
│   ├── models.py           Job, Segment, JobStats, ScheduleResult (+ metrics)
│   ├── algorithms.py       All scheduling algorithms (tick-based simulator)
│   ├── predictor.py        Exponential-averaging burst predictor
│   ├── smart.py            Workload analysis + Monte Carlo recommender
│   └── workload.py         Random generator, CSV <-> jobs
├── tests/test_algorithms.py
├── sample_jobs.csv
├── requirements.txt
└── run.bat
```

## CSV format

```
PID,Arrival,Burst,Priority,History
P1,0,8,3,7;9;8
P2,1,4,1,5;4;4
```
Lower priority number = higher priority. History holds the job's past CPU bursts (any of `;` `,` or space as the separator).

## How the smart part works

1. **Prediction.** A real OS does not know burst times in advance. Each job's next burst is
   predicted from its history by exponential averaging (α is set in the sidebar).
2. **Smart Adaptive Scheduler (SAS).** This preemptive policy uses only predictions:
   `score = predicted_remaining − aging_factor × time_waited`, and the lowest score runs.
   Short jobs go first, as in SRTF, which keeps average waiting time low. The aging term stops
   long jobs from starving. With `aging_factor < 1`, a running job's score always falls faster
   than any waiting job's, so it can only be preempted by a new arrival (no thrashing).
3. **Recommendation.** Every algorithm is simulated on 30 random "futures" drawn around the
   predictions, using each job's own historical spread. Metrics are averaged and scaled
   0 (best) to 1 (worst), then weighted by the chosen goal. The lowest score is recommended.
   The choice is then checked against the real bursts, which shows how good the
   recommendation actually was.

Demo tip: load the **Starvation demo** preset and compare SRTF with SAS under the
"Fair system" goal. SRTF makes the long job wait much longer, while SAS keeps the
average low and cuts the worst-case wait.
