"""Smart Job Scheduling System - Streamlit app.

Run:  streamlit run app.py
"""
from __future__ import annotations

import copy
import json
import time

import pandas as pd
import streamlit as st

from charts import gantt_chart, metric_bar
from scheduler.algorithms import ALGORITHMS, run_algorithm
from scheduler.models import Job
from scheduler.predictor import predict_bursts, prediction_error
from scheduler.smart import METRICS, PROFILES, recommend, score_results
from scheduler.workload import df_to_jobs, generate_workload, jobs_to_df

st.set_page_config(page_title="Smart Job Scheduling System", page_icon="⚙️", layout="wide")

# --------------------------------------------------------------------------- #
# Preset workloads (handy for the viva: results match the textbook)
# --------------------------------------------------------------------------- #
PRESETS = {
    "Textbook: FCFS convoy effect": [
        Job("P1", 0, 24, 2, [22, 25, 24]), Job("P2", 0, 3, 1, [3, 4, 3]),
        Job("P3", 0, 3, 3, [2, 3, 3])],
    "Textbook: SRTF example": [
        Job("P1", 0, 8, 3, [7, 9, 8]), Job("P2", 1, 4, 1, [5, 4, 4]),
        Job("P3", 2, 9, 4, [8, 10, 9]), Job("P4", 3, 5, 2, [4, 5, 6])],
    "Textbook: Priority example": [
        Job("P1", 0, 10, 3, [9, 11]), Job("P2", 0, 1, 1, [1, 2]), Job("P3", 0, 2, 4, [2, 2]),
        Job("P4", 0, 1, 5, [1, 1]), Job("P5", 0, 5, 2, [6, 5])],
    "Starvation demo (long job vs stream of short jobs)": (
        [Job("BIG", 0, 10, 5, [10, 9, 11])] +
        [Job(f"S{i + 1}", 2 * i + 1, 2, 1, [2, 3, 2]) for i in range(8)]),
}

# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
if "jobs_df" not in st.session_state:
    st.session_state.jobs_df = jobs_to_df(generate_workload(6, seed=42))
    st.session_state.editor_version = 0


def set_jobs(jobs):
    st.session_state.jobs_df = jobs_to_df(jobs)
    st.session_state.editor_version += 1   # forces the table editor to reload


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Workload")
    with st.expander("🎲 Random job set", expanded=False):
        n = st.slider("Number of jobs", 3, 15, 6)
        max_arr = st.slider("Max arrival time", 0, 30, 10)
        bmin, bmax = st.slider("Burst time range", 1, 30, (1, 12))
        pmax = st.slider("Priority levels", 1, 10, 5)
        noise = st.slider("Burst variability (history noise)", 0.0, 0.8, 0.25, 0.05,
                          help="How much a job's bursts vary run to run. Higher = harder to predict.")
        seed = st.number_input("Seed (same seed = same jobs)", 0, 99999, 42)
        if st.button("Generate", width="stretch"):
            set_jobs(generate_workload(n, max_arr, bmin, bmax, pmax, 4, noise, int(seed)))

    preset = st.selectbox("Load a preset", ["—"] + list(PRESETS))
    if st.button("Load preset", width="stretch", disabled=preset == "—"):
        set_jobs(copy.deepcopy(PRESETS[preset]))

    up = st.file_uploader("Upload CSV", type="csv",
                          help="Columns: PID, Arrival, Burst, Priority, History (e.g. 4;5;6)")
    if up is not None and st.session_state.get("last_upload") != up.file_id:
        try:
            set_jobs(df_to_jobs(pd.read_csv(up)))
            st.session_state.last_upload = up.file_id
        except Exception as e:  # show the problem instead of crashing
            st.error(f"Could not read CSV: {e}")

    st.header("🔧 Algorithm settings")
    quantum = st.number_input("Round Robin time quantum", 1, 20, 2)
    c1, c2 = st.columns(2)
    q0 = c1.number_input("MLFQ Q0 quantum", 1, 20, 4)
    q1 = c2.number_input("MLFQ Q1 quantum", 1, 40, 8)
    aging = st.number_input("Priority aging interval (0 = off)", 0, 50, 0,
                            help="Every N time units of waiting improves a job's priority by 1.")
    knowledge = st.radio("SJF / SRTF know…", ["Actual burst (textbook)", "Predicted burst (realistic)"],
                         help="Textbook SJF assumes burst times are known. A real OS only has predictions.")

    st.header("🧠 Smart settings")
    alpha = st.slider("Prediction α (weight of recent bursts)", 0.0, 1.0, 0.5, 0.05)
    tau_auto = st.checkbox("Initial guess τ₀ = mean of all history", value=True)
    tau0 = None if tau_auto else st.number_input("τ₀", 1.0, 100.0, 10.0)
    sas_aging = st.slider("SAS aging factor", 0.0, 0.9, 0.5, 0.05,
                          help="0 = pure predicted-SRTF. Higher = long waiting jobs gain priority faster.")

cfg = {
    "quantum": int(quantum), "mlfq_q0": int(q0), "mlfq_q1": int(q1),
    "aging_interval": int(aging), "sas_aging": float(sas_aging),
    "use_predicted": knowledge.startswith("Predicted"),
}

# --------------------------------------------------------------------------- #
# Header + tabs
# --------------------------------------------------------------------------- #
st.title("⚙️ Smart Job Scheduling System")
st.caption("CPU scheduling simulator with burst-time prediction, a proposed Smart Adaptive "
           "Scheduler (SAS) and automatic algorithm recommendation.")

tab_jobs, tab_sim, tab_cmp, tab_smart, tab_step, tab_about = st.tabs(
    ["📋 Jobs", "▶️ Simulate", "📊 Compare", "🧠 Smart Scheduler", "⏱️ Step-by-step", "📘 How it works"])


def stats_df(result):
    df = pd.DataFrame([{
        "PID": s.pid, "Arrival": s.arrival, "Burst": s.burst, "Priority": s.priority,
        "Start": s.start, "Completion": s.completion, "Turnaround": s.turnaround,
        "Waiting": s.waiting, "Response": s.response} for s in result.stats])
    avg = {"PID": "Average"}
    for c in ("Turnaround", "Waiting", "Response"):
        avg[c] = round(df[c].mean(), 2)
    return pd.concat([df, pd.DataFrame([avg])], ignore_index=True)


def show_metrics(result):
    c = st.columns(6)
    c[0].metric("Avg waiting", f"{result.avg_waiting:.2f}")
    c[1].metric("Avg turnaround", f"{result.avg_turnaround:.2f}")
    c[2].metric("Avg response", f"{result.avg_response:.2f}")
    c[3].metric("CPU utilisation", f"{result.cpu_utilization:.1f}%")
    c[4].metric("Throughput (jobs/unit)", f"{result.throughput:.3f}")
    c[5].metric("Context switches", result.context_switches)


# ---- Jobs ------------------------------------------------------------------
with tab_jobs:
    st.subheader("Job queue")
    st.write("Edit cells directly, add rows at the bottom, or use the sidebar to generate / load jobs. "
             "**Priority:** lower number = higher priority. **History:** the job's past CPU bursts, "
             "used to predict its next burst.")
    edited = st.data_editor(
        st.session_state.jobs_df, num_rows="dynamic", width="stretch",
        key=f"editor_{st.session_state.editor_version}",
        column_config={
            "PID": st.column_config.TextColumn(required=True),
            "Arrival": st.column_config.NumberColumn(min_value=0, step=1, required=True),
            "Burst": st.column_config.NumberColumn(min_value=1, step=1, required=True),
            "Priority": st.column_config.NumberColumn(min_value=0, step=1, default=1),
            "History": st.column_config.TextColumn(help="Past bursts separated by ; e.g. 4;6;5"),
        })
    try:
        jobs = df_to_jobs(edited)
        if not jobs:
            raise ValueError("Add at least one job.")
        pids = [j.pid for j in jobs]
        if len(set(pids)) != len(pids):
            raise ValueError("PIDs must be unique.")
        if any(j.burst < 1 for j in jobs):
            raise ValueError("Burst time must be at least 1.")
        if any(j.arrival < 0 for j in jobs):
            raise ValueError("Arrival time cannot be negative.")
    except ValueError as e:
        st.error(str(e))
        st.stop()

    predict_bursts(jobs, alpha, tau0)
    pids = [j.pid for j in jobs]
    err = prediction_error(jobs)

    st.subheader("Predicted burst times (exponential averaging)")
    pred_df = pd.DataFrame([{
        "PID": j.pid, "History": ", ".join(map(str, j.history)) or "—",
        "Predicted": j.predicted, "Actual": j.burst,
        "Error": round(j.predicted - j.burst, 2)} for j in jobs])
    st.dataframe(pred_df, width="stretch", hide_index=True)
    st.caption(f"τₙ₊₁ = α·tₙ + (1−α)·τₙ with α = {alpha}.  Mean absolute error = {err['mae']} "
               f"time units ({err['mape']}%).")
    st.download_button("⬇️ Download jobs CSV", jobs_to_df(jobs).to_csv(index=False),
                       "jobs.csv", "text/csv")

# ---- Simulate --------------------------------------------------------------
with tab_sim:
    algo = st.selectbox("Algorithm", ALGORITHMS, key="sim_algo")
    res = run_algorithm(algo, jobs, cfg)
    st.subheader(f"Gantt chart — {res.algorithm}")
    st.plotly_chart(gantt_chart(res.segments, pids), width="stretch", key="sim_gantt")
    show_metrics(res)
    st.subheader("Per-job results")
    st.dataframe(stats_df(res), width="stretch", hide_index=True)
    st.caption("Turnaround = Completion − Arrival · Waiting = Turnaround − Burst · "
               "Response = First start − Arrival")
    st.download_button("⬇️ Download results CSV", stats_df(res).to_csv(index=False),
                       f"results_{algo.replace(' ', '_')}.csv", "text/csv")

# ---- Compare ---------------------------------------------------------------
with tab_cmp:
    st.subheader("All algorithms on the same job set")
    results = {name: run_algorithm(name, jobs, cfg) for name in ALGORITHMS}
    summary = pd.DataFrame([r.summary() for r in results.values()])
    lower_better = ["Avg Waiting", "Avg Turnaround", "Avg Response", "Max Waiting", "Context Switches"]
    st.dataframe(
        summary.style.highlight_min(subset=lower_better, color="#cde2fb")
               .highlight_max(subset=["CPU Util %", "Throughput"], color="#cde2fb")
               .format(precision=2),
        width="stretch", hide_index=True)
    st.caption("Shaded cell = best value in that column.")

    names = list(results)
    grid = st.columns(2)
    for i, metric in enumerate(["Avg Waiting", "Avg Turnaround", "Avg Response", "Max Waiting"]):
        vals = [round(METRICS[metric](r), 2) for r in results.values()]
        best = names[vals.index(min(vals))]
        grid[i % 2].plotly_chart(metric_bar(names, vals, metric, best),
                                 width="stretch", key=f"cmp_{metric}")

    with st.expander("Gantt charts for every algorithm"):
        for name, r in results.items():
            st.markdown(f"**{r.algorithm}** — avg waiting {r.avg_waiting:.2f}")
            st.plotly_chart(gantt_chart(r.segments, pids, height=150),
                            width="stretch", key=f"g_{name}")

# ---- Smart scheduler -------------------------------------------------------
with tab_smart:
    st.subheader("🧠 Which algorithm suits this workload?")
    st.write("The system predicts each job's burst, simulates every algorithm on many possible "
             "futures (predictions ± the job's own variability), scores them on your goals, "
             "and recommends the best. It then checks the choice against the actual bursts.")

    profile = st.selectbox("System type / goal", list(PROFILES) + ["Custom"], key="profile")
    base = PROFILES.get(profile, PROFILES["Balanced"])
    weights = {}
    with st.expander("Objective weights", expanded=profile == "Custom"):
        cols = st.columns(len(METRICS))
        for c, m in zip(cols, METRICS):
            weights[m] = c.slider(m, 0, 5, base[m], key=f"w_{m}_{profile}",
                                  disabled=profile != "Custom")
    if sum(weights.values()) == 0:
        st.warning("Set at least one weight above 0.")
        st.stop()

    @st.cache_data(show_spinner="Simulating possible futures…")
    def cached_recommend(jobs_json: str, cfg_json: str, w_json: str):
        js = [Job(**d) for d in json.loads(jobs_json)]
        out = recommend(js, json.loads(cfg_json), json.loads(w_json))
        out["actual_results"] = None   # not needed by the UI below; keeps the cache light
        return out

    jobs_json = json.dumps([j.__dict__ for j in jobs])
    rec = cached_recommend(jobs_json, json.dumps(cfg, sort_keys=True), json.dumps(weights))
    chosen = rec["chosen"]

    st.success(f"### Recommended: **{chosen}**")
    pos = rec["chosen_actual_position"]
    if pos == 1:
        st.info(f"✅ Verified: on the actual burst times **{chosen}** was the best of "
                f"{len(ALGORITHMS)} algorithms.")
    else:
        st.info(f"On the actual burst times **{chosen}** ranked **#{pos}** of {len(ALGORITHMS)} "
                f"(actual best: {rec['actual_best']}). The plan was made from predictions only, "
                "so a small gap is expected.")

    wl = rec["workload"]
    c = st.columns(5)
    c[0].metric("Jobs", wl["jobs"])
    c[1].metric("Mean burst", wl["mean_burst"])
    c[2].metric("Burst variation (CV)", wl["burst_cv"])
    c[3].metric("CPU load", f"{wl['load']}×")
    c[4].metric("Prediction error (MAE)", f"{wl['pred_mae']} units")

    st.markdown("**Why this choice**")
    for r in rec["reasons"]:
        st.markdown(f"- {r}")

    chosen_res = run_algorithm(chosen, jobs, dict(cfg, use_predicted=True))
    st.markdown(f"**Gantt chart of {chosen} on the actual jobs**")
    st.plotly_chart(gantt_chart(chosen_res.segments, pids), width="stretch",
                    key="smart_gantt")

    l, r = st.columns(2)
    with l:
        st.markdown(f"**Planned ranking** (average of {rec['scenarios']} simulated futures)")
        st.dataframe(rec["plan_rank"], width="stretch", hide_index=True)
    with r:
        st.markdown("**Actual ranking** (real burst times)")
        st.dataframe(rec["actual_rank"], width="stretch", hide_index=True)
    st.caption("Score: each metric is scaled 0 (best) to 1 (worst) across algorithms, "
               "then combined with your weights. Lower score = better.")

# ---- Step-by-step ----------------------------------------------------------
with tab_step:
    st.subheader("Watch the scheduler tick by tick")
    algo_s = st.selectbox("Algorithm", ALGORITHMS, key="step_algo")
    res_s = run_algorithm(algo_s, jobs, cfg)
    t_end = res_s.timeline[-1]["time"] + 1

    def render(t):
        snap = res_s.timeline[min(t, t_end - 1)] if t < t_end else None
        chart_ph.plotly_chart(gantt_chart(res_s.segments, pids, upto=min(t + 1, t_end), x_max=t_end),
                              width="stretch", key=f"step_{t}_{time.time()}")
        if snap is None:
            info_ph.success(f"All jobs finished at t = {t_end}. "
                            f"Average waiting time = {res_s.avg_waiting:.2f}")
            return
        arrived = [j.pid for j in jobs if j.arrival == t]
        lines = [
            f"**Time {t} → {t + 1}**",
            f"- 🟢 Running: **{snap['running'] or 'CPU idle'}**",
            f"- ⏳ Ready queue: {', '.join(snap['ready']) or 'empty'}",
            f"- ✅ Completed: {', '.join(snap['done']) or 'none yet'}",
        ]
        if arrived:
            lines.insert(1, f"- 📥 Arrived now: {', '.join(arrived)}")
        info_ph.markdown("\n".join(lines))

    t = st.slider("Time", 0, t_end, 0, key=f"t_{algo_s}")
    play = st.button("▶️ Play from current time")
    chart_ph, info_ph = st.empty(), st.empty()
    if play:
        for tt in range(t, t_end + 1):
            render(tt)
            time.sleep(0.45)
    else:
        render(t)

# ---- About -----------------------------------------------------------------
with tab_about:
    st.markdown(r"""
### Algorithms
| Algorithm | Type | Rule |
|---|---|---|
| **FCFS** | Non-preemptive | Run jobs in order of arrival. Simple, but short jobs wait behind long ones (convoy effect). |
| **SJF** | Non-preemptive | Run the shortest job next. Lowest average waiting time among non-preemptive policies. |
| **SRTF** | Preemptive | Run the job with the shortest *remaining* time; a shorter arrival preempts. Optimal average waiting time, but long jobs can starve. |
| **Priority** | Both | Highest priority (lowest number) first. Optional **aging** raises the priority of waiting jobs. |
| **Round Robin** | Preemptive | Each job runs for one time quantum, then goes to the back of the queue. Good response time. |
| **MLFQ** | Preemptive | 3 queues (RR q0 → RR q1 → FCFS). Jobs that use a full quantum drop a level, so interactive jobs stay on top. |
| **Smart Adaptive (SAS)** | Preemptive | *Proposed.* Lowest score runs: `predicted remaining − aging × time waited`. |

### The smart part
**1. Burst prediction.** A real OS cannot know burst times in advance. Each job's next burst is
predicted from its past bursts by exponential averaging:

$$\tau_{n+1} = \alpha\, t_n + (1-\alpha)\,\tau_n$$

**2. Smart Adaptive Scheduler (SAS).** Works only on predictions. The predicted-remaining term keeps
SRTF's low average waiting time, and the aging term stops long jobs from starving. With an aging factor
below 1, a running job's score falls faster than any waiting job's, so it is only preempted by a new
arrival (no thrashing).

**3. Automatic recommendation.** Every algorithm is simulated on 30 possible futures built from the
predictions and each job's own variability. Each is scored against the chosen goal (batch, interactive,
fair or custom weights), and the most robust choice is recommended. The choice is then checked against the
actual bursts.

### Metrics
- **Turnaround** = Completion − Arrival  · **Waiting** = Turnaround − Burst  · **Response** = First run − Arrival
- **CPU utilisation** = busy time ÷ total time · **Throughput** = jobs ÷ total time
- **Max waiting** = the longest any single job waited (starvation indicator)
""")
