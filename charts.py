"""Plotly charts: Gantt chart and metric comparison bars."""
from __future__ import annotations

from typing import Dict, List, Optional

import plotly.graph_objects as go

from scheduler.models import Segment

# Categorical palette, assigned in fixed order (job i always gets slot i).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#6250d6", "#e34948"]
IDLE_COLOR = "#d9d8d2"
ACCENT = "#2a78d6"


def _text_color(hex_color: str) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#1a1a19" if lum > 0.45 else "#ffffff"


def color_map(pids: List[str]) -> Dict[str, str]:
    """Stable colour per job. With more than 8 jobs colours repeat, but every
    bar is also labelled with its PID, so identity never relies on colour."""
    return {pid: PALETTE[i % len(PALETTE)] for i, pid in enumerate(pids)}


def gantt_chart(segments: List[Segment], pids: List[str], upto: Optional[int] = None,
                x_max: Optional[int] = None, height: int = 190) -> go.Figure:
    colors = color_map(pids)
    fig = go.Figure()
    shown = []
    for seg in segments:
        start, end = seg.start, seg.end
        if upto is not None:
            if start >= upto:
                continue
            end = min(end, upto)
        shown.append((start, end))
        idle = seg.pid == "IDLE"
        fill = IDLE_COLOR if idle else colors.get(seg.pid, PALETTE[0])
        label = "Idle" if idle else seg.pid
        fig.add_trace(go.Bar(
            x=[end - start], base=[start], y=["CPU"], orientation="h", width=0.55,
            marker=dict(color=fill, line=dict(color="#ffffff", width=2)),
            text=[label], textposition="inside", insidetextanchor="middle",
            textfont=dict(color=_text_color(fill), size=13),
            hovertemplate=f"<b>{label}</b><br>Start: {start}<br>End: {end}"
                          f"<br>Duration: {end - start}<extra></extra>",
            showlegend=False,
        ))

    ticks = sorted({s for s, _ in shown} | {e for _, e in shown})
    lo = segments[0].start if segments else 0
    hi = x_max if x_max is not None else (segments[-1].end if segments else 1)
    xaxis = dict(title="Time", range=[lo - 0.2, hi + 0.2], showgrid=False, zeroline=False)
    if len(ticks) <= 40:
        xaxis.update(tickmode="array", tickvals=ticks)
    fig.update_layout(
        barmode="overlay", height=height, bargap=0.2,
        margin=dict(l=10, r=10, t=10, b=40),
        xaxis=xaxis, yaxis=dict(showticklabels=False, showgrid=False),
        uniformtext=dict(minsize=9, mode="hide"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def metric_bar(names: List[str], values: List[float], metric: str,
               highlight: Optional[str] = None, height: int = 300) -> go.Figure:
    """Single-series bar chart. The highlighted bar (e.g. the best algorithm)
    keeps the accent colour; the rest are muted, and the best is also named in
    its label so the highlight is not colour-only."""
    colors = [ACCENT if (highlight is None or n == highlight) else "#9ec5f4" for n in names]
    labels = [f"{v:g}" + ("  ★" if n == highlight else "") for n, v in zip(names, values)]
    fig = go.Figure(go.Bar(
        x=names, y=values, marker=dict(color=colors, line=dict(color="#ffffff", width=2)),
        text=labels, textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{x}</b><br>" + metric + ": %{y}<extra></extra>",
    ))
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=36, b=10),
        title=dict(text=metric, font=dict(size=15)), barcornerradius=4,
        yaxis=dict(gridcolor="rgba(128,128,128,0.15)", zeroline=False),
        xaxis=dict(tickangle=-30),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig
