"""Streamlit dashboard for the live intersection camera."""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import logging

import streamlit as st

from traffic_dashboard.config import (
    DEFAULT_LIGHT_ROI,
    DEFAULT_STREAM_URL,
    VEHICLE_CLASS_IDS,
    RuntimeConfig,
)
from traffic_dashboard.pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

st.set_page_config(
    page_title="Intersection Traffic",
    page_icon="🚦",
    layout="wide",
)


def _pipeline() -> Pipeline:
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = Pipeline()
    return st.session_state.pipeline


def _seconds(value: float) -> str:
    return f"{value:.1f} s"


pipeline = _pipeline()

st.title("Intersection traffic")
st.caption(
    "Counts the four lanes on the near approach, people on the crosswalks, and the red light. Parked cars are ignored."
)

with st.sidebar:
    st.header("Stream")
    url = st.text_input("YouTube URL", value=DEFAULT_STREAM_URL)
    quality = st.selectbox("Quality", ["720p", "480p", "360p", "best"], index=0)
    start_col, stop_col = st.columns(2)
    with start_col:
        if st.button("Start", type="primary", use_container_width=True):
            pipeline.start(url, quality)
    with stop_col:
        if st.button("Stop", use_container_width=True):
            pipeline.stop()
    if st.button("Reset counts", use_container_width=True):
        pipeline.reset_counts()

    st.header("Lanes")
    st.caption(
        "From the left: left only, left or straight, straight, right. "
        "The red outline is the parking lot and is not counted."
    )
    zone_shift = st.slider("Nudge lanes", -0.08, 0.08, 0.0, 0.005)

    st.header("Traffic light")
    st.caption("Drag the box until it covers one signal head.")
    light_x = st.slider("Box left", 0.0, 0.95, DEFAULT_LIGHT_ROI[0], 0.01)
    light_y = st.slider("Box top", 0.0, 0.95, DEFAULT_LIGHT_ROI[1], 0.01)
    light_w = st.slider("Box width", 0.02, 0.4, DEFAULT_LIGHT_ROI[2], 0.01)
    light_h = st.slider("Box height", 0.02, 0.4, DEFAULT_LIGHT_ROI[3], 0.01)
    red_ratio_min = st.slider("Red pixel ratio", 0.02, 0.40, 0.06, 0.01)
    show_red_mask = st.checkbox("Show red mask in the box")

    st.header("Detector")
    confidence = st.slider("Confidence", 0.15, 0.80, 0.35, 0.05)
    class_names = st.multiselect(
        "Vehicle types",
        list(VEHICLE_CLASS_IDS),
        default=list(VEHICLE_CLASS_IDS),
    )

pipeline.update_config(
    RuntimeConfig(
        zone_shift=zone_shift,
        light_x=light_x,
        light_y=light_y,
        light_w=light_w,
        light_h=light_h,
        red_ratio=red_ratio_min,
        confidence=confidence,
        classes=tuple(class_names),
        show_red_mask=show_red_mask,
    )
)


@st.fragment(run_every=0.25)
def live_view() -> None:
    frame, stats = pipeline.snapshot()
    cards = st.columns(4)
    cards[0].metric("Left turn", stats.left_turns)
    cards[1].metric("Straight", stats.straight)
    cards[2].metric("Right turn", stats.right_turns)
    cards[3].metric("People", stats.people_passed)
    lights = st.columns(3)
    lights[0].metric("Red light count", stats.red_appearances)
    if stats.red_active:
        lights[1].metric("Red light now", _seconds(stats.current_red_duration))
    else:
        lights[1].metric("Last red light", _seconds(stats.last_red_duration))
    lights[2].metric("Average red light", _seconds(stats.average_red_duration))

    if stats.error:
        st.error(stats.error)
    elif not pipeline.running and stats.status in {"Idle", "Stopped"}:
        st.info("Press Start to connect to the live camera. The first start downloads YOLOv8n.")

    if frame is not None:
        st.image(frame, use_container_width=True)
    else:
        st.markdown("Waiting for the first processed frame.")

    device_label = "MPS (Metal)" if stats.device == "mps" else stats.device.upper()
    st.caption(
        f"Status: {stats.status} · Device: {device_label} · "
        f"{stats.fps:.1f} fps · Tracks: {stats.tracks} · "
        f"Red pixels in box: {stats.red_ratio:.1%}"
    )

    with st.expander("Red light phases"):
        if stats.red_durations:
            st.dataframe(
                {
                    "Phase": list(range(1, len(stats.red_durations) + 1)),
                    "Duration (s)": [round(value, 1) for value in stats.red_durations],
                },
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.write("No completed red phase yet. A phase is stored when the light leaves red.")


live_view()
