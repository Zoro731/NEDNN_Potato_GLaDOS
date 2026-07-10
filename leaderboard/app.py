"""Streamlit leaderboard dashboard for the MEG encoding challenge."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from evaluate import score_submission, validate_submission

LEADERBOARD_PATH = Path(__file__).parent / "leaderboard.csv"
COLUMNS = ["timestamp", "group", "submission_name", "max_r", "mean_r", "median_r", "worst10_mean"]


def load_leaderboard() -> pd.DataFrame:
    if LEADERBOARD_PATH.exists():
        return pd.read_csv(LEADERBOARD_PATH, parse_dates=["timestamp"])
    return pd.DataFrame(columns=COLUMNS)


def save_leaderboard(df: pd.DataFrame) -> None:
    df.to_csv(LEADERBOARD_PATH, index=False)


# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="MEG Encoding Challenge — Leaderboard", layout="wide")
st.header("MEG Encoding Challenge 1 — Leaderboard")

# ── sidebar: submission form ──────────────────────────────────────────────────
with st.sidebar:
    st.subheader("submit a prediction")
    group_name = st.text_input("group name")
    submission_name = st.text_input("submission name")
    uploaded = st.file_uploader("prediction file (.npy)", type=["npy"])
    submit = st.button("submit", type="primary")

    if submit:
        if not group_name.strip():
            st.error("please enter a group name")
        elif not submission_name.strip():
            st.error("please enter a submission name")
        elif uploaded is None:
            st.error("please upload a .npy file")
        else:
            arr = np.load(uploaded)
            err = validate_submission(arr)
            if err:
                st.error(f"invalid submission: {err}")
            else:
                with st.spinner("scoring…"):
                    scores = score_submission(arr)

                row = {
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
                    "group": group_name.strip(),
                    "submission_name": submission_name.strip(),
                    "max_r": scores["max_r"],
                    "mean_r": scores["mean_r"],
                    "median_r": scores["median_r"],
                    "worst10_mean": scores["worst10_mean"],
                }
                df = load_leaderboard()
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                save_leaderboard(df)
                st.success(
                    f"scored! mean r = {scores['mean_r']:.4f} | "
                    f"max r = {scores['max_r']:.4f}"
                )
                st.rerun()

# ── main: leaderboard table ───────────────────────────────────────────────────
df = load_leaderboard()

if df.empty:
    st.info("no submissions yet — use the sidebar to submit a prediction.")
else:
    display = (
        df.sort_values("mean_r", ascending=False)
        .reset_index(drop=True)
        .rename(columns={"submission_name": "submission"})
    )
    display.index += 1  # 1-based rank

    fmt = {c: "{:.4f}" for c in ["max_r", "mean_r", "median_r", "worst10_mean"]}
    st.dataframe(
        display[["group", "submission", "mean_r", "median_r", "max_r", "worst10_mean", "timestamp"]]
        .style.format(fmt),
        use_container_width=True,
    )

    # ── scatter plot ──────────────────────────────────────────────────────────
    st.divider()
    fig = px.scatter(
        df,
        x="max_r",
        y="worst10_mean",
        color="group",
        hover_data={"submission_name": True, "mean_r": ":.4f", "group": False},
        labels={
            "max_r": "max encoding [r]",
            "worst10_mean": "worst-10-channel mean [r]",
        },
        template="simple_white",
    )
    fig.update_layout(
        legend_title_text="group",
        margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
