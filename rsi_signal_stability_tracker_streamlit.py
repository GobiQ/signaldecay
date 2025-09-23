# rsi_signal_stability_app.py
# Streamlit app to analyze stability/decay of RSI-based signals using absolute or percentile thresholds
# Author: ChatGPT
# Run: streamlit run rsi_signal_stability_app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date, timedelta
import yfinance as yf
from scipy import stats

# -----------------------------
# Utilities
# -----------------------------

def compute_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Wilder's RSI computed via EMA/ewm (alpha=1/length)."""
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    # Wilder's smoothing (EMA with alpha=1/length)
    gain = up.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    loss = down.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

@st.cache_data(show_spinner=False)
def load_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        return df
    df = df[['Close']].rename(columns={'Close': 'close'})
    df = df[~df.index.duplicated(keep='first')]
    return df

def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    return close.shift(-horizon) / close - 1.0

def rolling_percentile_threshold(series: pd.Series, window: int, percentile: float) -> pd.Series:
    """Return rolling RSI threshold (value) corresponding to the given percentile (0-100)."""
    q = np.clip(percentile/100.0, 0.0, 1.0)
    return series.rolling(window, min_periods=max(5, int(window*0.5))).quantile(q)

def rolling_signal_edge(event_returns: pd.Series, window: int, min_events: int = 10) -> pd.Series:
    """Rolling mean of forward returns *only at event dates* over a trailing window.
    event_returns has NaN on non-event days; values on event days equal the forward return realized for that event.
    """
    # Count of events in the rolling window
    event_counts = event_returns.notna().rolling(window, min_periods=1).sum()
    # Sum of returns across events within the window
    event_sum = event_returns.fillna(0.0).rolling(window, min_periods=1).sum()
    edge = event_sum / event_counts.replace(0, np.nan)
    edge[event_counts < min_events] = np.nan
    return edge

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="RSI Signal Stability Tracker", page_icon="📈", layout="wide")

st.title("📈 RSI Signal Stability / Decay Tracker")
st.caption("Analyze how an RSI-based entry condition performs over time. Choose an absolute RSI threshold or a percentile that translates to a rolling RSI figure.")

with st.sidebar:
    st.header("Controls")
    ticker = st.text_input("Ticker (Yahoo Finance)", value="SPY")
    today = date.today()
    default_start = date(today.year-8, 1, 1)  # ~8 years by default
    start_date = st.date_input("Start date", value=default_start, max_value=today - timedelta(days=1))
    end_date = st.date_input("End date", value=today)

    rsi_len = st.number_input("RSI length", min_value=2, max_value=200, value=14, step=1)

    signal_mode = st.radio("Signal type", options=["Absolute RSI", "Percentile RSI → RSI figure"], index=0)

    if signal_mode == "Absolute RSI":
        rsi_threshold = st.slider("RSI threshold", min_value=0.0, max_value=100.0, value=30.0, step=0.5)
    else:
        percentile = st.slider("Percentile threshold (0–100)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
        perc_window = st.number_input("Percentile rolling window (trading days)", min_value=30, max_value=1260, value=252, step=1)

    operator = st.radio("Condition", options=["RSI ≤ threshold", "RSI ≥ threshold"], index=0)

    horizon = st.number_input("Forward return horizon (trading days)", min_value=1, max_value=252, value=21, step=1)
    eval_window = st.number_input("Rolling evaluation window (days)", min_value=21, max_value=1260, value=126, step=1)
    min_ev = st.number_input("Min events in window to show edge", min_value=1, max_value=100, value=10, step=1)

    build_equity = st.checkbox("Show simple equity curve (long when condition true)", value=False)
    download_switch = st.checkbox("Enable CSV download of results", value=True)

# -----------------------------
# Data & Signal
# -----------------------------
if not ticker:
    st.warning("Enter a ticker symbol to begin.")
    st.stop()

prices = load_prices(ticker, str(start_date), str(end_date))
if prices.empty:
    st.error("No data returned. Check ticker or date range.")
    st.stop()

prices['rsi'] = compute_rsi(prices['close'], rsi_len)
prices['fwd_ret'] = forward_return(prices['close'], horizon)

# Compute threshold line (absolute or percentile-based)
if signal_mode == "Absolute RSI":
    prices['rsi_thresh'] = rsi_threshold
else:
    prices['rsi_thresh'] = rolling_percentile_threshold(prices['rsi'], window=perc_window, percentile=percentile)

# Build boolean signal based on operator
if "≤" in operator:
    prices['signal'] = prices['rsi'] <= prices['rsi_thresh']
else:
    prices['signal'] = prices['rsi'] >= prices['rsi_thresh']

# Event returns (forward returns measured at signal dates)
prices['event_ret'] = np.where(prices['signal'], prices['fwd_ret'], np.nan)

# Rolling signal edge (oscillates around 0 if no edge)
prices['rolling_edge'] = rolling_signal_edge(prices['event_ret'], window=eval_window, min_events=min_ev)

# Summary metrics
total_events = int(prices['signal'].sum())
valid_edge_points = int(prices['rolling_edge'].notna().sum())
edge_median = prices['rolling_edge'].median(skipna=True)
edge_mean = prices['rolling_edge'].mean(skipna=True)

# Compare event vs non-event forward returns (static over whole sample)
evt = prices.loc[prices['signal'], 'fwd_ret'].dropna()
nonevt = prices.loc[~prices['signal'], 'fwd_ret'].dropna()
if len(evt) > 1 and len(nonevt) > 1:
    # Welch t-test
    t_stat, p_val = stats.ttest_ind(evt, nonevt, equal_var=False)
else:
    t_stat, p_val = np.nan, np.nan

# -----------------------------
# Plots & Tables
# -----------------------------
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader("Rolling Signal Edge (mean forward return on event dates)")
    st.caption("Oscillating line around 0 indicates changing edge over time. Positive values suggest the condition tended to precede gains over the chosen horizon.")
    plot_df = prices[['rolling_edge']].copy()
    plot_df = plot_df.dropna()
    if plot_df.empty:
        st.info("Not enough events within the evaluation window to compute a rolling edge. Try reducing min events, shortening horizon, or extending the date range.")
    else:
        fig = px.line(plot_df, x=plot_df.index, y='rolling_edge', labels={'rolling_edge': f'Rolling mean of {horizon}D fwd returns'}, title=None)
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=420)
        st.plotly_chart(fig, use_container_width=True)

    if build_equity:
        st.subheader("Simple Equity Curve (long when condition true)")
        # Strategy: long next-day when today's signal is true (avoid look-ahead)
        ret = prices['close'].pct_change()
        strat_ret = ret * prices['signal'].shift(1).fillna(False).astype(float)
        eq = (1 + strat_ret.fillna(0)).cumprod()
        bench = (1 + ret.fillna(0)).cumprod()
        eq_df = pd.DataFrame({'Strategy': eq, 'Buy&Hold': bench})
        fig2 = px.line(eq_df, x=eq_df.index, y=['Strategy', 'Buy&Hold'])
        fig2.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=420)
        st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.subheader("Summary")
    st.metric("Total events", f"{total_events}")
    st.metric("Valid rolling-edge points", f"{valid_edge_points}")
    st.metric(f"Median rolling edge ({horizon}D fwd)", f"{edge_median:.4%}" if pd.notna(edge_median) else "—")
    st.metric(f"Mean rolling edge ({horizon}D fwd)", f"{edge_mean:.4%}" if pd.notna(edge_mean) else "—")

    st.markdown("---")
    st.markdown("**Event vs Non-Event (full sample)**")
    st.write({
        "Events (N)": int(len(evt)),
        "Non-events (N)": int(len(nonevt)),
        f"Mean evt {horizon}D fwd": f"{evt.mean():.4%}" if len(evt) else "—",
        f"Mean nonevt {horizon}D fwd": f"{nonevt.mean():.4%}" if len(nonevt) else "—",
        "Welch t-stat": None if np.isnan(t_stat) else float(t_stat),
        "p-value": None if np.isnan(p_val) else float(p_val),
    })

    st.markdown("---")
    st.markdown("**Threshold reference**")
    if signal_mode == "Absolute RSI":
        st.write(f"Using constant RSI threshold = **{rsi_threshold:.1f}**")
    else:
        st.write(
            f"Using **rolling percentile**: {percentile:.1f}th over {perc_window} days → daily RSI threshold varies by regime."
        )

# -----------------------------
# Data table & downloads
# -----------------------------
with st.expander("Show data / download"):
    show_cols = ['close', 'rsi', 'rsi_thresh', 'signal', 'fwd_ret', 'event_ret', 'rolling_edge']
    out = prices[show_cols].copy()
    st.dataframe(out.tail(1000))

    if download_switch:
        csv = out.to_csv(index=True).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{ticker}_rsi_signal_stability.csv",
            mime='text/csv'
        )

# -----------------------------
# Notes
# -----------------------------
st.markdown(
    """
    **Methodology**
    - RSI computed with Wilder's smoothing (EMA, α=1/length).
    - *Forward return horizon* is the % change from _t_ to _t+horizon_.
    - *Event* occurs when RSI is ≤/≥ the threshold (absolute or a rolling percentile-derived RSI value).
    - *Rolling signal edge* = mean forward return of events within a trailing window. Requires a minimum number of events to avoid noisy estimates.
    - *Equity curve* goes long on day _t+1_ if the signal is true at _t_ (to avoid look-ahead bias).
    """
)
