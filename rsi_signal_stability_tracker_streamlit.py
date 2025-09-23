# rsi_signal_stability_app.py
# Streamlit app to analyze stability/decay of RSI-based signals using absolute or percentile thresholds
# Author: ChatGPT
# Run: streamlit run rsi_signal_stability_app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
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

def _sanitize_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure no tz, no object numerics, drop non-finite."""
    if not df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_convert(None)
    # Replace infs, coerce numerics, drop NaNs
    df = df.replace([np.inf, -np.inf], np.nan)
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(how='any')
    return df

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="RSI Signal Stability Tracker", page_icon="📈", layout="wide")

st.title("📈 Cross-Asset RSI Signal Tracker")
st.caption("Analyze how an RSI-based entry condition on one asset performs when allocating to another asset. Choose an absolute RSI threshold or a percentile that translates to a rolling RSI figure.")

with st.sidebar:
    st.header("Controls")
    
    source_ticker = st.text_input("Source ticker (for RSI signal)", value="SPY", 
                                 help="The ticker used to calculate RSI and generate trading signals. This is where the RSI condition is evaluated.")
    target_ticker = st.text_input("Target ticker (to allocate / measure returns)", value="UVXY", 
                                 help="The ticker you allocate to when the RSI signal condition is TRUE. This is what you buy when the signal triggers.")
    comparison_ticker = st.text_input("Comparison ticker (held when signal condition is FALSE)", value="BIL", 
                                     help="The ticker you hold when the RSI signal condition is FALSE. This is your alternative allocation (e.g., cash, bonds, or another asset).")
    
    today = date.today()
    default_start = date(today.year-8, 1, 1)  # ~8 years by default
    
    auto_start = st.checkbox("Auto-adjust start date to earliest common date", value=True, 
                            help="Automatically set start date to the earliest date where all three tickers have data. This ensures maximum historical coverage for your analysis.")
    
    if auto_start:
        st.info("📅 Start date will be automatically adjusted to the earliest common date for all tickers.")
        start_date = st.date_input("Start date (will be auto-adjusted)", value=default_start, max_value=today - timedelta(days=1), disabled=True)
    else:
        start_date = st.date_input("Start date", value=default_start, max_value=today - timedelta(days=1))
    
    end_date = st.date_input("End date", value=today)

    rsi_len = st.number_input("RSI length", min_value=2, max_value=200, value=10, step=1,
                             help="Number of periods used to calculate RSI. Shorter periods (10-14) are more sensitive to recent price changes, while longer periods (20-30) are smoother and less noisy.")

    signal_mode = st.radio(
        "Signal type",
        options=["Absolute RSI", "Percentile RSI → RSI figure"],
        index=0,
        help="Absolute RSI: Use a fixed RSI threshold (e.g., 30 or 70). Percentile RSI: Use a dynamic threshold based on historical RSI percentiles (adapts to market conditions)."
    )

    if signal_mode == "Absolute RSI":
        rsi_threshold = st.slider("RSI threshold", min_value=0.0, max_value=100.0, value=30.0, step=0.5,
                                 help="Fixed RSI threshold for signal generation. Values below 30 are considered oversold, above 70 are overbought. Choose based on your strategy (mean reversion vs momentum).")
        perc_scope = None
        percentile = None
        perc_window = None
    else:
        percentile = st.slider("Percentile (0–100)", min_value=0.0, max_value=100.0, value=90.0, step=0.5,
                              help="Percentile threshold for dynamic RSI calculation. 90th percentile means the signal triggers when RSI is in the top 10% of historical values. Higher percentiles = more selective signals.")
        perc_scope = st.radio(
            "Percentile scope",
            options=["Whole dataset (fixed)", "Rolling (windowed)"],
            index=0,
            help="Whole dataset: one fixed threshold from the entire selected period. "
                 "Rolling: threshold recomputed from the last N trading days (changes over time)."
        )
        if perc_scope == "Rolling (windowed)":
            perc_window = st.number_input(
                "Rolling window (trading days)",
                min_value=30, max_value=1260, value=252, step=1,
                help="Number of trading days used to calculate the rolling percentile threshold. Longer windows provide more stable thresholds but adapt slower to regime changes."
            )
        else:
            perc_window = None

    operator = st.radio("Condition", options=["RSI ≤ threshold", "RSI ≥ threshold"], index=1,
                       help="RSI ≤ threshold: Signal triggers when RSI is at or below threshold (oversold/mean reversion). RSI ≥ threshold: Signal triggers when RSI is at or above threshold (overbought/momentum).")

    horizon = st.number_input("Forward return horizon (trading days)", min_value=1, max_value=252, value=3, step=1,
                             help="Number of trading days to look ahead when measuring returns. Shorter horizons (1-5 days) capture immediate effects, longer horizons (10-21 days) capture delayed effects.")
    eval_window = st.number_input("Rolling evaluation window (days)", min_value=21, max_value=1260, value=1008, step=1,
                                 help="Number of trading days used to calculate rolling signal edge. Longer windows provide more stable estimates but adapt slower to changing market conditions. 1008 days ≈ 4 years.")
    min_ev = st.number_input("Min events in window to show edge", min_value=1, max_value=100, value=3, step=1,
                            help="Minimum number of signal events required within the evaluation window to calculate rolling edge. Higher values ensure statistical significance but may create gaps in the analysis.")

    build_equity = st.checkbox("Show simple equity curve (long when condition true)", value=False,
                              help="Display equity curves comparing the switching strategy (target vs comparison) against buy-and-hold benchmarks. Shows cumulative performance over time.")
    download_switch = st.checkbox("Enable CSV download of results", value=True,
                                 help="Allow downloading the analysis results as a CSV file containing all calculated values (RSI, signals, returns, etc.) for further analysis.")

# -----------------------------
# Data & Signal
# -----------------------------
if not source_ticker or not target_ticker or not comparison_ticker:
    st.warning("Enter all three ticker symbols to begin.")
    st.stop()

# Auto-adjust start date to earliest common date if requested
if auto_start:
    # First, try to load data from a much earlier date to find the true earliest common date
    # Use a date far back enough to capture the earliest possible data for most tickers
    early_start = date(1990, 1, 1)
    
    # Load all three tickers from the early start date
    src = load_prices(source_ticker, str(early_start), str(end_date))
    tgt = load_prices(target_ticker, str(early_start), str(end_date))
    cmp = load_prices(comparison_ticker, str(early_start), str(end_date))
    
    if src.empty:
        st.error(f"No data for source ticker: {source_ticker}")
        st.stop()
    if tgt.empty:
        st.error(f"No data for target ticker: {target_ticker}")
        st.stop()
    if cmp.empty:
        st.error(f"No data for comparison ticker: {comparison_ticker}")
        st.stop()
    
    # Find the earliest common date where all three tickers have data
    earliest_common_date = max(
        src.index.min().date(),
        tgt.index.min().date(), 
        cmp.index.min().date()
    )
    
    # Update start_date to reflect the actual date being used
    original_start_date = start_date
    start_date = earliest_common_date
    
    # Always reload with the earliest common date to maximize data coverage
    st.info(f"📅 **Analysis period: {start_date} to {end_date}** "
            f"(extended from selected {original_start_date} to maximize data coverage for all three tickers)")
    
    # Reload data with the earliest possible start date
    src = load_prices(source_ticker, str(start_date), str(end_date))
    tgt = load_prices(target_ticker, str(start_date), str(end_date))
    cmp = load_prices(comparison_ticker, str(start_date), str(end_date))
else:
    # Load all three tickers with user's selected start date
    src = load_prices(source_ticker, str(start_date), str(end_date))
    tgt = load_prices(target_ticker, str(start_date), str(end_date))
    cmp = load_prices(comparison_ticker, str(start_date), str(end_date))
    
    if src.empty:
        st.error(f"No data for source ticker: {source_ticker}")
        st.stop()
    if tgt.empty:
        st.error(f"No data for target ticker: {target_ticker}")
        st.stop()
    if cmp.empty:
        st.error(f"No data for comparison ticker: {comparison_ticker}")
        st.stop()

# Ensure tz-naive DateTimeIndex
src.index = pd.to_datetime(src.index).tz_localize(None)
tgt.index = pd.to_datetime(tgt.index).tz_localize(None)
cmp.index = pd.to_datetime(cmp.index).tz_localize(None)

# Inner-join on trading days present in ALL series so the horizon aligns
# First, rename columns before joining to avoid confusion
src_renamed = src.rename(columns={'close': 'close_src'})
tgt_renamed = tgt.rename(columns={'close': 'close_tgt'})
cmp_renamed = cmp.rename(columns={'close': 'close_cmp'})

# Join all three dataframes
prices = src_renamed.join(tgt_renamed, how="inner")
prices = prices.join(cmp_renamed, how="inner")


prices['rsi'] = compute_rsi(prices['close_src'], rsi_len)
prices['fwd_ret'] = forward_return(prices['close_tgt'], horizon)

# Compute threshold line (absolute or percentile-based)
if signal_mode == "Absolute RSI":
    prices['rsi_thresh'] = rsi_threshold
    thresh_note = f"Using constant RSI threshold = **{rsi_threshold:.1f}**"
else:
    p = np.clip(percentile / 100.0, 0.0, 1.0)
    if perc_scope == "Whole dataset (fixed)":
        # One fixed value from the entire selected period
        fixed_thresh = float(prices['rsi'].quantile(p))
        prices['rsi_thresh'] = fixed_thresh
        thresh_note = (
            f"**Whole dataset p{percentile:.1f}** ⇒ RSI threshold **{fixed_thresh:.2f}** "
            "(fixed for all dates)"
        )
    else:
        # Rolling window percentile
        prices['rsi_thresh'] = rolling_percentile_threshold(
            prices['rsi'], window=int(perc_window), percentile=percentile
        )
        # Show the latest available threshold value
        last_valid_thresh = prices['rsi_thresh'].dropna().iloc[-1] if prices['rsi_thresh'].notna().any() else np.nan
        if np.isnan(last_valid_thresh):
            thresh_note = (
                f"**Rolling p{percentile:.1f} over {perc_window} days** ⇒ RSI threshold varies by date; "
                "insufficient data yet to compute a current value."
            )
        else:
            thresh_note = (
                f"**Rolling p{percentile:.1f} over {perc_window} days** ⇒ "
                f"current RSI threshold **{last_valid_thresh:.2f}** (varies over time)"
            )

# Build boolean signal based on operator
if "≤" in operator:
    prices['signal'] = prices['rsi'] <= prices['rsi_thresh']
else:
    prices['signal'] = prices['rsi'] >= prices['rsi_thresh']

# Optional: suppress signals before percentile threshold is defined (rolling case)
if signal_mode != "Absolute RSI" and perc_scope == "Rolling (windowed)":
    prices.loc[prices['rsi_thresh'].isna(), 'signal'] = False

# Event returns (forward returns measured at signal dates)
prices['event_ret'] = np.where(prices['signal'], prices['fwd_ret'], np.nan)

# Rolling signal edge (oscillates around 0 if no edge)
prices['rolling_edge'] = rolling_signal_edge(prices['event_ret'], window=eval_window, min_events=min_ev)


# Ensure numeric dtype for computed columns
for c in ['rsi', 'fwd_ret', 'event_ret', 'rolling_edge']:
    if c in prices.columns:
        prices[c] = pd.to_numeric(prices[c], errors='coerce')

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

    # Always show the full date range, even with NaN values
    plot_df = prices[['rolling_edge']].copy()
    # Clean all non-finite & enforce numeric, but keep NaN values for full date range
    plot_df['rolling_edge'] = pd.to_numeric(plot_df['rolling_edge'], errors='coerce')
    # Replace infs with NaN but don't drop rows
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan)


    if plot_df.empty:
        st.info("No data available to plot.")
    else:
        # Quick diagnostic for 2025 data
        data_2025 = plot_df[plot_df.index.year == 2025]
        if len(data_2025) > 0:
            st.write(f"📊 2025 data: {len(data_2025)} days, from {data_2025.index.min().date()} to {data_2025.index.max().date()}")
        else:
            st.write("📊 No 2025 data available in the analysis period")
        # Build via graph_objects (avoid PX's dataframe coercion path)
        dates = plot_df.index.to_pydatetime()
        vals = plot_df['rolling_edge'].to_numpy(dtype=float)
        
        # Create separate traces for valid data and NaN periods
        fig = go.Figure()
        
        # Add trace for valid rolling edge values (solid line)
        valid_mask = ~np.isnan(vals)
        if valid_mask.any():
            fig.add_trace(go.Scatter(
                x=dates[valid_mask], 
                y=vals[valid_mask], 
                mode='lines', 
                name='Rolling edge (valid data)',
                line=dict(color='blue', width=2)
            ))
        
        # Add shaded regions for NaN periods to indicate "insufficient data"
        nan_mask = np.isnan(vals)
        if nan_mask.any():
            # Find consecutive NaN periods and add them as shaded regions
            nan_indices = np.where(nan_mask)[0]
            if len(nan_indices) > 0:
                # Group consecutive NaN periods
                groups = []
                current_group = [nan_indices[0]]
                
                for i in range(1, len(nan_indices)):
                    if nan_indices[i] == nan_indices[i-1] + 1:
                        current_group.append(nan_indices[i])
                    else:
                        groups.append(current_group)
                        current_group = [nan_indices[i]]
                groups.append(current_group)
                
                # Add shaded regions for each group
                for group in groups:
                    if len(group) > 1:  # Only shade regions with multiple consecutive NaN values
                        start_date = dates[group[0]]
                        end_date = dates[group[-1]]
                        fig.add_vrect(
                            x0=start_date, x1=end_date,
                            fillcolor="lightgray", opacity=0.05,
                            layer="below", line_width=0,
                            annotation_text="Insufficient data", 
                            annotation_position="top left"
                        )
        
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=420,
            xaxis_title='Date',
            yaxis_title=f'Rolling mean of {horizon}D fwd returns (on {target_ticker})',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            # Stretch x-axis across full width for granular analysis
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                showline=True,
                linewidth=1,
                linecolor='black'
            ),
            # Optimize for wide display
            autosize=True,
            width=None  # Let it use full container width
        )
        st.plotly_chart(fig, use_container_width=True)

    if build_equity:
        st.subheader("Equity Curve: Target vs Comparison")

        # Daily returns (adjusted close pct changes) - ensure 1D arrays
        ret_tgt = prices['close_tgt'].pct_change().values.flatten()
        ret_cmp = prices['close_cmp'].pct_change().values.flatten()

        # When condition is true → take target returns, else comparison
        strat_ret = np.where(prices['signal'].shift(1), ret_tgt, ret_cmp)  # shift(1) = trade next day
        
        # Create Series with proper index alignment
        strat_ret = pd.Series(strat_ret, index=prices.index).fillna(0)

        eq_strat = (1 + strat_ret).cumprod()
        eq_tgt = (1 + pd.Series(ret_tgt, index=prices.index).fillna(0)).cumprod()
        eq_cmp = (1 + pd.Series(ret_cmp, index=prices.index).fillna(0)).cumprod()

        eq_df = pd.DataFrame({
            'Strategy': eq_strat,
            f'Buy&Hold {target_ticker}': eq_tgt,
            f'Buy&Hold {comparison_ticker}': eq_cmp
        })

        eq_df = _sanitize_for_plot(eq_df)

        if eq_df.empty or eq_df.shape[0] < 2:
            st.info("Not enough data to draw equity curve.")
        else:
            dates = eq_df.index.to_pydatetime()
            fig = go.Figure()
            for col in eq_df.columns:
                fig.add_trace(go.Scatter(x=dates, y=eq_df[col], mode='lines', name=col))
            fig.update_layout(
                margin=dict(l=10, r=10, t=30, b=10),
                height=420,
                xaxis_title='Date',
                yaxis_title='Equity (normalized)',
                # Stretch x-axis across full width for granular analysis
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='lightgray',
                    showline=True,
                    linewidth=1,
                    linecolor='black'
                ),
                # Optimize for wide display
                autosize=True,
                width=None  # Let it use full container width
            )
            st.plotly_chart(fig, use_container_width=True)

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
    st.write(f"Signal on **{source_ticker}**, returns on **{target_ticker}**.")
    st.write(thresh_note)

# -----------------------------
# Data table & downloads
# -----------------------------
with st.expander("Show data / download"):
    show_cols = [
        'close_src', 'close_tgt', 'close_cmp',
        'rsi', 'rsi_thresh', 'signal',
        'fwd_ret', 'event_ret', 'rolling_edge'
    ]
    out = prices[show_cols].copy()
    st.dataframe(out.tail(1000))

    if download_switch:
        csv = out.to_csv(index=True).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{source_ticker}_{target_ticker}_rsi_signal_stability.csv",
            mime='text/csv'
        )

# Debug expander (temporary, helpful on Cloud)
with st.expander("Debug (dtypes & head)—safe to remove later"):
    try:
        st.write("rolling_edge dtype:", prices['rolling_edge'].dtype)
        st.write("Any infs?:", np.isinf(prices['rolling_edge']).any())
        st.write("Head (non-null):")
        st.dataframe(prices[['rolling_edge']].dropna().head(5))
    except Exception as e:
        st.write("Debug exception:", repr(e))

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
