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

def rolling_win_rate(event_returns: pd.Series, window: int, min_events: int = 10, win_threshold: float = 0.0) -> pd.Series:
    """Rolling win rate of forward returns *only at event dates* over a trailing window."""
    is_event = event_returns.notna()
    is_win = (event_returns > win_threshold) & is_event
    wins = is_win.rolling(window, min_periods=1).sum()
    count = is_event.rolling(window, min_periods=1).sum()
    wr = wins / count.replace(0, np.nan)
    wr[count < min_events] = np.nan
    return wr

def segment_true_runs(mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Return list of (start_date, end_date) for each contiguous True run in mask.
    mask index must be sorted DateTimeIndex.
    """
    m = mask.fillna(False).astype(bool).to_numpy()
    idx = mask.index.to_numpy()
    runs = []
    in_run = False
    start = 0
    for i, v in enumerate(m):
        if v and not in_run:
            start = i
            in_run = True
        # close the run if v flipped to False OR we're at last bar
        last_bar = (i == len(m) - 1)
        if in_run and (not v or last_bar):
            end = i if v and last_bar else i - 1
            if end >= start:
                runs.append((pd.Timestamp(idx[start]), pd.Timestamp(idx[end])))
            in_run = False
    return runs

def build_event_table(prices: pd.DataFrame, alloc_bool: pd.Series) -> pd.DataFrame:
    """
    Build per-event stats for contiguous allocation runs.
    Returns a DataFrame with: start, end, duration, ret_tgt, ret_cmp, excess, MFE, MAE.
    """
    runs = segment_true_runs(alloc_bool)
    if not runs:
        return pd.DataFrame(columns=["start","end","duration","ret_tgt","ret_cmp","excess","MFE","MAE"])

    ret_tgt = prices["close_tgt"].pct_change()
    ret_cmp = prices["close_cmp"].pct_change()

    rows = []
    for start, end in runs:
        seg_tgt = ret_tgt.loc[start:end]
        seg_cmp = ret_cmp.loc[start:end]
        # cumulative returns over the allocated days
        cum_tgt = float((1.0 + seg_tgt.fillna(0)).prod() - 1.0)
        cum_cmp = float((1.0 + seg_cmp.fillna(0)).prod() - 1.0)
        excess  = cum_tgt - cum_cmp

        # MFE/MAE within the event (vs entry)
        path = (1.0 + seg_tgt.fillna(0)).cumprod()
        mfe = float(path.max() - 1.0) if len(path) else np.nan
        mae = float(path.min() - 1.0) if len(path) else np.nan

        rows.append({
            "start": start, "end": end,
            "duration": int(len(seg_tgt)),
            "ret_tgt": cum_tgt, "ret_cmp": cum_cmp, "excess": excess,
            "MFE": mfe, "MAE": mae
        })
    df = pd.DataFrame(rows).sort_values("end").reset_index(drop=True)
    return df

def calculate_tax_adjusted_equity(strategy_returns: pd.Series, tax_rate: float) -> pd.Series:
    """
    Apply year-end tax on POSITIVE yearly gains.
    Taxes reduce the base for future compounding (multiplicative adjustment at year-end).
    """
    tax = float(tax_rate) / 100.0
    eq = (1.0 + strategy_returns.fillna(0.0)).cumprod()
    eq_tax = eq.copy()

    # Track the start-of-year level AFTER prior taxes
    start_level = None
    years = eq.index.to_period("Y").unique()

    for y in years:
        mask = (eq.index.to_period("Y") == y)
        if not mask.any():
            continue

        start_idx = eq.index[mask][0]
        end_idx   = eq.index[mask][-1]

        # Start level is prior year end (after tax), or the first point in the series
        if start_level is None:
            start_level = eq_tax.loc[start_idx]

        end_level = eq_tax.loc[end_idx]
        year_gain = end_level - start_level

        if year_gain > 0:
            # After-tax end-of-year level
            end_after_tax = start_level + (1.0 - tax) * (end_level - start_level)
            # Scale FUTURE path by the ratio of after-tax to before-tax level
            factor = end_after_tax / end_level  # < 1 when taxed
            eq_tax.loc[end_idx:] = eq_tax.loc[end_idx:] * factor
            start_level = eq_tax.loc[end_idx]   # carry forward post-tax end as next year's start
        else:
            # No tax, just roll the end level forward
            start_level = end_level

    return eq_tax

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

def _clip_tails(x: pd.Series | np.ndarray, pct_each_side: float = 1.0) -> np.ndarray:
    """
    Clip extremes symmetrically for visualization (does not mutate source data).
    pct_each_side: e.g., 1.0 -> clips below 1st and above 99th percentile.
    """
    x = pd.Series(x).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty or pct_each_side <= 0:
        return x.to_numpy()
    lo, hi = np.nanpercentile(x, [pct_each_side, 100 - pct_each_side])
    return x.clip(lo, hi).to_numpy()

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Signal Decay", page_icon="📈", layout="wide")

st.title("Signal Decay")
st.caption("Signal Stability Analysis")

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
        rsi_threshold_input = st.text_input(
            "RSI threshold",
            value="80.0",
            help="Fixed RSI threshold for signal generation. Values below 30 are considered oversold, above 70 are overbought. Enter as decimal (e.g., 80.5)"
        )
        
        # Convert text input to float with validation
        try:
            rsi_threshold = float(rsi_threshold_input)
            if rsi_threshold < 0.0 or rsi_threshold > 100.0:
                st.warning("RSI threshold must be between 0.0 and 100.0. Using 80.0 as default.")
                rsi_threshold = 80.0
        except ValueError:
            st.warning("Invalid RSI threshold format. Using 80.0 as default.")
            rsi_threshold = 80.0
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

    edge_mode = st.radio(
        "Edge mode",
        ["Fixed horizon (days)", "Trade-to-exit (event-based)"],
        index=1,  # Make event-based the default
        help="Fixed horizon: score events by forward returns over a set number of days. "
             "Trade-to-exit: score each contiguous allocation period (from entry until exit)."
    )

    win_thresh_input = st.text_input(
        "Win threshold (return)",
        value="0.0",
        help="Return cutoff to count as a win (0.0 = break-even, 0.001 = +0.1%). Enter as decimal (e.g., 0.001 for 0.1%)"
    )
    
    # Convert text input to float with validation
    try:
        win_thresh = float(win_thresh_input)
        if win_thresh < -0.05 or win_thresh > 0.05:
            st.warning("Win threshold must be between -0.05 and 0.05. Using 0.0 as default.")
            win_thresh = 0.0
    except ValueError:
        st.warning("Invalid win threshold format. Using 0.0 as default.")
        win_thresh = 0.0
    show_wr_baseline = st.checkbox("Show baseline win rate", value=True)

    if edge_mode == "Fixed horizon (days)":
        horizon = st.number_input("Forward return horizon (trading days)", min_value=1, max_value=20, value=5, step=1,
                                 help="Number of trading days to look ahead when measuring returns. Shorter horizons (1-5 days) capture immediate effects, longer horizons (10-21 days) capture delayed effects.")
        eval_window = st.number_input("Rolling evaluation window (days)", min_value=21, max_value=252, value=63, step=1,
                                     help="Number of trading days used to calculate rolling signal edge. Longer windows provide more stable estimates but adapt slower to changing market conditions. 63 days ≈ 3 months.")
        min_ev = st.number_input("Min events in window to show edge", min_value=1, max_value=50, value=6, step=1,
                                help="Minimum number of signal events required within the evaluation window to calculate rolling edge. Higher values ensure statistical significance but may create gaps in the analysis.")
        edge_flavor = st.selectbox("Edge flavor", ["Target return", "Excess vs comparison"], index=1,
                                  help="Target return: measure absolute returns on target ticker. Excess vs comparison: measure target returns minus comparison returns (isolates pure target performance from market drift).")
    else:
        # Trade-to-exit mode - use different defaults
        horizon = 5  # Not used but needed for compatibility
        eval_window = 63  # Not used but needed for compatibility
        min_ev = st.number_input("Min events in window to show edge", min_value=1, max_value=50, value=6, step=1,
                                help="Minimum number of events required to compute rolling event-based edge. Higher values ensure statistical significance.")
        events_window = st.number_input(
            "Rolling events window (count)",
            min_value=3, max_value=200, value=20, step=1,
            help="Compute event-based edge as the rolling mean over the last N events."
        )

    build_equity = st.checkbox("Show simple equity curve (long when condition true)", value=True,
                              help="Display equity curves comparing the switching strategy (target vs comparison) against buy-and-hold benchmarks. Shows cumulative performance over time.")
    
    if build_equity:
        tax_rate = st.number_input("Yearly tax rate (%)", min_value=0.0, max_value=50.0, value=20.0, step=0.5,
                                  help="Capital gains tax rate applied at year-end rebalancing.")
    
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
    st.info(f"📅 **Analysis period: {start_date} to {end_date}**")
    
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
prices['fwd_ret_cmp'] = forward_return(prices['close_cmp'], horizon)

# Compute threshold line (absolute or percentile-based)
if signal_mode == "Absolute RSI":
    prices['rsi_thresh'] = rsi_threshold
    thresh_note = f"Using constant RSI threshold = **{rsi_threshold:.1f}**"
else:
    p = np.clip(percentile / 100.0, 0.0, 1.0)
    if perc_scope == "Whole dataset (fixed)":
        # One fixed value from the entire selected period
        if prices['rsi'].notna().sum() < 30:
            st.warning("Too few RSI observations to compute a stable whole-dataset percentile.")
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

# Calculate excess returns (target minus comparison) for signal events
prices['event_excess'] = np.where(prices['signal'], prices['fwd_ret'] - prices['fwd_ret_cmp'], np.nan)

# EOD decision at t -> allocated on day t+1
alloc_bool = prices['signal'].shift(1).fillna(False).astype(bool)

if edge_mode == "Fixed horizon (days)":
    # Same behavior as before
    prices['event_ret'] = np.where(prices['signal'], prices['fwd_ret'], np.nan)
    # Use selected edge flavor
    series_for_edge = prices['event_excess'] if edge_flavor == "Excess vs comparison" else prices['event_ret']
    prices['rolling_edge'] = rolling_signal_edge(
        series_for_edge, window=eval_window, min_events=min_ev
    )
    edge_mode_note = f"Fixed horizon edge over {eval_window} trading days; forward horizon = {horizon}."
    event_df = None
else:
    # Trade-to-exit (event-based)
    event_df = build_event_table(prices, alloc_bool)
    # Rolling event-based edge over last N events (use EXCESS vs comparison by default)
    if not event_df.empty:
        event_df["rolling_event_edge"] = (
            event_df["excess"].rolling(events_window, min_periods=min_ev).mean()
        )
        # Event-level win (default: EXCESS > win_thresh)
        event_df['is_win'] = (event_df['excess'] > win_thresh)
        event_df['rolling_event_wr'] = (
            event_df['is_win'].rolling(events_window, min_periods=min_ev).mean()
        )
    prices['event_ret'] = np.nan   # not used in this mode
    prices['rolling_edge'] = np.nan
    edge_mode_note = f"Event-based edge over last {events_window} events (minimum {min_ev} events)."

if edge_mode == "Fixed horizon (days)":
    prices['rolling_wr'] = rolling_win_rate(
        prices['event_ret'], window=eval_window, min_events=min_ev, win_threshold=win_thresh
    )
else:
    prices['rolling_wr'] = np.nan

# Rolling count of events used in rolling_edge
prices['rolling_event_count'] = prices['signal'].rolling(eval_window, min_periods=1).sum()

# Time in market (allocation ratio) for the equity logic
prices['time_in_market'] = prices['signal'].shift(1).fillna(False).rolling(252).mean()  # ~1y


# Ensure numeric dtype for computed columns
for c in ['rsi','fwd_ret','fwd_ret_cmp','event_ret','event_excess','rolling_edge','rolling_wr']:
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
    st.subheader("Signal Edge")
    st.caption(edge_mode_note)

    if edge_mode == "Fixed horizon (days)":
        # --- rolling-edge (calendar window) plot ---
        plot_df = prices[['rolling_edge']].copy()
        plot_df['rolling_edge'] = pd.to_numeric(plot_df['rolling_edge'], errors='coerce')
        plot_df = plot_df.replace([np.inf, -np.inf], np.nan)

        if plot_df.empty:
            st.info("No data available to plot.")
        else:
            dates = plot_df.index.to_pydatetime()
            vals = plot_df['rolling_edge'].to_numpy(dtype=float)

            fig = go.Figure()
            valid_mask = ~np.isnan(vals)
            if valid_mask.any():
                fig.add_trace(go.Scatter(x=dates[valid_mask], y=vals[valid_mask],
                                         mode='lines', name='Rolling edge'))
            # Dynamic y-axis label based on edge flavor
            y_axis_label = (f'Rolling mean excess (target−{comparison_ticker}) over {horizon}D'
                           if edge_flavor == "Excess vs comparison"
                           else f'Rolling mean of {horizon}D fwd returns (on {target_ticker})')
            
            fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=10), height=420,
                xaxis_title='Date',
                yaxis_title=y_axis_label,
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

    else:
        # --- event-based edge plot (x = event end date) ---
        if event_df is None or event_df.empty:
            st.info("No events detected (no allocated periods). Adjust thresholds or date range.")
        else:
            eplot = event_df[['end', 'rolling_event_edge']].dropna()
            if eplot.empty:
                st.info("Not enough events to compute rolling event-based edge.")
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=eplot['end'], y=eplot['rolling_event_edge'],
                    mode='lines+markers', name='Rolling event-based edge'
                ))
                fig.update_layout(
                    margin=dict(l=10, r=10, t=10, b=10), height=420,
                    xaxis_title='Event end date',
                    yaxis_title=f'Rolling mean excess return (last {events_window} events)',
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

        # Optional: show an event table preview
        with st.expander("Per-event table (entry→exit)"):
            if event_df is not None and not event_df.empty:
                st.dataframe(event_df.assign(
                    start=event_df['start'].dt.date,
                    end=event_df['end'].dt.date,
                    ret_tgt=lambda d: (d['ret_tgt']*100).round(2),
                    ret_cmp=lambda d: (d['ret_cmp']*100).round(2),
                    excess=lambda d: (d['excess']*100).round(2),
                    MFE=lambda d: (d['MFE']*100).round(2),
                    MAE=lambda d: (d['MAE']*100).round(2),
                ).rename(columns={
                    'ret_tgt':'ret_tgt(%)','ret_cmp':'ret_cmp(%)','excess':'excess(%)',
                }))
            else:
                st.info("No events to display.")

    # Win Rate Chart
    st.subheader("Win Rate")

    if edge_mode == "Fixed horizon (days)":
        wr_df = prices[['rolling_wr']].copy()
        wr_df['rolling_wr'] = pd.to_numeric(wr_df['rolling_wr'], errors='coerce')
        wr_df = wr_df.replace([np.inf, -np.inf], np.nan).dropna()

        if wr_df.empty:
            st.info("Not enough data to plot win rate.")
        else:
            fig_wr = go.Figure()
            fig_wr.add_trace(go.Scatter(
                x=wr_df.index, y=wr_df['rolling_wr'], mode='lines', name='Rolling win rate'
            ))

            if show_wr_baseline and prices['event_ret'].notna().any():
                baseline_wr = float((prices['event_ret'] > win_thresh).mean())
                fig_wr.add_hline(y=baseline_wr, line=dict(dash='dash'),
                                 annotation_text=f"Baseline {baseline_wr:.1%}")

            fig_wr.update_layout(
                margin=dict(l=10, r=10, t=10, b=10), height=300,
                xaxis_title='Date', yaxis_title='Win rate', yaxis=dict(range=[0, 1])
            )
            st.plotly_chart(fig_wr, use_container_width=True)

    else:
        # Event-based win rate (x = event end date)
        if event_df is None or event_df.empty or 'rolling_event_wr' not in event_df:
            st.info("Not enough events to plot win rate.")
        else:
            ewr = event_df[['end','rolling_event_wr']].dropna()
            if ewr.empty:
                st.info("Not enough events to plot win rate.")
            else:
                fig_wr = go.Figure()
                fig_wr.add_trace(go.Scatter(
                    x=ewr['end'], y=ewr['rolling_event_wr'], mode='lines+markers',
                    name=f'Rolling win rate (last {events_window} events)'
                ))

                if show_wr_baseline:
                    base_ev_wr = float(event_df['is_win'].mean()) if 'is_win' in event_df else np.nan
                    if np.isfinite(base_ev_wr):
                        fig_wr.add_hline(y=base_ev_wr, line=dict(dash='dash'),
                                         annotation_text=f"Baseline {base_ev_wr:.1%}")

                fig_wr.update_layout(
                    margin=dict(l=10, r=10, t=10, b=10), height=300,
                    xaxis_title='Event end date', yaxis_title='Win rate', yaxis=dict(range=[0, 1])
                )
                st.plotly_chart(fig_wr, use_container_width=True)

    if build_equity:
        st.subheader("Equity Curve: Target vs Comparison")

        # --- Equity Curve: Target vs Comparison (with taxes) ---
        
        # Daily returns (adjusted close pct changes) - ensure 1D arrays
        ret_tgt = prices['close_tgt'].pct_change().values.flatten()
        ret_cmp = prices['close_cmp'].pct_change().values.flatten()
        
        ret_tgt_s = pd.Series(ret_tgt, index=prices.index).fillna(0.0)
        ret_cmp_s = pd.Series(ret_cmp, index=prices.index).fillna(0.0)

        # EOD decision at t-1 → hold on day t
        alloc_bool = prices['signal'].shift(1).fillna(False).to_numpy(dtype=bool)
        strat_ret = pd.Series(np.where(alloc_bool, ret_tgt_s.values, ret_cmp_s.values), index=prices.index)

        # Pre-tax equity
        eq_strat = (1 + strat_ret).cumprod()
        eq_cmp   = (1 + ret_cmp_s).cumprod()

        # Tax-adjust only the strategy
        eq_strat_tax = calculate_tax_adjusted_equity(strat_ret, tax_rate)

        eq_df = pd.DataFrame({
            'Strategy (pre-tax)': eq_strat,
            f'Strategy (Tax {tax_rate:.0f}%)': eq_strat_tax,
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
            
            # Add year-end markers to show tax impact
            year_ends = eq_df.index.to_period("Y").drop_duplicates()
            for year in year_ends:
                # Get the last trading day of each year
                year_data = eq_df[eq_df.index.to_period("Y") == year]
                if not year_data.empty:
                    last_day = year_data.index[-1]
                    # Convert to datetime string for Plotly compatibility
                    last_day_str = last_day.strftime('%Y-%m-%d')
                    fig.add_vline(x=last_day_str, line_dash="dot", line_color="red", opacity=0.1)
            
            fig.update_layout(
                margin=dict(l=10, r=10, t=30, b=10),
                height=420,
                xaxis_title='Date',
                yaxis_title='Equity (normalized)',
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray', showline=True, linewidth=1, linecolor='black'),
                autosize=True,
                width=None
            )
            st.plotly_chart(fig, use_container_width=True)

    # Return Distribution
    st.subheader("Return Distribution")

    # Config inside an expander to avoid sidebar clutter
    with st.expander("Distribution settings", expanded=True):
        # Choice depends on edge mode
        if edge_mode == "Fixed horizon (days)":
            dist_metric = st.selectbox(
                "Metric",
                ["Event forward return (target)", "Event excess (target − comparison)"],
                index=0,
                help="Forward return is over your selected horizon; 'excess' subtracts the comparison's forward return."
            )
            overlay_baseline = st.checkbox(
                "Overlay baseline distribution", value=True,
                help="Baseline = same metric on all days (not just events)."
            )
        else:
            dist_metric = st.selectbox(
                "Metric",
                ["Per-event cumulative return (target)", "Per-event excess (target − comparison)"],
                index=1,
                help="Per-event = cumulative from entry to exit for each contiguous allocation."
            )
            overlay_baseline = False  # not applicable in event-based mode

        bins = st.number_input("Number of bins", min_value=20, max_value=200, value=60, step=5,
                               help="Number of histogram bins for distribution visualization")
        clip_pct = st.number_input(
            "Clip tails for visualization (each side, %)",
            min_value=0.0, max_value=5.0, value=1.0, step=0.1,
            help="Clips extreme tails to make the histogram readable on UVXY-like assets. Data itself is unchanged."
        )

    # Build series for the chosen metric
    if edge_mode == "Fixed horizon (days)":
        # Event samples
        if "forward return" in dist_metric:
            event_vals = prices.loc[prices['signal'], 'fwd_ret']
            base_vals  = prices['fwd_ret']  # all days baseline
            x_label = f"{horizon}D forward return (target: {target_ticker})"
        else:
            all_excess = prices['fwd_ret'] - prices['fwd_ret_cmp']
            event_vals = prices.loc[prices['signal'], 'event_excess']
            base_vals  = all_excess
            x_label = f"{horizon}D excess return (target − {comparison_ticker})"
        # Clip for display only
        ev = _clip_tails(event_vals, clip_pct)
        bv = _clip_tails(base_vals,  clip_pct) if overlay_baseline else None

        if ev.size < 2:
            st.info("Not enough event samples to plot a distribution. Try loosening the threshold or widening dates.")
        else:
            figd = go.Figure()
            figd.add_trace(go.Histogram(
                x=ev, name="Events", nbinsx=bins, histnorm="probability", opacity=0.75
            ))
            if overlay_baseline and bv is not None and bv.size > 2:
                figd.add_trace(go.Histogram(
                    x=bv, name="Baseline (all days)", nbinsx=bins, histnorm="probability", opacity=0.45
                ))
                figd.update_layout(barmode="overlay")

            # Reference lines
            ev_mean = float(np.nanmean(ev)) if ev.size else np.nan
            ev_median = float(np.nanmedian(ev)) if ev.size else np.nan
            figd.add_vline(x=0.0, line=dict(dash="dot"))
            if np.isfinite(ev_mean):
                figd.add_vline(x=ev_mean, line=dict(color="blue", dash="dash"),
                               annotation_text=f"Mean {ev_mean:.2%}", annotation_position="top right")
            if np.isfinite(ev_median):
                figd.add_vline(x=ev_median, line=dict(color="red", dash="dash"),
                               annotation_text=f"Median {ev_median:.2%}", annotation_position="top left")

            figd.update_layout(
                margin=dict(l=10, r=10, t=40, b=10), height=400,
                xaxis_title=x_label, yaxis_title="Probability"
            )
            st.plotly_chart(figd, use_container_width=True)

            # Distribution stats
            with st.expander("Distribution stats"):
                def _stats(x):
                    x = pd.Series(x).replace([np.inf,-np.inf], np.nan).dropna()
                    if x.empty: 
                        return {"N":0}
                    return {
                        "N": int(x.size),
                        "Mean": float(np.mean(x)),
                        "Median": float(np.median(x)),
                        "Std": float(np.std(x, ddof=1)) if x.size>1 else np.nan,
                        "P5": float(np.percentile(x,5)),
                        "P95": float(np.percentile(x,95)),
                    }
                st.write(_stats(ev))
                if overlay_baseline and bv is not None and bv.size>0:
                    st.write({"Baseline": _stats(bv)})

    else:
        # Event-based mode uses event_df
        if event_df is None or event_df.empty:
            st.info("No events yet to plot. Adjust thresholds or date range.")
        else:
            if "cumulative return (target)" in dist_metric:
                vals = event_df['ret_tgt'].to_numpy()
                x_label = "Per-event cumulative return (target)"
            else:
                vals = event_df['excess'].to_numpy()
                x_label = f"Per-event excess (target − {comparison_ticker})"

            vals = _clip_tails(vals, clip_pct)
            if vals.size < 2:
                st.info("Not enough events to plot a distribution.")
            else:
                figd = go.Figure()
                figd.add_trace(go.Histogram(
                    x=vals, name="Events", nbinsx=bins, histnorm="probability", opacity=0.75
                ))
                figd.add_vline(x=0.0, line=dict(dash="dot"))
                mean_v = float(np.nanmean(vals))
                median_v = float(np.nanmedian(vals))
                figd.add_vline(x=mean_v, line=dict(color="blue", dash="dash"),
                               annotation_text=f"Mean {mean_v:.2%}", annotation_position="top right")
                figd.add_vline(x=median_v, line=dict(color="red", dash="dash"),
                               annotation_text=f"Median {median_v:.2%}", annotation_position="top left")

                figd.update_layout(
                    margin=dict(l=10, r=10, t=40, b=10), height=400,
                    xaxis_title=x_label, yaxis_title="Probability"
                )
                st.plotly_chart(figd, use_container_width=True)

                # Distribution stats
                with st.expander("Distribution stats"):
                    def _stats(x):
                        x = pd.Series(x).replace([np.inf,-np.inf], np.nan).dropna()
                        if x.empty: 
                            return {"N":0}
                        return {
                            "N": int(x.size),
                            "Mean": float(np.mean(x)),
                            "Median": float(np.median(x)),
                            "Std": float(np.std(x, ddof=1)) if x.size>1 else np.nan,
                            "P5": float(np.percentile(x,5)),
                            "P95": float(np.percentile(x,95)),
                        }
                    st.write(_stats(vals))

with col2:
    if edge_mode == "Trade-to-exit (event-based)":
        st.subheader("Event-Based Summary")
        st.caption("Statistics for trade-to-exit mode")
        
        if event_df is not None and not event_df.empty:
            total_events = len(event_df)
            winning_events = len(event_df[event_df['excess'] > 0])
            win_rate = winning_events / total_events if total_events > 0 else 0
            
            st.metric("Total events", f"{total_events}")
            st.metric("Winning events", f"{winning_events}")
            st.metric("Win rate", f"{win_rate:.1%}")
            st.metric("Mean excess return", f"{event_df['excess'].mean():.2%}")
            st.metric("Median excess return", f"{event_df['excess'].median():.2%}")
            st.metric("Avg duration", f"{event_df['duration'].mean():.1f} days")
        else:
            st.info("No events detected")
    else:
        st.subheader("Summary")
        st.metric("Total events", f"{total_events}")
        st.metric("Valid rolling-edge points", f"{valid_edge_points}")
        st.metric(f"Median rolling edge ({horizon}D fwd)", f"{edge_median:.4%}" if pd.notna(edge_median) else "—")
        st.metric(f"Mean rolling edge ({horizon}D fwd)", f"{edge_mean:.4%}" if pd.notna(edge_mean) else "—")
        st.metric("Time in market (1y avg)", f"{prices['time_in_market'].iloc[-1]:.1%}" if prices['time_in_market'].notna().any() else "—")

    if edge_mode != "Trade-to-exit (event-based)":
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

    # Win rate display (only for fixed horizon mode)
    if edge_mode == "Fixed horizon (days)":
        st.markdown("---")
        st.subheader("Rolling Win Rate")
        st.caption("Percentage of positive returns in rolling window")
        
        wr_stats = prices['rolling_wr'].dropna()
        if len(wr_stats) > 0:
            st.metric("Mean win rate", f"{wr_stats.mean():.1%}")
            st.metric("Min win rate", f"{wr_stats.min():.1%}")
            st.metric("Max win rate", f"{wr_stats.max():.1%}")
            
            # Simple win rate plot - simplified to avoid rendering issues
            try:
                wr_plot = prices[['rolling_wr']].copy()
                wr_plot['rolling_wr'] = pd.to_numeric(wr_plot['rolling_wr'], errors='coerce')
                wr_plot = wr_plot.replace([np.inf, -np.inf], np.nan)
                
                if not wr_plot.empty and wr_plot['rolling_wr'].notna().any():
                    # Use a simple line chart
                    st.line_chart(wr_plot['rolling_wr'], height=200)
                else:
                    st.info("No win rate data to plot")
            except Exception as e:
                st.info(f"Win rate plot unavailable: {str(e)}")
    else:
        # For event-based mode, win rate is already shown in the Event-Based Summary above
        pass

# -----------------------------
# Data table & downloads
# -----------------------------
with st.expander("Show data / download"):
    show_cols = [
        'close_src', 'close_tgt', 'close_cmp',
        'rsi', 'rsi_thresh', 'signal',
        'fwd_ret', 'event_ret', 'rolling_edge', 'rolling_wr', 'rolling_event_count', 'time_in_market'
    ]
    out = prices[show_cols].copy()
    st.dataframe(out.tail(1000))

    if download_switch:
        # Use full dataset for CSV download, not just the displayed tail(1000)
        full_data = prices[show_cols].copy()
        st.write(f"📊 **CSV will contain {len(full_data)} rows** from {full_data.index.min().date()} to {full_data.index.max().date()}")
        csv = full_data.to_csv(index=True).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{source_ticker}_{target_ticker}_rsi_signal_stability.csv",
            mime='text/csv'
        )

    # Event table download for event-based mode
    if edge_mode == "Trade-to-exit (event-based)" and event_df is not None and not event_df.empty:
        csv_events = event_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download per-event CSV",
            data=csv_events,
            file_name=f"{source_ticker}_{target_ticker}_event_table.csv",
            mime="text/csv"
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
