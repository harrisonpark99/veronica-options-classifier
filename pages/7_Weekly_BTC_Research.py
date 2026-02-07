#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VERONICA – Weekly BTC Call Option Research System
Analyzes weekly BTC market trends and recommends optimal call option
strikes to maximize coupon income while minimizing strike-breach risk.
"""

import os, sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import certifi
from datetime import datetime, timezone

# TLS setup
for _v in ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE"):
    os.environ.pop(_v, None)
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.auth import require_auth, show_logout_button
from utils.options import (
    black_scholes_price, compute_rolling_volatility, compute_ema,
    forecast_volatility, get_close_prices_okx, get_ohlcv_data_okx,
)
from utils.okx_api import fetch_okx_ticker_price

# ═══════════════════════ Page Config ═══════════════════════════
st.set_page_config(
    page_title="Weekly BTC Research - VERONICA",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)
require_auth()

# ═══════════════════════ Constants ═════════════════════════════
RISK_FREE_RATE = 0.0
BTC_INST_ID = "BTC-USDT"
OTM_LEVELS = [5, 10, 15, 20, 25, 30]
EXPIRY_OPTIONS = [7, 14, 21, 28, 30, 60, 90]

# ═══════════════════════ Sidebar ══════════════════════════════
with st.sidebar:
    st.header("VERONICA")
    show_logout_button()
    st.markdown("---")
    st.subheader("Research Settings")

    expiry_days = st.selectbox("Expiry Period (days)", EXPIRY_OPTIONS, index=3)
    safety_pct = st.slider(
        "Safety Threshold (P(no-hit) %)", 50, 99, 85, step=5,
        help="Strikes with historical no-hit probability above this are marked 'safe'.",
    )
    backtest_weeks = st.slider("Backtest Lookback (weeks)", 12, 156, 52, step=4)
    hist_days = st.number_input(
        "Historical Data Depth (days)", 180, 4500, 730, step=30,
        help="Days of daily close data used for volatility & backtest.",
    )
    st.markdown("---")
    if st.button("Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache cleared.")


# ═══════════════════════ Helper Functions ══════════════════════

def _load_btc_ohlcv(target_limit: int):
    ohlcv = get_ohlcv_data_okx(BTC_INST_ID, bar="1D", target_limit=target_limit)
    if not ohlcv:
        return None
    df = pd.DataFrame(ohlcv)
    df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_localize(None)
    return df.sort_values("date").reset_index(drop=True)


def _weekly_change(df):
    if len(df) < 8:
        return 0.0, 0.0
    cur, prev = df["close"].iloc[-1], df["close"].iloc[-8]
    return ((cur - prev) / prev) * 100, cur - prev


def _vol_trend_label(rv7, rv30):
    if rv30 == 0:
        return "Stable", "off"
    ratio = rv7 / rv30
    if ratio > 1.10:
        return "Rising", "inverse"
    if ratio < 0.90:
        return "Falling", "normal"
    return "Stable", "off"


def _support_resistance(df):
    rows = []
    for period, label in [(7, "7D"), (14, "14D"), (30, "30D"), (90, "90D")]:
        sub = df.tail(period)
        rows.append({"Level": f"{label} High (Resistance)", "Price (USD)": sub["high"].max(), "Type": "Resistance"})
        rows.append({"Level": f"{label} Low (Support)", "Price (USD)": sub["low"].min(), "Type": "Support"})
    return pd.DataFrame(rows)


def _price_chart(df):
    recent = df.tail(90).copy()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=recent["date"], open=recent["open"], high=recent["high"],
        low=recent["low"], close=recent["close"],
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        name="BTC-USDT",
    ))
    fig.add_hline(y=recent["high"].max(), line_dash="dash", line_color="red",
                  annotation_text=f"90D High: ${recent['high'].max():,.0f}")
    fig.add_hline(y=recent["low"].min(), line_dash="dash", line_color="green",
                  annotation_text=f"90D Low: ${recent['low'].min():,.0f}")
    fig.update_layout(
        title="BTC-USDT Daily Chart (Last 90 Days)",
        yaxis_title="Price (USD)", xaxis_rangeslider_visible=False,
        plot_bgcolor="white", height=500,
        margin=dict(l=40, r=40, t=60, b=40),
        yaxis=dict(tickformat="$,.0f"),
    )
    return fig


# ── Strike Analysis helpers ─────────────────────────────────

def _historical_prob_not_hit(prices_arr, otm_pct, window):
    """Vectorized: fraction of rolling windows where max price stayed below strike."""
    n = len(prices_arr)
    if n <= window:
        return 0.0
    # forward-looking rolling max via reversed cummax trick
    rev = prices_arr[::-1].copy()
    rev_series = pd.Series(rev)
    rev_roll_max = rev_series.rolling(window=window + 1, min_periods=1).max().values
    fwd_max = rev_roll_max[::-1]  # fwd_max[i] = max(prices[i : i+window+1])
    # only valid windows
    valid = n - window
    entry_prices = prices_arr[:valid]
    window_maxes = fwd_max[:valid]
    strikes = entry_prices * (1.0 + otm_pct / 100.0)
    not_hit = np.sum(window_maxes < strikes)
    return float(not_hit) / valid


def _strike_table(spot, vol, expiry, prices_arr, otm_levels):
    T = expiry / 365.0
    rows = []
    for otm in otm_levels:
        strike = spot * (1 + otm / 100.0)
        prem = black_scholes_price(spot, strike, T, RISK_FREE_RATE, vol, "call")
        prem_pct = (prem / spot) * 100 if spot > 0 else 0
        ann_yield = prem_pct * (365.0 / expiry) if expiry > 0 else 0
        p_no = _historical_prob_not_hit(prices_arr, otm, expiry)
        if p_no >= 0.90:
            risk = "Low"
        elif p_no >= 0.75:
            risk = "Medium"
        else:
            risk = "High"
        rows.append({
            "OTM %": otm,
            "Strike (USD)": strike,
            "BS Premium (USD)": prem,
            "Premium %": prem_pct,
            "Ann. Yield %": ann_yield,
            "P(No-Hit)": p_no,
            "Risk": risk,
        })
    return pd.DataFrame(rows)


def _find_sweet_spot(sdf, threshold):
    safe = sdf[sdf["P(No-Hit)"] >= threshold]
    if safe.empty:
        return None
    return safe["Ann. Yield %"].idxmax()


# ── Backtest helpers ────────────────────────────────────────

def _run_backtest(prices_arr, otm_pct, expiry, num_weeks, vol):
    n = len(prices_arr)
    T = expiry / 365.0
    results = []
    for w in range(num_weeks):
        idx = n - 1 - (w * 7)
        if idx < 0 or idx + expiry >= n:
            continue
        entry = prices_arr[idx]
        strike = entry * (1 + otm_pct / 100.0)
        prem = black_scholes_price(entry, strike, T, RISK_FREE_RATE, vol, "call")
        prem_pct = (prem / entry) * 100 if entry > 0 else 0
        window_max = float(np.max(prices_arr[idx : idx + expiry + 1]))
        hit = window_max >= strike
        loss_pct = ((window_max - strike) / entry * 100) if hit else 0.0
        results.append({
            "Week #": num_weeks - w,
            "Entry Price": entry,
            "Strike": strike,
            "Premium (%)": prem_pct,
            "Window Max": window_max,
            "Hit": hit,
            "Coupon (%)": prem_pct,
            "Breach Loss (%)": loss_pct,
            "Net (%)": prem_pct - loss_pct,
        })
    return pd.DataFrame(results).sort_values("Week #").reset_index(drop=True)


def _bt_timeline_chart(bt):
    colors = ["#26a69a" if not h else "#ef5350" for h in bt["Hit"]]
    fig = go.Figure(go.Bar(
        x=bt["Week #"], y=bt["Net (%)"], marker_color=colors,
        hovertemplate="Week %{x}<br>Net: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Weekly Net P&L (Coupon - Breach Loss)",
        xaxis_title="Week #", yaxis_title="Net (%)",
        plot_bgcolor="white", height=400,
    )
    return fig


def _bt_cumulative_chart(bt):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bt["Week #"], y=bt["Coupon (%)"].cumsum(),
        mode="lines", name="Cumulative Coupon (%)",
        line=dict(color="#26a69a", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=bt["Week #"], y=bt["Breach Loss (%)"].cumsum(),
        mode="lines", name="Cumulative Breach Loss (%)",
        line=dict(color="#ef5350", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=bt["Week #"], y=bt["Net (%)"].cumsum(),
        mode="lines+markers", name="Cumulative Net (%)",
        line=dict(color="#1E88E5", width=2.5),
    ))
    fig.update_layout(
        title="Cumulative P&L Analysis",
        xaxis_title="Week #", yaxis_title="Cumulative (%)",
        plot_bgcolor="white", height=400, hovermode="x unified",
    )
    return fig


# ── Recommendation helpers ──────────────────────────────────

def _vol_regime(fv, lr_mean):
    if lr_mean == 0:
        return "NORMAL"
    ratio = fv / lr_mean
    if ratio < 0.85:
        return "LOW"
    if ratio > 1.15:
        return "HIGH"
    return "NORMAL"


def _generate_recommendation(spot, vol_data, strike_df, rv7, rv30, btc_df, threshold):
    regime = _vol_regime(vol_data["forecast_rv"], vol_data["long_run_mean"])
    warnings = []

    if rv30 > 0 and rv7 / rv30 > 1.10:
        warnings.append("Volatility is trending UP. Consider a more conservative (higher OTM) strike.")

    if btc_df is not None and len(btc_df) > 90:
        ath = btc_df["high"].max()
        if spot >= ath * 0.95:
            warnings.append(f"BTC is within 5% of its all-time high (${ath:,.0f}). Breakout risk is elevated.")

    safe = strike_df[strike_df["P(No-Hit)"] >= threshold].copy()

    if regime == "LOW":
        reasoning = "Volatility is below average — tighter strikes capture higher coupons with acceptable risk."
        selected = safe.iloc[0] if not safe.empty else strike_df.iloc[0]
        if safe.empty:
            warnings.append("No strike meets the safety threshold. Showing the lowest OTM available.")
    elif regime == "HIGH":
        reasoning = "Volatility is elevated — wider strikes prioritize safety over yield."
        selected = safe.iloc[-1] if not safe.empty else strike_df.iloc[-1]
        if safe.empty:
            warnings.append("No strike meets the safety threshold. Showing the highest OTM available.")
    else:
        reasoning = "Volatility is near its long-run average — balancing yield and safety."
        if not safe.empty:
            selected = safe.loc[safe["Ann. Yield %"].idxmax()]
        else:
            selected = strike_df.loc[strike_df["Ann. Yield %"].idxmax()]
            warnings.append("No strike meets the safety threshold.")

    return {
        "otm": selected["OTM %"],
        "strike": selected["Strike (USD)"],
        "prem_pct": selected["Premium %"],
        "ann_yield": selected["Ann. Yield %"],
        "safety": selected["P(No-Hit)"] * 100,
        "risk": selected["Risk"],
        "regime": regime,
        "reasoning": reasoning,
        "warnings": warnings,
    }


# ═══════════════════════ Data Loading ═════════════════════════

st.title("Weekly BTC Call Option Research")
st.caption("Maximize coupon income by finding strikes that are unlikely to be breached.")

with st.spinner("Loading BTC market data ..."):
    btc_spot = fetch_okx_ticker_price(BTC_INST_ID)
    btc_df = _load_btc_ohlcv(target_limit=max(hist_days, 730))
    close_prices = btc_df["close"].values if btc_df is not None else None

if btc_spot is None or close_prices is None or len(close_prices) < 60:
    st.error("Unable to load BTC data from OKX. Please try again later.")
    st.stop()

prices_arr = np.asarray(close_prices, dtype=np.float64)

# Pre-compute volatility (annualization=365 for crypto)
rolling_rv = compute_rolling_volatility(close_prices.tolist(), window=30, annualization_factor=365)
if len(rolling_rv) == 0:
    st.error("Not enough data to compute volatility. Increase the historical data depth.")
    st.stop()

recent_rv_90 = rolling_rv[-90:] if len(rolling_rv) >= 90 else rolling_rv
vol_data = forecast_volatility(rolling_rv, recent_rv_90, span=30, beta=0.5)

rv_7d_arr = compute_rolling_volatility(close_prices.tolist(), window=7, annualization_factor=365)
rv_7d_last = float(rv_7d_arr[-1]) if len(rv_7d_arr) > 0 else 0.0
rv_30d_last = float(rolling_rv[-1])

# Build strike table (shared across tabs)
strike_df = _strike_table(btc_spot, vol_data["forecast_rv"], expiry_days, prices_arr, OTM_LEVELS)
safety_threshold = safety_pct / 100.0

# ═══════════════════════ Tabs ═════════════════════════════════

tab_overview, tab_strike, tab_bt, tab_rec = st.tabs([
    "Market Overview", "Strike Analysis", "Historical Backtest", "Weekly Recommendation",
])

# ─── Tab 1: Market Overview ────────────────────────────────────
with tab_overview:
    st.subheader("Weekly Market Overview")

    pct_chg, abs_chg = _weekly_change(btc_df)
    trend_label, trend_delta = _vol_trend_label(rv_7d_last, rv_30d_last)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BTC Spot", f"${btc_spot:,.2f}")
    c2.metric("Weekly Change", f"{pct_chg:+.2f}%", delta=f"${abs_chg:+,.0f}")
    c3.metric("30D Realized Vol", f"{rv_30d_last:.2%}")
    c4.metric("7D Realized Vol", f"{rv_7d_last:.2%}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Vol Trend", trend_label, delta_color=trend_delta)
    c6.metric("Forecast Vol", f"{vol_data['forecast_rv']:.2%}")
    c7.metric("30D High", f"${btc_df['high'].tail(30).max():,.0f}")
    c8.metric("30D Low", f"${btc_df['low'].tail(30).min():,.0f}")

    st.plotly_chart(_price_chart(btc_df), use_container_width=True)

    st.subheader("Support & Resistance Levels")
    sr_df = _support_resistance(btc_df)
    sr_display = sr_df.copy()
    sr_display["Price (USD)"] = sr_display["Price (USD)"].map("${:,.0f}".format)
    st.dataframe(sr_display, use_container_width=True, hide_index=True)

# ─── Tab 2: Strike Analysis ───────────────────────────────────
with tab_strike:
    st.subheader("Strike Analysis")

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Current Spot", f"${btc_spot:,.2f}")
    mc2.metric("Forecast Volatility", f"{vol_data['forecast_rv']:.2%}")
    mc3.metric("Expiry", f"{expiry_days} days")

    sweet_idx = _find_sweet_spot(strike_df, safety_threshold)
    if sweet_idx is not None:
        sr = strike_df.loc[sweet_idx]
        st.success(
            f"**Sweet Spot: {sr['OTM %']}% OTM** — "
            f"Strike ${sr['Strike (USD)']:,.0f}  |  "
            f"Ann. Yield {sr['Ann. Yield %']:.2f}%  |  "
            f"Safety {sr['P(No-Hit)']:.1%}"
        )
    else:
        st.warning(f"No strike meets the {safety_pct}% safety threshold with current data.")

    disp = strike_df.copy()
    disp["OTM %"] = disp["OTM %"].map("{:.0f}%".format)
    disp["Strike (USD)"] = disp["Strike (USD)"].map("${:,.0f}".format)
    disp["BS Premium (USD)"] = disp["BS Premium (USD)"].map("${:,.2f}".format)
    disp["Premium %"] = disp["Premium %"].map("{:.3f}%".format)
    disp["Ann. Yield %"] = disp["Ann. Yield %"].map("{:.2f}%".format)
    disp["P(No-Hit)"] = disp["P(No-Hit)"].map("{:.1%}".format)
    st.dataframe(disp, use_container_width=True, hide_index=True)

    csv_bytes = strike_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV", data=csv_bytes,
        file_name=f"btc_strike_analysis_{expiry_days}d.csv", mime="text/csv",
    )

# ─── Tab 3: Historical Backtest ───────────────────────────────
with tab_bt:
    st.subheader("Historical Backtest")

    bt_col1, bt_col2 = st.columns([1, 3])
    with bt_col1:
        bt_otm = st.selectbox("OTM % to Backtest", OTM_LEVELS, index=2)
    with bt_col2:
        st.info(f"Simulating weekly entry of a **{bt_otm}% OTM** call with **{expiry_days}-day** expiry over the last **{backtest_weeks} weeks**.")

    bt_df = _run_backtest(prices_arr, bt_otm, expiry_days, backtest_weeks, vol_data["forecast_rv"])

    if bt_df.empty:
        st.warning("Not enough historical data for the selected backtest window.")
    else:
        total_w = len(bt_df)
        wins = int((~bt_df["Hit"]).sum())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Win Rate (No-Hit)", f"{wins / total_w:.1%}" if total_w else "N/A")
        m2.metric("Avg Coupon / Week", f"{bt_df['Coupon (%)'].mean():.3f}%")
        m3.metric("Total Net P&L", f"{bt_df['Net (%)'].sum():.2f}%")
        m4.metric("Worst Week", f"{bt_df['Net (%)'].min():.2f}%")

        st.plotly_chart(_bt_timeline_chart(bt_df), use_container_width=True)
        st.plotly_chart(_bt_cumulative_chart(bt_df), use_container_width=True)

        with st.expander("Detailed Backtest Data"):
            bt_disp = bt_df.copy()
            for c in ["Entry Price", "Strike", "Window Max"]:
                bt_disp[c] = bt_disp[c].map("${:,.0f}".format)
            for c in ["Premium (%)", "Coupon (%)", "Breach Loss (%)", "Net (%)"]:
                bt_disp[c] = bt_disp[c].map("{:.3f}%".format)
            bt_disp["Hit"] = bt_disp["Hit"].map({True: "YES", False: "no"})
            st.dataframe(bt_disp, use_container_width=True, hide_index=True)

        bt_csv = bt_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Backtest CSV", data=bt_csv,
            file_name=f"btc_backtest_{bt_otm}otm_{expiry_days}d.csv", mime="text/csv",
        )

# ─── Tab 4: Weekly Recommendation ─────────────────────────────
with tab_rec:
    st.subheader("Weekly Recommendation")

    rec = _generate_recommendation(
        btc_spot, vol_data, strike_df, rv_7d_last, rv_30d_last, btc_df, safety_threshold,
    )

    regime_map = {"LOW": ("LOW", "green"), "NORMAL": ("NORMAL", "orange"), "HIGH": ("HIGH", "red")}
    regime_label, regime_color = regime_map.get(rec["regime"], ("NORMAL", "orange"))
    st.info(f"**Volatility Regime: {regime_label}** — {rec['reasoning']}")

    st.markdown("---")

    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Recommended Strike", f"{rec['otm']}% OTM")
    rc2.metric("Strike Price", f"${rec['strike']:,.0f}")
    rc3.metric("Expected Coupon", f"{rec['prem_pct']:.3f}%")
    rc4.metric("Annualized Yield", f"{rec['ann_yield']:.2f}%")

    rc5, rc6 = st.columns(2)
    rc5.metric("Historical Safety", f"{rec['safety']:.1f}%")
    rc6.metric("Risk Level", rec["risk"])

    if rec["warnings"]:
        st.markdown("---")
        st.markdown("### Risk Warnings")
        for w in rec["warnings"]:
            st.warning(w)

    with st.expander("Methodology"):
        st.markdown(f"""
**How the recommendation is generated:**

1. **Volatility Regime** — The forecasted volatility (EMA + mean-reversion model) is compared to its
   long-run average. If it is >15% above average → HIGH; >15% below → LOW; otherwise NORMAL.

2. **Strike Selection:**
   - **LOW regime:** Select the *lowest* OTM% that still meets the safety threshold (maximize coupon).
   - **HIGH regime:** Select the *highest* OTM% that still meets the threshold (maximize safety).
   - **NORMAL regime:** Select the OTM% with the *highest annualized yield* among safe strikes.

3. **Safety Threshold:** Currently set to **{safety_pct}%** — meaning the strike must *not* have been
   breached in at least {safety_pct}% of all historical rolling windows.

4. **Risk Warnings** fire when:
   - 7-day RV exceeds 30-day RV by >10% (rising vol trend)
   - BTC is within 5% of its all-time high (breakout risk)

**Model Inputs:**
- Forecast Vol: **{vol_data['forecast_rv']:.2%}** (Long-run mean: {vol_data['long_run_mean']:.2%}, Recent EMA: {vol_data['ema_recent']:.2%})
- Annualization: 365 days (crypto markets trade 24/7)
- Risk-free rate: 0%
""")

    st.markdown("---")
    st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
