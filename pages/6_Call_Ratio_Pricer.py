#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VERONICA – Call Ratio Spread Pricer
Multi-asset 1×N call ratio spread pricing, P&L analysis, Greeks, and historical probabilities.
Supports Crypto (OKX) and Equity (yfinance) assets.
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
    forecast_volatility, get_close_prices_okx, to_okx_inst_id,
)
from utils.okx_api import fetch_okx_ticker_price

# ═══════════════════════ Page Config ═══════════════════════════
st.set_page_config(
    page_title="Call Ratio Pricer - VERONICA",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)
require_auth()


# ═══════════════════════ Helper Functions ═════════════════════

def _bs_greeks(S, K, T, r, sigma, option_type="call"):
    """Analytical BS price + Greeks for a single vanilla option."""
    from math import log as _ln, sqrt as _sq, exp as _ex
    from scipy.stats import norm as _N

    if T <= 0 or sigma <= 0:
        intr = max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
        d = 1.0 if (option_type == "call" and S > K) else (
            -1.0 if (option_type == "put" and S < K) else 0.0)
        return {"price": intr, "delta": d, "gamma": 0.0, "vega": 0.0, "theta": 0.0}

    d1 = (_ln(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * _sq(T))
    d2 = d1 - sigma * _sq(T)
    phi = _N.pdf(d1)

    if option_type == "call":
        price = S * _N.cdf(d1) - K * _ex(-r * T) * _N.cdf(d2)
        delta = _N.cdf(d1)
        theta_ann = -(S * phi * sigma) / (2 * _sq(T)) - r * K * _ex(-r * T) * _N.cdf(d2)
    else:
        price = K * _ex(-r * T) * _N.cdf(-d2) - S * _N.cdf(-d1)
        delta = _N.cdf(d1) - 1.0
        theta_ann = -(S * phi * sigma) / (2 * _sq(T)) + r * K * _ex(-r * T) * _N.cdf(-d2)

    gamma = phi / (S * sigma * _sq(T))
    vega_1pct = S * phi * _sq(T) * 0.01
    theta_day = theta_ann / 365.0

    return {"price": float(price), "delta": float(delta), "gamma": float(gamma),
            "vega": float(vega_1pct), "theta": float(theta_day)}


def _ratio_spread_greeks(S, K1, K2, T, r, sigma, ratio):
    """Portfolio Greeks for long 1 call @ K1, short `ratio` calls @ K2."""
    l1 = _bs_greeks(S, K1, T, r, sigma, "call")
    l2 = _bs_greeks(S, K2, T, r, sigma, "call")
    port = {k: l1[k] - ratio * l2[k] for k in ("price", "delta", "gamma", "vega", "theta")}
    return port, l1, l2


def _ratio_spread_pnl_at_expiry(spot_range, K1, K2, ratio, premium_paid):
    """Vectorised expiry P&L for 1xN call ratio spread."""
    s = np.asarray(spot_range, dtype=float)
    return np.maximum(s - K1, 0) - ratio * np.maximum(s - K2, 0) - premium_paid


def _ratio_spread_breakevens(K1, K2, ratio, premium_paid):
    """Lower BE, Upper BE, Max Profit for a 1xN call ratio spread."""
    lower_be = K1 + premium_paid
    max_profit = (K2 - K1) - premium_paid
    upper_be = None
    if ratio > 1 and max_profit > 0:
        upper_be = K2 + max_profit / (ratio - 1)
    return {"lower_be": lower_be, "upper_be": upper_be,
            "max_profit": max_profit, "max_profit_spot": K2}


def _ratio_historical_prob_zones(prices_arr, spot_ref, K1, K2, ratio, premium_paid, window):
    """Historical probability of landing in each P&L zone using proportional scaling."""
    n = len(prices_arr)
    if n <= window:
        return {"p_loss_below": 0, "p_profit": 0, "p_max_profit_zone": 0,
                "p_loss_above": 0, "total_windows": 0}
    valid = n - window
    entries = prices_arr[:valid]
    finals = prices_arr[window:window + valid]
    k1r, k2r, pr = K1 / spot_ref, K2 / spot_ref, premium_paid / spot_ref
    k1s = entries * k1r
    k2s = entries * k2r
    prems = entries * pr
    pnls = np.maximum(finals - k1s, 0) - ratio * np.maximum(finals - k2s, 0) - prems
    total = float(valid)
    return {
        "p_loss_below": float(np.sum((finals < k1s + prems) & (pnls < 0))) / total,
        "p_profit": float(np.sum(pnls > 0)) / total,
        "p_max_profit_zone": float(np.sum((finals >= k1s + prems) & (finals <= k2s))) / total,
        "p_loss_above": float(np.sum((finals > k2s) & (pnls < 0))) / total,
        "total_windows": valid,
    }


@st.cache_data(show_spinner=False, ttl=300)
def _fetch_equity_data(ticker, hist_days):
    """Fetch equity spot + historical closes via yfinance."""
    import yfinance as yf
    tk = yf.Ticker(ticker)
    hist = tk.history(period=f"{max(hist_days, 30)}d")
    if hist.empty:
        return None, None
    spot = float(hist["Close"].iloc[-1])
    closes = hist["Close"].tolist()
    return spot, closes


# ═══════════════════════ Sidebar ═════════════════════════════
with st.sidebar:
    st.header("VERONICA")
    show_logout_button()
    st.markdown("---")

    data_source = st.radio("Data Source", ["Crypto (OKX)", "Equity (yfinance)"],
                           key="crp_src")
    is_crypto = data_source == "Crypto (OKX)"

    if is_crypto:
        ticker = st.text_input("Ticker", value="BTC-USDT", key="crp_tk",
                               help="OKX format: BTC-USDT, ETH-USDT, SOL-USDT")
    else:
        ticker = st.text_input("Ticker", value="AAPL", key="crp_tk_eq",
                               help="Yahoo Finance ticker: AAPL, NVDA, TSLA")

    hist_days = st.number_input("Historical Days", 60, 4500, 730, key="crp_hd")
    risk_free = st.number_input("Risk-Free Rate (%)", 0.0, 20.0,
                                0.0 if is_crypto else 4.5, 0.1, key="crp_rf") / 100.0

    st.markdown("---")
    if st.button("Clear Cache", key="crp_cc"):
        st.cache_data.clear()
        st.rerun()

# ═══════════════════════ Data Loading ═════════════════════════

st.title("Call Ratio Spread Pricer")
st.caption("1×N call ratio spread pricing with full P&L, Greeks, and historical analysis.")

ann_factor = 365 if is_crypto else 252

with st.spinner(f"Loading {ticker} data ..."):
    if is_crypto:
        inst_id = to_okx_inst_id(ticker)
        spot = fetch_okx_ticker_price(inst_id)
        closes = get_close_prices_okx(inst_id, bar="1D", target_limit=hist_days)
    else:
        spot, closes = _fetch_equity_data(ticker, hist_days)

if spot is None or closes is None or len(closes) < 60:
    st.error(f"Unable to load data for **{ticker}**. Check the ticker and try again.")
    st.stop()

prices_arr = np.asarray(closes, dtype=np.float64)

# Volatility
rolling_rv = compute_rolling_volatility(closes, window=30, annualization_factor=ann_factor)
if len(rolling_rv) == 0:
    st.error("Not enough data to compute volatility.")
    st.stop()

recent_90 = rolling_rv[-90:] if len(rolling_rv) >= 90 else rolling_rv
vol_data = forecast_volatility(rolling_rv, recent_90, span=30, beta=0.5)
rv7_arr = compute_rolling_volatility(closes, window=7, annualization_factor=ann_factor)
rv7 = float(rv7_arr[-1]) if len(rv7_arr) > 0 else 0.0
rv30 = float(rolling_rv[-1])
forecast_vol = vol_data["forecast_rv"]
lr_mean = vol_data["long_run_mean"]

vol_ratio = forecast_vol / lr_mean if lr_mean > 0 else 1.0
vol_regime = "LOW" if vol_ratio < 0.85 else ("HIGH" if vol_ratio > 1.15 else "NORMAL")

# Derive asset symbol for labels
asset_sym = ticker.split("-")[0] if is_crypto else ticker.upper()

# ═══════════════════════ Asset Info ═══════════════════════════

st.markdown("---")
a1, a2, a3, a4, a5 = st.columns(5)
a1.metric(f"{asset_sym} Spot", f"${spot:,.2f}")
a2.metric("7D RV", f"{rv7 * 100:.1f}%")
a3.metric("30D RV", f"{rv30 * 100:.1f}%")
a4.metric("Forecast Vol", f"{forecast_vol * 100:.1f}%")
a5.metric("Vol Regime", vol_regime)

# ═══════════════════════ Trade Parameters ═════════════════════

st.markdown("---")
st.markdown("#### Trade Parameters")
rc1, rc2, rc3 = st.columns(3)

with rc1:
    k1m = st.radio("K1 Input", ["OTM %", "Dollar"], horizontal=True, key="crp_k1m")
    if k1m == "OTM %":
        k1otm = st.number_input("K1 OTM %", 0.0, 500.0, 14.0, 1.0, key="crp_k1o")
        K1 = spot * (1 + k1otm / 100.0)
    else:
        K1 = st.number_input("K1 Strike ($)", 0.0, value=float(round(spot * 1.14, -2 if spot > 100 else 0)),
                              step=1.0 if spot < 100 else 1000.0, key="crp_k1d")
    st.caption(f"K1 = ${K1:,.2f} ({(K1 / spot - 1) * 100:+.1f}%)")

with rc2:
    k2m = st.radio("K2 Input", ["OTM %", "Dollar"], horizontal=True, key="crp_k2m")
    if k2m == "OTM %":
        k2otm = st.number_input("K2 OTM %", 0.0, 500.0, 31.0, 1.0, key="crp_k2o")
        K2 = spot * (1 + k2otm / 100.0)
    else:
        K2 = st.number_input("K2 Strike ($)", 0.0, value=float(round(spot * 1.31, -2 if spot > 100 else 0)),
                              step=1.0 if spot < 100 else 1000.0, key="crp_k2d")
    st.caption(f"K2 = ${K2:,.2f} ({(K2 / spot - 1) * 100:+.1f}%)")

with rc3:
    ratio_n = st.number_input("Short Ratio (N)", 1.0, 10.0, 1.5, 0.1, key="crp_n",
                               help="Sell N calls at K2 per 1 bought at K1")
    ratio_exp = st.number_input("Expiry (days)", 1, 365, 28, key="crp_exp")
    spot_ref = st.number_input("Ref Spot ($)", 0.0, value=float(spot),
                                step=1.0 if spot < 100 else 100.0, key="crp_sref",
                                help="Spot at trade inception")

rp1, rp2, rp3 = st.columns(3)
with rp1:
    pmode = st.radio("Premium", ["BS Theoretical", "Manual Entry"], horizontal=True, key="crp_pm")
with rp2:
    man_prem = st.number_input("Net Premium Paid ($)", 0.0, value=0.0, step=1.0, key="crp_mp",
                                help="Actual net debit. Used only if Manual Entry selected.")
with rp3:
    hold_qty = st.number_input(f"{asset_sym} Holding (qty)", 0.0, 100000.0, 0.0, 0.1, key="crp_hq",
                                help=f"{asset_sym} held as margin. 0 = spread only.")

# ═══════════════════════ Validation & Compute ═════════════════

if K2 <= K1:
    st.error("K2 must be greater than K1.")
    st.stop()
if spot_ref <= 0:
    st.error("Ref Spot must be > 0.")
    st.stop()

T = ratio_exp / 365.0
vol = forecast_vol

pg, lg1, lg2 = _ratio_spread_greeks(spot_ref, K1, K2, T, risk_free, vol, ratio_n)
theo = pg["price"]
prem = max(theo, 0) if pmode == "BS Theoretical" else man_prem

with st.expander("Debug: pricing inputs"):
    st.write(f"spot_ref={spot_ref:,.2f}, K1={K1:,.2f}, K2={K2:,.2f}, T={T:.4f}, vol={vol:.4f}, r={risk_free:.4f}, ratio={ratio_n}")
    st.write(f"Long K1 price=${lg1['price']:,.4f}, Short K2 price=${lg2['price']:,.4f}")
    st.write(f"theo={theo:,.4f}, pmode={pmode}, prem={prem:,.4f}")
be = _ratio_spread_breakevens(K1, K2, ratio_n, prem)

# ═══════════════════════ Structure Summary ════════════════════

st.markdown("---")
st.markdown("#### Structure Summary")
s1, s2, s3, s4 = st.columns(4)
s1.metric("BS Theo Premium", f"${theo:,.2f}")
s2.metric("Premium Used", f"${prem:,.2f}")
s3.metric("Max Profit", f"${be['max_profit']:,.2f}", help=f"At K2 = ${K2:,.0f}")
s4.metric("Upper BE", f"${be['upper_be']:,.0f}" if be["upper_be"] else "N/A")

s5, s6, s7, s8 = st.columns(4)
s5.metric("Lower BE", f"${be['lower_be']:,.0f}")
s6.metric("K1 (Long)", f"${K1:,.2f}")
s7.metric(f"K2 (Short x{ratio_n:.1f})", f"${K2:,.2f}")
s8.metric("Expiry", f"{ratio_exp}D")

# ═══════════════════════ Per-Leg Pricing ══════════════════════

st.markdown("---")
st.markdown("#### Per-Leg BS Pricing")
lc1, lc2 = st.columns(2)
with lc1:
    st.markdown("**Long 1x Call @ K1**")
    st.write(f"- Price: ${lg1['price']:,.2f}")
    st.write(f"- Delta: {lg1['delta']:.4f}")
    st.write(f"- Gamma: {lg1['gamma']:.6f}")
    st.write(f"- Vega (1%): ${lg1['vega']:,.2f}")
    st.write(f"- Theta (/day): ${lg1['theta']:,.2f}")
with lc2:
    st.markdown(f"**Short {ratio_n:.1f}x Call @ K2**")
    st.write(f"- Price (unit): ${lg2['price']:,.2f}")
    st.write(f"- Price (x{ratio_n:.1f}): ${lg2['price'] * ratio_n:,.2f}")
    st.write(f"- Delta (unit): {lg2['delta']:.4f}")
    st.write(f"- Gamma (unit): {lg2['gamma']:.6f}")
    st.write(f"- Vega (1%, unit): ${lg2['vega']:,.2f}")
    st.write(f"- Theta (/day, unit): ${lg2['theta']:,.2f}")

# ═══════════════════════ Portfolio Greeks ═════════════════════

st.markdown("---")
st.markdown("#### Portfolio Greeks")
total_delta = pg['delta'] + hold_qty
g1, g2, g3, g4 = st.columns(4)
g1.metric("Delta (Total)", f"{total_delta:+.4f}",
           help=f"Spread {pg['delta']:+.4f} + {asset_sym} {hold_qty:+.1f}")
g2.metric("Gamma", f"{pg['gamma']:+.6f}")
g3.metric("Vega (1%)", f"${pg['vega']:+,.2f}")
g4.metric("Theta (/day)", f"${pg['theta']:+,.2f}")

# ═══════════════════════ Expiry P&L Chart ════════════════════

st.markdown("---")
st.markdown("#### Expiry P&L Profile")

margin_lo = min(spot_ref, K1) * 0.88
margin_hi = (be["upper_be"] * 1.08) if be["upper_be"] else K2 * 1.25
srange = np.linspace(margin_lo, margin_hi, 500)
spread_pnl = _ratio_spread_pnl_at_expiry(srange, K1, K2, ratio_n, prem)
hold_pnl = hold_qty * (srange - spot_ref)
combined_pnl = spread_pnl + hold_pnl

sp_pct = (srange / spot_ref - 1) * 100
spread_pct = spread_pnl / spot_ref * 100
hold_pct = hold_pnl / spot_ref * 100
comb_pct = combined_pnl / spot_ref * 100

fig = go.Figure()

if hold_qty > 0:
    fig.add_trace(go.Scatter(
        x=sp_pct, y=hold_pct, mode="lines", name=f"{asset_sym} Hold ({hold_qty:.1f})",
        line=dict(color="#FFA726", width=1.5, dash="dot"),
        hovertemplate="%{x:+.1f}% spot<br>" + asset_sym + ": %{y:+.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=sp_pct, y=spread_pct, mode="lines", name="Spread Only",
        line=dict(color="#90CAF9", width=1.5, dash="dash"),
        hovertemplate="%{x:+.1f}% spot<br>Spread: %{y:+.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=sp_pct, y=comb_pct, mode="lines", name="Combined",
        line=dict(color="#1E88E5", width=3),
        hovertemplate="%{x:+.1f}% spot<br>Combined: %{y:+.1f}%<extra></extra>",
    ))
else:
    fig.add_trace(go.Scatter(
        x=sp_pct, y=spread_pct, mode="lines", name="Spread P&L",
        line=dict(color="#1E88E5", width=2.5),
        hovertemplate="%{x:+.1f}% spot<br>P&L: %{y:+.1f}%<extra></extra>",
    ))

fig.add_hline(y=0, line_dash="dash", line_color="gray")

def _pct(v):
    return (v / spot_ref - 1) * 100

fig.add_vline(x=0, line_dash="dash", line_color="orange", annotation_text=f"Ref ${spot_ref:,.0f}")
fig.add_vline(x=_pct(K1), line_dash="dot", line_color="green", annotation_text=f"K1 ${K1:,.0f}")
fig.add_vline(x=_pct(K2), line_dash="dot", line_color="red", annotation_text=f"K2 ${K2:,.0f}")
fig.add_vline(x=_pct(be["lower_be"]), line_dash="dash", line_color="blue",
               annotation_text=f"LBE ${be['lower_be']:,.0f}")
if be["upper_be"]:
    fig.add_vline(x=_pct(be["upper_be"]), line_dash="dash", line_color="crimson",
                   annotation_text=f"UBE ${be['upper_be']:,.0f}")

if hold_qty > 0:
    sign_changes = np.where(np.diff(np.sign(combined_pnl)))[0]
    for sci in sign_changes:
        cbe = np.interp(0, [combined_pnl[sci], combined_pnl[sci + 1]],
                         [srange[sci], srange[sci + 1]])
        fig.add_vline(x=_pct(cbe), line_dash="dashdot", line_color="#7B1FA2",
                       annotation_text=f"CBE ${cbe:,.0f}")

show_pnl_pct = comb_pct if hold_qty > 0 else spread_pct
profit_mask = show_pnl_pct >= 0
if np.any(profit_mask):
    pz_x = sp_pct[profit_mask]
    fig.add_vrect(x0=float(pz_x[0]), x1=float(pz_x[-1]),
                   fillcolor="green", opacity=0.05, line_width=0)

title_suffix = f" + {hold_qty:.1f} {asset_sym} Hold" if hold_qty > 0 else ""
fig.update_layout(
    title=f"1x{ratio_n:.1f} Call Ratio{title_suffix} — Expiry P&L",
    xaxis_title="Spot Move from Ref (%)", yaxis_title="P&L (% of Ref Spot)",
    plot_bgcolor="white", height=550,
    margin=dict(l=40, r=40, t=60, b=40),
    xaxis=dict(ticksuffix="%", dtick=5),
    yaxis=dict(ticksuffix="%"),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════ P&L Table ═══════════════════════════

with st.expander("P&L Table"):
    step = max((K2 - K1) / 8, 1.0 if spot < 100 else 1000.0)
    key_levels = [spot_ref, K1, K2, be["lower_be"]] + ([be["upper_be"]] if be["upper_be"] else [])
    tbl_spots = np.array(sorted(set(
        [float(x) for x in np.arange(K1 * 0.85, margin_hi, step)] + key_levels
    )))
    tbl_spread = _ratio_spread_pnl_at_expiry(tbl_spots, K1, K2, ratio_n, prem)
    tbl_hold = hold_qty * (tbl_spots - spot_ref)
    tbl_comb = tbl_spread + tbl_hold
    tdf_data = {"Spot": [f"${v:,.2f}" for v in tbl_spots],
                "Spread P&L": [f"${v:+,.2f}" for v in tbl_spread]}
    if hold_qty > 0:
        tdf_data[f"{asset_sym} P&L"] = [f"${v:+,.2f}" for v in tbl_hold]
        tdf_data["Combined"] = [f"${v:+,.2f}" for v in tbl_comb]
        tdf_data["Combined %"] = [f"{v / spot_ref * 100:+.2f}%" for v in tbl_comb]
    else:
        tdf_data["P&L %"] = [f"{v / spot_ref * 100:+.2f}%" for v in tbl_spread]
    st.dataframe(pd.DataFrame(tdf_data), use_container_width=True, hide_index=True)

# ═══════════════════════ Delta Evolution ═════════════════════

st.markdown("---")
st.markdown("#### Delta Evolution")
d_spots = np.linspace(K1 * 0.8, K2 * 1.4, 200)
pd_arr, ld_arr, sd_arr = [], [], []
for ds in d_spots:
    dpg, dl1, dl2 = _ratio_spread_greeks(ds, K1, K2, T, risk_free, vol, ratio_n)
    pd_arr.append(dpg["delta"])
    ld_arr.append(dl1["delta"])
    sd_arr.append(-ratio_n * dl2["delta"])

fig_d = go.Figure()
if hold_qty > 0:
    td_arr = [d + hold_qty for d in pd_arr]
    fig_d.add_trace(go.Scatter(x=d_spots, y=td_arr, mode="lines",
                                name=f"Total (+ {asset_sym})", line=dict(color="#1E88E5", width=3)))
    fig_d.add_trace(go.Scatter(x=d_spots, y=pd_arr, mode="lines",
                                name="Spread Only", line=dict(color="#90CAF9", width=1.5, dash="dash")))
else:
    fig_d.add_trace(go.Scatter(x=d_spots, y=pd_arr, mode="lines",
                                name="Portfolio", line=dict(color="#1E88E5", width=2.5)))
fig_d.add_trace(go.Scatter(x=d_spots, y=ld_arr, mode="lines",
                            name="Long K1", line=dict(color="green", width=1.5, dash="dot")))
fig_d.add_trace(go.Scatter(x=d_spots, y=sd_arr, mode="lines",
                            name=f"Short K2 (x{ratio_n:.1f})",
                            line=dict(color="red", width=1.5, dash="dot")))
fig_d.add_hline(y=0, line_dash="dash", line_color="gray")
fig_d.add_vline(x=spot_ref, line_dash="dash", line_color="orange",
                 annotation_text=f"Ref ${spot_ref:,.0f}")
fig_d.add_vline(x=K1, line_dash="dot", line_color="green", annotation_text="K1")
fig_d.add_vline(x=K2, line_dash="dot", line_color="red", annotation_text="K2")
fig_d.update_layout(
    title="Delta vs Spot", xaxis_title=f"{asset_sym} Spot", yaxis_title="Delta",
    plot_bgcolor="white", height=400, hovermode="x unified",
    xaxis=dict(tickformat="$,.0f"),
)
st.plotly_chart(fig_d, use_container_width=True)

# ═══════════════════════ Scenario Analysis ═══════════════════

st.markdown("---")
st.markdown("#### Scenario Analysis (Spot x Vol)")
ss = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
vs = [-10, -5, 0, 5, 10]
base_val = pg["price"]
sc = {}
for v in vs:
    col = []
    for s in ss:
        ns = spot_ref * (1 + s / 100.0)
        nv = max(vol + v / 100.0, 0.01)
        npg, _, _ = _ratio_spread_greeks(ns, K1, K2, T, risk_free, nv, ratio_n)
        spread_chg = npg["price"] - base_val
        hold_chg = hold_qty * (ns - spot_ref)
        col.append(spread_chg + hold_chg)
    sc[f"Vol {v:+d}%pt"] = col
scdf = pd.DataFrame(sc, index=[f"Spot {s:+d}%" for s in ss])
sc_label = f" (incl. {asset_sym} hold)" if hold_qty > 0 else ""
st.caption(f"MTM P&L change from inception{sc_label}")
st.dataframe(scdf.map(lambda x: f"${x:+,.0f}"), use_container_width=True)

# ═══════════════════════ Historical Probability ══════════════

st.markdown("---")
st.markdown("#### Historical Probability")
pz = _ratio_historical_prob_zones(prices_arr, spot_ref, K1, K2, ratio_n, prem, ratio_exp)
h1, h2, h3, h4 = st.columns(4)
h1.metric("P(Profit)", f"{pz['p_profit']:.1%}")
h2.metric("P(Max Profit Zone)", f"{pz['p_max_profit_zone']:.1%}")
h3.metric("P(Loss Below K1)", f"{pz['p_loss_below']:.1%}")
h4.metric("P(Loss Above Upper BE)", f"{pz['p_loss_above']:.1%}" if ratio_n > 1 else "N/A")
st.caption(f"Based on {pz.get('total_windows', 0):,} rolling {ratio_exp}D windows "
           f"from {len(prices_arr):,} days of {asset_sym} history. "
           f"Annualization: {ann_factor} days.")

# ═══════════════════════ Auto-Save to Journal ════════════════

try:
    from utils.memory import save_analysis, get_memory_dir, load_lessons, append_lesson
    _mem_available = get_memory_dir() is not None
except ImportError:
    _mem_available = False

if _mem_available:
    _journal_key = f"crp_saved_{ticker}_{K1}_{K2}_{ratio_n}_{ratio_exp}_{spot_ref}"
    if _journal_key not in st.session_state:
        save_analysis({
            "type": "call_ratio_spread",
            "ticker": ticker,
            "source": data_source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "params": {"K1": K1, "K2": K2, "ratio": ratio_n, "expiry": ratio_exp,
                       "ref_spot": spot_ref, "vol": round(vol, 4), "rf": round(risk_free, 4)},
            "results": {"spot": spot, "premium": round(prem, 2), "theo": round(theo, 2),
                        "lower_be": round(be["lower_be"], 2),
                        "upper_be": round(be["upper_be"], 2) if be.get("upper_be") else None,
                        "max_profit": round(be["max_profit"], 2),
                        "delta": round(pg["delta"], 4),
                        "p_profit": round(pz["p_profit"], 4)},
        })
        st.session_state[_journal_key] = True

# ═══════════════════════ Download & Methodology ══════════════

st.markdown("---")
exp_df = pd.DataFrame({
    "Parameter": ["Asset", "Ticker", "Source", "K1", "K2", "Ratio", "Expiry", "Ref Spot",
                  "Premium Paid", "BS Theo Premium", "Lower BE", "Upper BE", "Max Profit",
                  "Delta", "Gamma", "Vega (1%)", "Theta (/day)",
                  "P(Profit)", "P(Max Profit Zone)", "P(Loss Below)", "P(Loss Above)"],
    "Value": [asset_sym, ticker, data_source, K1, K2, ratio_n, ratio_exp, spot_ref,
              prem, theo, be["lower_be"], be.get("upper_be", "N/A"), be["max_profit"],
              pg["delta"], pg["gamma"], pg["vega"], pg["theta"],
              pz["p_profit"], pz["p_max_profit_zone"], pz["p_loss_below"], pz["p_loss_above"]],
})
st.download_button("Download CSV", data=exp_df.to_csv(index=False).encode("utf-8"),
                   file_name=f"{asset_sym}_ratio_spread.csv", mime="text/csv", key="crp_dl")

with st.expander("Methodology"):
    st.markdown(f"""
**1x{ratio_n:.1f} Call Ratio Spread on {asset_sym}**

Buy 1 call @ K1 (${K1:,.2f}), sell {ratio_n:.1f} calls @ K2 (${K2:,.2f}).

**Payoff at Expiry:**
- Below K1: P&L = -premium = -${prem:,.2f}
- K1 to K2: P&L = (Spot - K1) - premium
- Above K2: P&L = (Spot - K1) - {ratio_n:.1f} x (Spot - K2) - premium

**Key Levels:**
- Lower BE = K1 + premium = **${be['lower_be']:,.2f}**
- Max Profit = (K2 - K1) - premium = **${be['max_profit']:,.2f}** at spot = ${K2:,.2f}
- Upper BE = K2 + max_profit / (N-1) = **{"${:,.0f}".format(be["upper_be"]) if be["upper_be"] else "N/A"}**

**Pricing:** Black-Scholes | Vol: {vol:.1%} (Forecast RV) | r: {risk_free:.1%}
**Data:** {data_source} | Annualization: {ann_factor} days
""")

st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

# ═══════════════════════ Trade Lessons ════════════════════════

if _mem_available:
    with st.expander("Trade Lessons"):
        existing_lessons = load_lessons()
        if existing_lessons.strip():
            st.markdown(existing_lessons)
        st.markdown("---")
        new_lesson = st.text_area("Add a lesson from this analysis:", key="crp_lesson",
                                   placeholder="e.g. HIGH vol regime에서 deep OTM ratio spread premium이 의미있게 나옴")
        if st.button("Save Lesson", key="crp_save_lesson") and new_lesson.strip():
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            append_lesson(f"\n## [{today}] {asset_sym} Call Ratio\n- {new_lesson.strip()}\n")
            st.success("Lesson saved.")
            st.rerun()
