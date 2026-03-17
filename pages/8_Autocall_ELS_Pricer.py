#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VERONICA – Autocall ELS Pricer
Forward pricing, Reverse solver (해찾기), and 2D Sweep for Autocall ELS structures.
Supports single stock and worst-of-2, with VKOSPI-calibrated IV term structure.
"""

import os, sys
import numpy as np
import pandas as pd
import streamlit as st
import time
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.auth import require_auth, show_logout_button

# ═══════════════════════ Page Config ═══════════════════════════
st.set_page_config(
    page_title="Autocall ELS Pricer - VERONICA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
require_auth()


# ═══════════════════════════════════════════════════════════════════
# STOCK DATABASE
# ═══════════════════════════════════════════════════════════════════
STOCKS = {
    "Samsung Electronics": {"vol": 0.6963, "beta": 1.05, "idio_vol": 0.20, "sector": "Technology"},
    "SK Hynix":            {"vol": 0.7644, "beta": 1.30, "idio_vol": 0.25, "sector": "Technology"},
    "Hyundai Motor":       {"vol": 0.9061, "beta": 1.10, "idio_vol": 0.25, "sector": "Consumer Cyclical"},
    "NAVER":               {"vol": 0.5063, "beta": 0.90, "idio_vol": 0.22, "sector": "Communication"},
    "Celltrion":           {"vol": 0.5170, "beta": 0.80, "idio_vol": 0.25, "sector": "Healthcare"},
    "SK Telecom":          {"vol": 0.6335, "beta": 0.65, "idio_vol": 0.20, "sector": "Communication"},
    "EWY (MSCI Korea ETF)": {"vol": 0.6500, "beta": 0.95, "idio_vol": 0.08, "sector": "ETF"},
    "XAUT (Tether Gold)":  {"vol": 0.3000, "beta": 0.00, "idio_vol": 0.30, "sector": "Commodity"},
    "Custom":              {"vol": 0.40,   "beta": 1.00, "idio_vol": 0.20, "sector": "Custom"},
}
USDKRW_VOL = 0.1150
LT_VOL = 0.28            # simple model long-term mean vol
KOSPI200_LT_VOL = 0.22   # KOSPI200 long-term average IV (VKOSPI 2025 avg 24%, FRED ~20%)
VKOSPI_DEFAULT = 0.65    # 2026-03 ~65%
KAPPA_DEFAULT = 6.0       # mean-reversion speed (KDI Heston calibration: 6.09)


# ═══════════════════════════════════════════════════════════════════
# VOL TERM STRUCTURE
# ═══════════════════════════════════════════════════════════════════
def implied_vol(realized_vol: float, tenor_months: int, lt_vol: float = LT_VOL) -> float:
    """(Legacy) Implied vol from realized vol with mean reversion term structure."""
    decay = max(0.05, 0.55 - 0.035 * tenor_months)
    return lt_vol + (realized_vol - lt_vol) * decay


def kospi200_iv(vkospi: float, tenor_months: int,
                lt_vol: float = KOSPI200_LT_VOL, kappa: float = KAPPA_DEFAULT) -> float:
    """KOSPI200 ATM IV at given tenor, anchored to VKOSPI (1m IV).
    σ_K(T) = σ_LT + (VKOSPI − σ_LT) × exp(−κT)
    """
    T = tenor_months / 12.0
    return lt_vol + (vkospi - lt_vol) * np.exp(-kappa * T)


def stock_iv_calibrated(vkospi: float, tenor_months: int,
                        beta: float, idio_vol: float,
                        lt_vol: float = KOSPI200_LT_VOL,
                        kappa: float = KAPPA_DEFAULT) -> float:
    """Individual stock IV from beta decomposition.
    σ_stock(T) = sqrt(β² × σ_KOSPI200(T)² + σ_idio²)
    """
    k_iv = kospi200_iv(vkospi, tenor_months, lt_vol, kappa)
    return np.sqrt(beta**2 * k_iv**2 + idio_vol**2)


# ═══════════════════════════════════════════════════════════════════
# CORE MC ENGINE
# ═══════════════════════════════════════════════════════════════════
def mc_autocall(
    structure: str,       # "single" or "worst_of_2"
    vol1: float,
    vol2: float,          # ignored if single
    correlation: float,   # ignored if single
    risk_free: float,
    apy: float,
    n_obs: int,
    tenor_months: int,
    call_level: float = 1.0,
    knock_in: float = 0.0,       # 0 = no KI
    capital_protection: float = 0.80,
    fx_vol1: float = 0.0,
    fx_corr1: float = 0.0,
    fx_vol2: float = 0.0,
    fx_corr2: float = 0.0,
    N: int = 60000,
    M: int = 200,
) -> dict:
    """
    Unified MC pricer for single / worst-of-2 autocall ELS.
    Returns dict with price, se, and detailed stats.
    """
    tenor_yr = tenor_months / 12.0
    T = tenor_yr
    obs_period = T / n_obs
    premiums = [apy * obs_period * (i + 1) for i in range(n_obs)]
    t_obs = [obs_period * (i + 1) for i in range(n_obs)]
    dt = T / M
    t_grid = np.linspace(0, T, M + 1)
    obs_indices = [np.argmin(np.abs(t_grid - t)) for t in t_obs]

    is_single = (structure == "single")
    n_assets = 1 if is_single else 2
    N_half = N // 2
    N_total = N_half * 2

    if is_single:
        quanto_adj = -fx_vol1 * vol1 * fx_corr1
        drift = (risk_free - 0.5 * vol1**2 + quanto_adj) * dt
        vsd = vol1 * np.sqrt(dt)

        Z = np.random.standard_normal((N_half, M))
        Z = np.concatenate([Z, -Z], axis=0)

        log_perf = np.zeros((N_total, M + 1))
        for t in range(M):
            log_perf[:, t+1] = log_perf[:, t] + drift + vsd * Z[:, t]
        perf = np.exp(log_perf)
        worst = perf  # single stock = itself
    else:
        corr_m = np.array([[1.0, correlation], [correlation, 1.0]])
        L = np.linalg.cholesky(corr_m)
        vols = np.array([vol1, vol2])
        qa = np.array([-fx_vol1*vol1*fx_corr1, -fx_vol2*vol2*fx_corr2])
        drifts = (risk_free + qa - 0.5 * vols**2) * dt
        vsd = vols * np.sqrt(dt)

        Z = np.random.standard_normal((N_half, M, 2)) @ L.T
        Z = np.concatenate([Z, -Z], axis=0)

        log_perf = np.zeros((N_total, M + 1, 2))
        for t in range(M):
            log_perf[:, t+1, :] = log_perf[:, t, :] + drifts + vsd * Z[:, t, :]
        perf = np.exp(log_perf)
        worst = np.min(perf, axis=2)

    # KI: continuous monitoring
    ki_hit = np.zeros(N_total, dtype=bool)
    if knock_in > 0:
        min_worst = np.min(worst, axis=1)
        ki_hit = min_worst < knock_in

    # Autocall processing
    payoffs = np.zeros(N_total)
    knocked_out = np.zeros(N_total, dtype=bool)
    ko_time = np.full(N_total, T)

    for obs_i, step_idx in enumerate(obs_indices):
        wp = worst[:, step_idx]
        trigger = (~knocked_out) & (wp >= call_level)
        t_yr = t_obs[obs_i]
        payoffs[trigger] = (1.0 + premiums[obs_i]) * np.exp(-risk_free * t_yr)
        ko_time[trigger] = t_yr
        knocked_out |= trigger

    # Terminal payoff
    nc = ~knocked_out
    if nc.any():
        dT = np.exp(-risk_free * T)
        fw = worst[:, obs_indices[-1]]

        if knock_in > 0:
            # KI hit → loss (worst perf, no floor unless cap_prot)
            ki_nc = nc & ki_hit
            no_ki_nc = nc & ~ki_hit
            if capital_protection < 1.0:
                payoffs[ki_nc] = np.maximum(capital_protection, fw[ki_nc]) * dT
            else:
                payoffs[ki_nc] = fw[ki_nc] * dT  # full loss
            # No KI hit → maturity redemption with premium
            payoffs[no_ki_nc] = (1.0 + premiums[-1]) * dT
        else:
            # No KI feature
            if capital_protection >= 1.0:
                # 100% protection → par at maturity (discounted)
                payoffs[nc] = 1.0 * dT
            else:
                # Partial protection → downside exposure
                payoffs[nc] = np.maximum(capital_protection, fw[nc]) * dT

    price = np.mean(payoffs) * 100
    se = np.std(payoffs) / np.sqrt(N_total) * 100
    ac_rate = knocked_out.sum() / N_total
    ki_rate = ki_hit.sum() / N_total if knock_in > 0 else 0
    loss_paths = nc & (ki_hit if knock_in > 0 else (worst[:, obs_indices[-1]] < 1.0))
    loss_rate = loss_paths.sum() / N_total
    avg_ko_time = ko_time[knocked_out].mean() if knocked_out.any() else T

    return {
        "price": price,
        "se": se,
        "margin": 100 - price,
        "autocall_pct": ac_rate * 100,
        "ki_hit_pct": ki_rate * 100,
        "loss_pct": loss_rate * 100,
        "avg_ko_months": avg_ko_time * 12,
    }


# ═══════════════════════════════════════════════════════════════════
# BATCH MC ENGINE (경로 1회 생성 → 다중 APY/CapProt 일괄 평가)
# ═══════════════════════════════════════════════════════════════════
def mc_autocall_batch(
    structure, vol1, vol2, correlation, risk_free,
    n_obs, tenor_months,
    call_level=1.0, knock_in=0.0,
    apy_list=None, cap_prot_list=None,
    fx_vol1=0, fx_corr1=0, fx_vol2=0, fx_corr2=0,
    N=20000, M=150,
):
    """Generate MC paths once, evaluate multiple (APY, cap_prot) combos."""
    if apy_list is None:
        apy_list = [0.10, 0.15, 0.20, 0.25]
    if cap_prot_list is None:
        cap_prot_list = [0.90, 0.95, 1.00]

    T = tenor_months / 12.0
    obs_period = T / n_obs
    dt = T / M
    t_grid = np.linspace(0, T, M + 1)
    t_obs = [obs_period * (i + 1) for i in range(n_obs)]
    obs_indices = [np.argmin(np.abs(t_grid - t)) for t in t_obs]

    is_single = (structure == "single")
    N_half = N // 2
    N_total = N_half * 2

    # --- Generate paths ---
    if is_single:
        qa = -fx_vol1 * vol1 * fx_corr1
        drift = (risk_free - 0.5 * vol1**2 + qa) * dt
        vsd = vol1 * np.sqrt(dt)
        Z = np.random.standard_normal((N_half, M))
        Z = np.concatenate([Z, -Z], axis=0)
        log_p = np.zeros((N_total, M + 1))
        for t in range(M):
            log_p[:, t + 1] = log_p[:, t] + drift + vsd * Z[:, t]
        worst = np.exp(log_p)
    else:
        corr_m = np.array([[1.0, correlation], [correlation, 1.0]])
        L = np.linalg.cholesky(corr_m)
        vols = np.array([vol1, vol2])
        qa = np.array([-fx_vol1 * vol1 * fx_corr1, -fx_vol2 * vol2 * fx_corr2])
        drifts = (risk_free + qa - 0.5 * vols**2) * dt
        vsd = vols * np.sqrt(dt)
        Z = np.random.standard_normal((N_half, M, 2)) @ L.T
        Z = np.concatenate([Z, -Z], axis=0)
        log_p = np.zeros((N_total, M + 1, 2))
        for t in range(M):
            log_p[:, t + 1, :] = log_p[:, t, :] + drifts + vsd * Z[:, t, :]
        perf = np.exp(log_p)
        worst = np.min(perf, axis=2)

    # --- KI ---
    ki_hit = np.zeros(N_total, dtype=bool)
    if knock_in > 0:
        ki_hit = np.min(worst, axis=1) < knock_in

    # --- Autocall timing (independent of APY/cap_prot) ---
    knocked_out = np.zeros(N_total, dtype=bool)
    ko_obs = np.full(N_total, -1, dtype=int)
    for obs_i, si in enumerate(obs_indices):
        trigger = (~knocked_out) & (worst[:, si] >= call_level)
        ko_obs[trigger] = obs_i
        knocked_out |= trigger

    nc = ~knocked_out
    fw = worst[:, obs_indices[-1]]
    dT_final = np.exp(-risk_free * T)
    ac_rate = knocked_out.sum() / N_total
    disc = [np.exp(-risk_free * t_obs[i]) for i in range(n_obs)]

    # --- Evaluate for each (apy, cap_prot) ---
    results = []
    for apy in apy_list:
        premiums = [apy * obs_period * (i + 1) for i in range(n_obs)]
        for cp in cap_prot_list:
            payoffs = np.zeros(N_total)
            for obs_i in range(n_obs):
                mask = ko_obs == obs_i
                payoffs[mask] = (1.0 + premiums[obs_i]) * disc[obs_i]
            if nc.any():
                if knock_in > 0:
                    ki_nc = nc & ki_hit
                    no_ki_nc = nc & ~ki_hit
                    if cp < 1.0:
                        payoffs[ki_nc] = np.maximum(cp, fw[ki_nc]) * dT_final
                    else:
                        payoffs[ki_nc] = fw[ki_nc] * dT_final
                    payoffs[no_ki_nc] = (1.0 + premiums[-1]) * dT_final
                else:
                    if cp >= 1.0:
                        payoffs[nc] = 1.0 * dT_final
                    else:
                        payoffs[nc] = np.maximum(cp, fw[nc]) * dT_final
            price = np.mean(payoffs) * 100
            se = np.std(payoffs) / np.sqrt(N_total) * 100
            loss_nc = nc & (ki_hit if knock_in > 0 else (fw < 1.0))
            results.append({
                "apy_pct": round(apy * 100, 1),
                "cap_prot_pct": round(cp * 100, 0),
                "price": round(price, 2),
                "se": round(se, 3),
                "margin": round(100 - price, 2),
                "autocall_pct": round(ac_rate * 100, 1),
                "loss_pct": round(loss_nc.sum() / N_total * 100, 1),
            })
    return results


def _calc_n_obs(tenor_months: int, obs_freq: str) -> int:
    if obs_freq == "monthly":
        return max(1, tenor_months)
    elif obs_freq == "biweekly":
        return max(1, int(tenor_months * 30.44 / 14))
    elif obs_freq == "quarterly":
        return max(1, tenor_months // 3)
    elif obs_freq == "semi-annual":
        return max(1, tenor_months // 6)
    return max(1, tenor_months)


# ═══════════════════════════════════════════════════════════════════
# MULTI-PARAM SWEEP
# ═══════════════════════════════════════════════════════════════════
def sweep_2d(param1_name, param1_values, param2_name, param2_values,
             base_params, N=30000):
    """2D parameter sweep returning price grid."""
    results = []
    for v1 in param1_values:
        for v2 in param2_values:
            params = {**base_params, param1_name: v1, param2_name: v2, "N": N}
            if param1_name == "tenor_months":
                params["n_obs"] = _calc_n_obs(int(v1), base_params.get("obs_freq", "monthly"))
            if param2_name == "tenor_months":
                params["n_obs"] = _calc_n_obs(int(v2), base_params.get("obs_freq", "monthly"))
            params.pop("obs_freq", None)
            r = mc_autocall(**params)
            results.append({param1_name: v1, param2_name: v2, **r})
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════
st.title("Autocall ELS Pricer")

# ─── Sidebar: Structure Setup ───
st.sidebar.header("Structure")
structure = st.sidebar.radio("Type", ["Worst-of-2", "Single Stock"], horizontal=True)
is_single = structure == "Single Stock"

stock_names = list(STOCKS.keys())
stock1_name = st.sidebar.selectbox("Stock 1", stock_names, index=2)  # Hyundai
realized_vol1 = STOCKS[stock1_name]["vol"]
beta1 = STOCKS[stock1_name]["beta"]
idio1 = STOCKS[stock1_name]["idio_vol"]
if stock1_name == "Custom":
    realized_vol1 = st.sidebar.number_input("Stock 1 Realized Vol", 0.10, 2.0, 0.40, 0.05)
    beta1 = st.sidebar.number_input("Stock 1 β", 0.1, 3.0, 1.0, 0.05, key="cb1")
    idio1 = st.sidebar.number_input("Stock 1 σ_idio (%)", 5.0, 100.0, 20.0, 1.0, key="ci1") / 100

if not is_single:
    stock2_name = st.sidebar.selectbox("Stock 2", stock_names, index=1)  # SK Hynix
    realized_vol2 = STOCKS[stock2_name]["vol"]
    beta2 = STOCKS[stock2_name]["beta"]
    idio2 = STOCKS[stock2_name]["idio_vol"]
    if stock2_name == "Custom":
        realized_vol2 = st.sidebar.number_input("Stock 2 Realized Vol", 0.10, 2.0, 0.40, 0.05)
        beta2 = st.sidebar.number_input("Stock 2 β", 0.1, 3.0, 1.0, 0.05, key="cb2")
        idio2 = st.sidebar.number_input("Stock 2 σ_idio (%)", 5.0, 100.0, 20.0, 1.0, key="ci2") / 100
    correlation = st.sidebar.slider("Correlation", 0.0, 0.99, 0.80, 0.05)
else:
    stock2_name = None
    realized_vol2 = 0.0
    beta2 = 1.0
    idio2 = 0.20
    correlation = 0.0

st.sidebar.header("Terms")
tenor_months = st.sidebar.slider("Tenor (months)", 1, 36, 6)
obs_freq = st.sidebar.selectbox("Observation", ["monthly", "biweekly", "quarterly", "semi-annual"])
n_obs = _calc_n_obs(tenor_months, obs_freq)
st.sidebar.caption(f"Observations: {n_obs}")

st.sidebar.header("Vol")
vol_mode = st.sidebar.radio("Vol Model", ["VKOSPI Calibrated", "Mean-Reversion", "Manual"])

# defaults (always defined)
vkospi_val = VKOSPI_DEFAULT
k200_lt_val = KOSPI200_LT_VOL
kappa_val = KAPPA_DEFAULT
lt_vol = LT_VOL

if vol_mode == "VKOSPI Calibrated":
    vkospi_pct = st.sidebar.number_input("VKOSPI (%)", 5.0, 150.0, VKOSPI_DEFAULT * 100, 1.0)
    vkospi_val = vkospi_pct / 100.0
    k200_lt_pct = st.sidebar.number_input("KOSPI200 LT Vol (%)", 10.0, 40.0, KOSPI200_LT_VOL * 100, 1.0)
    k200_lt_val = k200_lt_pct / 100.0
    kappa_val = st.sidebar.number_input("Mean-Rev κ", 0.5, 15.0, KAPPA_DEFAULT, 0.5)

    vol1 = stock_iv_calibrated(vkospi_val, tenor_months, beta1, idio1, k200_lt_val, kappa_val)
    vol2 = stock_iv_calibrated(vkospi_val, tenor_months, beta2, idio2, k200_lt_val, kappa_val) if not is_single else 0.0
    k_iv = kospi200_iv(vkospi_val, tenor_months, k200_lt_val, kappa_val)

    st.sidebar.caption(f"KOSPI200 IV({tenor_months}m): {k_iv*100:.1f}%")
    st.sidebar.caption(f"{stock1_name}: β={beta1} σ_idio={idio1*100:.0f}% → IV={vol1*100:.1f}%")
    if not is_single:
        st.sidebar.caption(f"{stock2_name}: β={beta2} σ_idio={idio2*100:.0f}% → IV={vol2*100:.1f}%")

    with st.sidebar.expander("KOSPI200 IV Term Structure"):
        ts_lines = []
        for t in [1, 3, 6, 9, 12, 18, 24, 36]:
            kiv = kospi200_iv(vkospi_val, t, k200_lt_val, kappa_val)
            ts_lines.append(f"{t}m: {kiv*100:.1f}%")
        st.write(" | ".join(ts_lines))

elif vol_mode == "Mean-Reversion":
    lt_vol = st.sidebar.number_input("Long-term vol", 0.10, 0.60, LT_VOL, 0.02)
    vol1 = implied_vol(realized_vol1, tenor_months, lt_vol)
    vol2 = implied_vol(realized_vol2, tenor_months, lt_vol) if not is_single else 0.0
    st.sidebar.caption(f"IV1: {vol1*100:.1f}% | IV2: {vol2*100:.1f}%")

else:  # Manual
    vol1 = st.sidebar.number_input("Vol 1", 0.05, 1.50, realized_vol1, 0.05)
    vol2 = st.sidebar.number_input("Vol 2", 0.05, 1.50, realized_vol2, 0.05) if not is_single else 0.0

st.sidebar.header("Currency / FX")
currency = st.sidebar.radio("Denomination", ["KRW", "USD"], horizontal=True)
if currency == "KRW":
    fx_vol1, fx_corr1, fx_vol2, fx_corr2 = 0, 0, 0, 0
else:
    fx_mode = st.sidebar.radio("USD asset", ["Stock 2 = USD", "Stock 1 = USD", "Both Foreign"])
    if fx_mode == "Stock 2 = USD":
        fx_vol1, fx_corr1 = USDKRW_VOL, 0.20
        fx_vol2, fx_corr2 = 0.0, 0.0
    elif fx_mode == "Stock 1 = USD":
        fx_vol1, fx_corr1 = 0.0, 0.0
        fx_vol2, fx_corr2 = USDKRW_VOL, 0.20
    else:
        fx_vol1, fx_corr1 = USDKRW_VOL, 0.20
        fx_vol2, fx_corr2 = USDKRW_VOL, 0.20

st.sidebar.header("MC Settings")
mc_paths = st.sidebar.select_slider("Paths", [10000, 20000, 40000, 60000, 100000, 200000], 60000)

# ─── Common params dict ───
base_params = dict(
    structure="single" if is_single else "worst_of_2",
    vol1=vol1, vol2=vol2, correlation=correlation,
    risk_free=0.035 if currency == "KRW" else 0.05,
    n_obs=n_obs, tenor_months=tenor_months,
    fx_vol1=fx_vol1, fx_corr1=fx_corr1,
    fx_vol2=fx_vol2, fx_corr2=fx_corr2,
    obs_freq=obs_freq,
)

risk_free = st.sidebar.number_input("Risk-free rate", 0.0, 0.15, base_params["risk_free"], 0.005)
base_params["risk_free"] = risk_free

# ─── Main Content: Tabs ───
tab1, tab2, tab3 = st.tabs(["Forward Pricer", "Reverse Solver (해찾기)", "Sweep / Heatmap"])

# ═══════════════════════════════════════════════
# TAB 1: FORWARD PRICER
# ═══════════════════════════════════════════════
with tab1:
    st.subheader("Forward Pricer")
    c1, c2, c3, c4 = st.columns(4)
    apy = c1.number_input("APY (%)", 0.0, 200.0, 20.0, 1.0) / 100
    call_level = c2.number_input("Call Level", 0.80, 1.30, 1.00, 0.05)
    knock_in = c3.number_input("Knock-In (0=없음)", 0.0, 0.95, 0.0, 0.05)
    cap_prot = c4.number_input("Capital Protection", 0.0, 1.0, 0.80, 0.05)

    if st.button("Price", type="primary", key="fwd_price"):
        with st.spinner("Running MC..."):
            params = {**base_params, "apy": apy, "call_level": call_level,
                      "knock_in": knock_in, "capital_protection": cap_prot, "N": mc_paths}
            params.pop("obs_freq", None)
            np.random.seed(42)
            result = mc_autocall(**params)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Price", f"{result['price']:.2f}")
        col2.metric("Margin", f"{result['margin']:.2f} pt")
        col3.metric("Autocall %", f"{result['autocall_pct']:.1f}%")
        col4.metric("Loss %", f"{result['loss_pct']:.1f}%")

        st.caption(f"SE: {result['se']:.3f} | KI hit: {result['ki_hit_pct']:.1f}% | Avg KO: {result['avg_ko_months']:.1f}m")

        # Vol info
        st.info(
            f"**Vol:** {vol1*100:.1f}% / {vol2*100:.1f}% | "
            f"**Corr:** {correlation} | **Obs:** {n_obs} ({obs_freq}) | "
            f"**Tenor:** {tenor_months}m | **r:** {risk_free*100:.1f}%"
        )

# ═══════════════════════════════════════════════
# TAB 2: 추천 조합 (해찾기)
# ═══════════════════════════════════════════════
with tab2:
    st.subheader("추천 조합 (해찾기)")
    st.markdown("Target price를 만족하는 최적 ELS 구조를 전 종목 조합에서 자동 탐색합니다.")

    # --- Controls ---
    rc1, rc2, rc3 = st.columns(3)
    target_price = rc1.number_input("Target Price", 80.0, 105.0, 95.0, 0.5, key="rec_target")
    rec_cl = rc2.number_input("Call Level (고정)", 0.80, 1.30, 1.00, 0.05, key="rec_cl")
    rec_ki = rc3.number_input("Knock-In (0=없음)", 0.0, 0.95, 0.0, 0.05, key="rec_ki")

    rc4, rc5 = st.columns(2)
    apy_targets = rc4.multiselect(
        "APY Targets (%)",
        [5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 25.0, 30.0],
        default=[10.0, 15.0, 20.0, 25.0],
        key="rec_apy",
    )
    min_cap_prot = rc5.number_input("Min Capital Protection (%)", 50, 100, 80, 1, key="rec_min_cp")
    min_cp_frac = min_cap_prot / 100.0

    rc6, rc7, rc8 = st.columns(3)
    tenor_min = rc6.number_input("Min Tenor (m)", 1, 36, 3, key="rec_tmin")
    tenor_max = rc7.number_input("Max Tenor (m)", 1, 36, 12, key="rec_tmax")
    rec_corr = rc8.number_input("Correlation", 0.0, 0.99, 0.80, 0.05, key="rec_corr")

    rec_obs = st.selectbox("Observation", ["monthly", "biweekly", "quarterly", "semi-annual"], key="rec_obs")
    include_single = st.checkbox("Include single stocks", False, key="rec_single")

    # Build cap_prot levels
    cp_levels = sorted(set(
        [round(x, 2) for x in np.arange(min_cp_frac, 1.001, 0.02)]
    ))
    # Build tenor levels
    tenor_levels = [t for t in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 18, 24, 36]
                    if tenor_min <= t <= tenor_max]
    if not tenor_levels:
        tenor_levels = [tenor_min]
    apy_list = [a / 100.0 for a in sorted(apy_targets)]

    real_stocks = {k: v for k, v in STOCKS.items() if k != "Custom"}
    pairs = list(itertools.combinations(real_stocks.keys(), 2))
    n_paths_gen = len(pairs) * len(tenor_levels)
    if include_single:
        n_paths_gen += len(real_stocks) * len(tenor_levels)
    n_evals = n_paths_gen * len(apy_list) * len(cp_levels)

    st.caption(
        f"Pairs: {len(pairs)} | Tenors: {tenor_levels} | "
        f"CapProt: {[int(c*100) for c in cp_levels]}% | "
        f"APY: {apy_targets}%"
    )
    st.caption(f"Path generations: {n_paths_gen} | Total evaluations: {n_evals}")

    if st.button("Find Recommendations", type="primary", key="rec_btn"):
        all_results = []
        progress = st.progress(0)
        status = st.empty()
        t0 = time.time()

        total_jobs = len(pairs) + (len(real_stocks) if include_single else 0)
        job_i = 0

        # --- WO-2 pairs ---
        for s1, s2 in pairs:
            rv1, rv2 = real_stocks[s1]["vol"], real_stocks[s2]["vol"]
            for tenor_m in tenor_levels:
                if vol_mode == "VKOSPI Calibrated":
                    iv1 = stock_iv_calibrated(vkospi_val, tenor_m,
                        real_stocks[s1]["beta"], real_stocks[s1]["idio_vol"],
                        k200_lt_val, kappa_val)
                    iv2 = stock_iv_calibrated(vkospi_val, tenor_m,
                        real_stocks[s2]["beta"], real_stocks[s2]["idio_vol"],
                        k200_lt_val, kappa_val)
                elif vol_mode == "Mean-Reversion":
                    iv1 = implied_vol(rv1, tenor_m, lt_vol)
                    iv2 = implied_vol(rv2, tenor_m, lt_vol)
                else:
                    iv1, iv2 = rv1, rv2
                n_obs_t = _calc_n_obs(tenor_m, rec_obs)

                np.random.seed(42)
                batch = mc_autocall_batch(
                    structure="worst_of_2",
                    vol1=iv1, vol2=iv2,
                    correlation=rec_corr,
                    risk_free=risk_free,
                    n_obs=n_obs_t, tenor_months=tenor_m,
                    call_level=rec_cl, knock_in=rec_ki,
                    apy_list=apy_list, cap_prot_list=cp_levels,
                    fx_vol1=fx_vol1, fx_corr1=fx_corr1,
                    fx_vol2=fx_vol2, fx_corr2=fx_corr2,
                    N=mc_paths // 3, M=150,
                )
                for r in batch:
                    r["underlying"] = f"{s1} / {s2}"
                    r["type"] = "WO-2"
                    r["tenor"] = tenor_m
                    r["stock1"] = s1
                    r["stock2"] = s2
                    r["rv1"] = round(rv1 * 100, 1)
                    r["rv2"] = round(rv2 * 100, 1)
                    r["iv1"] = round(iv1 * 100, 1)
                    r["iv2"] = round(iv2 * 100, 1)
                    r["n_obs"] = n_obs_t
                    r["obs_freq_str"] = rec_obs
                all_results.extend(batch)

            job_i += 1
            progress.progress(job_i / total_jobs)
            status.text(f"{s1} / {s2} ({job_i}/{total_jobs}) — {time.time()-t0:.0f}s")

        # --- Single stocks ---
        if include_single:
            for name, info in real_stocks.items():
                rv = info["vol"]
                for tenor_m in tenor_levels:
                    if vol_mode == "VKOSPI Calibrated":
                        iv = stock_iv_calibrated(vkospi_val, tenor_m,
                            real_stocks[name]["beta"], real_stocks[name]["idio_vol"],
                            k200_lt_val, kappa_val)
                    elif vol_mode == "Mean-Reversion":
                        iv = implied_vol(rv, tenor_m, lt_vol)
                    else:
                        iv = rv
                    n_obs_t = _calc_n_obs(tenor_m, rec_obs)

                    np.random.seed(42)
                    batch = mc_autocall_batch(
                        structure="single",
                        vol1=iv, vol2=0, correlation=0,
                        risk_free=risk_free,
                        n_obs=n_obs_t, tenor_months=tenor_m,
                        call_level=rec_cl, knock_in=rec_ki,
                        apy_list=apy_list, cap_prot_list=cp_levels,
                        N=mc_paths // 3, M=150,
                    )
                    for r in batch:
                        r["underlying"] = name
                        r["type"] = "Single"
                        r["tenor"] = tenor_m
                        r["stock1"] = name
                        r["stock2"] = "-"
                        r["rv1"] = round(rv * 100, 1)
                        r["rv2"] = 0
                        r["iv1"] = round(iv * 100, 1)
                        r["iv2"] = 0
                        r["n_obs"] = n_obs_t
                        r["obs_freq_str"] = rec_obs
                    all_results.extend(batch)

                job_i += 1
                progress.progress(job_i / total_jobs)
                status.text(f"{name} ({job_i}/{total_jobs}) — {time.time()-t0:.0f}s")

        progress.progress(1.0)
        elapsed = time.time() - t0
        status.text(f"Done — {elapsed:.0f}s")

        df_all = pd.DataFrame(all_results)
        feasible = df_all[df_all["price"] <= target_price].copy()

        # --- helper: coupon schedule ---
        def _coupon_schedule(apy_pct, tenor_m, n_obs_val):
            apy_dec = apy_pct / 100.0
            tenor_yr = tenor_m / 12.0
            obs_period = tenor_yr / n_obs_val
            return [apy_dec * obs_period * (i + 1) * 100 for i in range(n_obs_val)]

        def _coupon_str(apy_pct, tenor_m, n_obs_val):
            cs = _coupon_schedule(apy_pct, tenor_m, n_obs_val)
            if len(cs) <= 8:
                return " → ".join(f"{c:.2f}%" for c in cs)
            first3 = " → ".join(f"{c:.2f}%" for c in cs[:3])
            return f"{first3} → ... → {cs[-1]:.2f}% ({len(cs)}obs)"

        def _color_margin(val):
            try:
                v = float(val)
            except (ValueError, TypeError):
                return ""
            if v >= 7: return "background-color: #a8d5a2"
            if v >= 5: return "background-color: #d4edda"
            if v >= 3: return "background-color: #fff3cd"
            return ""

        if feasible.empty:
            st.warning(f"price ≤ {target_price}인 조합이 없습니다.")
            st.markdown("Target price를 올리거나, min capital protection을 낮추거나, 더 긴 tenor를 시도하세요.")
            closest = df_all.nsmallest(10, "price")
            st.markdown("**가장 가까운 10개:**")
            st.dataframe(
                closest[["underlying", "tenor", "apy_pct", "cap_prot_pct",
                         "stock1", "rv1", "iv1", "stock2", "rv2", "iv2",
                         "n_obs", "price", "margin", "autocall_pct", "loss_pct"]],
                use_container_width=True, hide_index=True,
            )
        else:
            st.success(f"총 {len(feasible)}개 feasible 조합 (price ≤ {target_price}) — {elapsed:.0f}s")

            # --- Best pick per APY ---
            st.markdown("### Best Pick per APY")
            best_rows = []
            for apy_val in sorted(apy_targets):
                sub = feasible[feasible["apy_pct"] == apy_val]
                if sub.empty:
                    best_rows.append({
                        "APY%": apy_val, "Underlying": "-", "Tenor": "-",
                        "CapProt%": "-", "Stock1 RV→IV": "-", "Stock2 RV→IV": "-",
                        "Price": "-", "Margin": "-", "AC%": "-", "Loss%": "-",
                        "Final Coupon": "-",
                    })
                else:
                    b = sub.loc[sub["margin"].idxmax()]
                    fc = _coupon_schedule(apy_val, int(b["tenor"]), int(b["n_obs"]))
                    best_rows.append({
                        "APY%": apy_val,
                        "Underlying": b["underlying"],
                        "Tenor": f"{int(b['tenor'])}m",
                        "CapProt%": f"{b['cap_prot_pct']:.0f}",
                        "Stock1 RV→IV": f"{b['rv1']:.1f}→{b['iv1']:.1f}%",
                        "Stock2 RV→IV": f"{b['rv2']:.1f}→{b['iv2']:.1f}%",
                        "Price": f"{b['price']:.2f}",
                        "Margin": f"{b['margin']:.2f}",
                        "AC%": f"{b['autocall_pct']:.1f}",
                        "Loss%": f"{b['loss_pct']:.1f}",
                        "Final Coupon": f"{fc[-1]:.2f}%",
                    })
            st.dataframe(pd.DataFrame(best_rows), use_container_width=True, hide_index=True)

            # --- Detailed tables per APY ---
            for apy_val in sorted(apy_targets):
                sub = feasible[feasible["apy_pct"] == apy_val].nlargest(20, "margin")
                if sub.empty:
                    st.markdown(f"### APY {apy_val}% — No feasible combinations")
                    continue

                st.markdown(f"### APY {apy_val}% — Top {len(sub)}")

                # Build display with vol detail + coupon
                rows = []
                for _, r in sub.iterrows():
                    cstr = _coupon_str(apy_val, int(r["tenor"]), int(r["n_obs"]))
                    rows.append({
                        "Underlying": r["underlying"],
                        "Tenor": f"{int(r['tenor'])}m",
                        "#Obs": int(r["n_obs"]),
                        "Stock1": r["stock1"],
                        "RV1→IV1": f"{r['rv1']:.1f}→{r['iv1']:.1f}%",
                        "Stock2": r["stock2"],
                        "RV2→IV2": f"{r['rv2']:.1f}→{r['iv2']:.1f}%",
                        "CapProt%": f"{r['cap_prot_pct']:.0f}",
                        "Price": r["price"],
                        "Margin(pt)": r["margin"],
                        "AC%": r["autocall_pct"],
                        "Loss%": r["loss_pct"],
                        "Coupons": cstr,
                    })

                display = pd.DataFrame(rows)
                display.index = range(1, len(display) + 1)
                styled = display.style.map(_color_margin, subset=["Margin(pt)"])
                st.dataframe(styled, use_container_width=True)

                # Coupon schedule detail per unique (tenor, n_obs)
                with st.expander(f"Coupon Schedule Detail — APY {apy_val}%"):
                    seen = set()
                    for _, r in sub.iterrows():
                        tenor_m = int(r["tenor"])
                        n_obs_r = int(r["n_obs"])
                        key = (tenor_m, n_obs_r)
                        if key in seen:
                            continue
                        seen.add(key)

                        apy_dec = apy_val / 100.0
                        tenor_yr = tenor_m / 12.0
                        obs_period = tenor_yr / n_obs_r

                        coupon_rows = []
                        for i in range(n_obs_r):
                            obs_month = obs_period * (i + 1) * 12
                            cum_prem = apy_dec * obs_period * (i + 1) * 100
                            per_period = apy_dec * obs_period * 100
                            coupon_rows.append({
                                "Obs#": i + 1,
                                "Month": f"{obs_month:.1f}",
                                "Period Coupon": f"{per_period:.3f}%",
                                "Cumulative": f"{cum_prem:.3f}%",
                                "Payoff if KO": f"{100 + cum_prem:.3f}",
                            })

                        st.markdown(f"**{tenor_m}개월 / {n_obs_r}회 관측 ({r['obs_freq_str']})**")
                        st.dataframe(
                            pd.DataFrame(coupon_rows),
                            use_container_width=True, hide_index=True,
                        )

            # --- Download ---
            csv = feasible.to_csv(index=False)
            st.download_button("Download All Results (CSV)", csv,
                               "autocall_recommendations.csv", "text/csv")

# ═══════════════════════════════════════════════
# TAB 3: SWEEP / HEATMAP
# ═══════════════════════════════════════════════
with tab3:
    st.subheader("2D Parameter Sweep")

    sc1, sc2 = st.columns(2)
    sweep_params_options = ["apy", "knock_in", "capital_protection", "call_level", "tenor_months", "correlation"]
    p1 = sc1.selectbox("Axis 1 (rows)", sweep_params_options, index=0)
    p2 = sc2.selectbox("Axis 2 (cols)", sweep_params_options, index=2)

    # Default ranges
    def default_range(p):
        if p == "apy": return "0.05, 0.10, 0.15, 0.20, 0.25, 0.30"
        if p == "knock_in": return "0, 0.50, 0.55, 0.60, 0.65, 0.70"
        if p == "capital_protection": return "0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00"
        if p == "call_level": return "0.90, 0.95, 1.00, 1.05, 1.10"
        if p == "tenor_months": return "1, 2, 3, 4, 5, 6, 9, 12"
        if p == "correlation": return "0.3, 0.5, 0.7, 0.8, 0.9"
        return "0.1, 0.2, 0.3"

    r1_str = sc1.text_input(f"{p1} values", default_range(p1))
    r2_str = sc2.text_input(f"{p2} values", default_range(p2))

    # Fixed params for sweep
    st.markdown("**Fixed parameters for sweep:**")
    fc1, fc2, fc3, fc4 = st.columns(4)
    sw_apy = fc1.number_input("APY %", 0.0, 200.0, 20.0, 1.0, key="sw_apy") / 100
    sw_cl = fc2.number_input("Call Level", 0.80, 1.30, 1.00, 0.05, key="sw_cl")
    sw_ki = fc3.number_input("Knock-In", 0.0, 0.95, 0.0, 0.05, key="sw_ki")
    sw_cp = fc4.number_input("Cap Prot", 0.0, 1.0, 0.80, 0.05, key="sw_cp")

    if st.button("Run Sweep", type="primary", key="sweep_btn"):
        try:
            vals1 = [float(x.strip()) for x in r1_str.split(",")]
            vals2 = [float(x.strip()) for x in r2_str.split(",")]
        except:
            st.error("Invalid value format. Use comma-separated numbers.")
            st.stop()

        sweep_base = {
            **base_params,
            "apy": sw_apy, "call_level": sw_cl,
            "knock_in": sw_ki, "capital_protection": sw_cp,
        }

        with st.spinner(f"Running {len(vals1)}×{len(vals2)} = {len(vals1)*len(vals2)} combinations..."):
            np.random.seed(42)
            df = sweep_2d(p1, vals1, p2, vals2, sweep_base, N=mc_paths // 2)

        # Pivot for heatmap
        pivot = df.pivot_table(index=p1, columns=p2, values="price")

        st.markdown("**Price Heatmap**")

        # Style: green if < 95, red if >= 100
        def color_price(val):
            if val < 95: return "background-color: #a8d5a2"
            elif val < 97: return "background-color: #fff3cd"
            elif val < 100: return "background-color: #ffdcdc"
            else: return "background-color: #f5a5a5"

        styled = pivot.style.format("{:.2f}").map(color_price)
        st.dataframe(styled, use_container_width=True)

        # Margin pivot
        margin_pivot = df.pivot_table(index=p1, columns=p2, values="margin")
        st.markdown("**Margin Heatmap (pt)**")
        def color_margin(val):
            if val >= 5: return "background-color: #a8d5a2"
            elif val >= 3: return "background-color: #fff3cd"
            elif val >= 0: return "background-color: #ffdcdc"
            else: return "background-color: #f5a5a5"
        styled_m = margin_pivot.style.format("{:.1f}").map(color_margin)
        st.dataframe(styled_m, use_container_width=True)

        # Full results table
        with st.expander("Full Results"):
            st.dataframe(df.round(2), use_container_width=True, hide_index=True)

        # CSV download
        csv = df.to_csv(index=False)
        st.download_button("Download CSV", csv, "autocall_sweep.csv", "text/csv")
