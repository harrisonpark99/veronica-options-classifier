# -*- coding: utf-8 -*-
"""
Shared options pricing & volatility utilities.
Extracted from pages/1_Option_Classifier.py for reuse across pages.
"""

import numpy as np
import requests
import certifi
import streamlit as st
from math import log, sqrt, exp
from scipy.stats import norm

OKX_BASE_URL = "https://www.okx.com"


# ── Ticker helpers ──────────────────────────────────────────────

def to_okx_inst_id(t: str) -> str:
    """Convert ticker to OKX instId format (e.g. BTCUSDT -> BTC-USDT)."""
    t = t.strip().upper()
    if "-" in t:
        return t
    if t.endswith("USDT") and len(t) > 4:
        return f"{t[:-4]}-USDT"
    return t


# ── OKX data fetching ──────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=300)
def get_close_prices_okx(inst_id: str, bar: str = "1D", target_limit: int = 4500):
    """
    Fetch close prices from OKX with pagination.
    Returns list of float close prices (oldest→newest), or None on error.
    """
    sess = requests.Session()
    all_candles = []
    after = None

    while len(all_candles) < target_limit:
        params = {"instId": inst_id, "bar": bar, "limit": "100"}
        if after is not None:
            params["after"] = str(after)

        url = f"{OKX_BASE_URL}/api/v5/market/history-candles"
        try:
            resp = sess.get(url, params=params, verify=certifi.where(), timeout=10)
        except Exception:
            return None

        if resp.status_code != 200:
            return None

        j = resp.json()
        if j.get("code") != "0":
            return None

        data = j.get("data", [])
        if not data:
            break

        all_candles.extend(data)
        oldest_ts = data[-1][0]
        after = int(oldest_ts)

        if len(data) < 2:
            break

    # Deduplicate & sort ascending
    seen = set()
    uniq = []
    for c in all_candles:
        ts = c[0]
        if ts in seen:
            continue
        seen.add(ts)
        uniq.append(c)

    uniq_sorted = sorted(uniq, key=lambda x: int(x[0]))
    close_prices = [float(c[4]) for c in uniq_sorted if len(c) > 4]

    if not close_prices:
        return None

    return close_prices


@st.cache_data(show_spinner=False, ttl=300)
def get_ohlcv_data_okx(inst_id: str, bar: str = "1D", target_limit: int = 4500):
    """
    Fetch full OHLCV candle data from OKX with pagination.
    Returns list of dicts [{ts, open, high, low, close, volume}, ...] oldest→newest,
    or None on error.
    """
    sess = requests.Session()
    all_candles = []
    after = None

    while len(all_candles) < target_limit:
        params = {"instId": inst_id, "bar": bar, "limit": "100"}
        if after is not None:
            params["after"] = str(after)

        url = f"{OKX_BASE_URL}/api/v5/market/history-candles"
        try:
            resp = sess.get(url, params=params, verify=certifi.where(), timeout=10)
        except Exception:
            return None

        if resp.status_code != 200:
            return None

        j = resp.json()
        if j.get("code") != "0":
            return None

        data = j.get("data", [])
        if not data:
            break

        all_candles.extend(data)
        oldest_ts = data[-1][0]
        after = int(oldest_ts)

        if len(data) < 2:
            break

    # Deduplicate & sort ascending
    seen = set()
    uniq = []
    for c in all_candles:
        ts = c[0]
        if ts in seen:
            continue
        seen.add(ts)
        uniq.append(c)

    uniq_sorted = sorted(uniq, key=lambda x: int(x[0]))

    rows = []
    for c in uniq_sorted:
        if len(c) < 7:
            continue
        rows.append({
            "ts": int(c[0]),
            "open": float(c[1]),
            "high": float(c[2]),
            "low": float(c[3]),
            "close": float(c[4]),
            "volume": float(c[5]),
        })

    return rows if rows else None


# ── Volatility calculations ─────────────────────────────────────

def compute_rolling_volatility(prices, window=30, annualization_factor=365):
    """
    Rolling realized volatility (annualized).
    Default annualization_factor=365 for crypto (24/7 markets).
    """
    vol_list = []
    n = len(prices)
    if n < window:
        return np.array([])
    for i in range(window, n + 1):
        window_prices = np.array(prices[i - window : i])
        log_returns = np.log(window_prices[1:] / window_prices[:-1])
        vol = np.std(log_returns, ddof=1) * sqrt(annualization_factor)
        vol_list.append(vol)
    return np.array(vol_list)


def compute_ema(data, span):
    """Exponential moving average – returns the final EMA value."""
    alpha = 2 / (span + 1)
    ema = data[0]
    for x in data[1:]:
        ema = alpha * x + (1 - alpha) * ema
    return ema


def forecast_volatility(rolling_rv, recent_rv, span=30, beta=0.5):
    """
    EMA + mean-reversion volatility forecast.
    Returns dict with forecast_rv, long_run_mean, ema_recent.
    """
    ema_recent = compute_ema(recent_rv, span=span)
    long_run_mean = float(np.mean(rolling_rv))
    fv = long_run_mean + beta * (ema_recent - long_run_mean)
    return {
        "forecast_rv": float(fv),
        "long_run_mean": long_run_mean,
        "ema_recent": float(ema_recent),
    }


# ── Black-Scholes pricing ───────────────────────────────────────

def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    """Standard Black-Scholes option price."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(0.0, S - K) if option_type.lower() == "call" else max(0.0, K - S)
        return intrinsic

    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type.lower() == "call":
        price = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        price = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return float(price)
