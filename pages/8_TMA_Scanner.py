#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VERONICA – TMA Scanner (Enhanced: Wyckoff/LarryW/Bulkowski/Livermore)
Scans Dow 30, S&P 100, Nasdaq 100, S&P 500
for Technical Merit Analysis (TMA) scores and momentum rankings.
"""

import os, sys, math, time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import certifi

# TLS setup
for _v in ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE"):
    os.environ.pop(_v, None)
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.auth import require_auth, show_logout_button

# ═══════════════════════ Page Config ═══════════════════════════
st.set_page_config(
    page_title="TMA Scanner - VERONICA",
    page_icon="📌",
    layout="wide",
    initial_sidebar_state="expanded",
)
require_auth()

# ═══════════════════════ Config / Utils ════════════════════════
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)

YAHOO_TICKER_MAP = {
    "BRK.B": "BRK-B",
    "BRK.A": "BRK-A",
    "BF.B": "BF-B",
}


def to_yahoo_ticker(t: str) -> str:
    t = t.strip().upper()
    return YAHOO_TICKER_MAP.get(t, t)


def from_yahoo_ticker(t: str) -> str:
    inv = {v: k for k, v in YAHOO_TICKER_MAP.items()}
    return inv.get(t, t)


def safe_read_html(url: str) -> List[pd.DataFrame]:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
    r.raise_for_status()
    return pd.read_html(r.text)


def zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd


def minmax01(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mn, mx = s.min(), s.max()
    if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)


def pct_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True)


# ═══════════════════════ Universe Fetchers ═════════════════════

@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_dow30() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
    tables = safe_read_html(url)
    for df in tables:
        cols = [c.lower() for c in df.columns.astype(str)]
        if any("symbol" in c for c in cols) and any("company" in c for c in cols):
            out = df.copy()
            out.columns = [str(c) for c in out.columns]
            sym_col = [c for c in out.columns if "Symbol" in c or "symbol" in c][0]
            name_col = [c for c in out.columns if "Company" in c or "company" in c][0]
            out = out[[name_col, sym_col]].rename(columns={name_col: "Name", sym_col: "Ticker"})
            out["Ticker"] = out["Ticker"].astype(str).str.strip().str.upper()
            out = out[out["Ticker"].str.len() > 0]
            out["Universe"] = "Dow 30"
            return out.reset_index(drop=True)
    raise ValueError("Dow 30 components table not found (Wikipedia layout changed).")


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_sp100() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/S%26P_100"
    tables = safe_read_html(url)
    for df in tables:
        cols = [c.lower() for c in df.columns.astype(str)]
        if any(c in ["symbol", "ticker"] for c in cols) and any("name" in c for c in cols):
            out = df.copy()
            sym_col = [c for c in out.columns if str(c).lower() in ["symbol", "ticker"]][0]
            name_col = [c for c in out.columns if "name" in str(c).lower()][0]
            out = out[[name_col, sym_col]].rename(columns={name_col: "Name", sym_col: "Ticker"})
            out["Ticker"] = out["Ticker"].astype(str).str.strip().str.upper()
            out = out[out["Ticker"].str.len() > 0]
            out["Universe"] = "S&P 100"
            return out.reset_index(drop=True)
    raise ValueError("S&P 100 components table not found (Wikipedia layout changed).")


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_nasdaq100() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = safe_read_html(url)
    for df in tables:
        cols = [c.lower() for c in df.columns.astype(str)]
        if any("ticker" in c for c in cols) and any("company" in c for c in cols):
            out = df.copy()
            tick_col = [c for c in out.columns if "Ticker" in str(c) or "ticker" in str(c)][0]
            comp_col = [c for c in out.columns if "Company" in str(c) or "company" in str(c)][0]
            out = out[[comp_col, tick_col]].rename(columns={comp_col: "Name", tick_col: "Ticker"})
            out["Ticker"] = out["Ticker"].astype(str).str.strip().str.upper()
            out = out[out["Ticker"].str.len() > 0]
            out["Universe"] = "Nasdaq 100"
            return out.reset_index(drop=True)
    raise ValueError("Nasdaq-100 components table not found (Wikipedia layout changed).")


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_sp500() -> pd.DataFrame:
    """S&P 500 components from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = safe_read_html(url)
    for df in tables:
        cols = [c.lower() for c in df.columns.astype(str)]
        if any("symbol" in c for c in cols) and any("security" in c for c in cols):
            out = df.copy()
            out.columns = [str(c) for c in out.columns]
            sym_col = [c for c in out.columns if "Symbol" in c or "symbol" in c][0]
            name_col = [c for c in out.columns if "Security" in c or "security" in c][0]
            out = out[[name_col, sym_col]].rename(columns={name_col: "Name", sym_col: "Ticker"})
            out["Ticker"] = out["Ticker"].astype(str).str.strip().str.upper()
            out = out[out["Ticker"].str.len() > 0]
            out["Universe"] = "S&P 500"
            return out.reset_index(drop=True)
    raise ValueError("S&P 500 components table not found (Wikipedia layout changed).")


# ═══════════════════════ Data Download (OHLCV) ═════════════════

@st.cache_data(ttl=60 * 30, show_spinner=False)
def download_yahoo(tickers: List[str], period: str = "2y"):
    """Returns (adj_close, volume, open, high, low, close) DataFrames."""
    yt = [to_yahoo_ticker(t) for t in tickers]
    data = yf.download(
        tickers=yt,
        period=period,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if data is None or data.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty, empty

    def _extract(field, fallback=None):
        if isinstance(data.columns, pd.MultiIndex):
            if field in data.columns.get_level_values(0):
                return data[field].copy()
            if fallback and fallback in data.columns.get_level_values(0):
                return data[fallback].copy()
            return pd.DataFrame(index=data.index)
        else:
            if field in data.columns:
                return data[[field]].rename(columns={field: yt[0]})
            if fallback and fallback in data.columns:
                return data[[fallback]].rename(columns={fallback: yt[0]})
            return pd.DataFrame(index=data.index)

    px = _extract("Adj Close", "Close")
    vol = _extract("Volume")
    op = _extract("Open")
    hi = _extract("High")
    lo = _extract("Low")
    cl = _extract("Close")

    def remap(df):
        df = df.copy()
        df.columns = [from_yahoo_ticker(str(c)) for c in df.columns]
        return df.sort_index()

    return remap(px), remap(vol), remap(op), remap(hi), remap(lo), remap(cl)


# ═══════════════════════ Momentum Scoring ══════════════════════

def compute_momentum_scores(px: pd.DataFrame, method: str) -> pd.DataFrame:
    if px.empty:
        return pd.DataFrame()

    valid_ratio = 1.0 - px.isna().mean()
    keep = valid_ratio[valid_ratio >= 0.85].index.tolist()
    px = px[keep].ffill().dropna(how="all")

    if px.shape[1] == 0 or px.shape[0] < 80:
        return pd.DataFrame()

    retd = px.pct_change().dropna()

    def total_return(days: int) -> pd.Series:
        if len(px) <= days:
            return pd.Series(np.nan, index=px.columns)
        return (px.iloc[-1] / px.iloc[-days - 1]) - 1.0

    r_1m = total_return(21)
    r_3m = total_return(63)
    r_6m = total_return(126)
    r_12m = total_return(252)

    vol_6m = retd.tail(126).std() * math.sqrt(252)

    df = pd.DataFrame({
        "Ret_1M": r_1m,
        "Ret_3M": r_3m,
        "Ret_6M": r_6m,
        "Ret_12M": r_12m,
        "Vol_6M_ann": vol_6m,
    })
    df["Mom_12_1"] = df["Ret_12M"] - df["Ret_1M"]

    if method == "Composite (Z: 1M/3M/6M/12M)":
        score = (
            0.1 * zscore(df["Ret_1M"]) +
            0.2 * zscore(df["Ret_3M"]) +
            0.3 * zscore(df["Ret_6M"]) +
            0.4 * zscore(df["Ret_12M"])
        )
    elif method == "12-1 Momentum":
        score = zscore(df["Mom_12_1"])
    elif method == "6M Return":
        score = zscore(df["Ret_6M"])
    elif method == "3M Return":
        score = zscore(df["Ret_3M"])
    elif method == "1M Return":
        score = zscore(df["Ret_1M"])
    elif method == "Composite - Vol Penalty":
        raw = (
            0.2 * zscore(df["Ret_3M"]) +
            0.3 * zscore(df["Ret_6M"]) +
            0.5 * zscore(df["Ret_12M"])
        )
        penalty = 0.35 * zscore(df["Vol_6M_ann"])
        score = raw - penalty
    else:
        score = zscore(df["Ret_6M"])

    df["Score"] = score
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Score"])
    return df.sort_values("Score", ascending=False)


# ═══════════════════════ Enhanced TMA Indicators ═══════════════

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def true_range(op: pd.Series, hi: pd.Series, lo: pd.Series, cl: pd.Series) -> pd.Series:
    prev = cl.shift(1)
    tr = pd.concat([(hi - lo).abs(), (hi - prev).abs(), (lo - prev).abs()], axis=1).max(axis=1)
    return tr


def calc_atr(op: pd.Series, hi: pd.Series, lo: pd.Series, cl: pd.Series, n: int = 14) -> pd.Series:
    tr = true_range(op, hi, lo, cl)
    return tr.rolling(n).mean()


def max_drawdown(close: pd.Series, window: int = 60) -> float:
    x = close.dropna()
    if len(x) < window + 5:
        return np.nan
    w = x.tail(window)
    peak = w.cummax()
    dd = (w / peak) - 1.0
    return float(dd.min())


def compute_regime_score(spy: pd.Series, qqq: pd.Series) -> float:
    def score_one(idx: pd.Series) -> int:
        idx = idx.dropna()
        if len(idx) < 260:
            return 0
        s50 = sma(idx, 50)
        s200 = sma(idx, 200)
        cond1 = int(idx.iloc[-1] > s200.iloc[-1])
        cond2 = int(idx.iloc[-1] > s50.iloc[-1])
        slope = s50.iloc[-1] - s50.iloc[-21] if len(s50.dropna()) > 230 else np.nan
        cond3 = int(np.isfinite(slope) and slope > 0)
        return 7 * cond1 + 4 * cond2 + 4 * cond3

    s_spy = score_one(spy)
    s_qqq = score_one(qqq)

    if s_spy == 0 and s_qqq == 0:
        return 0.0
    return float(0.65 * max(s_spy, s_qqq) + 0.35 * min(s_spy, s_qqq))


# ═══════════════════════ Enhanced TMA Scoring ══════════════════

def compute_tma_scores_enhanced(
    px_adj: pd.DataFrame,
    vol: pd.DataFrame,
    op: pd.DataFrame,
    hi: pd.DataFrame,
    lo: pd.DataFrame,
    cl: pd.DataFrame,
    benchmark_adj: pd.Series,
    regime_score_0_15: float,
    min_dollar_vol: float,
) -> pd.DataFrame:
    if px_adj.empty:
        return pd.DataFrame()

    px_adj = px_adj.sort_index().ffill()
    vol = vol.reindex(px_adj.index).sort_index().ffill()
    op = op.reindex(px_adj.index).sort_index().ffill()
    hi = hi.reindex(px_adj.index).sort_index().ffill()
    lo = lo.reindex(px_adj.index).sort_index().ffill()
    cl = cl.reindex(px_adj.index).sort_index().ffill()

    if len(px_adj) < 320:
        return pd.DataFrame()

    retd = px_adj.pct_change()

    def total_return(df, days):
        if len(df) <= days:
            return pd.Series(np.nan, index=df.columns)
        return (df.iloc[-1] / df.iloc[-days - 1]) - 1.0

    # --- Leadership (Livermore strengthened)
    r_6m = total_return(px_adj, 126)
    r_12m = total_return(px_adj, 252)

    bench = benchmark_adj.dropna()
    if len(bench) < 320:
        return pd.DataFrame()
    b6 = (bench.iloc[-1] / bench.iloc[-127]) - 1.0
    b12 = (bench.iloc[-1] / bench.iloc[-253]) - 1.0

    rs6 = r_6m - b6
    rs12 = r_12m - b12
    leadership = 15 * minmax01(pct_rank(rs6)) + 15 * minmax01(pct_rank(rs12))  # 0..30

    # Livermore bonus: 52-week high proximity (0..+5)
    high_52w = px_adj.rolling(252).max().iloc[-1]
    close_last = px_adj.iloc[-1]
    prox_52w = (close_last / high_52w).replace([np.inf, -np.inf], np.nan)
    prox52_score = 5 * ((prox_52w - 0.90) / 0.10).clip(0, 1).fillna(0)

    # --- Base: VCP + Pivot proximity + Bulkowski QA (depth/length)
    atr14 = pd.DataFrame({c: calc_atr(op[c], hi[c], lo[c], cl[c], 14) for c in cl.columns})
    atrp14 = (atr14.iloc[-1] / cl.iloc[-1]).replace([np.inf, -np.inf], np.nan)

    atrp_series = (atr14 / cl).replace([np.inf, -np.inf], np.nan)
    atr20 = atrp_series.rolling(20).mean().iloc[-1]
    atr60 = atrp_series.rolling(60).mean().iloc[-1]
    contraction_ratio = (atr20 / atr60).replace([np.inf, -np.inf], np.nan)
    vcp_raw = (1.0 - (contraction_ratio - 0.55) / (1.0 - 0.55)).clip(0, 1)
    vcp_score = 12 * vcp_raw.fillna(0)

    # Pivot proximity (Livermore/Darvas)
    pivot = px_adj.rolling(60).max().iloc[-1]
    pivot_prox = (close_last / pivot).replace([np.inf, -np.inf], np.nan)
    pivot_score = 5 * ((pivot_prox - 0.92) / (1.0 - 0.92)).clip(0, 1).fillna(0)

    # Bulkowski depth penalty & length score
    box_high = px_adj.rolling(60).max().iloc[-1]
    box_low = px_adj.rolling(60).min().iloc[-1]
    box_range_pct = (box_high / box_low - 1.0).replace([np.inf, -np.inf], np.nan)
    depth_pen = -5 * ((box_range_pct - 0.35) / (0.60 - 0.35)).clip(0, 1).fillna(0)

    band_lo = px_adj.rolling(60).min()
    band_hi = px_adj.rolling(60).max()
    in_band = ((px_adj >= band_lo) & (px_adj <= band_hi)).tail(80).mean()
    length_score = 3 * ((in_band - 0.70) / (0.95 - 0.70)).clip(0, 1).fillna(0)

    base_score = (vcp_score + pivot_score + length_score + depth_pen).clip(0, 25)

    # --- Demand: Wyckoff Absorption + Larry Williams Range Expansion + vol trend
    vol20 = vol.rolling(20).mean().iloc[-1]
    vol60 = vol.rolling(60).mean().iloc[-1]
    vol_ratio = (vol20 / vol60).replace([np.inf, -np.inf], np.nan)

    dollar_vol20 = (vol20 * close_last).replace([np.inf, -np.inf], np.nan)
    liq_ok = (dollar_vol20 >= min_dollar_vol).astype(float)

    vol_trend_score = 6 * ((vol_ratio - 0.8) / (1.3 - 0.8)).clip(0, 1).fillna(0)

    # Wyckoff absorption
    pivot_roll = px_adj.rolling(60).max()
    near_pivot = px_adj >= (0.95 * pivot_roll)
    up_day = retd > 0
    v20 = vol.tail(20)
    absorb_num = (v20.where(near_pivot.tail(20) & up_day.tail(20), 0)).sum()
    absorb_den = (v20.sum() + 1e-9)
    absorption_ratio = (absorb_num / absorb_den).replace([np.inf, -np.inf], np.nan)
    absorption_score = 6 * ((absorption_ratio - 0.10) / (0.35 - 0.10)).clip(0, 1).fillna(0)

    # Larry Williams range expansion
    tr_today = pd.DataFrame({c: true_range(op[c], hi[c], lo[c], cl[c]) for c in cl.columns}).iloc[-1]
    atr_today = atr14.iloc[-1]
    tr_atr = (tr_today / (atr_today + 1e-9)).replace([np.inf, -np.inf], np.nan)
    range_pos = ((cl.iloc[-1] - lo.iloc[-1]) / ((hi.iloc[-1] - lo.iloc[-1]) + 1e-9)).replace([np.inf, -np.inf], np.nan)
    exp_raw = ((tr_atr - 1.0) / (2.0 - 1.0)).clip(0, 1).fillna(0) * ((range_pos - 0.6) / (1.0 - 0.6)).clip(0, 1).fillna(0)
    range_expansion_score = 3 * exp_raw

    demand = ((vol_trend_score + absorption_score + range_expansion_score) * liq_ok).clip(0, 15)

    # --- Quality (trend alignment + stability)
    s20 = px_adj.rolling(20).mean().iloc[-1]
    s50 = px_adj.rolling(50).mean().iloc[-1]
    s200 = px_adj.rolling(200).mean().iloc[-1]
    align_full = ((s20 > s50) & (s50 > s200)).astype(float)
    align_mid = ((s20 > s50) | (s50 > s200)).astype(float)
    trend_score = 10 * (0.7 * align_full + 0.3 * align_mid)

    mdd60 = px_adj.apply(lambda s: max_drawdown(s, 60))
    stab = ((mdd60 - (-0.35)) / ((-0.10) - (-0.35))).clip(0, 1)
    stability_score = 5 * stab.fillna(0)
    quality = (trend_score + stability_score).clip(0, 15)

    # --- Risk Penalty: Wyckoff UT + volatility/liq penalties
    atrp_rank = atrp14.rank(pct=True)
    vol_pen = -3 * ((atrp_rank - 0.85) / (1.0 - 0.85)).clip(0, 1).fillna(0)

    absr20 = retd.tail(20).abs()
    big_move_freq = (absr20 > 0.12).mean()
    gap_pen = -2 * ((big_move_freq - 0.05) / (0.20 - 0.05)).clip(0, 1).fillna(0)

    # Wyckoff/Bulkowski UT (upthrust) penalty
    last10 = px_adj.tail(10)
    hi10 = hi.tail(10)
    pivot10 = pivot_roll.tail(10)
    failed = (hi10 > pivot10) & (last10 < pivot10)
    fail_freq = failed.mean()
    ut_pen = -5 * ((fail_freq - 0.10) / (0.40 - 0.10)).clip(0, 1).fillna(0)

    liq_pen = -10 * (1 - liq_ok)

    risk_penalty = (vol_pen + gap_pen + ut_pen + liq_pen).clip(-10, 0)

    # --- Regime (0..15)
    regime = pd.Series(regime_score_0_15, index=px_adj.columns).astype(float)

    # --- Total TMA
    tma = (regime + leadership + prox52_score + base_score + demand + quality + risk_penalty).clip(0, 100)

    out = pd.DataFrame({
        "TMA": tma,
        "Regime": regime,
        "Leadership": leadership,
        "52W_Bonus": prox52_score,
        "Base": base_score,
        "Demand": demand,
        "Quality": quality,
        "RiskPenalty": risk_penalty,
        "RS6": rs6,
        "RS12": rs12,
        "52W_Prox": prox_52w,
        "ContractionRatio": contraction_ratio,
        "PivotProx": pivot_prox,
        "BoxRange%": box_range_pct,
        "AbsorptionScore": absorption_score,
        "RangeExpScore": range_expansion_score,
        "UTPenalty": ut_pen,
        "VolRatio(20/60)": vol_ratio,
        "DollarVol20": dollar_vol20,
        "ATR%_14": atrp14,
        "MDD_60D": mdd60,
    }).replace([np.inf, -np.inf], np.nan).dropna(subset=["TMA"])

    return out.sort_values("TMA", ascending=False)


# ═══════════════════════ Sidebar ══════════════════════════════
with st.sidebar:
    st.header("VERONICA")
    show_logout_button()
    st.markdown("---")
    st.subheader("TMA Scanner Settings")

    period = st.selectbox("가격 데이터 기간", ["1y", "2y", "5y"], index=1)
    top_n = st.slider("Top N", min_value=5, max_value=50, value=10, step=1)
    min_dollar_vol_m = st.slider("최소 20일 평균 거래대금 (백만$)", 5, 200, 20, 5)
    min_dollar_vol = float(min_dollar_vol_m) * 1_000_000

    st.markdown("---")
    st.subheader("모멘텀 비교 설정")
    mom_method = st.selectbox(
        "모멘텀 정의(비교탭)",
        [
            "Composite (Z: 1M/3M/6M/12M)",
            "12-1 Momentum",
            "6M Return",
            "3M Return",
            "1M Return",
            "Composite - Vol Penalty",
        ],
        index=0,
    )

    st.markdown("---")
    st.subheader("체류 가중치")
    use_streak = st.toggle("체류 가중치 적용", value=True)
    streak_lookback = st.slider("체류 체크 기간 (거래일)", 3, 10, 5) if use_streak else 5
    streak_bonus_per_day = st.slider("체류일당 가점", 0.5, 2.0, 1.0, 0.5) if use_streak else 1.0
    if use_streak:
        st.caption(f"최근 {streak_lookback}일간 Top 20%에 연속 체류한 일수 × {streak_bonus_per_day}점 가산")

    st.caption("유니버스: Dow 30 / S&P 100 / Nasdaq 100 / S&P 500 (Wikipedia 기반)")
    st.markdown("---")
    if st.button("Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache cleared.")


# ═══════════════════════ Load Universes ════════════════════════

@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_all_universes() -> pd.DataFrame:
    u1 = fetch_dow30()
    u2 = fetch_sp100()
    u3 = fetch_nasdaq100()
    u4 = fetch_sp500()
    allu = pd.concat([u1, u2, u3, u4], ignore_index=True)
    allu["Ticker"] = allu["Ticker"].astype(str).str.strip().str.upper()
    allu["Name"] = allu["Name"].astype(str).str.strip()
    return allu


st.title("🏆 TMA Scanner (Enhanced)")

try:
    uni = load_all_universes()
except Exception as e:
    st.error(f"유니버스 로딩 실패: {e}")
    st.stop()

# Universe maps
universes: Dict[str, Tuple[pd.DataFrame, List[str]]] = {}
for u in uni["Universe"].unique():
    sub = uni[uni["Universe"] == u].copy()
    universes[u] = (sub, sub["Ticker"].tolist())

u_names = list(universes.keys())

membership = (
    uni.groupby("Ticker")["Universe"]
    .apply(lambda x: ", ".join(sorted(set(x.tolist()))))
    .to_dict()
)
name_map = (
    uni.groupby("Ticker")["Name"]
    .agg(lambda x: x.dropna().iloc[0] if len(x.dropna()) else "")
    .to_dict()
)

# Download prices once (tickers + benchmarks)
bench_tickers = ["SPY", "QQQ"]
all_tickers = sorted(set(uni["Ticker"].tolist() + bench_tickers))


def batched(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i:i + n] for i in range(0, len(lst), n)]


px_parts, vol_parts, op_parts, hi_parts, lo_parts, cl_parts = [], [], [], [], [], []
with st.spinner("가격/거래량 데이터를 다운로드 중... (무료: yfinance)"):
    for batch in batched(all_tickers, 120):
        px_b, vol_b, op_b, hi_b, lo_b, cl_b = download_yahoo(batch, period=period)
        if not px_b.empty:
            px_parts.append(px_b)
            vol_parts.append(vol_b)
            op_parts.append(op_b)
            hi_parts.append(hi_b)
            lo_parts.append(lo_b)
            cl_parts.append(cl_b)
        time.sleep(0.15)

if not px_parts:
    st.error("데이터 다운로드 실패. (네트워크/야후 제한/티커 이슈 가능)")
    st.stop()


def concat_dedup(parts: List[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(parts, axis=1)
    return df.loc[:, ~df.columns.duplicated()].sort_index()


px_all = concat_dedup(px_parts)
vol_all = concat_dedup(vol_parts)
op_all = concat_dedup(op_parts)
hi_all = concat_dedup(hi_parts)
lo_all = concat_dedup(lo_parts)
cl_all = concat_dedup(cl_parts)

# Bench series
spy = px_all["SPY"] if "SPY" in px_all.columns else pd.Series(dtype=float)
qqq = px_all["QQQ"] if "QQQ" in px_all.columns else pd.Series(dtype=float)

regime_0_15 = compute_regime_score(spy, qqq)
prev_regime_0_15 = compute_regime_score(spy.iloc[:-1], qqq.iloc[:-1])


# ═══════════════════════ Rank change helper ════════════════════

def build_prev_rank_map(scores_df: pd.DataFrame) -> Dict[str, int]:
    if scores_df.empty:
        return {}
    return {ticker: rank for rank, ticker in enumerate(scores_df.index, 1)}


def add_rank_columns(out: pd.DataFrame, prev_rank_map: Dict[str, int]) -> pd.DataFrame:
    out = out.copy()
    out["Rank"] = range(1, len(out) + 1)

    def _fmt(row):
        prev = prev_rank_map.get(row["Ticker"])
        if prev is None:
            return "NEW"
        chg = prev - row["Rank"]
        if chg > 0:
            return f"▲{chg}"
        elif chg < 0:
            return f"▼{abs(chg)}"
        return "─"

    out["전일대비"] = out.apply(_fmt, axis=1)
    return out


# ═══════════════════════ OHLCV slice helper ════════════════════

def _get_ohlcv(cols_exist):
    """Get all 6 OHLCV DataFrames for given ticker columns."""
    px_u = px_all[cols_exist].copy()
    vol_u = vol_all.reindex(columns=cols_exist)
    op_u = op_all.reindex(columns=cols_exist)
    hi_u = hi_all.reindex(columns=cols_exist)
    lo_u = lo_all.reindex(columns=cols_exist)
    cl_u = cl_all.reindex(columns=cols_exist)
    return px_u, vol_u, op_u, hi_u, lo_u, cl_u


def _call_tma(px_u, vol_u, op_u, hi_u, lo_u, cl_u, bench, regime):
    return compute_tma_scores_enhanced(
        px_adj=px_u, vol=vol_u,
        op=op_u, hi=hi_u, lo=lo_u, cl=cl_u,
        benchmark_adj=bench,
        regime_score_0_15=regime,
        min_dollar_vol=min_dollar_vol,
    )


# ═══════════════════════ Streak bonus helper ══════════════════

def compute_streak(px_u, vol_u, op_u, hi_u, lo_u, cl_u, bench, lookback):
    """
    Count consecutive Top-20% residency days (from yesterday backward).
    Returns pd.Series indexed by ticker with streak day count.
    """
    top20_sets = []
    for d in range(1, lookback + 1):
        if len(px_u) <= d + 320:
            top20_sets.append(set())
            continue
        tma_d = _call_tma(
            px_u.iloc[:-d], vol_u.iloc[:-d], op_u.iloc[:-d],
            hi_u.iloc[:-d], lo_u.iloc[:-d], cl_u.iloc[:-d],
            bench.iloc[:-d],
            compute_regime_score(spy.iloc[:-d], qqq.iloc[:-d]),
        )
        if tma_d.empty:
            top20_sets.append(set())
            continue
        n_top = max(1, int(len(tma_d) * 0.20))
        top20_sets.append(set(tma_d.head(n_top).index))

    streak = {}
    for ticker in px_u.columns:
        count = 0
        for s in top20_sets:
            if ticker in s:
                count += 1
            else:
                break
        streak[ticker] = count

    return pd.Series(streak)


# ═══════════════════════ Entry / Exit Signals ═════════════════

def compute_signals(tma_df):
    """Compute 5-tranche entry and 3-tranche exit trigger signals."""
    if tma_df.empty:
        return pd.DataFrame()

    pctile = tma_df["TMA"].rank(pct=True)
    sig = pd.DataFrame(index=tma_df.index)

    # ── Entry tranches (5분할 진입) ──
    sig["E1_TMA상위"] = (pctile >= 0.80).astype(int)
    sig["E2_베이스"] = (
        (tma_df["ContractionRatio"] < 0.80) & (tma_df["PivotProx"] > 0.92)
    ).astype(int)
    sig["E3_수요확인"] = (
        (tma_df["AbsorptionScore"] >= 3.0) & (tma_df["VolRatio(20/60)"] > 1.0)
    ).astype(int)
    sig["E4_추세확인"] = (
        (tma_df["Quality"] >= 10.0) & (tma_df["MDD_60D"] > -0.15)
    ).astype(int)
    sig["E5_돌파확인"] = (
        (tma_df["PivotProx"] > 0.97)
        & (tma_df["RangeExpScore"] > 1.5)
        & (tma_df["UTPenalty"] >= -1.0)
    ).astype(int)

    # ── Exit tranches (3분할 청산) — tight stop ──
    atrp_rank = tma_df["ATR%_14"].rank(pct=True)
    sig["X1_경고"] = (
        (tma_df["Quality"] < 10.0)
        | (tma_df["MDD_60D"] < -0.08)
        | (atrp_rank >= 0.80)
    ).astype(int)
    sig["X2_주의"] = (
        (pctile < 0.75)
        | (tma_df["Leadership"] < 15.0)
        | (tma_df["PivotProx"] < 0.90)
    ).astype(int)
    sig["X3_전량청산"] = (
        (pctile < 0.65)
        | (tma_df["RiskPenalty"] < -3.0)
        | (tma_df["MDD_60D"] < -0.15)
    ).astype(int)

    entry_cols = [c for c in sig.columns if c.startswith("E")]
    exit_cols = [c for c in sig.columns if c.startswith("X")]
    sig["EntryCount"] = sig[entry_cols].sum(axis=1).astype(int)
    sig["ExitCount"] = sig[exit_cols].sum(axis=1).astype(int)

    def _bar(n, total):
        return "\u2588" * int(n) + "\u2591" * (total - int(n))

    sig["\uc9c4\uc785\uc2dc\uadf8\ub110"] = sig["EntryCount"].apply(
        lambda n: f"{int(n)}/5 {_bar(n, 5)}"
    )
    sig["\uccad\uc0b0\uc2dc\uadf8\ub110"] = sig["ExitCount"].apply(
        lambda n: f"{int(n)}/3 {_bar(n, 3)}"
    )

    # Convert 0/1 to visual markers for display
    for c in entry_cols + exit_cols:
        sig[c] = sig[c].map({1: "\u2713", 0: ""})

    return sig


# ═══════════════════════ Column format helpers ═════════════════

TMA_SHOW_COLS = [
    "Rank", "전일대비", "Ticker", "Name", "Universes",
    "TMA", "StreakDays", "StreakBonus",
    "Regime", "Leadership", "52W_Bonus", "Base", "Demand", "Quality", "RiskPenalty",
    "RS6", "RS12",
    "ContractionRatio", "PivotProx", "BoxRange%",
    "AbsorptionScore", "RangeExpScore", "UTPenalty",
    "VolRatio(20/60)", "DollarVol20",
    "ATR%_14", "MDD_60D",
]

MOM_SHOW_COLS = [
    "Rank", "전일대비", "Ticker", "Name", "Universes",
    "Score", "Ret_1M", "Ret_3M", "Ret_6M", "Ret_12M", "Mom_12_1", "Vol_6M_ann",
]


def tma_fmt(cols) -> dict:
    fmt = {}
    for c in cols:
        if c in ("TMA", "Regime", "Leadership", "52W_Bonus", "Base", "Demand", "Quality",
                 "RiskPenalty", "AbsorptionScore", "RangeExpScore", "UTPenalty",
                 "StreakBonus"):
            fmt[c] = "{:.1f}"
        elif c == "StreakDays":
            fmt[c] = "{:.0f}"
        elif c in ("RS6", "RS12", "BoxRange%", "ATR%_14", "MDD_60D"):
            fmt[c] = "{:.2%}"
        elif c in ("ContractionRatio", "PivotProx", "VolRatio(20/60)"):
            fmt[c] = "{:.2f}"
        elif c == "DollarVol20":
            fmt[c] = "${:,.0f}"
    return fmt


def mom_fmt(cols) -> dict:
    fmt = {}
    for c in cols:
        if c == "Score":
            fmt[c] = "{:.3f}"
        elif c.startswith("Ret_") or c == "Mom_12_1":
            fmt[c] = "{:.2%}"
        elif c.startswith("Vol_"):
            fmt[c] = "{:.2%}"
    return fmt


# ═══════════════════════ Tabs ══════════════════════════════════

tabs = st.tabs(["🏆 TMA (유니버스별)", "🌐 통합 TMA Top", "📈 모멘텀 비교"])

# ── Tab 1: TMA per universe ──
with tabs[0]:
    st.subheader("TMA Top 랭킹 (유니버스별)")
    st.info(f"현재 Regime 점수(0~15): **{regime_0_15:.1f}**  (SPY/QQQ 기반)")

    def render_tma(universe_name: str, container):
        _, tickers = universes[universe_name]
        cols_exist = [t for t in tickers if t in px_all.columns and t in vol_all.columns]
        if len(cols_exist) < 10:
            container.warning(f"{universe_name}: 데이터 확보 티커 수가 너무 적습니다.")
            return

        px_u, vol_u, op_u, hi_u, lo_u, cl_u = _get_ohlcv(cols_exist)
        bench = spy.reindex(px_u.index).ffill()

        tma_result = _call_tma(px_u, vol_u, op_u, hi_u, lo_u, cl_u, bench, regime_0_15)
        if tma_result.empty:
            container.warning(f"{universe_name}: TMA 계산 불가(데이터 부족/결측).")
            return

        # Previous day rankings
        tma_prev = _call_tma(
            px_u.iloc[:-1], vol_u.iloc[:-1], op_u.iloc[:-1],
            hi_u.iloc[:-1], lo_u.iloc[:-1], cl_u.iloc[:-1],
            bench.iloc[:-1], prev_regime_0_15,
        )
        prev_map = build_prev_rank_map(tma_prev)

        out = tma_result.head(top_n).reset_index().rename(columns={"index": "Ticker"})
        out["Universes"] = out["Ticker"].map(membership).fillna(universe_name)
        out["Name"] = out["Ticker"].map(name_map).fillna("")
        out = add_rank_columns(out, prev_map)
        out = out[[c for c in TMA_SHOW_COLS if c in out.columns]]

        container.subheader(f"🏁 {universe_name} — TMA Top {top_n}")
        container.dataframe(out.style.format(tma_fmt(out.columns)), use_container_width=True)

    for i, uname in enumerate(u_names):
        if i % 2 == 0:
            tma_cols = st.columns(2)
        render_tma(uname, tma_cols[i % 2])

# ── Tab 2: Unified TMA Top ──
with tabs[1]:
    st.subheader(f"통합 TMA Top ({len(u_names)}개 유니버스 합산, 중복 제거)")
    st.caption("중복 티커는 하나로 합치고, '어느 유니버스에 속하는지'를 함께 표시합니다.")

    cols_exist = [t for t in uni["Ticker"].unique().tolist() if t in px_all.columns and t in vol_all.columns]
    if len(cols_exist) < 20:
        st.warning("통합 유니버스 데이터가 부족합니다.")
    else:
        px_u, vol_u, op_u, hi_u, lo_u, cl_u = _get_ohlcv(cols_exist)
        bench = spy.reindex(px_u.index).ffill()

        tma_all = _call_tma(px_u, vol_u, op_u, hi_u, lo_u, cl_u, bench, regime_0_15)
        if tma_all.empty:
            st.warning("통합 TMA 계산 불가(데이터 부족/결측).")
        else:
            # ── Streak bonus (통합 탭 전용) ──
            if use_streak:
                with st.spinner(f"체류 가중치 계산 중 (최근 {streak_lookback}일)..."):
                    streak_s = compute_streak(
                        px_u, vol_u, op_u, hi_u, lo_u, cl_u, bench, streak_lookback,
                    )
                tma_all["StreakDays"] = streak_s.reindex(tma_all.index).fillna(0).astype(int)
                tma_all["StreakBonus"] = tma_all["StreakDays"] * streak_bonus_per_day
                tma_all["TMA"] = tma_all["TMA"] + tma_all["StreakBonus"]
                tma_all = tma_all.sort_values("TMA", ascending=False)

            # Previous day ranking (raw TMA, streak 미적용)
            tma_all_prev = _call_tma(
                px_u.iloc[:-1], vol_u.iloc[:-1], op_u.iloc[:-1],
                hi_u.iloc[:-1], lo_u.iloc[:-1], cl_u.iloc[:-1],
                bench.iloc[:-1], prev_regime_0_15,
            )
            prev_map = build_prev_rank_map(tma_all_prev)

            out = tma_all.head(top_n).reset_index().rename(columns={"index": "Ticker"})
            out["Universes"] = out["Ticker"].map(membership).fillna("")
            out["Name"] = out["Ticker"].map(name_map).fillna("")
            out = add_rank_columns(out, prev_map)
            out = out[[c for c in TMA_SHOW_COLS if c in out.columns]]

            st.dataframe(out.style.format(tma_fmt(out.columns)), use_container_width=True)

            # ── Entry / Exit Signal Table ──
            st.markdown("---")
            st.subheader("📊 분할 진입/청산 시그널")
            st.caption("TMA 구성요소 기반 — 진입 5분할 / 청산 3분할 트리거 현황")

            signals = compute_signals(tma_all)
            sig_top = signals.reindex(tma_all.head(top_n).index)

            sig_display = tma_all.head(top_n).reset_index().rename(columns={"index": "Ticker"})
            sig_display["Name"] = sig_display["Ticker"].map(name_map).fillna("")

            sig_cols = [
                "진입시그널", "청산시그널",
                "E1_TMA상위", "E2_베이스", "E3_수요확인",
                "E4_추세확인", "E5_돌파확인",
                "X1_경고", "X2_주의", "X3_전량청산",
            ]
            for c in sig_cols:
                if c in sig_top.columns:
                    sig_display[c] = sig_top[c].values

            sig_show = [
                "Ticker", "Name", "TMA",
                "진입시그널", "청산시그널",
                "E1_TMA상위", "E2_베이스", "E3_수요확인",
                "E4_추세확인", "E5_돌파확인",
                "X1_경고", "X2_주의", "X3_전량청산",
            ]
            sig_display = sig_display[[c for c in sig_show if c in sig_display.columns]]

            st.dataframe(sig_display.style.format({"TMA": "{:.1f}"}), use_container_width=True)

            with st.expander("진입/청산 트리거 기준 설명"):
                st.markdown("""
**진입 트리거 (5분할) — 조건 충족 시 해당 분할만큼 매수:**

| 트리거 | 조건 | 의미 |
|--------|------|------|
| E1\_TMA상위 | TMA 상위 20% | TMA 종합점수 기준 상위권 진입 |
| E2\_베이스 | ContractionRatio < 0.80 & PivotProx > 0.92 | 변동성 수축 + 피봇 근접 (VCP 패턴) |
| E3\_수요확인 | AbsorptionScore ≥ 3.0 & VolRatio > 1.0 | Wyckoff 흡수 + 거래량 증가 |
| E4\_추세확인 | Quality ≥ 10 & MDD\_60D > -15% | MA 정배열 + 안정적 추세 |
| E5\_돌파확인 | PivotProx > 0.97 & RangeExpScore > 1.5 & UTPenalty ≥ -1 | 고점 돌파 시도 + 레인지 확장 |

**청산 트리거 (3분할) — 조건 충족 시 해당 분할만큼 매도 (Tight Stop):**

| 트리거 | 조건 | 의미 |
|--------|------|------|
| X1\_경고 | Quality < 10 또는 MDD\_60D < -8% 또는 ATR% 상위 20% | 품질 저하·낙폭 초기·변동성 급등 |
| X2\_주의 | TMA 하위 25% 또는 Leadership < 15 또는 PivotProx < 0.90 | 모멘텀 약화·피봇 이탈 |
| X3\_전량청산 | TMA 하위 35% 또는 RiskPenalty < -3 또는 MDD < -15% | 심각한 리스크·즉시 청산 |

**활용 예시:**
- 진입 1/5: E1만 충족 → 20% 포지션 진입
- 진입 3/5: E1+E2+E3 충족 → 60% 포지션
- 진입 5/5: 모든 트리거 충족 → 풀 포지션 (100%)
- 청산 1/3: X1 충족 → 보유분의 1/3 청산
- 청산 3/3: 모든 트리거 충족 → 전량 청산
""")

# ── Tab 3: Momentum comparison ──
with tabs[2]:
    st.subheader("모멘텀 랭킹 비교 (유니버스별)")

    def render_momentum(universe_name: str, container):
        _, tickers = universes[universe_name]
        cols_exist = [t for t in tickers if t in px_all.columns]
        if len(cols_exist) < 10:
            container.warning(f"{universe_name}: 데이터 확보 티커 수가 너무 적습니다.")
            return

        px_u = px_all[cols_exist].copy()
        mom = compute_momentum_scores(px_u, method=mom_method)
        if mom.empty:
            container.warning(f"{universe_name}: 모멘텀 계산 불가(데이터 부족/결측).")
            return

        mom_prev = compute_momentum_scores(px_u.iloc[:-1], method=mom_method)
        prev_map = build_prev_rank_map(mom_prev)

        out = mom.head(top_n).reset_index().rename(columns={"index": "Ticker"})
        out["Universes"] = out["Ticker"].map(membership).fillna(universe_name)
        out["Name"] = out["Ticker"].map(name_map).fillna("")
        out = add_rank_columns(out, prev_map)
        out = out[[c for c in MOM_SHOW_COLS if c in out.columns]]

        container.subheader(f"📌 {universe_name} — Momentum Top {top_n}")
        container.dataframe(out.style.format(mom_fmt(out.columns)), use_container_width=True)

    for i, uname in enumerate(u_names):
        if i % 2 == 0:
            mom_cols = st.columns(2)
        render_momentum(uname, mom_cols[i % 2])

