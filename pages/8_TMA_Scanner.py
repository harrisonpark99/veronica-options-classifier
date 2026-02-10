#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VERONICA – TMA Scanner + Momentum Compare
Scans Dow 30, S&P 100, Nasdaq 100, Russell Top-100 (IWB proxy)
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
def fetch_russell_top100_proxy() -> pd.DataFrame:
    """Russell Top-100 proxy via IWB ETF top 100 holdings."""
    url = "https://www.financecharts.com/etfs/IWB/holdings"
    tables = safe_read_html(url)
    for df in tables:
        cols = [c.lower() for c in df.columns.astype(str)]
        if any("ticker" in c for c in cols) and any("name" in c for c in cols):
            out = df.copy()
            name_col = [c for c in out.columns if "name" in str(c).lower()][0]
            tick_col = [c for c in out.columns if "ticker" in str(c).lower()][0]
            out = out[[name_col, tick_col]].rename(columns={name_col: "Name", tick_col: "Ticker"})
            out["Ticker"] = out["Ticker"].astype(str).str.strip().str.upper()
            out = out[out["Ticker"].str.len() > 0].head(100)
            out["Universe"] = "Russell (Top100 proxy via IWB)"
            return out.reset_index(drop=True)
    raise ValueError("IWB holdings table not found (site layout changed).")


# ═══════════════════════ Data Download ═════════════════════════

@st.cache_data(ttl=60 * 30, show_spinner=False)
def download_ohlcv_yahoo(tickers: List[str], period: str = "2y") -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        return pd.DataFrame(), pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        px = data["Adj Close"].copy() if "Adj Close" in data.columns.get_level_values(0) else data["Close"].copy()
        vol = data["Volume"].copy() if "Volume" in data.columns.get_level_values(0) else pd.DataFrame(index=px.index)
    else:
        px = data[["Adj Close"]].rename(columns={"Adj Close": yt[0]}) if "Adj Close" in data.columns else data[["Close"]].rename(columns={"Close": yt[0]})
        vol = data[["Volume"]].rename(columns={"Volume": yt[0]}) if "Volume" in data.columns else pd.DataFrame(index=px.index)

    px.columns = [from_yahoo_ticker(str(c)) for c in px.columns]
    vol.columns = [from_yahoo_ticker(str(c)) for c in vol.columns]
    return px.sort_index(), vol.sort_index()


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


# ═══════════════════════ TMA Scoring ═══════════════════════════

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def atr_pct(close: pd.Series, n: int = 14) -> pd.Series:
    tr = close.diff().abs()
    atr = tr.rolling(n).mean()
    return atr / close


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


def compute_tma_scores(
    px: pd.DataFrame,
    vol: pd.DataFrame,
    benchmark: pd.Series,
    regime_score_0_15: float,
    min_dollar_vol: float,
) -> pd.DataFrame:
    if px.empty:
        return pd.DataFrame()

    px = px.sort_index().ffill()
    vol = vol.reindex(px.index).sort_index().ffill()

    if len(px) < 260:
        return pd.DataFrame()

    retd = px.pct_change()

    def total_return(df: pd.DataFrame, days: int) -> pd.Series:
        if len(df) <= days:
            return pd.Series(np.nan, index=df.columns)
        return (df.iloc[-1] / df.iloc[-days - 1]) - 1.0

    r_6m = total_return(px, 126)
    r_12m = total_return(px, 252)

    bench = benchmark.dropna()
    if len(bench) < 260:
        return pd.DataFrame()
    b6 = (bench.iloc[-1] / bench.iloc[-127]) - 1.0
    b12 = (bench.iloc[-1] / bench.iloc[-253]) - 1.0

    rs6 = r_6m - b6
    rs12 = r_12m - b12

    rs6_norm = minmax01(rs6.rank(pct=True))
    rs12_norm = minmax01(rs12.rank(pct=True))
    leadership = 15 * rs6_norm + 15 * rs12_norm

    atr14 = px.apply(lambda s: atr_pct(s, 14))
    atr20 = atr14.rolling(20).mean().iloc[-1]
    atr60 = atr14.rolling(60).mean().iloc[-1]
    contraction_ratio = (atr20 / atr60).replace([np.inf, -np.inf], np.nan)

    vcp_raw = (1.0 - (contraction_ratio - 0.55) / (1.0 - 0.55)).clip(0, 1)
    vcp_score = 15 * vcp_raw.fillna(0)

    box_high = px.rolling(60).max().iloc[-1]
    box_low = px.rolling(60).min().iloc[-1]
    box_range_pct = (box_high / box_low - 1.0).replace([np.inf, -np.inf], np.nan)

    close_last = px.iloc[-1]
    close_to_high = (close_last / box_high).replace([np.inf, -np.inf], np.nan)

    range_ok = (1.0 - ((box_range_pct - 0.15) / (0.35 - 0.15))).clip(0, 1)
    prox_ok = ((close_to_high - 0.92) / (1.0 - 0.92)).clip(0, 1)

    box_score = 10 * (0.55 * prox_ok.fillna(0) + 0.45 * range_ok.fillna(0))
    base_score = vcp_score + box_score

    vol20 = vol.rolling(20).mean().iloc[-1]
    vol60 = vol.rolling(60).mean().iloc[-1]
    vol_ratio = (vol20 / vol60).replace([np.inf, -np.inf], np.nan)

    dollar_vol20 = (vol20 * close_last).replace([np.inf, -np.inf], np.nan)
    liq_ok = (dollar_vol20 >= min_dollar_vol).astype(float)

    vol_trend = ((vol_ratio - 0.8) / (1.3 - 0.8)).clip(0, 1).fillna(0)
    vol_trend_score = 10 * vol_trend

    r20 = retd.tail(20)
    v20 = vol.tail(20)
    up_vol = (v20.where(r20 > 0, 0)).sum()
    dn_vol = (v20.where(r20 < 0, 0)).sum()
    up_dom = (up_vol / (up_vol + dn_vol + 1e-9)).replace([np.inf, -np.inf], np.nan)
    up_dom_score = 5 * ((up_dom - 0.45) / (0.65 - 0.45)).clip(0, 1).fillna(0)

    demand = (vol_trend_score + up_dom_score) * liq_ok

    s20 = px.rolling(20).mean().iloc[-1]
    s50 = px.rolling(50).mean().iloc[-1]
    s200 = px.rolling(200).mean().iloc[-1]
    align_full = ((s20 > s50) & (s50 > s200)).astype(float)
    align_mid = ((s20 > s50) | (s50 > s200)).astype(float)
    trend_score = 10 * (0.7 * align_full + 0.3 * align_mid)

    mdd60 = px.apply(lambda s: max_drawdown(s, 60))
    stab = ((mdd60 - (-0.35)) / ((-0.10) - (-0.35))).clip(0, 1)
    stability_score = 5 * stab.fillna(0)
    quality = trend_score + stability_score

    atrp_last = atr14.iloc[-1]
    atrp_rank = atrp_last.rank(pct=True)
    vol_pen = -4 * ((atrp_rank - 0.85) / (1.0 - 0.85)).clip(0, 1).fillna(0)

    absr20 = retd.tail(20).abs()
    big_move_freq = (absr20 > 0.12).mean()
    gap_pen = -4 * ((big_move_freq - 0.05) / (0.20 - 0.05)).clip(0, 1).fillna(0)

    liq_pen = -10 * (1 - liq_ok)

    risk_penalty = (vol_pen + gap_pen + liq_pen).clip(-10, 0)

    regime = pd.Series(regime_score_0_15, index=px.columns).astype(float)

    tma_val = (regime + leadership + base_score + demand + quality + risk_penalty).clip(0, 100)

    out = pd.DataFrame({
        "TMA": tma_val,
        "Regime": regime,
        "Leadership": leadership,
        "Base": base_score,
        "Demand": demand,
        "Quality": quality,
        "RiskPenalty": risk_penalty,
        "RS6": rs6,
        "RS12": rs12,
        "Ret_6M": r_6m,
        "Ret_12M": r_12m,
        "ATR%_14": atrp_last,
        "ContractionRatio": contraction_ratio,
        "BoxRange%": box_range_pct,
        "Close/BoxHigh": close_to_high,
        "VolRatio(20/60)": vol_ratio,
        "DollarVol20": dollar_vol20,
        "MDD_60D": mdd60,
    })
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["TMA"])
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

    st.caption("러셀은 IWB 상위 100 홀딩 프록시(무료 공개 테이블).")
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
    u4 = fetch_russell_top100_proxy()
    allu = pd.concat([u1, u2, u3, u4], ignore_index=True)
    allu["Ticker"] = allu["Ticker"].astype(str).str.strip().str.upper()
    allu["Name"] = allu["Name"].astype(str).str.strip()
    return allu


st.title("📌 TMA Scanner + Momentum Compare")

try:
    uni = load_all_universes()
except Exception as e:
    st.error(f"유니버스 로딩 실패: {e}")
    st.stop()

# Universe maps
universes: Dict[str, Tuple[pd.DataFrame, List[str]]] = {}
for u in uni["Universe"].unique():
    sub = uni[uni["Universe"] == u].copy()
    tickers = sub["Ticker"].tolist()
    universes[u] = (sub, tickers)

u_names = list(universes.keys())

# Membership map: ticker -> "Universe1, Universe2, ..."
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


px_parts, vol_parts = [], []
with st.spinner("가격/거래량 데이터를 다운로드 중... (무료: yfinance)"):
    for batch in batched(all_tickers, 120):
        px_b, vol_b = download_ohlcv_yahoo(batch, period=period)
        if not px_b.empty:
            px_parts.append(px_b)
            vol_parts.append(vol_b)
        time.sleep(0.15)

if not px_parts:
    st.error("데이터 다운로드 실패. (네트워크/야후 제한/티커 이슈 가능)")
    st.stop()

px_all = pd.concat(px_parts, axis=1)
vol_all = pd.concat(vol_parts, axis=1)
px_all = px_all.loc[:, ~px_all.columns.duplicated()].sort_index()
vol_all = vol_all.loc[:, ~vol_all.columns.duplicated()].sort_index()

# Bench series
spy = px_all["SPY"] if "SPY" in px_all.columns else pd.Series(dtype=float)
qqq = px_all["QQQ"] if "QQQ" in px_all.columns else pd.Series(dtype=float)

regime_0_15 = compute_regime_score(spy, qqq)

# ═══════════════════════ Column format helper ══════════════════

TMA_SHOW_COLS = [
    "Ticker", "Name", "Universes",
    "TMA",
    "Regime", "Leadership", "Base", "Demand", "Quality", "RiskPenalty",
    "RS6", "RS12",
    "ContractionRatio", "BoxRange%", "Close/BoxHigh",
    "VolRatio(20/60)", "DollarVol20",
    "ATR%_14", "MDD_60D",
]

MOM_SHOW_COLS = [
    "Ticker", "Name", "Universes",
    "Score", "Ret_1M", "Ret_3M", "Ret_6M", "Ret_12M", "Mom_12_1", "Vol_6M_ann",
]


def tma_fmt(cols) -> dict:
    fmt = {}
    for c in cols:
        if c in ("TMA", "Regime", "Leadership", "Base", "Demand", "Quality", "RiskPenalty"):
            fmt[c] = "{:.1f}"
        elif c in ("RS6", "RS12"):
            fmt[c] = "{:.2%}"
        elif c in ("ContractionRatio", "Close/BoxHigh", "VolRatio(20/60)"):
            fmt[c] = "{:.2f}"
        elif c in ("BoxRange%", "ATR%_14", "MDD_60D"):
            fmt[c] = "{:.2%}"
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

    cols = st.columns(2)

    def render_tma(universe_name: str, container):
        sub_df, tickers = universes[universe_name]
        cols_exist = [t for t in tickers if t in px_all.columns]
        if len(cols_exist) < 10:
            container.warning(f"{universe_name}: 데이터 확보 티커 수가 너무 적습니다.")
            return

        px_u = px_all[cols_exist].copy()
        vol_u = vol_all[cols_exist].copy()
        bench = spy.reindex(px_u.index).ffill()

        tma_result = compute_tma_scores(px_u, vol_u, bench, regime_0_15, min_dollar_vol)
        if tma_result.empty:
            container.warning(f"{universe_name}: TMA 계산 불가(데이터 부족/결측 과다).")
            return

        out = tma_result.head(top_n).reset_index().rename(columns={"index": "Ticker"})
        out["Universes"] = out["Ticker"].map(membership).fillna(universe_name)
        out["Name"] = out["Ticker"].map(name_map).fillna("")
        out = out[[c for c in TMA_SHOW_COLS if c in out.columns]]

        container.subheader(f"🏁 {universe_name} — TMA Top {top_n}")
        container.dataframe(out.style.format(tma_fmt(out.columns)), use_container_width=True)

    render_tma(u_names[0], cols[0])
    render_tma(u_names[1], cols[1])
    render_tma(u_names[2], cols[0])
    render_tma(u_names[3], cols[1])

# ── Tab 2: Unified TMA Top ──
with tabs[1]:
    st.subheader("통합 TMA Top (4개 유니버스 합산, 중복 제거)")
    st.caption("중복 티커는 하나로 합치고, '어느 유니버스에 속하는지'를 함께 표시합니다.")

    cols_exist = [t for t in uni["Ticker"].unique().tolist() if t in px_all.columns]
    if len(cols_exist) < 20:
        st.warning("통합 유니버스 데이터가 부족합니다.")
    else:
        px_u = px_all[cols_exist].copy()
        vol_u = vol_all[cols_exist].copy()
        bench = spy.reindex(px_u.index).ffill()

        tma_all = compute_tma_scores(px_u, vol_u, bench, regime_0_15, min_dollar_vol)
        if tma_all.empty:
            st.warning("통합 TMA 계산 불가(데이터 부족/결측 과다).")
        else:
            out = tma_all.head(top_n).reset_index().rename(columns={"index": "Ticker"})
            out["Universes"] = out["Ticker"].map(membership).fillna("")
            out["Name"] = out["Ticker"].map(name_map).fillna("")
            out = out[[c for c in TMA_SHOW_COLS if c in out.columns]]

            st.dataframe(out.style.format(tma_fmt(out.columns)), use_container_width=True)

# ── Tab 3: Momentum comparison ──
with tabs[2]:
    st.subheader("모멘텀 랭킹 비교 (유니버스별)")
    cols = st.columns(2)

    def render_momentum(universe_name: str, container):
        sub_df, tickers = universes[universe_name]
        cols_exist = [t for t in tickers if t in px_all.columns]
        if len(cols_exist) < 10:
            container.warning(f"{universe_name}: 데이터 확보 티커 수가 너무 적습니다.")
            return

        px_u = px_all[cols_exist].copy()
        mom = compute_momentum_scores(px_u, method=mom_method)
        if mom.empty:
            container.warning(f"{universe_name}: 모멘텀 계산 불가(데이터 부족/결측 과다).")
            return

        out = mom.head(top_n).reset_index().rename(columns={"index": "Ticker"})
        out["Universes"] = out["Ticker"].map(membership).fillna(universe_name)
        out["Name"] = out["Ticker"].map(name_map).fillna("")
        out = out[[c for c in MOM_SHOW_COLS if c in out.columns]]

        container.subheader(f"📌 {universe_name} — Momentum Top {top_n}")
        container.dataframe(out.style.format(mom_fmt(out.columns)), use_container_width=True)

    render_momentum(u_names[0], cols[0])
    render_momentum(u_names[1], cols[1])
    render_momentum(u_names[2], cols[0])
    render_momentum(u_names[3], cols[1])
