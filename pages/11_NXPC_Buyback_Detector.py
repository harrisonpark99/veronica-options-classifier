"""
NXPC Buyback Detection Dashboard
=================================
Tier 1 venues: Binance, Bybit, OKX

Methodology:
1. Dollar Bars (instead of time bars) for statistically stable samples
2. VPIN (Volume-Synchronized Probability of Informed Trading) for directional flow
3. Cumulative Volume Delta (CVD) for net buying pressure tracking

References:
- Lopez de Prado, "Advances in Financial Machine Learning" (Ch. 2, Ch. 19)
- Easley, Lopez de Prado, O'Hara (2012), "Flow Toxicity and Liquidity in a HFT World"
"""

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone

# =====================================================
# 1. Config & Branding
# =====================================================
st.set_page_config(
    page_title="NXPC Buyback Detector | Presto",
    page_icon="🎯",
    layout="wide",
)

PRESTO_ORANGE = "#E8742A"
PRESTO_DARK   = "#2D2D2D"
BG_LIGHT      = "#FAFAFA"
GREEN         = "#16A34A"
RED           = "#DC2626"

st.markdown(
    f"""
<style>
    .main {{ background-color: {BG_LIGHT}; }}
    h1, h2, h3 {{ color: {PRESTO_DARK}; }}
    .stMetric {{ background: white; padding: 1rem;
                 border-left: 3px solid {PRESTO_ORANGE}; border-radius: 4px; }}
    div[data-testid="stSidebar"] {{ background-color: {PRESTO_DARK}; }}
    div[data-testid="stSidebar"] * {{ color: white !important; }}
    .accent {{ color: {PRESTO_ORANGE}; font-weight: 600; }}
</style>
""",
    unsafe_allow_html=True,
)

DB_PATH = Path(__file__).parent.parent / "nxpc_trades.db"


# =====================================================
# 2. Exchange API Adapters
# =====================================================
class BinanceAdapter:
    name = "binance"
    BASE = "https://api.binance.com"

    def fetch_trades(self, symbol="NXPCUSDT", start_ms=None, end_ms=None, limit=1000):
        url = f"{self.BASE}/api/v3/aggTrades"
        params = {"symbol": symbol, "limit": limit}
        if start_ms:
            params["startTime"] = start_ms
        if end_ms:
            params["endTime"] = end_ms
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["ts"]        = pd.to_datetime(df["T"], unit="ms", utc=True)
        df["price"]     = df["p"].astype(float)
        df["qty"]       = df["q"].astype(float)
        df["side"]      = (~df["m"]).map({True: "buy", False: "sell"})
        df["quote_qty"] = df["price"] * df["qty"]
        df["exchange"]  = self.name
        return df[["ts", "price", "qty", "side", "quote_qty", "exchange"]]


class BybitAdapter:
    name = "bybit"
    BASE = "https://api.bybit.com"

    def fetch_trades(self, symbol="NXPCUSDT", start_ms=None, end_ms=None, limit=1000):
        url    = f"{self.BASE}/v5/market/recent-trade"
        params = {"category": "spot", "symbol": symbol, "limit": min(limit, 1000)}
        r      = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json().get("result", {}).get("list", [])
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["ts"]        = pd.to_datetime(df["time"].astype(int), unit="ms", utc=True)
        df["price"]     = df["price"].astype(float)
        df["qty"]       = df["size"].astype(float)
        df["side"]      = df["side"].str.lower()
        df["quote_qty"] = df["price"] * df["qty"]
        df["exchange"]  = self.name
        df = df[["ts", "price", "qty", "side", "quote_qty", "exchange"]]
        if start_ms:
            df = df[df["ts"] >= pd.Timestamp(start_ms, unit="ms", tz="UTC")]
        if end_ms:
            df = df[df["ts"] <= pd.Timestamp(end_ms,   unit="ms", tz="UTC")]
        return df


class OKXAdapter:
    name = "okx"
    BASE = "https://www.okx.com"

    def fetch_trades(self, symbol="NXPC-USDT", start_ms=None, end_ms=None, limit=500):
        url    = f"{self.BASE}/api/v5/market/trades"
        params = {"instId": symbol, "limit": min(limit, 500)}
        r      = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["ts"]        = pd.to_datetime(df["ts"].astype(int), unit="ms", utc=True)
        df["price"]     = df["px"].astype(float)
        df["qty"]       = df["sz"].astype(float)
        df["side"]      = df["side"].str.lower()
        df["quote_qty"] = df["price"] * df["qty"]
        df["exchange"]  = self.name
        return df[["ts", "price", "qty", "side", "quote_qty", "exchange"]]


ADAPTERS: dict = {
    "Binance": (BinanceAdapter(), "NXPCUSDT"),
    "Bybit":   (BybitAdapter(),   "NXPCUSDT"),
    "OKX":     (OKXAdapter(),     "NXPC-USDT"),
}


# =====================================================
# 3. Data Persistence (SQLite)
# =====================================================
@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                ts        TIMESTAMP,
                price     REAL,
                qty       REAL,
                side      TEXT,
                quote_qty REAL,
                exchange  TEXT,
                PRIMARY KEY (ts, exchange, price, qty)
            )
            """
        )


def persist_trades(df: pd.DataFrame):
    if df.empty:
        return
    cols = ["ts", "price", "qty", "side", "quote_qty", "exchange"]
    rows = [
        (str(r.ts), r.price, r.qty, r.side, r.quote_qty, r.exchange)
        for r in df[cols].itertuples(index=False)
    ]
    with get_conn() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO trades (ts, price, qty, side, quote_qty, exchange) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )


def load_trades(exchange: str, hours_back: int = 24) -> pd.DataFrame:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).isoformat()
    with get_conn() as conn:
        df = pd.read_sql(
            "SELECT * FROM trades WHERE exchange = ? AND ts >= ? ORDER BY ts",
            conn,
            params=(exchange.lower(), cutoff),
        )
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], format="ISO8601", utc=True)
    return df


# =====================================================
# 4. Bar Construction (vectorized)
# =====================================================
def build_dollar_bars(trades: pd.DataFrame, dollar_threshold: float) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    trades = trades.sort_values("ts").reset_index(drop=True)
    cum = trades["quote_qty"].cumsum().values
    n_complete = int(cum[-1] / dollar_threshold)
    if n_complete == 0:
        return pd.DataFrame()
    boundaries = np.arange(1, n_complete + 1) * dollar_threshold
    bar_ids    = np.searchsorted(boundaries, cum, side="right")
    mask = bar_ids < n_complete
    t    = trades[mask].copy()
    t["_bar"] = bar_ids[mask]
    grp  = t.groupby("_bar")
    bars = pd.DataFrame({
        "ts":           grp["ts"].last(),
        "open":         grp["price"].first(),
        "high":         grp["price"].max(),
        "low":          grp["price"].min(),
        "close":        grp["price"].last(),
        "volume":       grp["qty"].sum(),
        "dollar_volume":grp["quote_qty"].sum(),
        "n_trades":     grp["qty"].count(),
    })
    buy_vol  = t[t["side"] == "buy"].groupby("_bar")["quote_qty"].sum()
    sell_vol = t[t["side"] == "sell"].groupby("_bar")["quote_qty"].sum()
    bars["buy_vol"]  = buy_vol.reindex(bars.index,  fill_value=0.0)
    bars["sell_vol"] = sell_vol.reindex(bars.index, fill_value=0.0)
    total = bars["buy_vol"] + bars["sell_vol"]
    bars["imbalance"] = np.where(
        total > 0, (bars["buy_vol"] - bars["sell_vol"]) / total, 0.0
    )
    return bars.reset_index(drop=True)


# =====================================================
# 5. VPIN & CVD
# =====================================================
def compute_vpin(bars: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    bars = bars.copy()
    if bars.empty or len(bars) < window:
        bars["vpin"] = np.nan
        return bars
    abs_imb    = (bars["buy_vol"] - bars["sell_vol"]).abs()
    total      = bars["buy_vol"] + bars["sell_vol"]
    bars["vpin"] = abs_imb.rolling(window).sum() / total.rolling(window).sum()
    return bars


def compute_cvd(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return bars
    bars = bars.copy()
    bars["delta"] = bars["buy_vol"] - bars["sell_vol"]
    bars["cvd"]   = bars["delta"].cumsum()
    return bars


def cvd_slope_significance(cvd_values: np.ndarray, delta_std: float) -> dict:
    n = len(cvd_values)
    if n < 5:
        return {"slope": 0.0, "t_stat": 0.0, "normalized_slope": 0.0, "significant": False}
    x      = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = cvd_values.mean()
    Sxx    = np.sum((x - x_mean) ** 2)
    Sxy    = np.sum((x - x_mean) * (cvd_values - y_mean))
    slope  = Sxy / Sxx if Sxx > 0 else 0.0
    intercept = y_mean - slope * x_mean
    residuals = cvd_values - (slope * x + intercept)
    rss    = np.sum(residuals ** 2)
    sigma2 = rss / (n - 2) if n > 2 else 0.0
    se_slope = np.sqrt(sigma2 / Sxx) if Sxx > 0 and sigma2 > 0 else 0.0
    t_stat   = slope / se_slope if se_slope > 0 else 0.0
    return {
        "slope":            slope,
        "t_stat":           t_stat,
        "normalized_slope": slope / delta_std if delta_std > 0 else 0.0,
        "significant":      t_stat > 2.0,
    }


# =====================================================
# 6. Buyback Score
# =====================================================
def buyback_score(bars, vpin_threshold=0.3, cvd_lookback=50, t_critical=2.0):
    empty = {
        "score": 0, "vpin_signal": False, "cvd_signal": False, "imb_signal": False,
        "vpin": 0.0, "cvd_slope": 0.0, "cvd_t_stat": 0.0,
        "cvd_norm_slope": 0.0, "mean_imbalance": 0.0,
    }
    if bars.empty or "vpin" not in bars.columns:
        return empty
    recent = bars.tail(cvd_lookback)
    if recent.empty:
        return empty
    latest_vpin = recent["vpin"].iloc[-1]
    if pd.isna(latest_vpin):
        latest_vpin = 0.0
    vpin_signal = latest_vpin > vpin_threshold
    delta_std   = recent["delta"].std() if "delta" in recent.columns else 0.0
    cvd_stats   = cvd_slope_significance(recent["cvd"].values, delta_std)
    cvd_signal  = cvd_stats["t_stat"] > t_critical and cvd_stats["slope"] > 0
    mean_imb    = recent["imbalance"].mean()
    imb_signal  = mean_imb > 0.1
    return {
        "score":          int(vpin_signal) + int(cvd_signal) + int(imb_signal),
        "vpin":           latest_vpin,
        "cvd_slope":      cvd_stats["slope"],
        "cvd_t_stat":     cvd_stats["t_stat"],
        "cvd_norm_slope": cvd_stats["normalized_slope"],
        "mean_imbalance": mean_imb,
        "vpin_signal":    vpin_signal,
        "cvd_signal":     cvd_signal,
        "imb_signal":     imb_signal,
    }


# =====================================================
# 7. Cached computation
# =====================================================
@st.cache_data(ttl=60, show_spinner=False)
def load_and_compute(label, hours_back, dollar_threshold, vpin_window, vpin_threshold, t_critical):
    trades = load_trades(label, hours_back=hours_back)
    if trades.empty:
        return None
    bars = build_dollar_bars(trades, dollar_threshold)
    bars = compute_vpin(bars, window=vpin_window)
    bars = compute_cvd(bars)
    sig  = buyback_score(bars, vpin_threshold=vpin_threshold, t_critical=t_critical)
    return {"trades": trades, "bars": bars, "signal": sig}


# =====================================================
# 8. Notification helper
# =====================================================
def fire_notification(alert_exchanges: list, scores: list):
    body = "  |  ".join(f"{ex}: {sc}/3" for ex, sc in zip(alert_exchanges, scores))
    js = f"""
    <script>
    (function() {{
        var title = "NXPC Buyback Alert";
        var body  = {json.dumps(body)};
        function doNotify() {{
            try {{ new Notification(title, {{ body: body }}); }} catch(e) {{}}
            try {{
                var ctx = new (window.AudioContext || window.webkitAudioContext)();
                var osc = ctx.createOscillator();
                var g   = ctx.createGain();
                osc.connect(g); g.connect(ctx.destination);
                osc.frequency.setValueAtTime(880, ctx.currentTime);
                osc.frequency.setValueAtTime(660, ctx.currentTime + 0.15);
                g.gain.setValueAtTime(0.3, ctx.currentTime);
                g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.5);
                osc.start(); osc.stop(ctx.currentTime + 0.5);
            }} catch(e) {{}}
        }}
        if (typeof Notification !== "undefined") {{
            if (Notification.permission === "granted") {{
                doNotify();
            }} else if (Notification.permission !== "denied") {{
                Notification.requestPermission().then(function(p) {{
                    if (p === "granted") doNotify();
                }});
            }}
        }}
    }})();
    </script>
    """
    components.html(js, height=0)


# =====================================================
# 9. UI
# =====================================================
init_db()

st.markdown(
    f"<h1>NXPC <span class='accent'>Buyback Detector</span></h1>",
    unsafe_allow_html=True,
)
st.caption("Dollar bars + VPIN + CVD  ·  Binance / Bybit / OKX")

# ── Sidebar: analysis controls ────────────────────────────────────────────
st.sidebar.header("Settings")

dollar_threshold = st.sidebar.number_input(
    "Dollar bar size (USD)", min_value=1_000, max_value=500_000, value=50_000, step=5_000,
)
vpin_window    = st.sidebar.slider("VPIN window (# bars)", 10, 200, 50)
vpin_threshold = st.sidebar.slider("VPIN alert threshold", 0.1, 0.9, 0.3, 0.05)
t_critical     = st.sidebar.slider("CVD t-stat threshold", 1.0, 4.0, 2.0, 0.1)
hours_back     = st.sidebar.slider("Lookback (hours)", 1, 168, 24)

st.sidebar.divider()

# ── Sidebar: auto-refresh controls ───────────────────────────────────────
auto_refresh = st.sidebar.toggle("Auto-refresh", value=False)
if auto_refresh:
    refresh_sec = st.sidebar.select_slider(
        "Refresh interval",
        options=[30, 60, 120, 300, 600],
        value=60,
        format_func=lambda x: f"{x}s" if x < 60 else f"{x // 60}m",
    )
    alert_min_score = st.sidebar.slider("Alert score threshold", 1, 3, 2)
else:
    refresh_sec     = None
    alert_min_score = 2

st.sidebar.divider()

# ── Sidebar: fetch + notification permission ──────────────────────────────
fetch_clicked = st.sidebar.button("Fetch now", use_container_width=True, type="primary")

if st.sidebar.button("Enable browser alerts", use_container_width=True):
    components.html(
        """<script>
        if ('Notification' in window && Notification.permission !== 'granted') {
            Notification.requestPermission();
        }
        </script>""",
        height=0,
    )

# ── Manual fetch handler ──────────────────────────────────────────────────
if fetch_clicked:
    with st.spinner("Fetching from Tier 1 venues..."):
        for label, (adapter, symbol) in ADAPTERS.items():
            try:
                df = adapter.fetch_trades(symbol=symbol, limit=1000)
                persist_trades(df)
                st.sidebar.success(f"{label}: {len(df)} trades saved")
            except Exception as e:
                st.sidebar.error(f"{label}: {str(e)[:70]}")
            time.sleep(0.2)
    st.cache_data.clear()


# ── Auto-refreshing data panel ────────────────────────────────────────────
@st.fragment(run_every=refresh_sec)
def data_panel():
    # Auto-fetch silently on each refresh cycle
    if auto_refresh:
        for label, (adapter, symbol) in ADAPTERS.items():
            try:
                df = adapter.fetch_trades(symbol=symbol, limit=1000)
                persist_trades(df)
            except Exception:
                pass
            time.sleep(0.2)
        st.cache_data.clear()

    # Compute
    results = {}
    for label in ADAPTERS:
        results[label] = load_and_compute(
            label, hours_back, float(dollar_threshold),
            vpin_window, vpin_threshold, t_critical,
        )

    # Fire alert when threshold reached
    alert_exchanges = [
        lbl for lbl, r in results.items()
        if r and r["signal"]["score"] >= alert_min_score
    ]
    if alert_exchanges:
        scores = [results[lbl]["signal"]["score"] for lbl in alert_exchanges]
        fire_notification(alert_exchanges, scores)

    # ── Tabs ──────────────────────────────────────────────────────────────
    tabs = st.tabs(["Overview"] + list(ADAPTERS.keys()))

    with tabs[0]:
        if auto_refresh:
            st.info(
                f"Auto-refreshing every "
                f"{'{}s'.format(refresh_sec) if refresh_sec < 60 else '{}m'.format(refresh_sec // 60)}"
                f"  |  Alert threshold: {alert_min_score}/3"
                f"  |  Last: {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC"
            )

        cols = st.columns(3)
        for i, label in enumerate(ADAPTERS):
            with cols[i]:
                r = results[label]
                if r is None:
                    st.metric(label, "No data", "Fetch trades first")
                    continue
                sig   = r["signal"]
                score = sig["score"]
                emoji = "🔴" if score >= 3 else ("🟡" if score == 2 else "🟢")
                st.metric(f"{emoji} {label}", f"Score: {score}/3", f"VPIN: {sig['vpin']:.2f}")

        st.divider()
        st.subheader("Signal breakdown")
        rows_data = []
        for label, r in results.items():
            if r is None:
                continue
            s = r["signal"]
            rows_data.append({
                "Exchange":       label,
                "Bars":           len(r["bars"]),
                "VPIN":           f"{s['vpin']:.3f}",
                "VPIN > thr":     "v" if s["vpin_signal"] else "-",
                "CVD t-stat":     f"{s['cvd_t_stat']:+.2f}",
                "CVD norm.slope": f"{s['cvd_norm_slope']:+.3f}",
                "CVD signif.":    "v" if s["cvd_signal"] else "-",
                "Mean imb.":      f"{s['mean_imbalance']:+.3f}",
                "Imb > 0.1":      "v" if s["imb_signal"] else "-",
                "Score":          f"{s['score']}/3",
            })
        if rows_data:
            st.dataframe(pd.DataFrame(rows_data), use_container_width=True, hide_index=True)
            st.caption("Score >= 2 suggests likely buyback activity.")
        else:
            st.info("Fetch trades from the sidebar to begin.")

    for i, label in enumerate(ADAPTERS, start=1):
        with tabs[i]:
            r = results[label]
            if r is None:
                st.info(f"No data for {label}. Use 'Fetch now' in the sidebar.")
                continue

            bars   = r["bars"]
            trades = r["trades"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Trades",       f"{len(trades):,}")
            c2.metric("Dollar bars",  f"{len(bars):,}")
            c3.metric("Last price",   f"${trades['price'].iloc[-1]:.4f}" if not trades.empty else "-")
            c4.metric("Period $ vol", f"${trades['quote_qty'].sum():,.0f}")

            if bars.empty or "vpin" not in bars.columns:
                st.warning("Not enough trades. Lower the bar size or fetch more data.")
                continue

            fig = make_subplots(
                rows=3, cols=1, shared_xaxes=True,
                row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.05,
                subplot_titles=("Price (dollar bars)", "VPIN", "CVD (bars=delta, line=cumulative)"),
            )
            fig.add_trace(
                go.Candlestick(
                    x=bars["ts"], open=bars["open"], high=bars["high"],
                    low=bars["low"], close=bars["close"],
                    increasing_line_color=GREEN, decreasing_line_color=RED, name="Price",
                ), row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(x=bars["ts"], y=bars["vpin"],
                           line=dict(color=PRESTO_ORANGE, width=2), name="VPIN"),
                row=2, col=1,
            )
            fig.add_hline(y=vpin_threshold, line_dash="dash", line_color=RED, row=2, col=1)
            delta_colors = [GREEN if v >= 0 else RED for v in bars["delta"]]
            fig.add_trace(
                go.Bar(x=bars["ts"], y=bars["delta"], marker_color=delta_colors, name="Delta"),
                row=3, col=1,
            )
            fig.add_trace(
                go.Scatter(x=bars["ts"], y=bars["cvd"],
                           line=dict(color=PRESTO_DARK, width=1.5), name="CVD"),
                row=3, col=1,
            )
            fig.update_layout(
                height=720, showlegend=False, xaxis_rangeslider_visible=False,
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Recent dollar bars (last 20)"):
                display_cols = ["ts", "close", "dollar_volume", "buy_vol", "sell_vol",
                                "imbalance", "vpin", "delta", "cvd"]
                st.dataframe(bars.tail(20)[display_cols], use_container_width=True, hide_index=True)


data_panel()

st.divider()
st.caption(
    "Presto Labs  |  Internal Use Only  |  "
    "Methodology: Lopez de Prado (2018), Easley et al. (2012)"
)
