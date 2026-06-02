"""
NXPC Buyback Monitor — Cloud Edition
======================================
1차 vs 2차 트랜치 | TWAP 슬라이싱 | 대형 매도 감지
Data: Binance klines (historical) + aggTrades (recent 48h)
Internal — Presto Labs
"""

from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import urllib3

urllib3.disable_warnings()

# ── Constants ────────────────────────────────────────────────────────────────
SYMBOL = "NXPCUSDT"
T1_START = pd.Timestamp("2026-05-25 07:00", tz="UTC")
T1_END = pd.Timestamp("2026-05-31 12:00", tz="UTC")
T2_START = pd.Timestamp("2026-06-01 10:00", tz="UTC")
BASELINE = 10_667.0  # $/hr organic baseline (05/24 18:00~05/25 07:00 UTC avg)

C_ORANGE = "#E8742A"
C_DARK = "#1A1A2E"
C_BLUE = "#3B82F6"
C_GREEN = "#10B981"
C_RED = "#EF4444"
C_BG = "#0F172A"
C_CARD = "#1E293B"
C_BORDER = "#334155"
C_TEXT = "#F1F5F9"
C_MUTED = "#94A3B8"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NXPC Buyback Monitor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    f"""
<style>
  .stApp {{ background-color: {C_BG}; }}
  .main .block-container {{ padding: 1.5rem 2rem 2rem; max-width: 1400px; }}
  section[data-testid="stSidebar"] {{ background: {C_CARD}; }}
  header[data-testid="stHeader"] {{ background: transparent; }}

  .metric-card {{
    background: {C_CARD}; border: 1px solid {C_BORDER};
    border-radius: 12px; padding: 1.1rem 1.3rem; height: 100%;
  }}
  .metric-label {{
    font-size: 11px; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: {C_MUTED}; margin-bottom: 6px;
  }}
  .metric-value {{
    font-size: 26px; font-weight: 700; color: {C_TEXT};
    line-height: 1.1; white-space: nowrap;
  }}
  .metric-sub   {{ font-size: 12px; color: {C_MUTED}; margin-top: 4px; }}
  .metric-delta-up   {{ color: {C_GREEN}; font-size: 12px; font-weight: 600; margin-top: 4px; }}
  .metric-delta-down {{ color: {C_RED};   font-size: 12px; font-weight: 600; margin-top: 4px; }}
  .metric-delta-neu  {{ color: {C_MUTED}; font-size: 12px; font-weight: 600; margin-top: 4px; }}

  .badge-alert {{ display:inline-block; background:#7F1D1D; color:#FCA5A5;
    padding:2px 10px; border-radius:20px; font-size:11px; font-weight:700; }}
  .badge-ok    {{ display:inline-block; background:#064E3B; color:#6EE7B7;
    padding:2px 10px; border-radius:20px; font-size:11px; font-weight:700; }}
  .badge-warn  {{ display:inline-block; background:#78350F; color:#FCD34D;
    padding:2px 10px; border-radius:20px; font-size:11px; font-weight:700; }}

  .section-header {{
    font-size: 12px; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: {C_MUTED};
    border-bottom: 1px solid {C_BORDER}; padding-bottom: 8px; margin: 1.5rem 0 1rem;
  }}
  .tranche-card {{
    background: {C_CARD}; border: 1px solid {C_BORDER};
    border-radius: 12px; padding: 1.2rem 1.4rem;
  }}
  .tranche-title {{
    font-size: 12px; font-weight: 700; color: {C_MUTED};
    text-transform: uppercase; letter-spacing: .08em; margin-bottom: 12px;
  }}
  .tranche-row  {{ display: flex; gap: 1.5rem; flex-wrap: wrap; }}
  .tranche-item {{ flex: 1; min-width: 80px; }}
  .tranche-item-label {{ font-size: 11px; color: {C_MUTED}; margin-bottom: 2px; }}
  .tranche-item-value {{ font-size: 20px; font-weight: 700; color: {C_TEXT}; white-space: nowrap; }}
  .tranche-item-sub   {{ font-size: 11px; color: {C_MUTED}; }}

  .dot-green {{ display:inline-block; width:8px; height:8px; border-radius:50%;
    background:{C_GREEN}; margin-right:6px; box-shadow:0 0 6px {C_GREEN}; }}
  .dot-yellow {{ display:inline-block; width:8px; height:8px; border-radius:50%;
    background:#FBBF24; margin-right:6px; box-shadow:0 0 6px #FBBF24; }}

  hr {{ border-color: {C_BORDER} !important; }}
  [data-testid="stMetric"] {{ display: none; }}
</style>
""",
    unsafe_allow_html=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt(n: float, d: int = 1) -> str:
    if abs(n) >= 1_000_000:
        return f"${n/1_000_000:.{d}f}M"
    if abs(n) >= 1_000:
        return f"${n/1_000:.{d}f}K"
    return f"${n:.0f}"


def fmt_day(n: float) -> str:
    return fmt(n) + "/day"


def metric_html(label, value, sub="", delta="", delta_dir="neu") -> str:
    d_html = f'<div class="metric-delta-{delta_dir}">{delta}</div>' if delta else ""
    s_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return f"""<div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      {s_html}{d_html}
    </div>"""


def tranche_html(title, color, items, badge="") -> str:
    b_html = f'<span style="float:right">{badge}</span>' if badge else ""
    rows = "".join(
        f"""<div class="tranche-item">
      <div class="tranche-item-label">{l}</div>
      <div class="tranche-item-value" style="color:{vc}">{v}</div>
      <div class="tranche-item-sub">{s}</div>
    </div>"""
        for l, v, s, vc in items
    )
    return f"""<div class="tranche-card">
      <div class="tranche-title" style="color:{color}">{title}{b_html}</div>
      <div class="tranche-row">{rows}</div>
    </div>"""


def plotly_cfg() -> dict:
    return dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C_TEXT, family="Inter, system-ui, sans-serif", size=12),
        margin=dict(l=10, r=10, t=32, b=32),
        xaxis=dict(gridcolor=C_BORDER, linecolor=C_BORDER, zerolinecolor=C_BORDER),
        yaxis=dict(gridcolor=C_BORDER, linecolor=C_BORDER, zerolinecolor=C_BORDER),
        legend=dict(
            orientation="h", x=0, y=1.12, bgcolor="rgba(0,0,0,0)", font=dict(size=11)
        ),
    )


# ── API helpers ───────────────────────────────────────────────────────────────
_BINANCE_HOSTS = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
]


def _get(url, params, retries=2):
    """Try Binance mirror hosts automatically on 451/5xx errors."""
    path = url.split("binance.com", 1)[-1] if "binance.com" in url else None
    hosts = _BINANCE_HOSTS if path else [url.rsplit("/", len(url.split("/")) - 3)[0]]
    last_exc = None
    for host in hosts:
        full_url = (host + path) if path else url
        for _ in range(retries):
            try:
                r = requests.get(full_url, params=params, timeout=15, verify=False)
                r.raise_for_status()
                return r.json()
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code in (451, 403):
                    last_exc = e
                    break  # try next host
                last_exc = e
            except Exception as e:
                last_exc = e
    raise last_exc


@st.cache_data(ttl=120, show_spinner=False)
def fetch_klines(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """Fetch 1h klines — returns hourly buy_quote (taker buy volume in USDT)."""
    rows, start_ms = [], int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)
    while start_ms < end_ms:
        batch = _get(
            "https://api.binance.com/api/v3/klines",
            {
                "symbol": SYMBOL,
                "interval": "1h",
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 1000,
            },
        )
        if not batch:
            break
        rows.extend(batch)
        last = batch[-1][0]
        if len(batch) < 1000 or last >= end_ms:
            break
        start_ms = last + 3_600_000
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(
        rows,
        columns=[
            "ts_open",
            "open",
            "high",
            "low",
            "close",
            "vol",
            "ts_close",
            "quote_vol",
            "n_trades",
            "tbuy_base",
            "tbuy_quote",
            "ignore",
        ],
    )
    df["ts"] = pd.to_datetime(df["ts_open"], unit="ms", utc=True)
    df["kst"] = df["ts"] + pd.Timedelta(hours=9)
    df["buy_quote"] = df["tbuy_quote"].astype(float)
    df["close"] = df["close"].astype(float)
    return df[["ts", "kst", "buy_quote", "close"]].copy()


@st.cache_data(ttl=60, show_spinner=False)
def fetch_agg_trades(hours_back: int = 48) -> pd.DataFrame:
    """Fetch recent aggTrades for TWAP + large sell analysis."""
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    start_ms = now_ms - hours_back * 3_600_000
    rows, cur = [], start_ms
    while cur < now_ms:
        batch = _get(
            "https://api.binance.com/api/v3/aggTrades",
            {
                "symbol": SYMBOL,
                "startTime": cur,
                "endTime": min(cur + 3_600_000, now_ms),
                "limit": 1000,
            },
        )
        if not batch:
            cur += 3_600_000
            continue
        rows.extend(batch)
        last = batch[-1]["T"]
        cur = last + 1 if len(batch) == 1000 else cur + 3_600_000
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["T"], unit="ms", utc=True)
    df["kst"] = df["ts"] + pd.Timedelta(hours=9)
    df["price"] = df["p"].astype(float)
    df["qty"] = df["q"].astype(float)
    df["side"] = (~df["m"]).map({True: "buy", False: "sell"})
    df["quote"] = df["price"] * df["qty"]
    return df[["ts", "kst", "price", "qty", "side", "quote"]].copy()


@st.cache_data(ttl=30, show_spinner=False)
def fetch_current_price() -> float:
    try:
        data = _get("https://api.binance.com/api/v3/ticker/price", {"symbol": SYMBOL})
        return float(data["price"])
    except Exception:
        # CoinGecko fallback (no geo-restriction)
        cg = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": "nxpc", "vs_currencies": "usd"},
            timeout=10,
        )
        cg.raise_for_status()
        return float(cg.json()["nxpc"]["usd"])


# ── Computation ───────────────────────────────────────────────────────────────
def hourly_excess(klines: pd.DataFrame) -> pd.DataFrame:
    h = klines.copy()
    h["excess"] = (h["buy_quote"] - BASELINE).clip(lower=0)
    h["hod"] = h["kst"].dt.hour
    return h


def compute_vpin(df: pd.DataFrame) -> float:
    if len(df) < 50:
        return float("nan")
    df = df.copy().reset_index(drop=True)
    df["cum"] = df["quote"].cumsum()
    df["bar"] = (df["cum"] // 50_000).astype(int)
    buy_q = df["quote"].where(df["side"] == "buy", 0)
    sell_q = df["quote"].where(df["side"] == "sell", 0)
    bars = pd.DataFrame(
        {
            "buy": buy_q.groupby(df["bar"]).sum(),
            "sell": sell_q.groupby(df["bar"]).sum(),
        }
    )
    total = bars["buy"] + bars["sell"]
    vpin = (bars["buy"] - bars["sell"]).abs().rolling(
        10, min_periods=5
    ).sum() / total.rolling(10, min_periods=5).sum()
    v = vpin.iloc[-1]
    return float(v) if not np.isnan(v) else float("nan")


def klines_pace(klines: pd.DataFrame, hours: int, now: pd.Timestamp) -> float:
    cut = now - pd.Timedelta(hours=hours)
    sub = klines[klines["ts"] >= max(cut, T2_START)]
    if sub.empty:
        return 0.0
    hrs = (now - max(cut, T2_START)).total_seconds() / 3600
    return (
        max(0.0, sub["buy_quote"].sum() - BASELINE * hrs) / hrs * 24 if hrs > 0 else 0.0
    )


def detect_twap(df: pd.DataFrame):
    if len(df) < 10:
        return pd.DataFrame(), pd.DataFrame()
    buys = df[df["side"] == "buy"].sort_values("ts").copy()
    buys["gap"] = buys["ts"].diff().dt.total_seconds()
    buys["burst_id"] = (buys["gap"] > 0.5).cumsum()
    bursts = (
        buys.groupby("burst_id")
        .agg(
            ts=("ts", "first"),
            kst=("kst", "first"),
            quote=("quote", "sum"),
            n=("quote", "count"),
        )
        .reset_index(drop=True)
    )
    bursts["inter_gap"] = bursts["ts"].diff().dt.total_seconds()
    twap = bursts[bursts["inter_gap"].between(5, 120)].copy()
    return bursts, twap


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"<div style='color:{C_ORANGE};font-weight:700;font-size:14px;margin-bottom:12px'>⚙ Controls</div>",
        unsafe_allow_html=True,
    )
    auto_refresh = st.toggle("Auto-refresh (90s)", value=False)
    if st.button("Refresh now", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.caption(f"Baseline: ${BASELINE:,.0f}/hr")
    st.caption("T1 start: 05/25 16:00 KST")
    st.caption("T2 start: 06/01 19:00 KST")
    st.caption("klines: 1h interval")
    st.caption("TWAP: recent 48h aggTrades")


# ── Main ──────────────────────────────────────────────────────────────────────
@st.fragment(run_every=90 if auto_refresh else None)
def render():
    if auto_refresh:
        st.cache_data.clear()

    now = pd.Timestamp.now(tz="UTC")

    with st.spinner("데이터 로딩 중..."):
        try:
            price = fetch_current_price()
            kl_t1 = fetch_klines(T1_START, T1_END)
            kl_t2 = fetch_klines(T2_START, now)
            trades = fetch_agg_trades(48)
        except Exception as e:
            st.error(f"API 오류: {e}")
            return

    if kl_t1.empty or kl_t2.empty:
        st.error("klines 데이터 없음. 잠시 후 다시 시도해주세요.")
        return

    # ── Compute ───────────────────────────────────────────────────────────────
    h1 = hourly_excess(kl_t1)
    h2 = hourly_excess(kl_t2)

    t1_days = (T1_END - T1_START).total_seconds() / 86400
    t2_hrs = (now - T2_START).total_seconds() / 3600
    t1_excess = float(h1["excess"].sum())
    t2_excess = float(h2["excess"].sum())
    t1_daily = t1_excess / t1_days
    t2_daily = t2_excess / (t2_hrs / 24) if t2_hrs > 0 else 0

    vpin = compute_vpin(trades) if not trades.empty else float("nan")
    p1h = klines_pace(kl_t2, 1, now)
    p3h = klines_pace(kl_t2, 3, now)
    p6h = klines_pace(kl_t2, 6, now)

    # ── Header ────────────────────────────────────────────────────────────────
    kst_str = (now + pd.Timedelta(hours=9)).strftime("%m/%d %H:%M")
    vpin_badge = (
        '<span class="badge-alert">ALERT</span>'
        if not np.isnan(vpin) and vpin >= 0.30
        else (
            '<span class="badge-warn">WATCH</span>'
            if not np.isnan(vpin) and vpin >= 0.22
            else '<span class="badge-ok">NORMAL</span>'
        )
    )
    dot = (
        '<span class="dot-green"></span>'
        if auto_refresh
        else '<span class="dot-yellow"></span>'
    )

    st.markdown(
        f"""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:.5rem">
      <div>
        <span style="font-size:22px;font-weight:800;color:{C_TEXT}">NXPC Buyback Monitor</span>
        <span style="font-size:13px;color:{C_MUTED};margin-left:12px">Binance · Presto Labs</span>
      </div>
      <div style="font-size:12px;color:{C_MUTED}">{dot}{kst_str} KST &nbsp;·&nbsp; VPIN {vpin_badge}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── Top metrics ───────────────────────────────────────────────────────────
    cols = st.columns(5, gap="small")
    vpin_str = f"{vpin:.3f}" if not np.isnan(vpin) else "N/A"
    pace_ratio = f"{p1h/t1_daily*100:.0f}% vs 1차" if t1_daily > 0 else ""
    pace_dir = "up" if p1h > t1_daily else "down" if p1h < t1_daily * 0.5 else "neu"
    vpin_dir = "down" if not np.isnan(vpin) and vpin >= 0.30 else "neu"

    with cols[0]:
        st.markdown(
            metric_html("현재가", f"${price:.4f}", "NXPC/USDT"), unsafe_allow_html=True
        )
    with cols[1]:
        st.markdown(
            metric_html(
                "VPIN",
                vpin_str,
                "0.30 = Alert",
                "⚠ 경보" if not np.isnan(vpin) and vpin >= 0.30 else "정상",
                vpin_dir,
            ),
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            metric_html("Pace 1h", fmt_day(p1h), "최근 1h 환산", pace_ratio, pace_dir),
            unsafe_allow_html=True,
        )
    with cols[3]:
        st.markdown(
            metric_html("Pace 3h", fmt_day(p3h), "최근 3h 환산"), unsafe_allow_html=True
        )
    with cols[4]:
        st.markdown(
            metric_html("Pace 6h", fmt_day(p6h), "최근 6h 환산"), unsafe_allow_html=True
        )

    st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

    # ── Tranche cards ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">트랜치 현황</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")

    with c1:
        st.markdown(
            tranche_html(
                "1차 트랜치  ·  05/25 ~ 05/31",
                C_ORANGE,
                [
                    (
                        "기간",
                        f"{t1_days:.1f}일",
                        "05/25 16:00 → 05/31 21:00 KST",
                        C_TEXT,
                    ),
                    ("누적 Excess", fmt(t1_excess, 2), "vs organic baseline", C_ORANGE),
                    ("일평균", fmt_day(t1_daily), "기준 페이스", C_TEXT),
                ],
                badge='<span class="badge-ok">완료</span>',
            ),
            unsafe_allow_html=True,
        )

    with c2:
        ratio = t2_daily / t1_daily * 100 if t1_daily > 0 else 0
        rc = C_GREEN if ratio >= 80 else C_ORANGE if ratio >= 50 else C_RED
        st.markdown(
            tranche_html(
                f"2차 트랜치  ·  06/01 ~ 진행중  ({t2_hrs:.0f}h 경과)",
                C_BLUE,
                [
                    ("경과", f"{t2_hrs:.0f}h", f"({t2_hrs/24:.1f}일)", C_TEXT),
                    ("누적 Excess", fmt(t2_excess, 2), "vs organic baseline", C_BLUE),
                    ("일평균", fmt_day(t2_daily), f"1차 대비 {ratio:.0f}%", rc),
                ],
                badge='<span class="badge-warn">진행중</span>',
            ),
            unsafe_allow_html=True,
        )

    # ── KST HOD chart ─────────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">KST 시간대별 초과 매수</div>',
        unsafe_allow_html=True,
    )

    hod1 = h1.groupby("hod")["excess"].mean()
    hod2 = h2.groupby("hod")["excess"].mean()
    hrs = list(range(24))
    xlbl = [f"{h:02d}:00" for h in hrs]

    fig_hod = go.Figure()
    fig_hod.add_trace(
        go.Bar(
            x=xlbl,
            y=[float(hod1.get(h, 0)) for h in hrs],
            name="1차 평균",
            marker_color=C_ORANGE,
            opacity=0.65,
            width=0.35,
            offset=-0.18,
            hovertemplate="1차 %{x}<br>$%{y:,.0f}<extra></extra>",
        )
    )
    fig_hod.add_trace(
        go.Bar(
            x=xlbl,
            y=[float(hod2.get(h, 0)) for h in hrs],
            name="2차 평균",
            marker_color=C_BLUE,
            opacity=0.9,
            width=0.35,
            offset=0.18,
            hovertemplate="2차 %{x}<br>$%{y:,.0f}<extra></extra>",
        )
    )
    fig_hod.add_vrect(
        x0="09:00",
        x1="18:00",
        fillcolor="rgba(255,255,255,0.03)",
        layer="below",
        line_width=0,
        annotation_text="KST 영업시간",
        annotation_font=dict(color=C_MUTED, size=10),
        annotation_position="top left",
    )
    cfg = plotly_cfg()
    cfg.update(
        height=300,
        barmode="overlay",
        yaxis=dict(
            title="Excess $/hr",
            tickformat="$,.0f",
            gridcolor=C_BORDER,
            linecolor=C_BORDER,
            zerolinecolor=C_BORDER,
        ),
    )
    fig_hod.update_layout(**cfg)
    st.plotly_chart(fig_hod, use_container_width=True, config={"displayModeBar": False})

    with st.expander("2차 시간별 상세 테이블"):
        disp = h2[["kst", "buy_quote", "excess"]].copy()
        disp["KST"] = disp["kst"].dt.strftime("%m/%d %H:%M")
        disp["매수$"] = disp["buy_quote"].apply(lambda x: f"${x:,.0f}")
        disp["초과$"] = disp["excess"].apply(lambda x: f"${x:,.0f}")
        st.dataframe(
            disp[["KST", "매수$", "초과$"]],
            use_container_width=True,
            hide_index=True,
            height=300,
        )

    # ── TWAP (recent 48h aggTrades) ───────────────────────────────────────────
    st.markdown(
        '<div class="section-header">TWAP 슬라이싱 분석  ·  최근 48h</div>',
        unsafe_allow_html=True,
    )

    if trades.empty:
        st.info("aggTrades 데이터 없음")
    else:
        t2_cut = now - pd.Timedelta(hours=48)
        t1_cut = now - pd.Timedelta(hours=96)
        tr_t2 = trades[trades["ts"] >= t2_cut]
        tr_t1 = trades[(trades["ts"] >= t1_cut) & (trades["ts"] < t2_cut)]

        bursts2, twap2 = detect_twap(tr_t2)
        bursts1, twap1 = detect_twap(tr_t1)

        def twap_stats(tw, bur, days):
            if tw.empty or days <= 0:
                return None
            ig = tw["inter_gap"].dropna()
            cv = float(ig.std() / ig.mean()) if len(ig) > 2 else float("nan")
            return {
                "n": len(tw),
                "ratio": len(tw) / max(len(bur), 1) * 100,
                "cv": cv,
                "gap": float(ig.mean()),
                "slice": float(tw["quote"].mean()),
                "daily": float(tw["quote"].sum() / days),
            }

        s1 = twap_stats(twap1, bursts1, 2)
        s2 = twap_stats(twap2, bursts2, 2)

        def twap_card(col, s, label, color):
            with col:
                if not s:
                    st.info("데이터 부족")
                    return
                cv_c = (
                    C_GREEN if s["cv"] < 0.85 else C_ORANGE if s["cv"] < 1.2 else C_RED
                )
                cv_l = (
                    "규칙적"
                    if s["cv"] < 0.85
                    else "보통" if s["cv"] < 1.2 else "불규칙"
                )
                st.markdown(
                    tranche_html(
                        label,
                        color,
                        [
                            (
                                "TWAP 버스트",
                                f"{s['n']:,}개",
                                f"전체의 {s['ratio']:.0f}%",
                                C_TEXT,
                            ),
                            ("Gap CV", f"{s['cv']:.3f}", cv_l, cv_c),
                            ("평균 Gap", f"{s['gap']:.1f}s", "버스트 간격", C_TEXT),
                            ("슬라이스 평균", fmt(s["slice"]), "버스트당", color),
                            (
                                "일평균 TWAP",
                                fmt_day(s["daily"]),
                                "TWAP 집행 추정",
                                C_TEXT,
                            ),
                        ],
                    ),
                    unsafe_allow_html=True,
                )

        c1, c2 = st.columns(2, gap="medium")
        twap_card(c1, s1, "직전 48h (06/01 이전)", C_ORANGE)
        twap_card(c2, s2, "최근 48h (현재 기준)", C_BLUE)

        # Gap distribution
        bins = [0, 5, 10, 20, 30, 60, 120, 300, 9999]
        blbls = ["<5s", "5-10s", "10-20s", "20-30s", "30-60s", "60-120s", "2-5m", ">5m"]

        def gap_pct(tw):
            if tw.empty:
                return [0] * len(blbls)
            ig = tw["inter_gap"].dropna()
            cut = pd.cut(ig, bins=bins, labels=blbls)
            cnt = cut.value_counts().sort_index()
            tot = max(len(ig), 1)
            return [cnt.get(l, 0) / tot * 100 for l in blbls]

        cg, ch = st.columns(2, gap="medium")

        with cg:
            fg = go.Figure()
            fg.add_trace(
                go.Bar(
                    x=blbls,
                    y=gap_pct(twap1),
                    name="직전 48h",
                    marker_color=C_ORANGE,
                    opacity=0.65,
                    hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
                )
            )
            fg.add_trace(
                go.Bar(
                    x=blbls,
                    y=gap_pct(twap2),
                    name="최근 48h",
                    marker_color=C_BLUE,
                    opacity=0.9,
                    hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
                )
            )
            c = plotly_cfg()
            c.update(
                height=240,
                barmode="group",
                title=dict(
                    text="Inter-burst Gap 분포", font=dict(size=13, color=C_MUTED), x=0
                ),
                yaxis=dict(
                    title="비율 (%)",
                    ticksuffix="%",
                    gridcolor=C_BORDER,
                    linecolor=C_BORDER,
                    zerolinecolor=C_BORDER,
                ),
            )
            fg.update_layout(**c)
            st.plotly_chart(
                fg, use_container_width=True, config={"displayModeBar": False}
            )

        with ch:

            def hod_twap(tw, days):
                if tw.empty or days <= 0:
                    return pd.Series(dtype=float)
                t = tw.copy()
                t["h"] = t["ts"].dt.tz_convert("Asia/Seoul").dt.hour
                return t.groupby("h")["quote"].sum() / days

            ht1 = hod_twap(twap1, 2)
            ht2 = hod_twap(twap2, 2)
            fh = go.Figure()
            fh.add_trace(
                go.Bar(
                    x=xlbl,
                    y=[float(ht1.get(h, 0)) for h in hrs],
                    name="직전 48h",
                    marker_color=C_ORANGE,
                    opacity=0.65,
                    hovertemplate="%{x}<br>$%{y:,.0f}/day<extra></extra>",
                )
            )
            fh.add_trace(
                go.Bar(
                    x=xlbl,
                    y=[float(ht2.get(h, 0)) for h in hrs],
                    name="최근 48h",
                    marker_color=C_BLUE,
                    opacity=0.9,
                    hovertemplate="%{x}<br>$%{y:,.0f}/day<extra></extra>",
                )
            )
            fh.add_vrect(
                x0="09:00",
                x1="18:00",
                fillcolor="rgba(255,255,255,0.03)",
                layer="below",
                line_width=0,
            )
            c2 = plotly_cfg()
            c2.update(
                height=240,
                barmode="group",
                title=dict(
                    text="KST 시간대별 TWAP 강도",
                    font=dict(size=13, color=C_MUTED),
                    x=0,
                ),
                yaxis=dict(
                    title="$/day",
                    tickformat="$,.0f",
                    gridcolor=C_BORDER,
                    linecolor=C_BORDER,
                    zerolinecolor=C_BORDER,
                ),
            )
            fh.update_layout(**c2)
            st.plotly_chart(
                fh, use_container_width=True, config={"displayModeBar": False}
            )

    # ── Large sell detector ───────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">대형 매도 감지  ·  최근 48h  ($1,000+)</div>',
        unsafe_allow_html=True,
    )

    if trades.empty:
        st.info("데이터 없음")
    else:
        sells = trades[(trades["side"] == "sell") & (trades["quote"] >= 1000)].copy()
        sells = sells.sort_values("quote", ascending=False)

        if sells.empty:
            st.markdown(
                f'<div style="color:{C_MUTED};padding:1rem;text-align:center">최근 48h 내 $1,000+ 단일 매도 없음</div>',
                unsafe_allow_html=True,
            )
        else:
            c1, c2 = st.columns([1, 1.6], gap="medium")
            with c1:
                top = sells.head(20).copy()
                top["KST"] = (top["ts"] + pd.Timedelta(hours=9)).dt.strftime(
                    "%m/%d %H:%M"
                )
                top["가격"] = top["price"].apply(lambda x: f"${x:.4f}")
                top["NXPC"] = top["qty"].apply(lambda x: f"{x:,.0f}")
                top["금액"] = top["quote"].apply(fmt)
                st.dataframe(
                    top[["KST", "가격", "NXPC", "금액"]],
                    use_container_width=True,
                    hide_index=True,
                    height=340,
                )
            with c2:
                s_hr = (
                    sells.assign(
                        kst_h=(sells["ts"] + pd.Timedelta(hours=9)).dt.floor("h")
                    )
                    .groupby("kst_h")["quote"]
                    .sum()
                    .reset_index()
                )
                fs = go.Figure()
                fs.add_trace(
                    go.Bar(
                        x=s_hr["kst_h"].dt.strftime("%m/%d %H:%M"),
                        y=s_hr["quote"],
                        marker_color=C_RED,
                        opacity=0.85,
                        hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
                    )
                )
                cs = plotly_cfg()
                cs.update(
                    height=340,
                    title=dict(
                        text="시간별 대형 매도 합계",
                        font=dict(size=13, color=C_MUTED),
                        x=0,
                    ),
                    yaxis=dict(
                        tickformat="$,.0f",
                        gridcolor=C_BORDER,
                        linecolor=C_BORDER,
                        zerolinecolor=C_BORDER,
                    ),
                    xaxis=dict(
                        tickangle=-35,
                        tickfont=dict(size=10),
                        gridcolor=C_BORDER,
                        linecolor=C_BORDER,
                        zerolinecolor=C_BORDER,
                    ),
                )
                fs.update_layout(**cs)
                st.plotly_chart(
                    fs, use_container_width=True, config={"displayModeBar": False}
                )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        f"""
    <div style="margin-top:2rem;padding-top:1rem;border-top:1px solid {C_BORDER};
      color:{C_MUTED};font-size:11px;display:flex;justify-content:space-between">
      <span>Presto Labs  ·  Internal Use Only</span>
      <span>Baseline ${BASELINE:,.0f}/hr  ·  klines 1h  ·  TWAP gap 5–120s</span>
    </div>
    """,
        unsafe_allow_html=True,
    )


render()
