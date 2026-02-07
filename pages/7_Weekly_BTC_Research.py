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
import streamlit.components.v1 as components
import plotly.graph_objects as go
import certifi
from datetime import datetime, timezone, timedelta

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


def _market_sentiment(pct_chg, rv7, rv30, vol_regime):
    """Return (sentiment_label, short_narrative) based on market conditions."""
    if pct_chg < -5:
        tone = "bearish"
    elif pct_chg < -1:
        tone = "cautious"
    elif pct_chg > 5:
        tone = "bullish"
    elif pct_chg > 1:
        tone = "constructive"
    else:
        tone = "neutral"

    vol_dir = "rising" if (rv30 > 0 and rv7 / rv30 > 1.10) else "falling" if (rv30 > 0 and rv7 / rv30 < 0.90) else "stable"

    narratives = {
        ("bearish", "rising"): "Markets under pressure as volatility spikes. Key support levels are being tested — caution warranted but structured products can capitalize on elevated premiums.",
        ("bearish", "stable"): "Digital assets continue to slide with sentiment turning negative. Support levels are trying to stem losses — a good environment for coupon-generating strategies.",
        ("bearish", "falling"): "Selling pressure persists but volatility is compressing. Consider locking in current premium levels before vol normalizes further.",
        ("cautious", "rising"): "Markets trading lower with uptick in volatility. Elevated premiums offer attractive coupon opportunities for structured products.",
        ("cautious", "stable"): "Slight weakness across digital assets with vol holding steady. Covered call structures look compelling at current levels.",
        ("cautious", "falling"): "Mild pullback with declining vol. Tighter strikes can capture reasonable yield as the market consolidates.",
        ("neutral", "rising"): "Range-bound price action with vol ticking higher. Premium sellers benefit — structured products offer yield pick-up over simple spot holdings.",
        ("neutral", "stable"): "Quiet markets with stable volatility. Consistent coupon generation via covered calls remains the play.",
        ("neutral", "falling"): "Calm markets and declining vol. Consider shorter-dated structures before premiums compress further.",
        ("constructive", "rising"): "Positive momentum with rising volatility — juicy premiums available. Wider strikes recommended for upside participation + coupon.",
        ("constructive", "stable"): "Markets grinding higher with steady vol. Good conditions for call overwriting strategies.",
        ("constructive", "falling"): "Bullish drift with vol coming in. Lock in yield via shorter expiries while premiums remain.",
        ("bullish", "rising"): "Strong rally with vol elevated. Use wider OTM strikes to capture coupon while leaving room for continued upside.",
        ("bullish", "stable"): "Bullish momentum continues. Wider strikes ensure participation in the move while still generating income.",
        ("bullish", "falling"): "Strong move higher with vol declining. Premium levels still attractive — act before further vol compression.",
    }
    narrative = narratives.get((tone, vol_dir), "Markets moving — reach out for tailored structures.")
    return tone, narrative


def _build_sales_memo_en(spot, btc_df, vol_data, multi_recs, pct_chg, rv7, rv30, safety_pct,
                         headline, custom_narrative, extra_lines):
    """Generate an English Telegram-ready sales memo."""
    regime = _vol_regime(vol_data["forecast_rv"], vol_data["long_run_mean"])
    _, auto_narrative = _market_sentiment(pct_chg, rv7, rv30, regime)
    narrative = custom_narrative if custom_narrative.strip() else auto_narrative
    today = datetime.now(timezone.utc).date()

    high_30d = btc_df["high"].tail(30).max()
    low_30d = btc_df["low"].tail(30).min()

    lines = [
        f"📊 {headline}",
        "",
        narrative,
        "",
        f"BTC spot ref ${spot:,.0f} | 30D range ${low_30d:,.0f} – ${high_30d:,.0f}",
        f"Vol regime: {regime} | 7D RV {rv7:.1%} | 30D RV {rv30:.1%}",
        "",
        "── Recommended Structures (spot ref ${:,.0f}) ──".format(spot),
        "",
    ]

    for exp in [7, 14, 21, 28]:
        if exp not in multi_recs:
            continue
        r = multi_recs[exp]["rec"]
        expiry_date = today + timedelta(days=exp)
        date_str = expiry_date.strftime("%d%b").upper()
        otm = r["otm"]
        strike_usd = r["strike"]
        prem_usd = r["prem_pct"] / 100.0 * spot
        ann_yield = r["ann_yield"]
        no_hit = r["safety"]

        lines.append(
            f"_ BTC {date_str} {otm}% OTM (${strike_usd:,.0f}) Call = "
            f"${prem_usd:,.0f} per BTC "
            f"({ann_yield:.1f}% ann. | {no_hit:.0f}% no-hit)"
        )

    if extra_lines.strip():
        lines.append("")
        for el in extra_lines.strip().split("\n"):
            lines.append(el)

    lines += [
        "",
        "Reach out for custom structures or sizing.",
        "",
        f"— VERONICA Research Desk | {today.strftime('%d %b %Y')}",
    ]
    return "\n".join(lines)


def _build_sales_memo_kr(spot, btc_df, vol_data, multi_recs, pct_chg, rv7, rv30, safety_pct,
                         headline_kr, custom_narrative_kr, extra_lines_kr):
    """Generate a Korean Telegram-ready sales memo."""
    regime = _vol_regime(vol_data["forecast_rv"], vol_data["long_run_mean"])
    today = datetime.now(timezone.utc).date()

    high_30d = btc_df["high"].tail(30).max()
    low_30d = btc_df["low"].tail(30).min()

    # Korean auto-narrative by tone
    tone_label, _ = _market_sentiment(pct_chg, rv7, rv30, regime)
    vol_dir = "상승" if (rv30 > 0 and rv7 / rv30 > 1.10) else "하락" if (rv30 > 0 and rv7 / rv30 < 0.90) else "안정"

    kr_narratives = {
        "bearish": f"디지털 자산 전반적으로 약세가 지속되고 있습니다. 변동성 {vol_dir} 국면에서 구조화 상품을 통한 쿠폰 수취 전략이 유효합니다.",
        "cautious": f"시장이 소폭 하락세를 보이고 있으며, 변동성은 {vol_dir} 추세입니다. 커버드콜 구조가 매력적인 구간입니다.",
        "neutral": f"BTC가 박스권에서 움직이고 있으며 변동성은 {vol_dir}세입니다. 안정적 쿠폰 수익을 위한 콜 매도 전략을 추천드립니다.",
        "constructive": f"시장이 점진적 상승세를 보이고 있습니다. 변동성 {vol_dir} 국면에서 넓은 스트라이크로 상승 참여와 쿠폰을 동시에 확보할 수 있습니다.",
        "bullish": f"강한 상승 모멘텀이 이어지고 있습니다. 변동성 {vol_dir} 환경에서 높은 OTM 스트라이크를 통해 상승 여력을 남기면서 쿠폰을 확보하세요.",
    }
    narrative = custom_narrative_kr if custom_narrative_kr.strip() else kr_narratives.get(tone_label, "시장 동향을 반영한 맞춤 구조를 제안드립니다.")

    regime_kr = {"LOW": "저변동성", "NORMAL": "보통", "HIGH": "고변동성"}.get(regime, regime)

    lines = [
        f"📊 {headline_kr}",
        "",
        narrative,
        "",
        f"BTC 현재가 ${spot:,.0f} | 30일 레인지 ${low_30d:,.0f} – ${high_30d:,.0f}",
        f"변동성 환경: {regime_kr} | 7일 RV {rv7:.1%} | 30일 RV {rv30:.1%}",
        "",
        "── 추천 구조 (기준가 ${:,.0f}) ──".format(spot),
        "",
    ]

    for exp in [7, 14, 21, 28]:
        if exp not in multi_recs:
            continue
        r = multi_recs[exp]["rec"]
        expiry_date = today + timedelta(days=exp)
        date_str = expiry_date.strftime("%m/%d")
        otm = r["otm"]
        strike_usd = r["strike"]
        prem_usd = r["prem_pct"] / 100.0 * spot
        ann_yield = r["ann_yield"]
        no_hit = r["safety"]

        lines.append(
            f"_ BTC {date_str} 만기 {otm}% OTM (${strike_usd:,.0f}) 콜 = "
            f"BTC당 ${prem_usd:,.0f} "
            f"(연환산 {ann_yield:.1f}% | 미도달 확률 {no_hit:.0f}%)"
        )

    if extra_lines_kr.strip():
        lines.append("")
        for el in extra_lines_kr.strip().split("\n"):
            lines.append(el)

    lines += [
        "",
        "맞춤 구조 및 사이징 문의는 연락주시기 바랍니다.",
        "",
        f"— VERONICA Research Desk | {today.strftime('%Y년 %m월 %d일')}",
    ]
    return "\n".join(lines)


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

tab_overview, tab_strike, tab_bt, tab_rec, tab_memo = st.tabs([
    "Market Overview", "Strike Analysis", "Historical Backtest", "Weekly Recommendation", "Sales Memo",
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

    # Volatility regime (shared across all expiries)
    regime = _vol_regime(vol_data["forecast_rv"], vol_data["long_run_mean"])
    regime_map = {"LOW": "LOW", "NORMAL": "NORMAL", "HIGH": "HIGH"}
    rec_single = _generate_recommendation(
        btc_spot, vol_data, strike_df, rv_7d_last, rv_30d_last, btc_df, safety_threshold,
    )
    st.info(f"**Volatility Regime: {regime}** — {rec_single['reasoning']}")

    # ── Risk Warnings (global) ──
    if rec_single["warnings"]:
        for w in rec_single["warnings"]:
            st.warning(w)

    st.markdown("---")

    # ── Multi-Expiry Analysis ──
    st.subheader("Multi-Expiry Comparison")
    st.caption("Recommended strike for each expiry period based on current volatility regime and safety threshold.")

    REC_EXPIRIES = [7, 14, 21, 28]

    # Build strike tables and recommendations for each expiry
    multi_rows = []
    multi_recs = {}
    for exp in REC_EXPIRIES:
        sdf = _strike_table(btc_spot, vol_data["forecast_rv"], exp, prices_arr, OTM_LEVELS)
        rec = _generate_recommendation(
            btc_spot, vol_data, sdf, rv_7d_last, rv_30d_last, btc_df, safety_threshold,
        )
        multi_recs[exp] = {"rec": rec, "strike_df": sdf}
        multi_rows.append({
            "Expiry": f"{exp}D",
            "Rec. OTM %": f"{rec['otm']}%",
            "Strike (USD)": rec["strike"],
            "Coupon (%)": rec["prem_pct"],
            "Ann. Yield (%)": rec["ann_yield"],
            "Safety (%)": rec["safety"],
            "Risk": rec["risk"],
        })

    summary_df = pd.DataFrame(multi_rows)

    # Display summary metrics per expiry in columns
    cols = st.columns(len(REC_EXPIRIES))
    for i, exp in enumerate(REC_EXPIRIES):
        r = multi_recs[exp]["rec"]
        with cols[i]:
            st.markdown(f"#### {exp}-Day Expiry")
            st.metric("Strike", f"{r['otm']}% OTM (${r['strike']:,.0f})")
            st.metric("Coupon", f"{r['prem_pct']:.3f}%")
            st.metric("Ann. Yield", f"{r['ann_yield']:.2f}%")
            st.metric("Safety", f"{r['safety']:.1f}%")
            risk_icon = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(r["risk"], "⚪")
            st.markdown(f"Risk: {risk_icon} **{r['risk']}**")

    st.markdown("---")

    # Comparison table
    st.subheader("Side-by-Side Summary")
    disp_summary = summary_df.copy()
    disp_summary["Strike (USD)"] = disp_summary["Strike (USD)"].map("${:,.0f}".format)
    disp_summary["Coupon (%)"] = disp_summary["Coupon (%)"].map("{:.3f}%".format)
    disp_summary["Ann. Yield (%)"] = disp_summary["Ann. Yield (%)"].map("{:.2f}%".format)
    disp_summary["Safety (%)"] = disp_summary["Safety (%)"].map("{:.1f}%".format)
    st.dataframe(disp_summary, use_container_width=True, hide_index=True)

    csv_rec = summary_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Recommendation CSV", data=csv_rec,
        file_name="btc_multi_expiry_recommendation.csv", mime="text/csv",
    )

    st.markdown("---")

    # ── Per-expiry detail drilldown ──
    st.subheader("Expiry Drilldown")
    detail_exp = st.selectbox("Select expiry to view full strike table", REC_EXPIRIES, index=3, key="rec_detail_exp")
    detail_sdf = multi_recs[detail_exp]["strike_df"]
    detail_rec = multi_recs[detail_exp]["rec"]

    sweet_idx = _find_sweet_spot(detail_sdf, safety_threshold)
    if sweet_idx is not None:
        sw = detail_sdf.loc[sweet_idx]
        st.success(
            f"**Sweet Spot ({detail_exp}D): {sw['OTM %']}% OTM** — "
            f"Strike ${sw['Strike (USD)']:,.0f}  |  "
            f"Ann. Yield {sw['Ann. Yield %']:.2f}%  |  "
            f"Safety {sw['P(No-Hit)']:.1%}"
        )

    detail_disp = detail_sdf.copy()
    detail_disp["OTM %"] = detail_disp["OTM %"].map("{:.0f}%".format)
    detail_disp["Strike (USD)"] = detail_disp["Strike (USD)"].map("${:,.0f}".format)
    detail_disp["BS Premium (USD)"] = detail_disp["BS Premium (USD)"].map("${:,.2f}".format)
    detail_disp["Premium %"] = detail_disp["Premium %"].map("{:.3f}%".format)
    detail_disp["Ann. Yield %"] = detail_disp["Ann. Yield %"].map("{:.2f}%".format)
    detail_disp["P(No-Hit)"] = detail_disp["P(No-Hit)"].map("{:.1%}".format)
    st.dataframe(detail_disp, use_container_width=True, hide_index=True)

    # ── Yield vs Safety scatter chart across all expiries ──
    st.markdown("---")
    st.subheader("Yield vs Safety Map")
    scatter_fig = go.Figure()
    colors_exp = {"7": "#FF6B6B", "14": "#FFA94D", "21": "#51CF66", "28": "#339AF0"}
    for exp in REC_EXPIRIES:
        sdf = multi_recs[exp]["strike_df"]
        scatter_fig.add_trace(go.Scatter(
            x=sdf["P(No-Hit)"] * 100,
            y=sdf["Ann. Yield %"],
            mode="markers+text",
            text=sdf["OTM %"].astype(str) + "%",
            textposition="top center",
            name=f"{exp}D Expiry",
            marker=dict(size=12, color=colors_exp.get(str(exp), "#888")),
            hovertemplate=(
                f"{exp}D Expiry<br>"
                "OTM: %{text}<br>"
                "Safety: %{x:.1f}%<br>"
                "Ann. Yield: %{y:.2f}%<extra></extra>"
            ),
        ))
    scatter_fig.add_vline(x=safety_pct, line_dash="dash", line_color="gray",
                          annotation_text=f"Safety Threshold ({safety_pct}%)")
    scatter_fig.update_layout(
        title="Annualized Yield vs Historical Safety (All Expiries)",
        xaxis_title="P(No-Hit) %", yaxis_title="Annualized Yield %",
        plot_bgcolor="white", height=500, hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(scatter_fig, use_container_width=True)

    st.markdown("---")

    with st.expander("Methodology"):
        st.markdown(f"""
**How the recommendation is generated:**

1. **Volatility Regime** — The forecasted volatility (EMA + mean-reversion model) is compared to its
   long-run average. If it is >15% above average → HIGH; >15% below → LOW; otherwise NORMAL.

2. **Strike Selection (applied independently per expiry):**
   - **LOW regime:** Select the *lowest* OTM% that still meets the safety threshold (maximize coupon).
   - **HIGH regime:** Select the *highest* OTM% that still meets the threshold (maximize safety).
   - **NORMAL regime:** Select the OTM% with the *highest annualized yield* among safe strikes.

3. **Safety Threshold:** Currently set to **{safety_pct}%** — meaning the strike must *not* have been
   breached in at least {safety_pct}% of all historical rolling windows of that expiry length.

4. **Risk Warnings** fire when:
   - 7-day RV exceeds 30-day RV by >10% (rising vol trend)
   - BTC is within 5% of its all-time high (breakout risk)

**Model Inputs:**
- Forecast Vol: **{vol_data['forecast_rv']:.2%}** (Long-run mean: {vol_data['long_run_mean']:.2%}, Recent EMA: {vol_data['ema_recent']:.2%})
- Annualization: 365 days (crypto markets trade 24/7)
- Risk-free rate: 0%
""")

    st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

# ─── Tab 5: Sales Memo ────────────────────────────────────────
with tab_memo:
    st.subheader("Sales Memo Generator")
    st.caption("Generate Telegram-ready market updates in English & Korean. Edit, then copy.")

    # Auto-detect sentiment
    pct_chg_memo, _ = _weekly_change(btc_df)
    regime_memo = _vol_regime(vol_data["forecast_rv"], vol_data["long_run_mean"])
    tone_label, auto_narrative_en = _market_sentiment(pct_chg_memo, rv_7d_last, rv_30d_last, regime_memo)

    tone_display = {
        "bearish": "Bearish 🔴", "cautious": "Cautious 🟠", "neutral": "Neutral ⚪",
        "constructive": "Constructive 🟢", "bullish": "Bullish 🟢",
    }
    st.info(f"**Auto-detected tone: {tone_display.get(tone_label, tone_label)}** — You can override the narratives below.")

    # ── Inputs: EN + KR side by side ──
    en_col, kr_col = st.columns(2)
    with en_col:
        st.markdown("#### English")
        memo_headline_en = st.text_input(
            "Headline (EN)", value="Weekly BTC Options Desk Update", key="hl_en",
        )
        memo_narrative_en = st.text_area(
            "Narrative (EN) — blank = auto", value="", height=80,
            placeholder=auto_narrative_en, key="narr_en",
        )
        memo_extra_en = st.text_area(
            "Additional Lines (EN)", value="", height=60,
            placeholder="_ ETH 3AUG 3500 Call = $128 offer per ETH", key="extra_en",
        )

    with kr_col:
        st.markdown("#### 한국어")
        memo_headline_kr = st.text_input(
            "헤드라인 (KR)", value="주간 BTC 옵션 데스크 업데이트", key="hl_kr",
        )
        memo_narrative_kr = st.text_area(
            "내러티브 (KR) — 빈칸 = 자동 생성", value="", height=80,
            placeholder="시장 코멘터리를 입력하세요...", key="narr_kr",
        )
        memo_extra_kr = st.text_area(
            "추가 라인 (KR)", value="", height=60,
            placeholder="_ ETH 08/03 만기 3500 콜 = ETH당 $128", key="extra_kr",
        )

    # Build recs
    memo_recs = {}
    for exp in [7, 14, 21, 28]:
        sdf = _strike_table(btc_spot, vol_data["forecast_rv"], exp, prices_arr, OTM_LEVELS)
        rec = _generate_recommendation(
            btc_spot, vol_data, sdf, rv_7d_last, rv_30d_last, btc_df, safety_threshold,
        )
        memo_recs[exp] = {"rec": rec, "strike_df": sdf}

    memo_args = (btc_spot, btc_df, vol_data, memo_recs, pct_chg_memo, rv_7d_last, rv_30d_last, safety_pct)

    memo_en = _build_sales_memo_en(*memo_args, memo_headline_en, memo_narrative_en, memo_extra_en)
    memo_kr = _build_sales_memo_kr(*memo_args, memo_headline_kr, memo_narrative_kr, memo_extra_kr)

    st.markdown("---")

    # ── Preview: EN and KR side by side ──
    import base64 as _b64
    prev_en, prev_kr = st.columns(2)

    with prev_en:
        st.markdown("### English Preview")
        st.code(memo_en, language=None)

        _encoded_en = _b64.b64encode(memo_en.encode()).decode()
        _js_en = f"""
        <textarea id="memo-en" style="position:absolute;left:-9999px">{memo_en}</textarea>
        <button onclick="
            var t=document.getElementById('memo-en');
            t.style.position='static';t.select();
            document.execCommand('copy');
            t.style.position='absolute';t.style.left='-9999px';
            this.innerText='Copied!';
            setTimeout(()=>this.innerText='Copy English',2000);
        " style="
            background:#1E88E5;color:white;border:none;padding:8px 20px;
            border-radius:6px;cursor:pointer;font-size:14px;font-weight:600;
            margin-right:8px;
        ">Copy English</button>
        """
        components.html(_js_en, height=50, key="copy_en")

        st.download_button(
            "Download EN .txt", data=memo_en.encode("utf-8"),
            file_name="btc_sales_memo_en.txt", mime="text/plain", key="dl_en",
        )

    with prev_kr:
        st.markdown("### 한국어 미리보기")
        st.code(memo_kr, language=None)

        _encoded_kr = _b64.b64encode(memo_kr.encode()).decode()
        _js_kr = f"""
        <textarea id="memo-kr" style="position:absolute;left:-9999px">{memo_kr}</textarea>
        <button onclick="
            var t=document.getElementById('memo-kr');
            t.style.position='static';t.select();
            document.execCommand('copy');
            t.style.position='absolute';t.style.left='-9999px';
            this.innerText='복사 완료!';
            setTimeout(()=>this.innerText='한국어 복사',2000);
        " style="
            background:#1E88E5;color:white;border:none;padding:8px 20px;
            border-radius:6px;cursor:pointer;font-size:14px;font-weight:600;
            margin-right:8px;
        ">한국어 복사</button>
        """
        components.html(_js_kr, height=50, key="copy_kr")

        st.download_button(
            "Download KR .txt", data=memo_kr.encode("utf-8"),
            file_name="btc_sales_memo_kr.txt", mime="text/plain", key="dl_kr",
        )
