# -*- coding: utf-8 -*-
"""
VERONICA - Custom Candle Return Visualizer
OKX API를 이용한 커스텀 캔들 수익률 시각화
"""

# 1) 인증서 환경변수는 가장 먼저 설정
import os
import certifi

for var in ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE"):
    if var in os.environ:
        del os.environ[var]
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

import streamlit as st
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone

# Import utilities
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.auth import require_auth, show_logout_button

# ================== Page Config ==================
st.set_page_config(
    page_title="Custom Candle Returns - VERONICA",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== Auth Check ==================
require_auth()


# ================== Session Setup ==================
def _make_session():
    """requests 세션을 만들어 certifi를 기본 verify로 고정 + 재시도 설정"""
    s = requests.Session()
    s.verify = certifi.where()
    retry = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "veronica-candle-return-app/1.0"})
    return s


SESSION = _make_session()
OKX_BASE = "https://www.okx.com"


# ================== API Functions ==================
def norm_inst_id(symbol: str) -> str:
    """심볼 정규화 (예: BTCUSDT -> BTC-USDT)"""
    if not symbol:
        return ""
    s = symbol.upper().strip()
    # 이미 OKX 형식인 경우
    if "-" in s:
        return s
    # USDT 페어 변환
    if s.endswith("USDT"):
        base = s[:-4]
        return f"{base}-USDT"
    return s


@st.cache_data(ttl=300, show_spinner=False)
def get_okx_klines(symbol="BTC-USDT", start=None, end=None):
    """OKX API: 일봉 데이터 조회 (페이지네이션 포함)"""
    inst_id = norm_inst_id(symbol)
    if not inst_id:
        return None, "Invalid symbol"

    all_candles = []

    # 최신 데이터부터 시작
    anchor = None
    max_pages = 20  # 최대 20페이지 (약 4000일)

    for page in range(max_pages):
        params = {"instId": inst_id, "bar": "1D", "limit": 200}
        if anchor:
            params["before"] = anchor

        # 최근 데이터는 /market/candles, 과거 데이터는 /market/history-candles
        endpoint = "/api/v5/market/candles" if page == 0 else "/api/v5/market/history-candles"

        try:
            resp = SESSION.get(f"{OKX_BASE}{endpoint}", params=params, timeout=15)
        except requests.exceptions.SSLError as e:
            return None, f"TLS/SSL error: {e}"
        except requests.exceptions.RequestException as e:
            return None, f"Network error: {e}"

        if resp.status_code != 200:
            return None, f"API request failed: {resp.status_code} - {resp.text[:200]}"

        try:
            js = resp.json()
            if js.get("code") != "0":
                return None, f"API error: {js.get('msg', 'Unknown error')}"

            data = js.get("data", [])
            if not data:
                break

            all_candles.extend(data)

            # 다음 페이지를 위한 anchor 설정 (가장 오래된 타임스탬프)
            try:
                timestamps = [int(row[0]) for row in data]
                anchor = min(timestamps) - 1
            except:
                break

            # 시작 날짜에 도달했으면 중단
            if start:
                start_ms = int(pd.to_datetime(start).timestamp() * 1000)
                if anchor < start_ms:
                    break

        except Exception as e:
            return None, f"Data parsing error: {str(e)}"

    if not all_candles:
        return None, "No data returned from API"

    try:
        # OKX 캔들 형식: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
        df = pd.DataFrame(all_candles, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "vol_ccy", "vol_ccy_quote", "confirm"
        ])

        # 타임존 처리: UTC로 파싱 후 naive로 변환
        df["date"] = pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True).dt.tz_localize(None)
        df["close"] = df["close"].astype(float)
        df = df.sort_values("date").reset_index(drop=True)
        df.set_index("date", inplace=True)
        df = df[["close"]]

        if start:
            df = df[df.index >= pd.to_datetime(start)]
        if end:
            df = df[df.index <= pd.to_datetime(end)]
        return df, None
    except Exception as e:
        return None, f"Data parsing error: {str(e)}"


def generate_custom_returns(df, candle_size):
    """Custom 수익률 계산"""
    returns, labels = [], []
    n = len(df)
    for i in range(0, n - candle_size + 1, candle_size):
        chunk = df.iloc[i:i + candle_size]
        if len(chunk) < candle_size:
            continue
        start_price = chunk['close'].iloc[0]
        end_price = chunk['close'].iloc[-1]
        ret = ((end_price - start_price) / start_price) * 100
        label = f"{chunk.index[0].strftime('%Y-%m-%d')} → {chunk.index[-1].strftime('%Y-%m-%d')}"
        returns.append(ret)
        labels.append(label)
    return labels, returns


# ================== UI ==================
st.title("📈 Custom Candle Return Visualizer")
st.caption("OKX API를 이용한 커스텀 캔들 수익률 시각화")

# Sidebar
with st.sidebar:
    st.header("VERONICA")
    show_logout_button()

    st.markdown("---")
    st.header("설정")

    ticker = st.text_input("Ticker (예: BTC-USDT, ETH-USDT)", value="BTC-USDT")
    candle_days = st.slider("Custom Candle Size (일)", 1, 90, 14)

    st.markdown("---")
    st.subheader("날짜 범위")

    default_end = datetime.today()
    default_start = default_end - timedelta(days=90)

    start_date = st.date_input("시작일", value=default_start)
    end_date = st.date_input("종료일", value=default_end)

# Main content
st.markdown("OKX에서 일봉 데이터를 가져와 커스텀 캔들 수익률을 계산합니다.")

col1, col2 = st.columns([1, 4])
with col1:
    generate_clicked = st.button("Generate Chart", type="primary", use_container_width=True)

if generate_clicked:
    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
    except Exception:
        st.error("Invalid date format.")
        st.stop()

    with st.spinner(f"Fetching {ticker.upper()} data from OKX..."):
        df, err = get_okx_klines(ticker, start=start, end=end)

    if err:
        st.error(f"{err}")
        st.stop()
    if df is None or df.empty or df["close"].isna().all():
        st.warning("No valid close price data available.")
        st.stop()

    labels, returns = generate_custom_returns(df, candle_days)

    if not returns:
        st.warning("Not enough data to generate returns with the selected candle size.")
        st.stop()

    orange = "#FF5C00"
    gray = "#A6A6A6"
    colors = [orange if r >= 0 else gray for r in returns]

    # 차트 생성
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=returns,
        marker_color=colors,
        hovertemplate='Period: %{x}<br>Return: %{y:.2f}%'
    ))
    fig.update_layout(
        title=f"{ticker.upper()} {candle_days}-Day Returns ({start.date()} to {end.date()})",
        xaxis_title="Period",
        yaxis_title="Return (%)",
        plot_bgcolor='white',
        hovermode='x unified',
        height=500,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    st.success(f"{ticker.upper()} return chart generated")
    st.plotly_chart(fig, use_container_width=True)

    # 테이블
    st.subheader("Return Data")
    table = pd.DataFrame({
        "Period": labels,
        "Return (%)": [round(r, 2) for r in returns]
    })
    st.dataframe(table, use_container_width=True)

    # 통계 요약
    st.subheader("Summary Statistics")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.metric("Total Periods", len(returns))
    with col_s2:
        st.metric("Avg Return", f"{sum(returns)/len(returns):.2f}%")
    with col_s3:
        st.metric("Max Return", f"{max(returns):.2f}%")
    with col_s4:
        st.metric("Min Return", f"{min(returns):.2f}%")

    # CSV 다운로드
    csv = table.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"{ticker}_{candle_days}d_returns.csv",
        mime="text/csv"
    )
else:
    st.info("좌측 사이드바에서 설정 후 'Generate Chart' 버튼을 클릭하세요.")
