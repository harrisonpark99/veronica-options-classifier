#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 15:20:39 2025
@author: harrisonpark

옵션 가격 계산기 (Black-Scholes & Volatility Forecast)
- Coinglass(만료) → OKX Market Data API로 대체 버전
"""

import streamlit as st
import requests
import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm
import certifi  # TLS 인증서 번들 경로 문제 해결용

# ================== Auth Check ==================
if "auth_ok" not in st.session_state or not st.session_state.auth_ok:
    st.warning("로그인이 필요합니다.")
    st.switch_page("app.py")
    st.stop()

# 무위험 금리는 0%로 고정
risk_free_rate = 0.0

st.title("옵션 가격 계산기 (Black-Scholes & Volatility Forecast)")
st.markdown("이 앱은 **OKX API**와 계산된 또는 직접 입력한 연율화 변동성을 이용하여 옵션 가격을 산출합니다.")

# 1. 사용자 입력: 티커
ticker = st.text_input("티커를 입력하세요 (예: BTCUSDT 또는 BTC-USDT)", value="BTCUSDT")

def to_okx_inst_id(t: str) -> str:
    """
    입력을 OKX instId 형식으로 변환.
    - BTCUSDT -> BTC-USDT
    - BTC-USDT 그대로 사용
    """
    t = t.strip().upper()
    if "-" in t:
        return t
    if t.endswith("USDT") and len(t) > 4:
        return f"{t[:-4]}-USDT"
    return t

OKX_BASE_URL = "https://www.okx.com"

# 2. OKX 데이터 요청 및 종가 추출
@st.cache_data(show_spinner=False)
def get_close_prices_okx(inst_id: str, bar: str = "1D", target_limit: int = 4500):
    """
    OKX 캔들 데이터는 public endpoint로 조회 가능.
    - history-candles는 limit이 작아서(일반적으로 100) pagination(after)로 과거 데이터를 계속 가져옴
    - 응답 candle 형태: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
      여기서 c(인덱스 4)가 종가
    """
    sess = requests.Session()

    all_candles = []
    after = None

    # 과거로 계속 paging하면서 목표 개수만큼 수집
    while len(all_candles) < target_limit:
        params = {"instId": inst_id, "bar": bar, "limit": "100"}
        if after is not None:
            params["after"] = str(after)

        url = f"{OKX_BASE_URL}/api/v5/market/history-candles"
        resp = sess.get(url, params=params, verify=certifi.where(), timeout=10)

        if resp.status_code != 200:
            st.error(f"OKX API 요청 실패: {resp.status_code}")
            return None

        j = resp.json()
        if j.get("code") != "0":
            st.error(f"OKX API 응답 오류: {j.get('msg')}")
            return None

        data = j.get("data", [])
        if not data:
            break  # 더 이상 과거 데이터 없음

        all_candles.extend(data)

        # 다음 페이지를 위해 이번 batch에서 가장 오래된 ts(ms)를 after로 세팅
        # (응답이 최신->과거 순으로 오기 때문에 data[-1]이 가장 오래된 값인 경우가 일반적)
        oldest_ts = data[-1][0]
        after = int(oldest_ts)

        # 안전장치: 너무 적게 오면 종료
        if len(data) < 2:
            break

    # 경계 중복 제거 (paging에서 중복이 생길 수 있음)
    seen = set()
    uniq = []
    for c in all_candles:
        ts = c[0]
        if ts in seen:
            continue
        seen.add(ts)
        uniq.append(c)

    # 시간 오름차순(과거->최신) 정렬 후 종가만 추출
    uniq_sorted = sorted(uniq, key=lambda x: int(x[0]))
    close_prices = [float(c[4]) for c in uniq_sorted if len(c) > 4]

    if not close_prices:
        st.error("유효한 종가 데이터가 없습니다.")
        return None

    return close_prices

okx_inst_id = to_okx_inst_id(ticker)
st.write(f"OKX instId: **{okx_inst_id}**")

close_prices = get_close_prices_okx(okx_inst_id, bar="1D", target_limit=4500)
if close_prices is None:
    st.stop()

S = close_prices[-1]
st.write(f"**현재 가격 (S):** {S:.6f}")

# 3. rolling volatility 계산 (30일 기준, 연율화: 252 거래일)
def compute_rolling_volatility(prices, window=30, annualization_factor=252):
    vol_list = []
    n = len(prices)
    if n < window:
        raise ValueError("가격 데이터의 길이가 window보다 짧습니다.")
    for i in range(window, n + 1):
        window_prices = np.array(prices[i - window : i])
        log_returns = np.log(window_prices[1:] / window_prices[:-1])
        vol = np.std(log_returns, ddof=1) * sqrt(annualization_factor)
        vol_list.append(vol)
    return np.array(vol_list)

rolling_rv = compute_rolling_volatility(close_prices, window=30, annualization_factor=252)
st.write(f"전체 rolling RV 데이터 개수: {len(rolling_rv)}")

# 4. 최근 90일치(약 3개월) 데이터 선택
recent_rv = rolling_rv[-90:] if len(rolling_rv) >= 90 else rolling_rv
st.write(f"최근 rolling RV 데이터 개수: {len(recent_rv)}")

# 5. 변동성 선택: 계산된 예측치 vs. 직접 입력
vol_option = st.radio("변동성 선택", ("계산된 예측 변동성 (EMA+평균 회귀)", "직접 입력"))

if vol_option == "계산된 예측 변동성 (EMA+평균 회귀)":

    def compute_ema(data, span):
        alpha = 2 / (span + 1)
        ema = data[0]
        for x in data[1:]:
            ema = alpha * x + (1 - alpha) * ema
        return ema

    ema_recent = compute_ema(recent_rv, span=30)
    long_run_mean = np.mean(rolling_rv)
    beta = 0.5  # 감쇠 계수
    forecast_rv = long_run_mean + beta * (ema_recent - long_run_mean)
    vol = float(forecast_rv)

    st.write(f"장기 평균 RV: {long_run_mean:.6f}")
    st.write(f"최근 RV EMA (스팬=30): {ema_recent:.6f}")
    st.write(f"예측된 변동성 (EMA+평균 회귀): {forecast_rv:.6f}")

else:
    vol = st.number_input(
        "연율화 변동성 값을 입력하세요 (예: 0.5 for 50%)",
        value=0.5,
        step=0.01,
    )

st.write(f"**선택된 변동성 (vol):** {vol:.6f}")

# 6. 옵션 매개변수: 옵션 종류, OTM %, 만기일(일수)
option_type = st.selectbox("옵션 종류", ("call", "put"))
otm_percent = st.number_input("OTM % (예: 15 for 15%)", value=15.0, step=1.0)

if option_type == "call":
    K = S * (1 + otm_percent / 100)
else:
    K = S * (1 - otm_percent / 100)

expiry_days = st.number_input("옵션 만기 (일수)", value=30.0, step=1.0)
T = expiry_days / 365  # 연 단위 만기

st.write(f"**조정된 행사가 (K):** {K:.6f}")
st.write(f"**옵션 만기 (T, 연 단위):** {T:.6f}")

# 7. Black-Scholes 옵션 가격 계산 함수
def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    # 방어: T=0 또는 sigma=0이면 분모가 0이 되므로 처리
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

option_price = black_scholes_price(S, K, T, risk_free_rate, vol, option_type=option_type)
premium_percent = (option_price / S) * 100 if S != 0 else 0.0

st.markdown("### 옵션 가격 결과")
st.write(f"**{option_type.capitalize()} 옵션 가격:** {option_price:.6f}")
st.write(f"**옵션 프리미엄 (%):** {premium_percent:.2f}%")
