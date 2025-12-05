#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 15:20:39 2025

@author: harrisonpark
"""

import streamlit as st
import requests
import numpy as np
from math import log, sqrt, exp
from scipy.stats import norm
import certifi  # ✅ TLS 인증서 번들 경로 문제 해결용

# ================== Auth Check ==================
if "auth_ok" not in st.session_state or not st.session_state.auth_ok:
    st.warning("로그인이 필요합니다.")
    st.switch_page("app.py")
    st.stop()

# 무위험 금리는 0%로 고정
risk_free_rate = 0.0

st.title("옵션 가격 계산기 (Black-Scholes & Volatility Forecast)")
st.markdown("이 앱은 Coinglass API와 계산된 또는 직접 입력한 연율화 변동성을 이용하여 옵션 가격을 산출합니다.")

# 1. 사용자 입력: 티커 및 API 관련
ticker = st.text_input("티커를 입력하세요 (예: BTCUSDT 또는 ETHUSDT)", value="BTCUSDT")

# Coinglass API Key (본인의 API Key로 교체 가능)
COINGLASS_API_KEY = "8cca09baf8cc497dbb5a4caba45a41f6"
headers_cg = {"CG-API-KEY": COINGLASS_API_KEY}

# 2. Coinglass 데이터 요청 및 종가 추출
@st.cache_data(show_spinner=False)
def get_close_prices(ticker):
    url = (
        "https://open-api-v3.coinglass.com/api/price/ohlc-history?"
        f"exchange=Binance&symbol={ticker}&type=futures&interval=1d&limit=4500"
    )
    # 🔑 certifi.where()를 verify 인자로 넣어서 TLS CA 인증서 경로 문제를 우회
    response = requests.get(
        url,
        headers=headers_cg,
        verify=certifi.where(),
        timeout=10,
    )
    if response.status_code != 200:
        st.error(f"Coinglass API 요청 실패: {response.status_code}")
        return None
    json_data = response.json()
    if json_data.get("code") != "0":
        st.error(f"Coinglass API 응답 오류: {json_data.get('msg')}")
        return None
    data = json_data.get("data", [])
    if not data:
        st.error("데이터가 없습니다.")
        return None
    close_prices = [float(item["c"]) for item in data if "c" in item]
    if not close_prices:
        st.error("유효한 종가 데이터가 없습니다.")
        return None
    return close_prices

close_prices = get_close_prices(ticker)
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

rolling_rv = compute_rolling_volatility(
    close_prices, window=30, annualization_factor=252
)
st.write(f"전체 rolling RV 데이터 개수: {len(rolling_rv)}")

# 4. 최근 90일치(약 3개월) 데이터 선택
if len(rolling_rv) >= 90:
    recent_rv = rolling_rv[-90:]
else:
    recent_rv = rolling_rv
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
    vol = forecast_rv
    st.write(f"장기 평균 RV: {long_run_mean:.6f}")
    st.write(f"최근 RV EMA (스팬=30): {ema_recent:.6f}")
    st.write(f"예측된 변동성 (EMA+평균 회귀): {forecast_rv:.6f}")
else:
    vol = st.number_input(
        "연율화 변동성 값을 입력하세요 (예: 0.5 for 50%)", value=0.5, step=0.01
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
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if option_type.lower() == "call":
        price = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        price = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return price

option_price = black_scholes_price(
    S, K, T, risk_free_rate, vol, option_type=option_type
)
premium_percent = (option_price / S) * 100

st.markdown("### 옵션 가격 결과")
st.write(f"**{option_type.capitalize()} 옵션 가격:** {option_price:.6f}")
st.write(f"**옵션 프리미엄 (%):** {premium_percent:.2f}%")
