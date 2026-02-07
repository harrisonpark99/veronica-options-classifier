#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
옵션 가격 계산기 (Black-Scholes & Volatility Forecast)
- OKX Market Data API 기반
"""

import os, sys
import numpy as np
import streamlit as st
import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.auth import require_auth, show_logout_button
from utils.options import (
    to_okx_inst_id, get_close_prices_okx,
    compute_rolling_volatility, compute_ema,
    black_scholes_price,
)

# ================== Auth Check ==================
require_auth()

# 무위험 금리는 0%로 고정
risk_free_rate = 0.0

st.title("옵션 가격 계산기 (Black-Scholes & Volatility Forecast)")
st.markdown("이 앱은 **OKX API**와 계산된 또는 직접 입력한 연율화 변동성을 이용하여 옵션 가격을 산출합니다.")

# 1. 사용자 입력: 티커
ticker = st.text_input("티커를 입력하세요 (예: BTCUSDT 또는 BTC-USDT)", value="BTCUSDT")

okx_inst_id = to_okx_inst_id(ticker)
st.write(f"OKX instId: **{okx_inst_id}**")

close_prices = get_close_prices_okx(okx_inst_id, bar="1D", target_limit=4500)
if close_prices is None:
    st.error("OKX에서 데이터를 가져올 수 없습니다.")
    st.stop()

S = close_prices[-1]
st.write(f"**현재 가격 (S):** {S:.6f}")

# 3. rolling volatility 계산 (30일 기준, 연율화: 252 거래일)
rolling_rv = compute_rolling_volatility(close_prices, window=30, annualization_factor=252)
st.write(f"전체 rolling RV 데이터 개수: {len(rolling_rv)}")

# 4. 최근 90일치(약 3개월) 데이터 선택
recent_rv = rolling_rv[-90:] if len(rolling_rv) >= 90 else rolling_rv
st.write(f"최근 rolling RV 데이터 개수: {len(recent_rv)}")

# 5. 변동성 선택: 계산된 예측치 vs. 직접 입력
vol_option = st.radio("변동성 선택", ("계산된 예측 변동성 (EMA+평균 회귀)", "직접 입력"))

if vol_option == "계산된 예측 변동성 (EMA+평균 회귀)":
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

# 7. Black-Scholes 옵션 가격 계산
option_price = black_scholes_price(S, K, T, risk_free_rate, vol, option_type=option_type)
premium_percent = (option_price / S) * 100 if S != 0 else 0.0

st.markdown("### 옵션 가격 결과")
st.write(f"**{option_type.capitalize()} 옵션 가격:** {option_price:.6f}")
st.write(f"**옵션 프리미엄 (%):** {premium_percent:.2f}%")
