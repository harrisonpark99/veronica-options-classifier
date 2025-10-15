# -*- coding: utf-8 -*-
"""
VERONICA XUNKE SUPPORT (Patched & Refactored)
- TLS 인증서 자동 설정 (certifi)
- 비밀번호 게이트 (st.secrets)
- session_state 초기화
- Binance 지역 제한 감지 / 예외 처리
- 거래소 접근성 진단 도구 (probe_exchanges)
"""

# ================== Imports & Globals ==================
import os, io, re, hmac, hashlib, time, difflib, urllib.parse, requests
import pandas as pd
import numpy as np
import streamlit as st
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, date, timezone, timedelta
from dateutil.relativedelta import relativedelta
from zoneinfo import ZoneInfo
import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
KST = ZoneInfo("Asia/Seoul")

# ================== Password Gate ==================
st.set_page_config(page_title="CSV 옵션 딜 분류기 (Patched)", layout="wide")

APP_PASSWORD = str(st.secrets.get("APP_PASSWORD", os.environ.get("APP_PASSWORD", ""))).strip()
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

if not st.session_state.auth_ok:
    st.title("🔐 패스워드를 입력하세요")
    pw = st.text_input("Password", type="password", placeholder="prestosnt!")
    if st.button("확인", type="primary"):
        if hmac.compare_digest((pw or "").strip(), APP_PASSWORD):
            st.session_state.auth_ok = True
            st.rerun()
        else:
            st.error("패스워드가 올바르지 않습니다.")
    st.stop()

# ================== Config & Session Init ==================
@dataclass
class AppConfig:
    trade_field: str = "Created Time"
    coupon_parties: str = "Emporio Holdings, Novaflow Labs Ltd."
    coupon_quotes: str = "USDT, USD"
    covered_call_parties: str = "Marblex, Near_Siftung, Layer3_Operations"
    exclude_mm: bool = False
    exclude_unknown: bool = False

for key, val in {
    "df_raw": None,
    "file_hash": None,
    "debug_mode": False,
    "_last_probe": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ================== Connectivity Probes ==================
BLOCK_KEYWORDS = ["eligibility", "restricted", "unavailable", "service unavailable", "forbidden", "not allowed", "blocked"]

def _looks_blocked(text: str) -> bool:
    return any(k in (text or "").lower() for k in BLOCK_KEYWORDS)

@st.cache_data(ttl=20)
def probe_endpoint(name: str, url: str, params: Optional[Dict[str, Any]] = None,
                   headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    start = time.time()
    try:
        r = requests.get(url, params=params or {}, headers=headers or {}, timeout=8, verify=certifi.where())
        latency = int((time.time() - start) * 1000)
        preview = (r.text or "")[:180]
        blocked = _looks_blocked(preview) or (r.status_code in (401, 403, 418, 451))
        return {
            "Exchange": name,
            "Status": r.status_code,
            "Blocked?": "yes" if blocked else "no",
            "Latency(ms)": latency,
            "Preview": preview[:180],
            "OK": (r.status_code == 200 and not blocked)
        }
    except Exception as e:
        return {"Exchange": name, "Status": "EXC", "Blocked?": "unknown", "Latency(ms)": "-", "Preview": str(e)[:180], "OK": False}

@st.cache_data(ttl=20)
def probe_exchanges() -> pd.DataFrame:
    rows = []
    rows.append(probe_endpoint("Binance", "https://api.binance.com/api/v3/ticker/price", params={"symbol": "BTCUSDT"}))
    rows.append(probe_endpoint("Bybit", "https://api.bybit.com/v5/market/tickers", params={"category": "spot", "symbol": "BTCUSDT"}))
    rows.append(probe_endpoint("OKX", "https://www.okx.com/api/v5/market/ticker", params={"instId": "BTC-USDT"}))
    rows.append(probe_endpoint("Coinbase", "https://api.exchange.coinbase.com/products/BTC-USD/ticker"))
    rows.append(probe_endpoint("Coingecko", "https://api.coingecko.com/api/v3/simple/price", params={"ids": "bitcoin", "vs_currencies": "usd"}))
    return pd.DataFrame(rows)

# ================== Streamlit UI ==================
st.title("VERONICA SUPPORT · 거래소 접근성 진단 포함")
st.caption("패스워드 보호, TLS 인증, Binance 제한 감지 및 다중 거래소 프로브 포함 버전")

with st.expander("🌐 거래소 접근성 진단", expanded=True):
    st.caption("현재 서버에서 접근 가능한 거래소 API를 점검합니다 (Binance / Bybit / OKX / Coinbase / Coingecko).")
    if st.button("🔎 진단 실행 / 새로고침", key="probe_run", type="primary"):
        st.session_state["_last_probe"] = probe_exchanges()
    df = st.session_state.get("_last_probe")
    if isinstance(df, pd.DataFrame):
        st.dataframe(df, use_container_width=True)
        ok = df[df["OK"] == True]["Exchange"].tolist()
        if ok:
            st.success("✅ 접근 가능: " + ", ".join(ok))
        else:
            st.error("❌ 모든 주요 거래소 API 접근이 제한되었습니다.")
        st.download_button("💾 결과 CSV 다운로드", df.to_csv(index=False).encode("utf-8"),
                           file_name="exchange_probe.csv", mime="text/csv")
    else:
        st.info("‘🔎 진단 실행’을 눌러 테스트하세요.")

st.caption("이후 접근 가능한 거래소 순서로 API 폴백 체인을 구성하세요.")
