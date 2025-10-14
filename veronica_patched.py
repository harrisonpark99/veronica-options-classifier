# -*- coding: utf-8 -*-
"""
VERONICA XUNKE SUPPORT (Patched & Refactored)
- TLS 인증서 경로 자동 설정 (certifi)
- 코드 모듈화(블록화) / 간결화
- 기능 유지: 분류, Coinglass 종가(과거), Binance 현재가(실시간), 집계/필터/다운로드, 디버그 툴
- 🔐 추가: st.secrets 기반 비밀번호 게이트
"""

# ================== Bootstrap & Globals ==================
import os
import re
import io
import hashlib
import difflib
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date, timezone, timedelta

# TLS 인증서 경로 자동 설정 (requests가 신뢰 루트 못 찾는 환경 대응)
import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

import requests
import pandas as pd
import numpy as np
import streamlit as st
from dateutil.relativedelta import relativedelta
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")

# ================== Auth Gate (with st.secrets) ==================
st.set_page_config(page_title="CSV 옵션 딜 분류기 (Patched)", layout="wide", initial_sidebar_state="expanded")

# st.secrets 로부터 비밀번호 불러오기 (환경변수 fallback 포함)
APP_PASSWORD = st.secrets.get("APP_PASSWORD", os.environ.get("APP_PASSWORD", ""))

if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

if not st.session_state.auth_ok:
    st.title("🔐 패스워드를 입력하세요")
    st.caption("인증 후에 메인 화면으로 이동합니다.")
    pw = st.text_input("Password", type="password", placeholder="패스워드 입력")
    col_ok, col_sp = st.columns([1, 3])
    with col_ok:
        confirm = st.button("확인", type="primary")
    if confirm:
        if pw == APP_PASSWORD and APP_PASSWORD:
            st.session_state.auth_ok = True
            st.rerun()
        else:
            st.error("패스워드가 올바르지 않거나 설정되어 있지 않습니다. 관리자에게 문의하세요.")
    st.stop()

# ================== App Config ==================
@dataclass
class AppConfig:
    trade_field: str = "Created Time"  # or "Initiation Time"
    coupon_parties: str = "Emporio Holdings, Novaflow Labs Ltd."
    coupon_quotes: str = "USDT, USD"
    covered_call_parties: str = "Marblex, Near_Siftung, Layer3_Operations"
    exclude_mm: bool = False
    exclude_unknown: bool = False

    def save_to_session(self):
        st.session_state.config = asdict(self)

    @classmethod
    def load_from_session(cls):
        return cls(**st.session_state.get("config", asdict(cls())))

# ================== Constants & Regex ==================
SYMBOL_REGEX = re.compile(
    r'^(?P<issuer>[^.]+)\.(?P<style>[A-Z_]+)\.(?P<base>[A-Z0-9]+)-(?P<quote>[A-Z0-9]+)\.(?P<expiry>\d{8})\.(?P<option>[CP])\.(?P<series>\d+)$'
)
DATE_YMD_RE = re.compile(r'(\d{4})-(\d{2})-(\d{2})')

TOKEN_ALIASES = {
    "WETH": "ETH",
    "BCH.SV": "BSV",
    "BCHSV": "BSV",
}

# Coinglass API Key 우선순위: secrets.toml > 환경변수 (하드코딩 기본값 제거)
DEFAULT_COINGLASS_API_KEY = ""  # <- 보안을 위해 빈 값 유지
def _get_secret(name: str, default: str = "") -> str:
    """Streamlit secrets가 없을 때도 안전하게 읽기."""
    try:
        return st.secrets.get(name, default)
    except Exception:
        return os.environ.get(name, default)

COINGLASS_API_KEY = _get_secret("COINGLASS_API_KEY", "") or DEFAULT_COINGLASS_API_KEY
CG_HEADERS = {"CG-API-KEY": COINGLASS_API_KEY}

# ================== Utilities (Parsing / Dates / Normalization) ==================
def parse_symbol(symbol: str) -> Dict:
    if not isinstance(symbol, str):
        return {}
    m = SYMBOL_REGEX.match(symbol)
    return m.groupdict() if m else {}

def extract_iso_date_to_str(date_str: str) -> str:
    if not isinstance(date_str, str):
        return ""
    m = DATE_YMD_RE.search(date_str)
    if not m:
        return ""
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return f"{mo}/{d}/{y}"

def extract_iso_date_to_date(date_str: str) -> Optional[date]:
    if not isinstance(date_str, str):
        return None
    m = DATE_YMD_RE.search(date_str)
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return date(y, mo, d)

def yyyymmdd_to_mdy_str(yyyymmdd: str) -> str:
    if not isinstance(yyyymmdd, str) or len(yyyymmdd) != 8:
        return ""
    y = int(yyyymmdd[:4]); m = int(yyyymmdd[4:6]); d = int(yyyymmdd[6:8])
    return f"{m}/{d}/{y}"

def yyyymmdd_to_date(yyyymmdd: str) -> Optional[date]:
    if not isinstance(yyyymmdd:=(yyyymmdd), str) or len(yyyymmdd) != 8:
        return None
    y = int(yyyymmdd[:4]); m = int(yyyymmdd[4:6]); d = int(yyyymmdd[6:8])
    return date(y, m, d)

def normalize_party_list(txt: str) -> Set[str]:
    return {s.strip().lower() for s in (txt or '').split(',') if s.strip()}

def normalize_quote_list(txt: str) -> Set[str]:
    return {s.strip().upper() for s in (txt or '').split(',') if s.strip()}

def calculate_month_difference(start: datetime, end: datetime) -> float:
    try:
        delta = relativedelta(end, start)
        return delta.years * 12 + delta.months + delta.days / 30.0
    except Exception:
        return (end - start).days / 30

def make_pair_symbol(token: str) -> str:
    """BTC -> BTCUSDT, USD/USDT/USDC -> USDT (1.0 취급)"""
    if not token:
        return ""
    tk = re.sub(r'[^A-Z0-9]', '', str(token).upper())
    tk = TOKEN_ALIASES.get(tk, tk)
    if tk in {"USD", "USDT", "USDC"}:
        return "USDT"
    return f"{tk}USDT"

def resolve_trade_utc_date(ts_val) -> date:
    if ts_val is None or pd.isna(ts_val):
        kst_today = datetime.now(KST).date()
        return datetime(kst_today.year, kst_today.month, kst_today.day, tzinfo=KST).astimezone(timezone.utc).date()
    ts = pd.to_datetime(ts_val, errors="coerce")
    if pd.isna(ts):
        kst_today = datetime.now(KST).date()
        return datetime(kst_today.year, kst_today.month, kst_today.day, tzinfo=KST).astimezone(timezone.utc).date()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=KST)
    return ts.astimezone(timezone.utc).date()

# ================== External Services ==================
# ---- Coinglass (Historical Close) ----
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_coinglass_ohlc_binance(symbol: str, market_type: str = "spot", interval: str = "1d", limit: int = 2000):
    if not COINGLASS_API_KEY:
        st.warning("Coinglass API Key가 설정되지 않았습니다.")
        return None
    url = (
        "https://open-api-v3.coinglass.com/api/price/ohlc-history"
        f"?exchange=Binance&symbol={symbol}&type={market_type}&interval={interval}&limit={limit}"
    )
    try:
        r = requests.get(url, headers=CG_HEADERS, timeout=15, verify=certifi.where())
        if r.status_code != 200:
            st.warning(f"Coinglass 요청 실패: {r.status_code} {r.text[:120]}")
            return None
        data = r.json()
        if data.get("code") != "0":
            st.warning(f"Coinglass 응답 오류: {data.get('msg')}")
            return None
        return data.get("data", []) or None
    except Exception as e:
        st.warning(f"Coinglass 호출 예외: {str(e)[:120]}")
        return None

def _normalize_ts_ms(v) -> Optional[int]:
    if v is None:
        return None
    try:
        x = float(v)
        if x < 1e12:
            x *= 1000.0
        return int(x)
    except Exception:
        pass
    try:
        dt = pd.to_datetime(v, errors="coerce", utc=True)
        if pd.isna(dt):
            return None
        return int(dt.value // 10**6)
    except Exception:
        return None

def find_close_for_utc_date_from_rows(rows, utc_date: date) -> Optional[float]:
    if not rows:
        return None
    day_start = datetime(utc_date.year, utc_date.month, utc_date.day, tzinfo=timezone.utc)
    start_ms = int(day_start.timestamp() * 1000)
    end_ms = start_ms + 86_400_000 - 1

    sample = rows[0]
    t_key = "t" if "t" in sample else ("time" if "time" in sample else None)
    if t_key is None:
        return None

    for item in rows:
        tms = _normalize_ts_ms(item.get(t_key))
        if tms is None:
            continue
        if start_ms <= tms <= end_ms:
            try:
                return float(item.get("c"))
            except Exception:
                pass

    best = None; best_dist = 1e18; mid_ms = start_ms + 43_200_000
    for item in rows:
        tms = _normalize_ts_ms(item.get(t_key))
        if tms is None:
            continue
        dist = abs(tms - mid_ms)
        if dist < best_dist:
            try:
                c = float(item.get("c")); best, best_dist = c, dist
            except Exception:
                continue
    if best is not None and best_dist <= 129_600_000:
        return best
    return None

def get_close_on_trade_date_coinglass(pair_symbol: str, trade_ts) -> Optional[float]:
    if not pair_symbol:
        return None
    if pair_symbol == "USDT":
        return 1.0
    utc_d = resolve_trade_utc_date(trade_ts)
    rows_spot = fetch_coinglass_ohlc_binance(pair_symbol, market_type="spot", interval="1d", limit=2000)
    px = find_close_for_utc_date_from_rows(rows_spot, utc_d) if rows_spot else None
    if px is None:
        rows_fut = fetch_coinglass_ohlc_binance(pair_symbol, market_type="futures", interval="1d", limit=2000)
        px = find_close_for_utc_date_from_rows(rows_fut, utc_d) if rows_fut else None
    if px is None:
        for delta in (1, -1, 2, -2, 3, -3):
            alt = utc_d + timedelta(days=delta)
            px = find_close_for_utc_date_from_rows(rows_spot, alt) if rows_spot else None
            if px is not None: break
            px = find_close_for_utc_date_from_rows(rows_fut, alt) if 'rows_fut' in locals() and rows_fut else None
            if px is not None: break
    return px

@st.cache_data(show_spinner=False, ttl=3600)
def get_batch_prices_coinglass(pair_ts_pairs: List[Tuple[str, object]]) -> Dict[Tuple[str, date], float]:
    prices: Dict[Tuple[str, date], float] = {}
    uniq = list({(p, pd.to_datetime(ts, errors="coerce")) for p, ts in pair_ts_pairs})
    progress_bar = st.progress(0)
    total = len(uniq) or 1
    with ThreadPoolExecutor(max_workers=5) as ex:
        fut = {ex.submit(get_close_on_trade_date_coinglass, p, ts): (p, ts) for (p, ts) in uniq}
        for i, f in enumerate(as_completed(fut), 1):
            p, ts = fut[f]
            try:
                px = f.result()
                if px is not None:
                    prices[(p, resolve_trade_utc_date(ts))] = px
            except Exception as e:
                st.warning(f"가격 조회 실패({p}): {str(e)[:80]}")
            finally:
                progress_bar.progress(i/total)
    progress_bar.empty()
    return prices

# ---- Binance (Current Price) ----
@st.cache_data(show_spinner=False, ttl=30)
def fetch_binance_ticker_price(pair_symbol: str) -> Optional[float]:
    if not pair_symbol or pair_symbol == "USDT":
        return 1.0 if pair_symbol == "USDT" else None
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbol": pair_symbol}, timeout=8, verify=certifi.where()
        )
        if r.status_code == 200:
            return float(r.json().get("price"))
        return None
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=30)
def get_batch_current_prices(pair_symbols: List[str]) -> Dict[str, Optional[float]]:
    uniq = sorted({s for s in pair_symbols if s})
    results: Dict[str, Optional[float]] = {s: (1.0 if s == "USDT" else None) for s in uniq}
    if not uniq:
        return {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        fut_map = {ex.submit(fetch_binance_ticker_price, s): s for s in uniq if s != "USDT"}
        for fut in as_completed(fut_map):
            sym = fut_map[fut]
            try:
                results[sym] = fut.result()
            except Exception:
                results[sym] = None
    return results

# ---- Debug Helpers ----
@st.cache_data(show_spinner=False, ttl=15)
def debug_fetch_binance_symbol(sym: str) -> Dict[str, object]:
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbol": sym}, timeout=8, verify=certifi.where()
        )
        return {"API Symbol": sym, "endpoint": "/ticker/price", "status": r.status_code, "ok": r.status_code==200, "response": r.text[:300]}
    except Exception as e:
        return {"API Symbol": sym, "endpoint": "/ticker/price", "status": "EXCEPTION", "ok": False, "response": str(e)[:300]}

@st.cache_data(show_spinner=False, ttl=15)
def debug_check_symbol_exists(sym: str) -> Dict[str, object]:
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/exchangeInfo",
            params={"symbol": sym}, timeout=8, verify=certifi.where()
        )
        return {"API Symbol": sym, "endpoint": "/exchangeInfo", "status": r.status_code, "ok": r.status_code==200, "response": r.text[:300]}
    except Exception as e:
        return {"API Symbol": sym, "endpoint": "/exchangeInfo", "status": "EXCEPTION", "ok": False, "response": str(e)[:300]}

def build_current_price_debug_table(pair_symbols: List[str]) -> pd.DataFrame:
    syms = sorted(set([s for s in pair_symbols if s]))
    rows = []
    for s in syms:
        r1 = debug_fetch_binance_symbol(s)
        r2 = debug_check_symbol_exists(s)
        rows.append({
            "API Symbol": s,
            "ticker_price_ok": r1.get("ok"),
            "ticker_status": r1.get("status"),
            "ticker_preview": r1.get("response"),
            "exists_ok": r2.get("ok"),
            "exists_status": r2.get("status"),
            "exists_preview": r2.get("response"),
        })
    return pd.DataFrame(rows)

# ---- Binance (Historical close per trade date) ----
@st.cache_data(show_spinner=False, ttl=3600)
def get_binance_daily_close_for_date(pair_symbol: str, utc_date: date) -> Tuple[Optional[float], str]:
    """1d 캔들에서 특정 UTC 일자의 종가를 Binance에서 조회."""
    if not pair_symbol:
        return None, "empty_symbol"
    if pair_symbol == "USDT":
        return 1.0, "stablecoin"

    day_start = datetime(utc_date.year, utc_date.month, utc_date.day, tzinfo=timezone.utc)
    start_ms = int(day_start.timestamp() * 1000)
    end_ms = start_ms + 86_400_000 - 1

    try:
        r = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={
                "symbol": pair_symbol,
                "interval": "1d",
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 2,
            },
            timeout=10,
            verify=certifi.where(),
        )
        if r.status_code != 200:
            return None, f"kline:{r.status_code}:{r.text[:120]}"
        js = r.json()
        if not isinstance(js, list) or not js:
            return None, "kline:empty"
        # kline: [ openTime, open, high, low, close, volume, closeTime, ... ]
        close_px = None
        for k in js:
            try:
                close_px = float(k[4])
            except Exception:
                continue
        return (close_px, "ok") if close_px is not None else (None, "kline:no_close")
    except Exception as e:
        return None, f"EXC:{str(e)[:120]}"

@st.cache_data(show_spinner=False, ttl=3600)
def get_batch_binance_closes(pair_date_pairs: List[Tuple[str, date]]) -> Dict[Tuple[str, date], flo]()_
