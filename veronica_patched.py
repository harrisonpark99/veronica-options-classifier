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
def get_batch_binance_closes(pair_date_pairs: List[Tuple[str, date]]) -> Dict[Tuple[str, date], float]:
    """여러 (심볼, UTC일자)의 일별 종가 배치 조회."""
    uniq = list({(sym, d) for sym, d in pair_date_pairs})
    out: Dict[Tuple[str, date], float] = {}
    if not uniq:
        return out
    with ThreadPoolExecutor(max_workers=8) as ex:
        fut = {ex.submit(get_binance_daily_close_for_date, sym, d): (sym, d) for sym, d in uniq}
        for f in as_completed(fut):
            key = fut[f]
            try:
                px, _ = f.result()
                if px is not None:
                    out[key] = px
            except Exception:
                pass
    return out

# ================== Business Logic (Classification & Aggregation) ==================
def infer_product_type(base: str, option: str, quote: str, counterparty: str, symbol: str,
                       coupon_whitelist: Set[str], coupon_quote_set: Set[str],
                       covered_call_whitelist: Set[str]) -> str:
    cp_norm = (counterparty or '').strip().lower()
    sym = (symbol or '').lower()
    if 'bonus' in sym:
        return f"{(quote or '').upper()} Bonus Coupon"
    if 'sharkfin' in sym:
        return f"{base} Sharkfin"
    if option == 'P':
        return f"{base} Put"
    if option == 'C':
        return f"{base} Covered Call" if cp_norm in covered_call_whitelist else f"{base} MM Call"
    return "Unknown"

def apply_output_filters(df: pd.DataFrame, exclude_mm: bool, exclude_unknown: bool) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if exclude_mm and "Product Type" in out.columns:
        out = out[~out["Product Type"].astype(str).str.contains(r'\bMM\b', case=False, na=False)]
    if exclude_unknown and "Product Type" in out.columns:
        pt = out["Product Type"].astype(str)
        out = out[~(pt.str.strip().eq("") | pt.str.strip().str.lower().eq("unknown"))]
    return out

def aggregate_by_product_type(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Product Type", "Token Amount Sum"])
    tmp = df.copy()
    tmp["Token Amount Num"] = (
        tmp["Token Amount"].astype(str).str.replace(",", "", regex=False).str.replace(" ", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )
    agg = (tmp.groupby("Product Type", dropna=False)["Token Amount Num"].sum().reset_index()
           .rename(columns={"Token Amount Num": "Token Amount Sum"}).sort_values("Token Amount Sum", ascending=False))
    agg["Product Type"] = agg["Product Type"].fillna("Unknown")
    agg["Token Amount Sum"] = agg["Token Amount Sum"].map('{:,.2f}'.format)
    return agg

def aggregate_qty_month_by_counterparty(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Counterparty", "Qty * Month (USD) Sum"])
    tmp = df.copy()
    if "Product Type" in tmp.columns:
        tmp = tmp[~tmp["Product Type"].astype(str).str.contains(r'\bMM\b', case=False, na=False)]
    col = "Qty * Month (USD)" if "Qty * Month (USD)" in tmp.columns else "Qty * Month"
    if col not in tmp.columns:
        return pd.DataFrame(columns=["Counterparty", "Qty * Month (USD) Sum"])
    tmp["QxM_USD_Num"] = tmp[col].astype(str).str.replace(",", "", regex=False).str.replace(" ", "", regex=False).pipe(pd.to_numeric, errors="coerce")
    agg = (tmp.groupby("Counterparty", dropna=False)["QxM_USD_Num"].sum().reset_index()
           .rename(columns={"QxM_USD_Num": "Qty * Month (USD) Sum"}).sort_values("Qty * Month (USD) Sum", ascending=False))
    agg["Counterparty"] = agg["Counterparty"].fillna("Unknown")
    agg["Qty * Month (USD) Sum"] = agg["Qty * Month (USD) Sum"].map('{:,.2f}'.format)
    return agg

# ================== IO Helpers (CSV) ==================
def read_csv_safely(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    for enc in ("utf-8", "utf-8-sig", "cp949", "latin1"):
        for sep in (None, ",", ";", "\t", "|"):
            bio = io.BytesIO(raw)
            try:
                df = pd.read_csv(bio, encoding=enc, sep=sep, engine="python" if sep is None else None, low_memory=False)
                return optimize_dataframe(df)
            except Exception:
                continue
    raise ValueError("CSV 파싱 실패: 인코딩/구분자를 확인하세요.")

def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        t = df[col].dtype
        try:
            if 'float' in str(t):
                df[col] = pd.to_numeric(df[col], downcast='float')
            elif 'int' in str(t):
                df[col] = pd.to_numeric(df[col], downcast='integer')
        except Exception:
            pass
    return df

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return b"" if df is None else df.to_csv(index=False).encode("utf-8")

# ================== Core Pipeline ==================
def classify_core(df_raw: pd.DataFrame, config: AppConfig, progress_placeholder=None) -> Dict:
    coupon_whitelist = normalize_party_list(config.coupon_parties)
    coupon_quote_set = normalize_quote_list(config.coupon_quotes)
    covered_call_whitelist = normalize_party_list(config.covered_call_parties)
    today = datetime.now(KST).date()

    # 1) 심볼/타임스탬프 수집
    pair_ts_pairs: List[Tuple[str, object]] = []
    current_symbols: List[str] = []

    for _, r in df_raw.iterrows():
        sym = str(r.get("Symbol", ""))
        parsed = parse_symbol(sym)
        base, quote, opt = parsed.get("base", ""), parsed.get("quote", ""), parsed.get("option", "")
        cp = r.get("Counterparty", "")
        ptype = infer_product_type(base, opt, quote, cp, sym, coupon_whitelist, coupon_quote_set, covered_call_whitelist)
        token_type = quote if ("Bonus Coupon" in ptype) else base
        pair_symbol = make_pair_symbol(token_type)

        start_ts = pd.to_datetime(r.get(config.trade_field, pd.NaT), errors="coerce", utc=False)
        trade_ts = start_ts if pd.notna(start_ts) else pd.to_datetime(r.get("Expiry Time", pd.NaT), errors="coerce", utc=False)
        trade_ts = trade_ts if pd.notna(trade_ts) else datetime.now(KST)

        if pair_symbol:
            if pair_symbol != "USDT":
                pair_ts_pairs.append((pair_symbol, trade_ts))
            current_symbols.append(pair_symbol)

    # 2) 과거 종가 배치
    if progress_placeholder:
        progress_placeholder.info(f"📊 가격 데이터 조회 중... (고유 조합 {len(set(pair_ts_pairs))}개)")
    price_cache = get_batch_prices_coinglass(pair_ts_pairs)

    # 3) 현재가 배치
    current_price_map = get_batch_current_prices(current_symbols)

    # 3.5) 거래일 종가 배치 (Binance)
    pair_date_pairs = [
        (p, resolve_trade_utc_date(ts))
        for (p, ts) in pair_ts_pairs
        if p and p != "USDT" and pd.notna(pd.to_datetime(ts, errors="coerce"))
    ]
    trade_close_map = get_batch_binance_closes(pair_date_pairs)

    # 4) 레코드 변환
    rows = []
    if progress_placeholder:
        pbar = progress_placeholder.progress(0); ptxt = progress_placeholder.empty()
    total = len(df_raw) or 1

    for i, (_, r) in enumerate(df_raw.iterrows(), 1):
        if progress_placeholder:
            pbar.progress(i/total); ptxt.text(f"처리 중... {i}/{total}")
        sym = str(r.get("Symbol", ""))
        parsed = parse_symbol(sym)
        base, quote, opt = parsed.get("base", ""), parsed.get("quote", ""), parsed.get("option", "")
        cp = r.get("Counterparty", "")
        ptype = infer_product_type(base, opt, quote, cp, sym, coupon_whitelist, coupon_quote_set, covered_call_whitelist)

        expiry_ts = pd.to_datetime(r.get("Expiry Time", pd.NaT), errors="coerce", utc=False)
        start_ts = pd.to_datetime(r.get(config.trade_field, pd.NaT), errors="coerce", utc=False)
        month_diff = calculate_month_difference(start_ts, expiry_ts) if (pd.notna(expiry_ts) and pd.notna(start_ts)) else None

        qty_raw = str(r.get("Qty", ""))
        try:
            qty_num = pd.to_numeric(qty_raw.replace(",", "").replace(" ", ""), errors="coerce")
        except Exception:
            qty_num = pd.NA

        token_type = quote if ("Bonus Coupon" in ptype) else base
        pair_symbol = make_pair_symbol(token_type)

        # 거래일 UTC 일자 (항상 계산)
        key_date = resolve_trade_utc_date(
            start_ts if pd.notna(start_ts) else (expiry_ts if pd.notna(expiry_ts) else datetime.now(KST))
        )

        price_close = None; qty_usd_trade = None
        if pd.notna(qty_num):
            if pair_symbol == "USDT":
                price_close = 1.0; qty_usd_trade = float(qty_num)
            else:
                key_date = resolve_trade_utc_date(start_ts if pd.notna(start_ts) else (expiry_ts if pd.notna(expiry_ts) else datetime.now(KST)))
                price_close = price_cache.get((pair_symbol, key_date))
                qty_usd_trade = (float(qty_num) * float(price_close)) if price_close is not None else None

        trade_date_bin_px = 1.0 if pair_symbol == "USDT" else trade_close_map.get((pair_symbol, key_date))
        qty_usd_trade_bin = (float(qty_num) * float(trade_date_bin_px)) if (pd.notna(qty_num) and (trade_date_bin_px is not None)) else None
        cur_px = current_price_map.get(pair_symbol, None)
        qty_usd_cur = (float(qty_num) * float(cur_px)) if (pd.notna(qty_num) and cur_px is not None) else None

        qxm = (float(qty_usd_trade_bin) * float(month_diff)) if (qty_usd_trade_bin is not None and month_diff is not None) else None

        exp_str_from_iso = extract_iso_date_to_str(r.get("Expiry Time", "")) or yyyymmdd_to_mdy_str(parsed.get("expiry", ""))
        exp_date = extract_iso_date_to_date(r.get("Expiry Time", "")) or yyyymmdd_to_date(parsed.get("expiry", ""))
        trade_date_str = extract_iso_date_to_str(r.get(config.trade_field, ""))

        rows.append({
            "Counterparty": cp,
            "Product Type": ptype,
            "Token Type": token_type,
            "API Symbol": pair_symbol,
            "Token Amount": qty_raw,
            "Qty": qty_raw,
            "Current Price (USD)": cur_px,
            "Trade Date Price (USD, Binance)": trade_date_bin_px,
            "Qty USD (Current)": qty_usd_cur,
            "Month Difference": month_diff,
            "Qty * Month (USD)": qxm,
            "Trade Date": trade_date_str,
            "Expiry Date": exp_str_from_iso,
            "_expiry_date_obj": exp_date,
            "_trade_date_obj": start_ts.date() if pd.notna(start_ts) else None,
        })

    if progress_placeholder:
        pbar.empty(); ptxt.empty()

    out = pd.DataFrame(rows)
    # ---- Display trimming & column order ----
    # Drop requested columns from output tables (keep internal computation only)
    out = out.drop(columns=["Price Close (USD on Trade Date)", "Qty USD (on Trade Date)"], errors="ignore")

    # Reorder columns so that 'Current Price Debug' appears at the end
    desired_order = [
        "Counterparty", "Product Type", "Token Type", "API Symbol",
        "Token Amount", "Qty",
        "Current Price (USD)", "Trade Date Price (USD, Binance)", "Qty USD (Current)",
        "Month Difference", "Qty * Month (USD)",
        "Trade Date", "Expiry Date",
        "Current Price Debug"
    ]
    cols = [c for c in desired_order if c in out.columns] + [c for c in out.columns if c not in desired_order]
    out = out[cols]

    if not out.empty and pd.api.types.is_object_dtype(out["_expiry_date_obj"]):
        out["_expiry_date_obj"] = pd.to_datetime(out["_expiry_date_obj"]).dt.tz_localize(None)

    # 만기 필터
    today_ts = pd.to_datetime(today)
    nonexp = out[out["_expiry_date_obj"].notna() & (out["_expiry_date_obj"] >= today_ts)].copy()

    def month_offset(y: int, m: int, k: int):
        nm = m + k
        return y + (nm - 1)//12, ((nm - 1)%12)+1

    y, m = today.year, today.month
    y1, m1 = month_offset(y, m, 1); y2, m2 = month_offset(y, m, 2); y3, m3 = month_offset(y, m, 3)

    def filter_by_month(df_in: pd.DataFrame, yy: int, mm: int):
        if df_in.empty:
            return df_in
        cond = df_in["_expiry_date_obj"].notna() & (df_in["_expiry_date_obj"].dt.year == yy) & (df_in["_expiry_date_obj"].dt.month == mm)
        return df_in[cond].copy()

    m1_df = filter_by_month(out, y1, m1); m2_df = filter_by_month(out, y2, m2); m3_df = filter_by_month(out, y3, m3)

    def display(df_in: pd.DataFrame) -> pd.DataFrame:
        return df_in if (df_in is None or df_in.empty) else df_in.drop(columns=["_expiry_date_obj", "_trade_date_obj"], errors="ignore")

    full_display = apply_output_filters(display(out), config.exclude_mm, config.exclude_unknown)
    nonexp_display = apply_output_filters(display(nonexp), config.exclude_mm, config.exclude_unknown)
    m1_display = apply_output_filters(display(m1_df), config.exclude_mm, config.exclude_unknown)
    m2_display = apply_output_filters(display(m2_df), config.exclude_mm, config.exclude_unknown)
    m3_display = apply_output_filters(display(m3_df), config.exclude_mm, config.exclude_unknown)

    # 집계
    agg_nonexp = aggregate_by_product_type(nonexp_display)
    agg_m1 = aggregate_by_product_type(m1_display)
    agg_m2 = aggregate_by_product_type(m2_display)
    agg_m3 = aggregate_by_product_type(m3_display)

    start_of_year = date(today.year, 1, 1)
    out_with_filters = apply_output_filters(out, config.exclude_mm, config.exclude_unknown)
    year_mask = out_with_filters["_trade_date_obj"].notna() & (pd.to_datetime(out_with_filters["_trade_date_obj"]).dt.date >= start_of_year)
    full_display_this_year = out_with_filters[year_mask].copy()
    agg_qty_month_cp = aggregate_qty_month_by_counterparty(full_display_this_year)

    # 현재가 스냅샷
    unique_syms = sorted(set(full_display["API Symbol"].dropna().astype(str).replace("", pd.NA).dropna().tolist())) if isinstance(full_display, pd.DataFrame) and not full_display.empty else []
    current_price_table = pd.DataFrame([{ "API Symbol": s, "Current Price (USD)": current_price_map.get(s)} for s in unique_syms])

    return {
        "full": full_display,
        "nonexp": nonexp_display,
        "m1": m1_display,
        "m2": m2_display,
        "m3": m3_display,
        "agg_nonexp": agg_nonexp,
        "agg_m1": agg_m1,
        "agg_m2": agg_m2,
        "agg_m3": agg_m3,
        "agg_qty_month_cp": agg_qty_month_cp,
        "current_prices": current_price_table,
        "msg": "✅ 완료! Coinglass 종가 + Binance 현재가 반영 (TLS 인증서 자동 설정 포함)",
        "today_info": f"오늘(Asia/Seoul): {today.isoformat()}"
    }

# ================== UI (Streamlit) ==================
st.title("VERONICA XUNKE SUPPORT · Patched")
st.caption("TLS 인증서 자동 설정, 코드 모듈화, 기능 유지 · 🔐 내부 접근 보호")

# Sidebar
with st.sidebar:
    st.header("⚙️ 설정")
    config = AppConfig.load_from_session()

    uploaded = st.file_uploader("📁 CSV 업로드", type=["csv"])
    if uploaded is not None:
        try:
            raw = uploaded.getvalue(); file_hash = hashlib.md5(raw).hexdigest()
            if st.session_state.get("file_hash") != file_hash:
                with st.spinner("CSV 파일 로드 중..."):
                    df_raw = read_csv_safely(uploaded)
                    # Validate
                    required = ['Symbol', 'Counterparty', 'Qty']
                    missing_required = [c for c in required if c not in df_raw.columns]
                    if missing_required:
                        st.error(f"❌ 필수 컬럼 누락: {', '.join(missing_required)}"); st.stop()
                    # Optional warn
                    optional = ['Expiry Time', 'Created Time', 'Initiation Time']
                    missing_optional = [c for c in optional if c not in df_raw.columns]
                    if missing_optional:
                        st.warning(f"⚠️ 선택적 컬럼 누락: {', '.join(missing_optional)} - 일부 기능 제한될 수 있음")
                    st.success("✅ 데이터 검증 완료")
                    st.session_state.df_raw = df_raw
                    st.session_state.file_hash = file_hash
        except Exception as e:
            st.error(f"CSV 로드 실패: {e}"); st.stop()

    config.trade_field = st.radio("📅 Trade Date 기준", ["Created Time", "Initiation Time"], index=0 if config.trade_field == "Created Time" else 1)

    st.markdown("---")
    st.subheader("🎯 분류 설정")
    config.coupon_parties = st.text_input("Bonus Coupon 카운터파티", config.coupon_parties)
    config.coupon_quotes = st.text_input("Bonus Coupon 결제통화", config.coupon_quotes)
    config.covered_call_parties = st.text_input("Covered Call 카운터파티", config.covered_call_parties)

    st.markdown("---")
    st.subheader("🔍 필터 & 디버그")
    col1, col2, col3 = st.columns(3)
    with col1:
        config.exclude_mm = st.checkbox("MM 제외", config.exclude_mm)
    with col2:
        config.exclude_unknown = st.checkbox("Unknown 제외", config.exclude_unknown)
    with col3:
        debug_mode = st.checkbox("🧪 디버그 모드", value=st.session_state.get("debug_mode", False))
        st.session_state.debug_mode = debug_mode

    # Counterparty 자동완성
    if "df_raw" in st.session_state and "Counterparty" in st.session_state.df_raw.columns:
        vals = (st.session_state.df_raw["Counterparty"].dropna().astype(str).map(lambda s: s.strip()).replace("", pd.NA).dropna().unique().tolist())
        st.session_state.cp_catalog = sorted(set(vals), key=lambda s: s.lower())
    cp_search = st.text_input("🔍 Counterparty 검색", placeholder="입력하면서 자동완성...")

    def suggest(q: str, catalog: List[str], limit: int = 10) -> List[str]:
        if not catalog: return []
        if not q: return catalog[:limit]
        ql = q.lower()
        partial = [c for c in catalog if ql in c.lower()]
        fuzzy = difflib.get_close_matches(q, catalog, n=limit, cutoff=0.5)
        out, seen = [], set()
        for it in partial + fuzzy:
            if it not in seen:
                out.append(it); seen.add(it)
            if len(out) >= limit: break
        return out

    suggestions = suggest(cp_search, st.session_state.get("cp_catalog", []), 10)
    if suggestions:
        st.markdown("**추천 목록:**")
        for i, sopt in enumerate(suggestions[:5]):
            c1, c2 = st.columns([3,1])
            with c1: st.text(sopt)
            with c2:
                if st.button("➕", key=f"add_{i}", use_container_width=True):
                    cur = [s.strip() for s in config.covered_call_parties.split(",") if s.strip()]
                    if sopt not in cur:
                        cur.append(sopt); config.covered_call_parties = ", ".join(cur); config.save_to_session(); st.rerun()
    if cp_search:
        if st.button(f"➕ '{cp_search}' 직접 추가", use_container_width=True):
            cur = [s.strip() for s in config.covered_call_parties.split(",") if s.strip()]
            if cp_search not in cur:
                cur.append(cp_search); config.covered_call_parties = ", ".join(cur); config.save_to_session(); st.rerun()

    # 저장
    config.save_to_session()

# Main
st.caption("⚡ 업로드 후 좌측 필터를 조정하면 아래 표/요약이 갱신됩니다.")
if "df_raw" not in st.session_state:
    st.info("📂 좌측 사이드바에서 CSV를 업로드하세요.")
    st.stop()

df_raw = st.session_state.df_raw
with st.expander("원본 데이터 미리보기", expanded=False):
    st.dataframe(df_raw.head(50), use_container_width=True)
    mem_bytes = df_raw.memory_usage(deep=True).sum() if hasattr(df_raw, 'memory_usage') else 0
    st.caption(f"rows={len(df_raw):,}, memory≈{mem_bytes/1_048_576:,.2f} MB")

# 실행 버튼 & 캐시 키
run_col1, run_col2 = st.columns([1,3])
with run_col1:
    run_clicked = st.button("🚀 분류 실행 / 새로고침", type="primary")
with run_col2:
    auto_run = st.toggle("업로드 시 자동 실행", value=st.session_state.get("auto_run", True)); st.session_state.auto_run = auto_run

def _hash_config(cfg: AppConfig) -> str:
    try:
        payload = f"{cfg.trade_field}|{cfg.coupon_parties}|{cfg.coupon_quotes}|{cfg.covered_call_parties}|{cfg.exclude_mm}|{cfg.exclude_unknown}"
        return hashlib.md5(payload.encode("utf-8")).hexdigest()
    except Exception:
        return str(datetime.utcnow().timestamp())

need_run = run_clicked or st.session_state.get("last_keys") != (st.session_state.get("file_hash"), _hash_config(AppConfig.load_from_session()))
if auto_run:
    need_run = True

if need_run:
    try:
        progress_area = st.empty()
        with st.spinner("분류 중..."):
            result = classify_core(df_raw, AppConfig.load_from_session(), progress_placeholder=progress_area)
        st.session_state.last_result = result
        st.session_state.last_keys = (st.session_state.get("file_hash"), _hash_config(AppConfig.load_from_session()))
    except Exception as e:
        st.error(f"처리 중 오류: {e}"); st.stop()
else:
    result = st.session_state.get("last_result")

if not result:
    st.stop()

st.success(result.get("msg", "완료"))
st.caption(result.get("today_info", ""))

# Tabs
(tab_all, tab_nonexp, tab_m1, tab_m2, tab_m3, tab_summary, tab_cp, tab_px, tab_debug) = st.tabs([
    "전체", "미만기", "M+1", "M+2", "M+3", "요약(합계)", "Counterparty 합계", "현재가 스냅샷", "디버그"
])

def table_with_download(df: Optional[pd.DataFrame], label: str, key: str):
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        st.info("데이터가 없습니다."); return
    st.dataframe(df, use_container_width=True)
    st.download_button("💾 CSV 다운로드", data=to_csv_bytes(df), file_name=f"{label}.csv", mime="text/csv", key=f"dl_{key}")

with tab_all:
    st.subheader("모든 레코드")
    table_with_download(result["full"], "all_records", "all")

with tab_nonexp:
    st.subheader("미만기 레코드")
    table_with_download(result["nonexp"], "non_expired", "nonexp")

with tab_m1:
    st.subheader("만기 M+1")
    table_with_download(result["m1"], "m_plus_1", "m1")

with tab_m2:
    st.subheader("만기 M+2")
    table_with_download(result["m2"], "m_plus_2", "m2")

with tab_m3:
    st.subheader("만기 M+3")
    table_with_download(result["m3"], "m_plus_3", "m3")

with tab_summary:
    st.subheader("요약(합계)")
    c1, c2 = st.columns(2)
    try:
        total_rows = len(result["full"]) if isinstance(result.get("full"), pd.DataFrame) else 0
        nonexp_rows = len(result["nonexp"]) if isinstance(result.get("nonexp"), pd.DataFrame) else 0
        c1.metric("전체 건수", f"{total_rows:,}"); c2.metric("미만기 건수", f"{nonexp_rows:,}")
    except Exception:
        pass
    st.markdown("**미만기 Product Type별 Token Amount 합계**"); table_with_download(result["agg_nonexp"], "agg_nonexpired_by_product_type", "agg_nonexp")
    st.markdown("**M+1 요약**"); table_with_download(result["agg_m1"], "agg_m1_by_product_type", "agg_m1")
    st.markdown("**M+2 요약**"); table_with_download(result["agg_m2"], "agg_m2_by_product_type", "agg_m2")
    st.markdown("**M+3 요약**"); table_with_download(result["agg_m3"], "agg_m3_by_product_type", "agg_m3")

with tab_cp:
    st.subheader("Counterparty별 Qty * Month (USD) 합계 (올해, MM 제외)")
    table_with_download(result["agg_qty_month_cp"], "agg_qty_month_by_counterparty", "agg_cp")

with tab_px:
    st.subheader("API Symbol 현재가 스냅샷 (Binance)")
    table_with_download(result["current_prices"], "current_prices_snapshot", "px")

with tab_debug:
    st.subheader("현재가 None 진단 도구")
    if st.session_state.get("debug_mode", False):
        unique_syms = (result["current_prices"]["API Symbol"].dropna().astype(str).unique().tolist() if isinstance(result.get("current_prices"), pd.DataFrame) and not result["current_prices"].empty else [])
        st.markdown("**(1) 배치 진단 테이블**")
        dbg_df = build_current_price_debug_table(unique_syms) if unique_syms else pd.DataFrame()
        table_with_download(dbg_df, "current_price_debug", "dbg_batch")
        st.markdown("**(2) 개별 심볼 점검**")
        colx, coly = st.columns([2,1])
        with colx:
            test_sym = st.text_input("API Symbol 입력 (예: BTCUSDT)")
        with coly:
            if st.button("🔎 테스트") and test_sym:
                r1 = debug_fetch_binance_symbol(test_sym); r2 = debug_check_symbol_exists(test_sym)
                st.write({"ticker": r1, "exists": r2})
    else:
        st.info("사이드바에서 '🧪 디버그 모드'를 켜면 진단 도구가 활성화됩니다.")

st.caption("※ 열 순서 변경/숨김은 표 우측 상단 메뉴에서 조정 가능합니다.")
