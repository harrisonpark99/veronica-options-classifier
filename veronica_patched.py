# -*- coding: utf-8 -*-
"""
VERONICA XUNKE SUPPORT (OKX-swapped)
- Binance API 제거, OKX API로 대체
- TLS 인증서 경로 자동 설정 (certifi)
- 기능 유지: 분류, (선택)Coinglass 과거종가, OKX 현재가/과거종가, 집계/필터/다운로드, 디버그 툴
- 🔐 st.secrets 기반 비밀번호 게이트 (안전비교 + 공백/개행 제거)
- 🛡️ 업로드/세션 가드 보강 (NoneType.head 방지, 실패 시 기존 데이터 유지)
"""

# ================== Bootstrap & Globals ==================
import os
import re
import io
import hmac
import hashlib
import difflib
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date, timezone, timedelta

# TLS 인증서 경로 자동 설정
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
st.set_page_config(page_title="CSV 옵션 딜 분류기 (OKX)", layout="wide", initial_sidebar_state="expanded")

APP_PASSWORD = st.secrets.get("APP_PASSWORD", os.environ.get("APP_PASSWORD", ""))

if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

if not st.session_state.auth_ok:
    st.title("🔐 패스워드를 입력하세요")
    st.caption("인증 후에 메인 화면으로 이동합니다.")
    pw = st.text_input("Password", type="password", placeholder="패스워드 입력")
    confirm = st.button("확인", type="primary")

    if confirm:
        if not str(APP_PASSWORD).strip():
            st.error("서버에 비밀번호가 설정되어 있지 않습니다. 관리자에게 문의하세요.")
        else:
            pw_norm = (pw or "").strip()
            app_pw_norm = str(APP_PASSWORD).strip()
            if hmac.compare_digest(pw_norm, app_pw_norm):
                st.session_state.auth_ok = True
                st.rerun()
            else:
                st.error("패스워드가 올바르지 않습니다.")
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

# ===== (선택) Coinglass API (그대로 유지하고 싶으면 secrets에 키 넣기) =====
def _get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, default)
    except Exception:
        return os.environ.get(name, default)

COINGLASS_API_KEY = _get_secret("COINGLASS_API_KEY", "")
CG_HEADERS = {"CG-API-KEY": COINGLASS_API_KEY} if COINGLASS_API_KEY else {}

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
    if not isinstance(yyyymmdd, str) or len(yyyymmdd) != 8:
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
    """
    입력 토큰을 OKX instId 포맷으로 변환.
    BTC -> BTC-USDT
    USD/USDT/USDC -> USDT (1.0 취급, API 호출 생략)
    """
    if not token:
        return ""
    tk = re.sub(r'[^A-Z0-9]', '', str(token).upper())
    tk = TOKEN_ALIASES.get(tk, tk)
    if tk in {"USD", "USDT", "USDC"}:
        return "USDT"  # sentinel
    return f"{tk}-USDT"

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

# ================== (선택) Coinglass – 그대로 유지 ==================
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_coinglass_ohlc(exchange: str, symbol: str, market_type: str = "spot", interval: str = "1d", limit: int = 2000):
    if not COINGLASS_API_KEY:
        return None
    url = (
        "https://open-api-v3.coinglass.com/api/price/ohlc-history"
        f"?exchange={exchange}&symbol={symbol}&type={market_type}&interval={interval}&limit={limit}"
    )
    try:
        r = requests.get(url, headers=CG_HEADERS, timeout=15, verify=certifi.where())
        if r.status_code != 200:
            return None
        js = r.json()
        if js.get("code") != "0":
            return None
        return js.get("data", []) or None
    except Exception:
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

    # closest to midday fallback (±1.5d)
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

@st.cache_data(show_spinner=False, ttl=3600)
def get_batch_prices_coinglass(pair_ts_pairs: List[Tuple[str, object]]) -> Dict[Tuple[str, date], float]:
    """(선택) Coinglass 과거 종가 캐시 — 키가 없으면 빈 dict"""
    if not COINGLASS_API_KEY:
        return {}
    prices: Dict[Tuple[str, date], float] = {}
    uniq = list({(p, pd.to_datetime(ts, errors="coerce")) for p, ts in pair_ts_pairs})
    if not uniq:
        return prices
    with ThreadPoolExecutor(max_workers=5) as ex:
        fut = {ex.submit(_coinglass_one, p, ts): (p, ts) for (p, ts) in uniq}
        for f in as_completed(fut):
            p, ts = fut[f]
            try:
                px = f.result()
                if px is not None:
                    prices[(p, resolve_trade_utc_date(ts))] = px
            except Exception:
                pass
    return prices

def _coinglass_one(pair_symbol: str, trade_ts) -> Optional[float]:
    if not pair_symbol or pair_symbol == "USDT":
        return 1.0 if pair_symbol == "USDT" else None
    utc_d = resolve_trade_utc_date(trade_ts)
    # Binance → OKX 순으로 예시 (원하면 더 추가)
    for ex in ["OKX", "Binance", "Bybit", "Bitget"]:
        rows_spot = fetch_coinglass_ohlc(ex, pair_symbol, market_type="spot", interval="1d", limit=2000)
        px = find_close_for_utc_date_from_rows(rows_spot, utc_d) if rows_spot else None
        if px is None:
            rows_fut = fetch_coinglass_ohlc(ex, pair_symbol, market_type="futures", interval="1d", limit=2000)
            px = find_close_for_utc_date_from_rows(rows_fut, utc_d) if rows_fut else None
        if px is not None:
            return px
    return None

# ================== OKX API (Binance 대체) ==================
OKX_BASE = "https://www.okx.com"

@st.cache_data(show_spinner=False, ttl=30)
def fetch_okx_ticker_price(inst_id: str) -> Optional[float]:
    """OKX 현재가 — /market/ticker"""
    if not inst_id or inst_id == "USDT":
        return 1.0 if inst_id == "USDT" else None
    try:
        r = requests.get(f"{OKX_BASE}/api/v5/market/ticker",
                         params={"instId": inst_id}, timeout=8, verify=certifi.where())
        if r.status_code != 200:
            return None
        js = r.json()
        if js.get("code") == "0" and js.get("data"):
            return float(js["data"][0]["last"])
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False, ttl=15)
def debug_fetch_okx_ticker(inst_id: str) -> Dict[str, object]:
    try:
        r = requests.get(f"{OKX_BASE}/api/v5/market/ticker",
                         params={"instId": inst_id}, timeout=8, verify=certifi.where())
        return {
            "instId": inst_id,
            "endpoint": "/api/v5/market/ticker",
            "status": r.status_code,
            "ok": (r.status_code == 200),
            "response": r.text[:300]
        }
    except Exception as e:
        return {"instId": inst_id, "endpoint": "/api/v5/market/ticker", "status": "EXC", "ok": False, "response": str(e)[:300]}

@st.cache_data(show_spinner=False, ttl=15)
def debug_check_okx_instrument(inst_id: str) -> Dict[str, object]:
    try:
        r = requests.get(f"{OKX_BASE}/api/v5/public/instruments",
                         params={"instType": "SPOT", "instId": inst_id}, timeout=8, verify=certifi.where())
        ok = False; preview = ""
        if r.status_code == 200:
            js = r.json(); ok = (js.get("code") == "0" and len(js.get("data", [])) > 0)
            preview = r.text[:300]
        return {"instId": inst_id, "endpoint": "/api/v5/public/instruments", "status": r.status_code, "ok": ok, "response": preview}
    except Exception as e:
        return {"instId": inst_id, "endpoint": "/api/v5/public/instruments", "status": "EXC", "ok": False, "response": str(e)[:300]}

def build_current_price_debug_table_okx(inst_ids: List[str]) -> pd.DataFrame:
    ids = sorted(set([s for s in inst_ids if s]))
    rows = []
    for inst in ids:
        r1 = debug_fetch_okx_ticker(inst)
        r2 = debug_check_okx_instrument(inst)
        rows.append({
            "OKX instId": inst,
            "ticker_ok": r1.get("ok"),
            "ticker_status": r1.get("status"),
            "ticker_preview": r1.get("response"),
            "inst_exists_ok": r2.get("ok"),
            "inst_status": r2.get("status"),
            "inst_preview": r2.get("response"),
        })
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False, ttl=3600)
def get_okx_daily_close_for_date(inst_id: str, utc_date: date) -> Tuple[Optional[float], str]:
    """
    OKX 1D 캔들에서 특정 UTC 일자의 종가를 조회.
    /api/v5/market/candles?instId=BTC-USDT&bar=1D&after=...&before=...
    반환: (close, "ok") or (None, reason)
    """
    if not inst_id:
        return None, "empty"
    if inst_id == "USDT":
        return 1.0, "stablecoin"

    day_start = datetime(utc_date.year, utc_date.month, utc_date.day, tzinfo=timezone.utc)
    start_ms = int(day_start.timestamp() * 1000)
    end_ms = start_ms + 86_400_000 - 1

    try:
        # OKX: after < ts <= before 형식이므로 살짝 여유 버퍼
        params = {
            "instId": inst_id,
            "bar": "1D",
            "after": start_ms - 1,
            "before": end_ms + 1,
            "limit": 100,
        }
        r = requests.get(f"{OKX_BASE}/api/v5/market/candles", params=params, timeout=10, verify=certifi.where())
        if r.status_code != 200:
            return None, f"http:{r.status_code}"
        js = r.json()
        if js.get("code") != "0" or not js.get("data"):
            return None, "empty"
        # data: list of [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
        close_px = None
        for row in js["data"]:
            ts_ms = int(row[0])
            if start_ms <= ts_ms <= end_ms:
                try:
                    close_px = float(row[4])
                    break
                except Exception:
                    pass
        if close_px is not None:
            return close_px, "ok"

        # 근접 일자(±3일) 보정
        best = None; best_dist = 10**18
        target = start_ms + 43_200_000
        for row in js["data"]:
            ts_ms = int(row[0]); dist = abs(ts_ms - target)
            if dist < best_dist:
                try:
                    best = float(row[4]); best_dist = dist
                except Exception:
                    pass
        if best is not None and best_dist <= 129_600_000:
            return best, "nearest"
        return None, "not_found"
    except Exception as e:
        return None, f"EXC:{str(e)[:80]}"

@st.cache_data(show_spinner=False, ttl=3600)
def get_batch_okx_closes(pair_date_pairs: List[Tuple[str, date]]) -> Dict[Tuple[str, date], float]:
    """여러 (instId, UTC일자)의 일별 종가 배치 조회 — OKX"""
    uniq = list({(sym, d) for sym, d in pair_date_pairs})
    out: Dict[Tuple[str, date], float] = {}
    if not uniq:
        return out
    with ThreadPoolExecutor(max_workers=8) as ex:
        fut = {ex.submit(get_okx_daily_close_for_date, sym, d): (sym, d) for sym, d in uniq}
        for f in as_completed(fut):
            key = fut[f]
            try:
                px, _ = f.result()
                if px is not None:
                    out[key] = px
            except Exception:
                pass
    return out

@st.cache_data(show_spinner=False, ttl=30)
def get_batch_current_prices_okx(inst_ids: List[str]) -> Dict[str, Optional[float]]:
    uniq = sorted({s for s in inst_ids if s})
    results: Dict[str, Optional[float]] = {s: (1.0 if s == "USDT" else None) for s in uniq}
    if not uniq:
        return {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        fut_map = {ex.submit(fetch_okx_ticker_price, s): s for s in uniq if s != "USDT"}
        for fut in as_completed(fut_map):
            sym = fut_map[fut]
            try:
                results[sym] = fut.result()
            except Exception:
                results[sym] = None
    return results

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
        inst_id = make_pair_symbol(token_type)  # OKX instId

        start_ts = pd.to_datetime(r.get(config.trade_field, pd.NaT), errors="coerce", utc=False)
        trade_ts = start_ts if pd.notna(start_ts) else pd.to_datetime(r.get("Expiry Time", pd.NaT), errors="coerce", utc=False)
        trade_ts = trade_ts if pd.notna(trade_ts) else datetime.now(KST)

        if inst_id:
            if inst_id != "USDT":
                pair_ts_pairs.append((inst_id, trade_ts))
            current_symbols.append(inst_id)

    # (선택) Coinglass 과거 종가 캐시
    if progress_placeholder:
        progress_placeholder.info(f"📊 가격 데이터 조회 중... (고유 조합 {len(set(pair_ts_pairs))}개)")
    price_cache = get_batch_prices_coinglass(pair_ts_pairs) if COINGLASS_API_KEY else {}

    # 3) 현재가 배치 (OKX)
    current_price_map = get_batch_current_prices_okx(current_symbols)

    # 3.5) 거래일 종가 배치 (OKX)
    pair_date_pairs = [
        (p, resolve_trade_utc_date(ts))
        for (p, ts) in pair_ts_pairs
        if p and p != "USDT" and pd.notna(pd.to_datetime(ts, errors="coerce"))
    ]
    trade_close_map = get_batch_okx_closes(pair_date_pairs)

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
        inst_id = make_pair_symbol(token_type)

        # 거래일 UTC 일자
        key_date = resolve_trade_utc_date(
            start_ts if pd.notna(start_ts) else (expiry_ts if pd.notna(expiry_ts) else datetime.now(KST))
        )

        # (선택) Coinglass — 안 쓰면 price_close는 None
        price_close = None; qty_usd_trade = None
        if pd.notna(qty_num):
            if inst_id == "USDT":
                price_close = 1.0; qty_usd_trade = float(qty_num)
            else:
                px_cg = price_cache.get((inst_id, key_date)) if price_cache else None
                price_close = px_cg
                qty_usd_trade = (float(qty_num) * float(price_close)) if price_close is not None else None

        trade_date_okx_px = 1.0 if inst_id == "USDT" else trade_close_map.get((inst_id, key_date))
        qty_usd_trade_okx = (float(qty_num) * float(trade_date_okx_px)) if (pd.notna(qty_num) and (trade_date_okx_px is not None)) else None
        cur_px = current_price_map.get(inst_id, None)
        qty_usd_cur = (float(qty_num) * float(cur_px)) if (pd.notna(qty_num) and cur_px is not None) else None

        qxm = (float(qty_usd_trade_okx) * float(month_diff)) if (qty_usd_trade_okx is not None and month_diff is not None) else None

        exp_str_from_iso = extract_iso_date_to_str(r.get("Expiry Time", "")) or yyyymmdd_to_mdy_str(parsed.get("expiry", ""))
        exp_date = extract_iso_date_to_date(r.get("Expiry Time", "")) or yyyymmdd_to_date(parsed.get("expiry", ""))
        trade_date_str = extract_iso_date_to_str(r.get(config.trade_field, ""))

        rows.append({
            "Counterparty": cp,
            "Product Type": ptype,
            "Token Type": token_type,
            "API Symbol": inst_id,  # OKX instId
            "Token Amount": qty_raw,
            "Qty": qty_raw,
            "Current Price (USD)": cur_px,
            "Trade Date Price (USD, OKX)": trade_date_okx_px,
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

    # 내부 계산용 컬럼 제거 (Coinglass 기반 컬럼은 화면에서 숨김)
    out = out.drop(columns=["Price Close (USD on Trade Date)", "Qty USD (on Trade Date)"], errors="ignore")

    # 표시 컬럼 순서
    desired_order = [
        "Counterparty", "Product Type", "Token Type", "API Symbol",
        "Token Amount", "Qty",
        "Current Price (USD)", "Trade Date Price (USD, OKX)", "Qty USD (Current)",
        "Month Difference", "Qty * Month (USD)",
        "Trade Date", "Expiry Date",
        "Current Price Debug"
    ]
    cols = [c for c in desired_order if c in out.columns] + [c for c in out.columns if c not in desired_order]
    out = out[cols]

    if not out.empty and pd.api.types.is_object_dtype(out["_expiry_date_obj"]):
        out["_expiry_date_obj"] = pd.to_datetime(out["_expiry_date_obj"]).dt.tz_localize(None)

    # 만기 필터
    today = datetime.now(KST).date()
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
        "msg": "✅ 완료! OKX 현재가 + OKX 거래일 종가 적용 (Coinglass는 선택적)",
    }

# ================== UI (Streamlit) ==================
st.title("VERONICA XUNKE SUPPORT · OKX")
st.caption("OKX API 기반 현재가/과거가, TLS 인증서 자동 설정, 코드 모듈화, 🔐 내부 접근 보호")

# Sidebar
with st.sidebar:
    st.header("⚙️ 설정")
    config = AppConfig.load_from_session()

    # (선택) 데이터 리셋 버튼
    if st.button("데이터 리셋", help="불러온 CSV와 캐시 초기화"):
        st.session_state.pop("df_raw", None)
        st.session_state.pop("file_hash", None)
        st.session_state.pop("last_result", None)
        st.session_state.pop("last_keys", None)
        st.rerun()

    uploaded = st.file_uploader("📁 CSV 업로드", type=["csv"])
    if uploaded is not None:
        try:
            raw = uploaded.getvalue()
            file_hash = hashlib.md5(raw).hexdigest()
            if st.session_state.get("file_hash") != file_hash:
                with st.spinner("CSV 파일 로드 중..."):
                    df_candidate = read_csv_safely(uploaded)

                    # 필수 컬럼 체크
                    required = ['Symbol', 'Counterparty', 'Qty']
                    missing_required = [c for c in required if c not in df_candidate.columns]
                    if missing_required:
                        st.error(f"❌ 필수 컬럼 누락: {', '.join(missing_required)}")
                        st.stop()

                    # 선택 컬럼 경고
                    optional = ['Expiry Time', 'Created Time', 'Initiation Time']
                    missing_optional = [c for c in optional if c not in df_candidate.columns]
                    if missing_optional:
                        st.warning(f"⚠️ 선택적 컬럼 누락: {', '.join(missing_optional)} - 일부 기능 제한될 수 있음")

                    st.success("✅ 데이터 검증 완료")
                    # 성공시에만 저장
                    st.session_state.df_raw = df_candidate
                    st.session_state.file_hash = file_hash
        except Exception as e:
            st.error(f"CSV 로드 실패: {e}")
            st.stop()

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
        debug_mode = st.checkbox("🧪 디버그 모드(OKX)", value=st.session_state.get("debug_mode", False))
        st.session_state.debug_mode = debug_mode

    # Counterparty 자동완성
    if isinstance(st.session_state.get("df_raw"), pd.DataFrame) and "Counterparty" in st.session_state.df_raw.columns:
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

# ✅ None 방지: 존재 + 타입 확인
df_raw = st.session_state.get("df_raw", None)
if not isinstance(df_raw, pd.DataFrame):
    st.info("📂 좌측 사이드바에서 CSV를 업로드하세요.")
    st.stop()

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
        st.error(f"처리 중 오류: {e}")
        st.stop()
else:
    result = st.session_state.get("last_result")

if not result:
    st.stop()

st.success(result.get("msg", "완료"))

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
    st.mark다운("**M+2 요약**"); table_with_download(result["agg_m2"], "agg_m2_by_product_type", "agg_m2")
    st.markdown("**M+3 요약**"); table_with_download(result["agg_m3"], "agg_m3_by_product_type", "agg_m3")

with tab_cp:
    st.subheader("Counterparty별 Qty * Month (USD) 합계 (올해, MM 제외)")
    table_with_download(result["agg_qty_month_cp"], "agg_qty_month_by_counterparty", "agg_cp")

with tab_px:
    st.subheader("API Symbol 현재가 스냅샷 (OKX)")
    table_with_download(result["current_prices"], "current_prices_snapshot", "px")

with tab_debug:
    st.subheader("현재가/심볼 진단 도구 (OKX)")
    if st.session_state.get("debug_mode", False):
        unique_ids = (result["current_prices"]["API Symbol"].dropna().astype(str).unique().tolist()
                      if isinstance(result.get("current_prices"), pd.DataFrame) and not result["current_prices"].empty else [])
        st.markdown("**(1) 배치 진단 테이블 (OKX)**")
        dbg_df = build_current_price_debug_table_okx(unique_ids) if unique_ids else pd.DataFrame()
        table_with_download(dbg_df, "okx_price_debug", "dbg_okx_batch")
        st.markdown("**(2) 개별 심볼 점검 (OKX instId 포맷: BTC-USDT)**")
        colx, coly = st.columns([2,1])
        with colx:
            test_inst = st.text_input("OKX instId 입력 (예: BTC-USDT)")
        with coly:
            if st.button("🔎 테스트") and test_inst:
                r1 = debug_fetch_okx_ticker(test_inst); r2 = debug_check_okx_instrument(test_inst)
                st.write({"ticker": r1, "instrument": r2})
    else:
        st.info("사이드바에서 '🧪 디버그 모드(OKX)'를 켜면 진단 도구가 활성화됩니다.")

st.caption("※ 열 순서 변경/숨김은 표 우측 상단 메뉴에서 조정 가능합니다.")
