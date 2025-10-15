# -*- coding: utf-8 -*-
"""
VERONICA XUNKE SUPPORT (Patched, Trade-Date Close Fixed)
- 🔐 st.secrets 비밀번호 게이트
- 💪 현재가: 벌크 티커 → 누락분 개별 폴백
- 📅 거래일 종가: candles + history-candles 페이지네이션으로 안정 조회
- 🛠 종가 매칭 개선: 선택 캔들(ts) 기준 in-range 판정, ±2일 허용, 조기 종료 완화
- 기능 유지: 분류, 집계/필터/다운로드, 디버그 툴
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
OKX_BASE = "https://www.okx.com"

# ================== Auth Gate (with st.secrets) ==================
st.set_page_config(page_title="CSV 옵션 딜 분류기 (OKX Patched)", layout="wide", initial_sidebar_state="expanded")

APP_PASSWORD = st.secrets.get("APP_PASSWORD", os.environ.get("APP_PASSWORD", ""))

if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

if not st.session_state.auth_ok:
    st.title("🔐 패스워드를 입력하세요")
    st.caption("인증 후 메인 화면으로 이동합니다.")
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
    # NEW: tz-naive 타임스탬프 가정(KST/UTC)
    naive_ts_timezone: str = "KST"

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

# ================== Utilities ==================
def norm_inst_id(s: str) -> str:
    """OKX instId 표준화: 공백 제거, 대문자, '_'→'-', 스테이블은 'USDT' 고정."""
    if not isinstance(s, str):
        return ""
    up = s.strip().upper().replace("_", "-")
    if up in {"USDT", "USDC", "USD"}:
        return "USDT"
    return up

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
    """BTC -> BTC-USDT, USD/USDT/USDC -> USDT (1.0 취급)"""
    if not token:
        return ""
    tk = re.sub(r'[^A-Z0-9]', '', str(token).upper())
    tk = TOKEN_ALIASES.get(tk, tk)
    if tk in {"USD", "USDT", "USDC"}:
        return "USDT"
    return norm_inst_id(f"{tk}-USDT")

def resolve_trade_utc_date(ts_val, naive_tz: str = "KST") -> date:
    """
    tz-naive로 들어온 타임스탬프를 지정 타임존(KST/UTC)으로 가정해 UTC date로 변환.
    """
    tz_assumed = KST if str(naive_tz).upper() == "KST" else timezone.utc

    def _today_in_tz(tzinfo):
        kst_today = datetime.now(tzinfo).date()
        return datetime(kst_today.year, kst_today.month, kst_today.day, tzinfo=tzinfo).astimezone(timezone.utc).date()

    if ts_val is None or pd.isna(ts_val):
        return _today_in_tz(tz_assumed)

    ts = pd.to_datetime(ts_val, errors="coerce")
    if pd.isna(ts):
        return _today_in_tz(tz_assumed)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=tz_assumed)
    return ts.astimezone(timezone.utc).date()

# ================== OKX Market API ==================
@st.cache_data(show_spinner=False, ttl=10)
def fetch_okx_tickers_bulk_spot() -> Dict[str, float]:
    """OKX 스팟 전종목 티커 일괄 조회 → {INSTID: last}"""
    out: Dict[str, float] = {}
    try:
        r = requests.get(
            f"{OKX_BASE}/api/v5/market/tickers",
            params={"instType": "SPOT"},
            timeout=10,
            verify=certifi.where(),
        )
        if r.status_code != 200:
            return out
        js = r.json()
        if js.get("code") != "0":
            return out
        for row in js.get("data", []):
            inst = norm_inst_id(row.get("instId", ""))
            last = row.get("last")
            if inst and last not in (None, ""):
                try:
                    out[inst] = float(last)
                except Exception:
                    pass
    except Exception:
        pass
    return out

@st.cache_data(show_spinner=False, ttl=30)
def fetch_okx_ticker_price(inst_id: str) -> Optional[float]:
    """OKX 현재가 단건 — /market/ticker (재시도 포함)"""
    iid = norm_inst_id(inst_id)
    if not iid:
        return None
    if iid == "USDT":
        return 1.0
    params = {"instId": iid}
    for _ in range(2):
        try:
            r = requests.get(f"{OKX_BASE}/api/v5/market/ticker",
                             params=params, timeout=8, verify=certifi.where())
            if r.status_code != 200:
                continue
            js = r.json()
            if js.get("code") == "0" and js.get("data"):
                last = js["data"][0].get("last")
                if last not in (None, ""):
                    return float(last)
        except Exception:
            pass
    return None

@st.cache_data(show_spinner=False, ttl=15)
def get_batch_current_prices_okx(inst_ids: List[str]) -> Dict[str, Optional[float]]:
    """벌크 → 누락분 개별 폴백."""
    uniq = sorted({norm_inst_id(s) for s in inst_ids if s})
    results: Dict[str, Optional[float]] = {iid: (1.0 if iid == "USDT" else None) for iid in uniq}
    if not uniq:
        return {}

    bulk = fetch_okx_tickers_bulk_spot()
    for iid in uniq:
        if iid != "USDT" and iid in bulk:
            results[iid] = bulk[iid]

    missing = [iid for iid in uniq if iid != "USDT" and results[iid] is None]
    if missing:
        with ThreadPoolExecutor(max_workers=4) as ex:
            fut_map = {ex.submit(fetch_okx_ticker_price, iid): iid for iid in missing}
            for fut in as_completed(fut_map):
                iid = fut_map[fut]
                try:
                    px = fut.result()
                    if px is not None:
                        results[iid] = px
                except Exception:
                    pass
    return results

# ---- 캔들 페이징 (공용) ----
@st.cache_data(show_spinner=False, ttl=900)
def _okx_fetch_candles_page(inst_id: str, *, bar: str = "1D",
                            limit: int = 200, before: Optional[int] = None,
                            use_history: bool = False) -> List[List]:
    """
    OKX 1페이지 캔들 조회.
    - use_history=False  → /market/candles (최근 구간)
    - use_history=True   → /market/history-candles (과거 아카이브)
    반환: [[ts_ms, o, h, l, c, ...], ...]  (보통 최신→오래된 순)
    """
    iid = norm_inst_id(inst_id)
    if not iid:
        return []

    endpoint = "/api/v5/market/history-candles" if use_history else "/api/v5/market/candles"
    params = {"instId": iid, "bar": bar, "limit": limit}
    if before is not None:
        params["before"] = int(before)

    try:
        r = requests.get(f"{OKX_BASE}{endpoint}", params=params, timeout=10, verify=certifi.where())
        if r.status_code != 200:
            return []
        js = r.json()
        if js.get("code") != "0":
            return []
        data = js.get("data", [])
        try:
            data.sort(key=lambda k: int(k[0]), reverse=True)
        except Exception:
            pass
        return data
    except Exception:
        return []

# ---- 거래일(UTC) 종가 조회 (페이지네이션 포함) ----
@st.cache_data(show_spinner=False, ttl=3600)
def get_okx_daily_close_for_date(inst_id: str, utc_day: date) -> Tuple[Optional[float], str]:
    """
    특정 UTC 일자(00:00~23:59:59)의 1D 종가를 OKX에서 조회.
    1) /market/candles(최근)에서 먼저 찾고
    2) 없으면 /market/history-candles 를 before 커서로 여러 페이지 내려가며 찾는다.
    - 개선: 선택 캔들의 ts로 in-range 판정, ±2일 허용, 조기 종료 완화
    """
    iid = norm_inst_id(inst_id)
    if not iid:
        return None, "empty_symbol"
    if iid == "USDT":
        return 1.0, "stablecoin"

    # 타깃 일자 범위(ms)
    day_start = datetime(utc_day.year, utc_day.month, utc_day.day, tzinfo=timezone.utc)
    start_ms = int(day_start.timestamp() * 1000)
    end_ms = start_ms + 86_400_000 - 1
    mid_ms = start_ms + 43_200_000
    NEAREST_TOL = 172_800_000  # ±2 days

    def pick_from_batch(batch: List[List]) -> Tuple[Optional[float], Optional[int], Optional[int], Optional[int]]:
        """
        배치에서 정확 일자 매칭 or 근접값(±2d) 선택
        반환: (price, matched_ts, min_ts, max_ts)
        """
        if not batch:
            return None, None, None, None
        min_ts = None
        max_ts = None
        best = None
        best_dist = 1e18
        matched_ts = None

        for row in batch:
            try:
                ts_ms = int(row[0])
                close_px = float(row[4])
            except Exception:
                continue
            if max_ts is None or ts_ms > max_ts:
                max_ts = ts_ms
            if min_ts is None or ts_ms < min_ts:
                min_ts = ts_ms

            # 정확 매칭
            if start_ms <= ts_ms <= end_ms:
                matched_ts = ts_ms
                return close_px, matched_ts, min_ts, max_ts

            # 근접(±2일) 보정
            dist = abs(ts_ms - mid_ms)
            if dist < best_dist and dist <= NEAREST_TOL:
                best = close_px
                best_dist = dist
                matched_ts = ts_ms

        return best, matched_ts, min_ts, max_ts

    # 1) 최신 구간에서 시도
    recent = _okx_fetch_candles_page(iid, bar="1D", limit=200, use_history=False)
    px, matched_ts, min_ts, max_ts = pick_from_batch(recent)
    in_range = (matched_ts is not None) and (start_ms <= matched_ts <= end_ms)
    if px is not None:
        return (px, "ok" if in_range else "nearest")

    # 2) history로 페이지네이션 (과거 내려가며 탐색)
    anchor = (min_ts - 1) if min_ts else int(datetime.now(timezone.utc).timestamp() * 1000)
    MAX_PAGES = 20  # 20 * 200일 ≈ 4000일(10년+) 커버
    for _ in range(MAX_PAGES):
        batch = _okx_fetch_candles_page(iid, bar="1D", limit=200, before=anchor, use_history=True)
        px, matched_ts, batch_min, batch_max = pick_from_batch(batch)
        in_range_hist = (matched_ts is not None) and (start_ms <= matched_ts <= end_ms)
        if px is not None:
            return (px, "ok" if in_range_hist else "nearest")
        if not batch:
            break
        # 다음 페이지를 위해 더 과거로 anchor 이동
        anchor = (batch_min - 1) if batch_min else (anchor - 86_400_000 * 200)

        # ⛔️ 기존: 타깃-2일보다 과거면 조기 종료 → 제거(최대 페이지 수로만 제한)

    return None, "not_found"

@st.cache_data(show_spinner=False, ttl=3600)
def get_batch_okx_closes(pair_date_pairs: List[Tuple[str, date]]) -> Dict[Tuple[str, date], float]:
    """여러 (instId, UTC일자)의 1D 종가 배치 조회(페이지네이션 포함)."""
    uniq = list({(norm_inst_id(sym), d) for sym, d in pair_date_pairs})
    out: Dict[Tuple[str, date], float] = {}
    if not uniq:
        return out
    with ThreadPoolExecutor(max_workers=6) as ex:
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

# ---- Debug (OKX) ----
@st.cache_data(show_spinner=False, ttl=10)
def debug_fetch_okx_ticker(inst_id: str) -> Dict[str, object]:
    iid = norm_inst_id(inst_id)
    try:
        r = requests.get(f"{OKX_BASE}/api/v5/market/ticker",
                         params={"instId": iid}, timeout=8, verify=certifi.where())
        return {
            "OKX instId": iid, "endpoint": "/market/ticker",
            "ticker_status": r.status_code, "ticker_ok": (r.status_code == 200),
            "ticker_preview": r.text[:300]
        }
    except Exception as e:
        return {
            "OKX instId": iid, "endpoint": "/market/ticker",
            "ticker_status": "EXC", "ticker_ok": False,
            "ticker_preview": str(e)[:300]
        }

def build_current_price_debug_table_okx(inst_ids: List[str]) -> pd.DataFrame:
    ids = sorted(set(norm_inst_id(s) for s in inst_ids if s))
    rows = []
    for s in ids:
        r1 = debug_fetch_okx_ticker(s)
        rows.append(r1)
    return pd.DataFrame(rows)

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
    raw = uploaded.getvalue()
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
    current_inst_ids: List[str] = []

    for _, r in df_raw.iterrows():
        sym = str(r.get("Symbol", ""))
        parsed = parse_symbol(sym)
        base, quote, opt = parsed.get("base", ""), parsed.get("quote", ""), parsed.get("option", "")
        cp = r.get("Counterparty", "")
        ptype = infer_product_type(base, opt, quote, cp, sym, coupon_whitelist, coupon_quote_set, covered_call_whitelist)
        token_type = quote if ("Bonus Coupon" in ptype) else base
        inst_id = norm_inst_id(make_pair_symbol(token_type))

        start_ts = pd.to_datetime(r.get(config.trade_field, pd.NaT), errors="coerce", utc=False)
        trade_ts = start_ts if pd.notna(start_ts) else pd.to_datetime(r.get("Expiry Time", pd.NaT), errors="coerce", utc=False)
        trade_ts = trade_ts if pd.notna(trade_ts) else datetime.now(KST)

        if inst_id:
            if inst_id != "USDT":
                pair_ts_pairs.append((inst_id, trade_ts))
            current_inst_ids.append(inst_id)

    # 2) 과거(거래일) 종가 배치 (OKX)
    if progress_placeholder:
        progress_placeholder.info(f"📊 가격 데이터 조회 중... (고유 조합 {len(set(pair_ts_pairs))}개)")
    pair_date_pairs = [
        (p, resolve_trade_utc_date(ts, config.naive_ts_timezone))
        for (p, ts) in pair_ts_pairs
        if p and p != "USDT" and pd.notna(pd.to_datetime(ts, errors="coerce"))
    ]
    trade_close_map = get_batch_okx_closes(pair_date_pairs)

    # 3) 현재가 배치 (OKX)
    current_price_map = get_batch_current_prices_okx(current_inst_ids)

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
        inst_id = norm_inst_id(make_pair_symbol(token_type))

        # 거래일 UTC 일자
        key_date = resolve_trade_utc_date(
            start_ts if pd.notna(start_ts) else (expiry_ts if pd.notna(expiry_ts) else datetime.now(KST)),
            config.naive_ts_timezone
        )

        # 가격 계산
        trade_date_px = 1.0 if inst_id == "USDT" else trade_close_map.get((inst_id, key_date))
        qty_usd_trade = (float(qty_num) * float(trade_date_px)) if (pd.notna(qty_num) and (trade_date_px is not None)) else None
        cur_px = current_price_map.get(inst_id, None)
        qty_usd_cur = (float(qty_num) * float(cur_px)) if (pd.notna(qty_num) and cur_px is not None) else None
        qxm = (float(qty_usd_trade) * float(month_diff)) if (qty_usd_trade is not None and month_diff is not None) else None

        exp_str_from_iso = extract_iso_date_to_str(r.get("Expiry Time", "")) or yyyymmdd_to_mdy_str(parsed.get("expiry", ""))
        exp_date = extract_iso_date_to_date(r.get("Expiry Time", "")) or yyyymmdd_to_date(parsed.get("expiry", ""))
        trade_date_str = extract_iso_date_to_str(r.get(config.trade_field, ""))

        rows.append({
            "Counterparty": cp,
            "Product Type": ptype,
            "Token Type": token_type,
            "API Symbol": inst_id,
            "Token Amount": qty_raw,
            "Qty": qty_raw,
            "Current Price (USD)": cur_px,
            "Trade Date Price (USD, OKX)": trade_date_px,
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
    desired_order = [
        "Counterparty", "Product Type", "Token Type", "API Symbol",
        "Token Amount", "Qty",
        "Current Price (USD)", "Trade Date Price (USD, OKX)", "Qty USD (Current)",
        "Month Difference", "Qty * Month (USD)",
        "Trade Date", "Expiry Date",
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
    year_mask = out_with_filters["_trade_date_obj"].notna() & (p_]()_
