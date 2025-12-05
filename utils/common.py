# -*- coding: utf-8 -*-
"""VERONICA Common Utilities"""

import os
import re
import io
from dataclasses import dataclass, asdict
from typing import Dict, Set, Optional
from datetime import datetime, date, timezone

import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

import pandas as pd
import streamlit as st
from dateutil.relativedelta import relativedelta
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")

# ================== Regex Patterns ==================
SYMBOL_REGEX = re.compile(
    r'^(?P<issuer>[^.]+)\.(?P<style>[A-Z_]+)\.(?P<base>[A-Z0-9]+)-(?P<quote>[A-Z0-9]+)\.(?P<expiry>\d{8})\.(?P<option>[CP])\.(?P<series>\d+)$'
)
DATE_YMD_RE = re.compile(r'(\d{4})-(\d{2})-(\d{2})')

TOKEN_ALIASES = {
    "WETH": "ETH",
    "BCH.SV": "BSV",
    "BCHSV": "BSV",
}


# ================== App Config ==================
@dataclass
class AppConfig:
    trade_field: str = "Created Time"  # or "Initiation Time"
    coupon_parties: str = "Emporio Holdings, Novaflow Labs Ltd."
    coupon_quotes: str = "USDT, USD"
    covered_call_parties: str = "Marblex, Near_Siftung, Layer3_Operations"
    exclude_mm: bool = False
    exclude_unknown: bool = False
    naive_ts_timezone: str = "KST"

    def save_to_session(self):
        st.session_state.config = asdict(self)

    @classmethod
    def load_from_session(cls):
        return cls(**st.session_state.get("config", asdict(cls())))


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
    y = int(yyyymmdd[:4])
    m = int(yyyymmdd[4:6])
    d = int(yyyymmdd[6:8])
    return f"{m}/{d}/{y}"


def yyyymmdd_to_date(yyyymmdd: str) -> Optional[date]:
    if not isinstance(yyyymmdd, str) or len(yyyymmdd) != 8:
        return None
    y = int(yyyymmdd[:4])
    m = int(yyyymmdd[4:6])
    d = int(yyyymmdd[6:8])
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
