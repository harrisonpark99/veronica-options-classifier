# -*- coding: utf-8 -*-
"""
VERONICA - Data Processing Module
Options Classifier data processing utilities.
"""

import re
import io
from datetime import datetime, date, timezone
from typing import Dict, List, Optional, Set

import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

from .config import KST, AppConfig
from .okx_api import (
    norm_inst_id, make_pair_symbol,
    get_batch_okx_closes, get_batch_current_prices_okx
)

# ================== Constants & Regex ==================
SYMBOL_REGEX = re.compile(
    r'^(?P<issuer>[^.]+)\.(?P<style>[A-Z_]+)\.(?P<base>[A-Z0-9]+)-(?P<quote>[A-Z0-9]+)\.(?P<expiry>\d{8})\.(?P<option>[CP])\.(?P<series>\d+)$'
)
DATE_YMD_RE = re.compile(r'(\d{4})-(\d{2})-(\d{2})')


# ================== Utilities ==================
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


# ================== Classification Logic ==================
def infer_product_type(
    base: str, option: str, quote: str, counterparty: str, symbol: str,
    coupon_whitelist: Set[str], coupon_quote_set: Set[str],
    covered_call_whitelist: Set[str]
) -> str:
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


# ================== Aggregation Functions ==================
def aggregate_by_product_type(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Product Type", "Token Amount Sum"])
    tmp = df.copy()
    tmp["Token Amount Num"] = (
        tmp["Token Amount"].astype(str).str.replace(",", "", regex=False).str.replace(" ", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )
    agg = (
        tmp.groupby("Product Type", dropna=False)["Token Amount Num"]
        .sum().reset_index()
        .rename(columns={"Token Amount Num": "Token Amount Sum"})
        .sort_values("Token Amount Sum", ascending=False)
    )
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
    tmp["QxM_USD_Num"] = (
        tmp[col].astype(str).str.replace(",", "", regex=False).str.replace(" ", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )
    agg = (
        tmp.groupby("Counterparty", dropna=False)["QxM_USD_Num"]
        .sum().reset_index()
        .rename(columns={"QxM_USD_Num": "Qty * Month (USD) Sum"})
        .sort_values("Qty * Month (USD) Sum", ascending=False)
    )
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
                df = pd.read_csv(
                    bio, encoding=enc, sep=sep,
                    engine="python" if sep is None else None,
                    low_memory=False
                )
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
    pair_ts_pairs: List[tuple] = []
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
        progress_placeholder.info(f"가격 데이터 조회 중... (고유 조합 {len(set(pair_ts_pairs))}개)")
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
        pbar = progress_placeholder.progress(0)
        ptxt = progress_placeholder.empty()
    total = len(df_raw) or 1

    for i, (_, r) in enumerate(df_raw.iterrows(), 1):
        if progress_placeholder:
            pbar.progress(i/total)
            ptxt.text(f"처리 중... {i}/{total}")

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
        pbar.empty()
        ptxt.empty()

    out = pd.DataFrame(rows)

    # Display trimming & column order
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
    y1, m1 = month_offset(y, m, 1)
    y2, m2 = month_offset(y, m, 2)
    y3, m3 = month_offset(y, m, 3)

    def filter_by_month(df_in: pd.DataFrame, yy: int, mm: int):
        if df_in.empty:
            return df_in
        cond = df_in["_expiry_date_obj"].notna() & (df_in["_expiry_date_obj"].dt.year == yy) & (df_in["_expiry_date_obj"].dt.month == mm)
        return df_in[cond].copy()

    m1_df = filter_by_month(out, y1, m1)
    m2_df = filter_by_month(out, y2, m2)
    m3_df = filter_by_month(out, y3, m3)

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
    unique_syms = sorted(set(
        norm_inst_id(s) for s in full_display["API Symbol"].dropna().astype(str).replace("", pd.NA).dropna().tolist()
    )) if isinstance(full_display, pd.DataFrame) and not full_display.empty else []
    current_price_table = pd.DataFrame([{"API Symbol": s, "Current Price (USD)": current_price_map.get(s)} for s in unique_syms])

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
        "msg": "완료! OKX 종가/현재가 반영",
        "today_info": f"오늘(Asia/Seoul): {today.isoformat()}"
    }
