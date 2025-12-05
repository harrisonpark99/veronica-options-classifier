# -*- coding: utf-8 -*-
"""
VERONICA - Option Classifier Page
CSV 옵션 딜 분류 및 집계
"""

import os
import hashlib
import difflib
from typing import List, Set
from datetime import datetime, date

import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

import pandas as pd
import streamlit as st

# Import utilities
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.auth import require_auth, show_logout_button
from utils.common import (
    KST, AppConfig, parse_symbol, extract_iso_date_to_str, extract_iso_date_to_date,
    yyyymmdd_to_mdy_str, yyyymmdd_to_date, normalize_party_list, normalize_quote_list,
    calculate_month_difference, make_pair_symbol, resolve_trade_utc_date,
    read_csv_safely, to_csv_bytes, norm_inst_id
)
from utils.okx_api import get_batch_current_prices_okx, get_batch_okx_closes

# ================== Page Config ==================
st.set_page_config(
    page_title="Option Classifier - VERONICA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== Auth Check ==================
require_auth()

# ================== Business Logic ==================
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


def classify_core(df_raw: pd.DataFrame, config: AppConfig, progress_placeholder=None) -> dict:
    coupon_whitelist = normalize_party_list(config.coupon_parties)
    coupon_quote_set = normalize_quote_list(config.coupon_quotes)
    covered_call_whitelist = normalize_party_list(config.covered_call_parties)
    today = datetime.now(KST).date()

    # 1) 심볼/타임스탬프 수집
    pair_ts_pairs = []
    current_inst_ids = []

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
            pbar.progress(i / total)
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

    # Column order
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
        return y + (nm - 1) // 12, ((nm - 1) % 12) + 1

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
        "current_prices": current_price_table,
        "msg": "완료! 분류 및 집계가 완료되었습니다.",
        "today_info": f"오늘(Asia/Seoul): {today.isoformat()}"
    }


# ================== UI ==================
st.title("📊 Option Classifier")
st.caption("CSV 옵션 딜 분류 및 집계")

# Sidebar
with st.sidebar:
    st.header("VERONICA")
    show_logout_button()

    st.markdown("---")
    st.header("설정")
    config = AppConfig.load_from_session()

    # 캐시 초기화 버튼
    if st.button("캐시/세션 리셋", use_container_width=True):
        st.cache_data.clear()
        for k in ["df_raw", "file_hash", "last_result", "last_keys"]:
            st.session_state.pop(k, None)
        st.success("캐시/세션이 초기화되었습니다.")

    uploaded = st.file_uploader("CSV 업로드", type=["csv"])
    if uploaded is not None:
        try:
            raw = uploaded.getvalue()
            file_hash = hashlib.md5(raw).hexdigest()
            if st.session_state.get("file_hash") != file_hash:
                with st.spinner("CSV 파일 로드 중..."):
                    df_raw = read_csv_safely(uploaded)
                    required = ['Symbol', 'Counterparty', 'Qty']
                    missing_required = [c for c in required if c not in df_raw.columns]
                    if missing_required:
                        st.error(f"필수 컬럼 누락: {', '.join(missing_required)}")
                        st.stop()
                    optional = ['Expiry Time', 'Created Time', 'Initiation Time']
                    missing_optional = [c for c in optional if c not in df_raw.columns]
                    if missing_optional:
                        st.warning(f"선택적 컬럼 누락: {', '.join(missing_optional)}")
                    st.success("데이터 검증 완료")
                    st.session_state.df_raw = df_raw
                    st.session_state.file_hash = file_hash
        except Exception as e:
            st.error(f"CSV 로드 실패: {e}")
            st.stop()

    config.trade_field = st.radio("Trade Date 기준", ["Created Time", "Initiation Time"], index=0 if config.trade_field == "Created Time" else 1)
    config.naive_ts_timezone = st.radio("tz-naive 타임스탬프 가정", ["KST", "UTC"], index=0 if config.naive_ts_timezone.upper() == "KST" else 1)

    st.markdown("---")
    st.subheader("분류 설정")
    config.coupon_parties = st.text_input("Bonus Coupon 카운터파티", config.coupon_parties)
    config.coupon_quotes = st.text_input("Bonus Coupon 결제통화", config.coupon_quotes)
    config.covered_call_parties = st.text_input("Covered Call 카운터파티", config.covered_call_parties)

    st.markdown("---")
    st.subheader("필터")
    col1, col2 = st.columns(2)
    with col1:
        config.exclude_mm = st.checkbox("MM 제외", config.exclude_mm)
    with col2:
        config.exclude_unknown = st.checkbox("Unknown 제외", config.exclude_unknown)

    # Counterparty 자동완성
    if "df_raw" in st.session_state and "Counterparty" in st.session_state.df_raw.columns:
        vals = (st.session_state.df_raw["Counterparty"].dropna().astype(str).map(lambda s: s.strip()).replace("", pd.NA).dropna().unique().tolist())
        st.session_state.cp_catalog = sorted(set(vals), key=lambda s: s.lower())

    cp_search = st.text_input("Counterparty 검색", placeholder="입력하면서 자동완성...")

    def suggest(q: str, catalog: List[str], limit: int = 10) -> List[str]:
        if not catalog:
            return []
        if not q:
            return catalog[:limit]
        ql = q.lower()
        partial = [c for c in catalog if ql in c.lower()]
        fuzzy = difflib.get_close_matches(q, catalog, n=limit, cutoff=0.5)
        out, seen = [], set()
        for it in partial + fuzzy:
            if it not in seen:
                out.append(it)
                seen.add(it)
            if len(out) >= limit:
                break
        return out

    suggestions = suggest(cp_search, st.session_state.get("cp_catalog", []), 10)
    if suggestions:
        st.markdown("**추천 목록:**")
        for i, sopt in enumerate(suggestions[:5]):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.text(sopt)
            with c2:
                if st.button("➕", key=f"add_{i}", use_container_width=True):
                    cur = [s.strip() for s in config.covered_call_parties.split(",") if s.strip()]
                    if sopt not in cur:
                        cur.append(sopt)
                        config.covered_call_parties = ", ".join(cur)
                        config.save_to_session()
                        st.rerun()

    if cp_search:
        if st.button(f"'{cp_search}' 직접 추가", use_container_width=True):
            cur = [s.strip() for s in config.covered_call_parties.split(",") if s.strip()]
            if cp_search not in cur:
                cur.append(cp_search)
                config.covered_call_parties = ", ".join(cur)
                config.save_to_session()
                st.rerun()

    config.save_to_session()

# Main
st.caption("업로드 후 좌측 필터를 조정하면 아래 표/요약이 갱신됩니다.")

if "df_raw" not in st.session_state or st.session_state.get("df_raw") is None:
    st.info("좌측 사이드바에서 CSV를 업로드하세요.")
    st.stop()

df_raw = st.session_state.df_raw
with st.expander("원본 데이터 미리보기", expanded=False):
    st.dataframe(df_raw.head(50), use_container_width=True)
    mem_bytes = df_raw.memory_usage(deep=True).sum() if hasattr(df_raw, 'memory_usage') else 0
    st.caption(f"rows={len(df_raw):,}, memory~{mem_bytes / 1_048_576:,.2f} MB")

# 실행 버튼
run_col1, run_col2 = st.columns([1, 3])
with run_col1:
    run_clicked = st.button("분류 실행 / 새로고침", type="primary")
with run_col2:
    auto_run = st.toggle("업로드 시 자동 실행", value=st.session_state.get("auto_run", True))
    st.session_state.auto_run = auto_run


def _hash_config(cfg: AppConfig) -> str:
    try:
        payload = f"{cfg.trade_field}|{cfg.coupon_parties}|{cfg.coupon_quotes}|{cfg.covered_call_parties}|{cfg.exclude_mm}|{cfg.exclude_unknown}|{cfg.naive_ts_timezone}"
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
st.caption(result.get("today_info", ""))

# Tabs
tab_all, tab_nonexp, tab_m1, tab_m2, tab_m3, tab_summary, tab_px = st.tabs([
    "전체", "미만기", "M+1", "M+2", "M+3", "요약(합계)", "현재가 스냅샷"
])


def table_with_download(df, label: str, key: str):
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        st.info("데이터가 없습니다.")
        return
    st.dataframe(df, use_container_width=True)
    st.download_button("CSV 다운로드", data=to_csv_bytes(df), file_name=f"{label}.csv", mime="text/csv", key=f"dl_{key}")


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
        c1.metric("전체 건수", f"{total_rows:,}")
        c2.metric("미만기 건수", f"{nonexp_rows:,}")
    except Exception:
        pass
    st.markdown("**미만기 Product Type별 Token Amount 합계**")
    table_with_download(result["agg_nonexp"], "agg_nonexpired_by_product_type", "agg_nonexp")
    st.markdown("**M+1 요약**")
    table_with_download(result["agg_m1"], "agg_m1_by_product_type", "agg_m1")
    st.markdown("**M+2 요약**")
    table_with_download(result["agg_m2"], "agg_m2_by_product_type", "agg_m2")
    st.markdown("**M+3 요약**")
    table_with_download(result["agg_m3"], "agg_m3_by_product_type", "agg_m3")

with tab_px:
    st.subheader("API Symbol 현재가 스냅샷 (OKX)")
    table_with_download(result["current_prices"], "current_prices_snapshot_okx", "px")

st.caption("열 순서 변경/숨김은 표 우측 상단 메뉴에서 조정 가능합니다.")
