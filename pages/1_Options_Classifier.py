# -*- coding: utf-8 -*-
"""
VERONICA - Options Classifier Page
CSV 옵션 딜 분류 및 분석 도구
"""

import hashlib
import difflib
from datetime import datetime
from typing import List, Optional

import pandas as pd
import streamlit as st

# Page config
st.set_page_config(
    page_title="Options Classifier - VERONICA",
    page_icon="V",
    layout="wide",
    initial_sidebar_state="expanded"
)

from utils.auth import require_auth
from utils.config import AppConfig, KST
from utils.data_processing import (
    read_csv_safely, to_csv_bytes, classify_core
)
from utils.okx_api import (
    build_current_price_debug_table_okx, debug_fetch_okx_ticker
)

# Require authentication
require_auth()

# Title
st.title("Options Classifier")
st.caption("OKX 단일 소스로 가격 조회 · 모듈화 · 내부 접근 보호")


# ================== Sidebar ==================
with st.sidebar:
    st.header("설정")
    config = AppConfig.load_from_session()

    # 캐시 초기화 버튼
    if st.button("캐시/세션 리셋", use_container_width=True):
        st.cache_data.clear()
        for k in ["df_raw", "file_hash", "last_result", "last_keys"]:
            st.session_state.pop(k, None)
        st.success("캐시/세션이 초기화되었습니다. CSV를 다시 업로드하세요.")

    uploaded = st.file_uploader("CSV 업로드", type=["csv"])
    if uploaded is not None:
        try:
            raw = uploaded.getvalue()
            file_hash = hashlib.md5(raw).hexdigest()
            if st.session_state.get("file_hash") != file_hash:
                with st.spinner("CSV 파일 로드 중..."):
                    df_raw = read_csv_safely(uploaded)
                    # Validate
                    required = ['Symbol', 'Counterparty', 'Qty']
                    missing_required = [c for c in required if c not in df_raw.columns]
                    if missing_required:
                        st.error(f"필수 컬럼 누락: {', '.join(missing_required)}")
                        st.stop()
                    # Optional warn
                    optional = ['Expiry Time', 'Created Time', 'Initiation Time']
                    missing_optional = [c for c in optional if c not in df_raw.columns]
                    if missing_optional:
                        st.warning(f"선택적 컬럼 누락: {', '.join(missing_optional)} - 일부 기능 제한될 수 있음")
                    st.success("데이터 검증 완료")
                    st.session_state.df_raw = df_raw
                    st.session_state.file_hash = file_hash
        except Exception as e:
            st.error(f"CSV 로드 실패: {e}")
            st.stop()

    config.trade_field = st.radio(
        "Trade Date 기준",
        ["Created Time", "Initiation Time"],
        index=0 if config.trade_field == "Created Time" else 1
    )

    config.naive_ts_timezone = st.radio(
        "tz-naive 타임스탬프 가정",
        ["KST", "UTC"],
        index=0 if config.naive_ts_timezone.upper() == "KST" else 1
    )

    st.markdown("---")
    st.subheader("분류 설정")
    config.coupon_parties = st.text_input("Bonus Coupon 카운터파티", config.coupon_parties)
    config.coupon_quotes = st.text_input("Bonus Coupon 결제통화", config.coupon_quotes)
    config.covered_call_parties = st.text_input("Covered Call 카운터파티", config.covered_call_parties)

    st.markdown("---")
    st.subheader("필터 & 디버그")
    col1, col2, col3 = st.columns(3)
    with col1:
        config.exclude_mm = st.checkbox("MM 제외", config.exclude_mm)
    with col2:
        config.exclude_unknown = st.checkbox("Unknown 제외", config.exclude_unknown)
    with col3:
        debug_mode = st.checkbox("디버그 모드", value=st.session_state.get("debug_mode", False))
        st.session_state.debug_mode = debug_mode

    # Counterparty 자동완성
    if "df_raw" in st.session_state and "Counterparty" in st.session_state.df_raw.columns:
        vals = (
            st.session_state.df_raw["Counterparty"]
            .dropna().astype(str)
            .map(lambda s: s.strip())
            .replace("", pd.NA).dropna()
            .unique().tolist()
        )
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
        if st.button(f"➕ '{cp_search}' 직접 추가", use_container_width=True):
            cur = [s.strip() for s in config.covered_call_parties.split(",") if s.strip()]
            if cp_search not in cur:
                cur.append(cp_search)
                config.covered_call_parties = ", ".join(cur)
                config.save_to_session()
                st.rerun()

    # 저장
    config.save_to_session()

    st.markdown("---")
    if st.button("← 홈으로", use_container_width=True):
        st.switch_page("Home.py")


# ================== Main Content ==================
st.caption("업로드 후 좌측 필터를 조정하면 아래 표/요약이 갱신됩니다.")

if "df_raw" not in st.session_state or st.session_state.get("df_raw") is None:
    st.info("좌측 사이드바에서 CSV를 업로드하세요.")
    st.stop()

df_raw = st.session_state.df_raw

with st.expander("원본 데이터 미리보기", expanded=False):
    st.dataframe(df_raw.head(50), use_container_width=True)
    mem_bytes = df_raw.memory_usage(deep=True).sum() if hasattr(df_raw, 'memory_usage') else 0
    st.caption(f"rows={len(df_raw):,}, memory≈{mem_bytes/1_048_576:,.2f} MB")

# 실행 버튼 & 캐시 키
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


need_run = run_clicked or st.session_state.get("last_keys") != (
    st.session_state.get("file_hash"),
    _hash_config(AppConfig.load_from_session())
)
if auto_run:
    need_run = True

if need_run:
    try:
        progress_area = st.empty()
        with st.spinner("분류 중..."):
            result = classify_core(df_raw, AppConfig.load_from_session(), progress_placeholder=progress_area)
        st.session_state.last_result = result
        st.session_state.last_keys = (
            st.session_state.get("file_hash"),
            _hash_config(AppConfig.load_from_session())
        )
    except Exception as e:
        st.error(f"처리 중 오류: {e}")
        st.stop()
else:
    result = st.session_state.get("last_result")

if not result:
    st.stop()

st.success(result.get("msg", "완료"))
st.caption(result.get("today_info", ""))


# ================== Tabs ==================
def table_with_download(df: Optional[pd.DataFrame], label: str, key: str):
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        st.info("데이터가 없습니다.")
        return
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "CSV 다운로드",
        data=to_csv_bytes(df),
        file_name=f"{label}.csv",
        mime="text/csv",
        key=f"dl_{key}"
    )


(tab_all, tab_nonexp, tab_m1, tab_m2, tab_m3, tab_summary, tab_cp, tab_px, tab_debug) = st.tabs([
    "전체", "미만기", "M+1", "M+2", "M+3", "요약(합계)", "Counterparty 합계", "현재가 스냅샷(OKX)", "디버그"
])

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

with tab_cp:
    st.subheader("Counterparty별 Qty * Month (USD) 합계 (올해, MM 제외)")
    table_with_download(result["agg_qty_month_cp"], "agg_qty_month_by_counterparty", "agg_cp")

with tab_px:
    st.subheader("API Symbol 현재가 스냅샷 (OKX)")
    table_with_download(result["current_prices"], "current_prices_snapshot_okx", "px")

with tab_debug:
    st.subheader("현재가/엔드포인트 진단 (OKX)")
    if st.session_state.get("debug_mode", False):
        unique_ids = (
            result["current_prices"]["API Symbol"].dropna().astype(str).unique().tolist()
            if isinstance(result.get("current_prices"), pd.DataFrame) and not result["current_prices"].empty
            else []
        )
        dbg_df = build_current_price_debug_table_okx(unique_ids) if unique_ids else pd.DataFrame()
        table_with_download(dbg_df, "okx_debug", "dbg_okx")
        st.markdown("**개별 심볼 점검**")
        colx, coly = st.columns([2, 1])
        with colx:
            test_sym = st.text_input("OKX instId 입력 (예: BTC-USDT)")
        with coly:
            if st.button("테스트") and test_sym:
                st.write(debug_fetch_okx_ticker(test_sym))
    else:
        st.info("사이드바에서 '디버그 모드'를 켜면 진단 도구가 활성화됩니다.")

st.caption("※ 열 순서 변경/숨김은 표 우측 상단 메뉴에서 조정 가능합니다.")
