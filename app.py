# -*- coding: utf-8 -*-
"""
VERONICA - Main Entry Point
Multi-page Streamlit Application with Authentication
"""

import os
import streamlit as st

# TLS 인증서 경로 자동 설정
import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

# ================== Page Config ==================
st.set_page_config(
    page_title="VERONICA",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== Auth ==================
APP_PASSWORD = st.secrets.get("APP_PASSWORD", os.environ.get("APP_PASSWORD", ""))

if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False


def show_login_page():
    """Display the login page."""
    st.markdown(
        """
        <style>
        .main-title {
            text-align: center;
            font-size: 4rem;
            font-weight: bold;
            color: #1E88E5;
            margin-bottom: 0.5rem;
        }
        .sub-title {
            text-align: center;
            font-size: 1.2rem;
            color: #666;
            margin-bottom: 2rem;
        }
        .feature-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<h1 class="main-title">VERONICA</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Virtual Enhanced Research & Operations Network for Institutional Crypto Analytics</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("---")
        st.markdown("### 로그인")

        pw = st.text_input(
            "Password",
            type="password",
            placeholder="패스워드를 입력하세요",
            key="main_password"
        )

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("로그인", type="primary", use_container_width=True):
                if pw == APP_PASSWORD and APP_PASSWORD:
                    st.session_state.auth_ok = True
                    st.rerun()
                else:
                    st.error("패스워드가 올바르지 않거나 설정되어 있지 않습니다.")

        st.markdown("---")

        # Feature Overview
        st.markdown("### 제공 기능")

        col_f1, col_f2, col_f3 = st.columns(3)
        col_f4, col_f5, col_f6 = st.columns(3)

        with col_f1:
            st.markdown(
                """
                **Option Classifier**
                - CSV 옵션 딜 분류
                - 상품 유형 자동 분류
                - 만기별 집계 및 필터링
                - 다운로드 지원
                """
            )

        with col_f2:
            st.markdown(
                """
                **Xunke Support**
                - OKX 실시간 가격 조회
                - 거래일 종가 조회
                - Qty * Month (USD) 계산
                - 디버그 도구
                """
            )

        with col_f3:
            st.markdown(
                """
                **Custom Candle Returns**
                - OKX 일봉 데이터
                - 커스텀 캔들 수익률
                - 기간별 차트 시각화
                - 통계 요약 및 다운로드
                """
            )

        with col_f4:
            st.markdown(
                """
                **Strategy PnL Summary**
                - 전략별 DTD/MTD/YTD PnL
                - 카테고리별 집계
                - Top Movers 분석
                - CSV 다운로드
                """
            )

        with col_f5:
            st.markdown(
                """
                **Create Invoice**
                - 체결내역 CSV 업로드
                - 다중 거래소 지원
                - Invoice 템플릿 자동 생성
                - XLSX 다운로드
                """
            )

        with col_f6:
            st.markdown(
                """
                **Ledger Tool**
                - 클라이언트 렛저 관리
                - Coupon / Reclass / AES Trade
                - Undo (Reversal) 지원
                - Excel / PDF 내보내기
                """
            )

        st.markdown("---")
        st.caption("문의: 관리자에게 연락하세요")


def show_dashboard():
    """Display the main dashboard after login."""
    st.markdown(
        """
        <style>
        .dashboard-title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: bold;
            color: #1E88E5;
        }
        .welcome-text {
            text-align: center;
            font-size: 1.1rem;
            color: #666;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Sidebar
    with st.sidebar:
        st.markdown("### VERONICA")
        st.markdown("---")

        if st.button("로그아웃", use_container_width=True):
            st.session_state.auth_ok = False
            for key in ["df_raw", "file_hash", "last_result", "last_keys"]:
                st.session_state.pop(key, None)
            st.rerun()

        st.markdown("---")
        st.markdown("**페이지 바로가기**")
        st.page_link("pages/1_Option_Classifier.py", label="Option Classifier", icon="📊")
        st.page_link("pages/2_Xunke_Support.py", label="Xunke Support", icon="💹")
        st.page_link("pages/3_Custom_Candle_Returns.py", label="Custom Candle Returns", icon="📈")
        st.page_link("pages/4_Strategy_PnL_Summary.py", label="Strategy PnL Summary", icon="💰")
        st.page_link("pages/5_Create_Invoice.py", label="Create Invoice", icon="📄")
        st.page_link("pages/6_Ledger_Tool.py", label="Ledger Tool", icon="📒")

    # Main content
    st.markdown('<h1 class="dashboard-title">VERONICA Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="welcome-text">환영합니다! 좌측 사이드바 또는 아래 카드에서 원하는 기능을 선택하세요.</p>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    col5, col6 = st.columns(2)

    with col1:
        st.markdown("### 📊 Option Classifier")
        st.markdown(
            """
            CSV 파일을 업로드하여 옵션 딜을 자동으로 분류하고 집계합니다.

            **주요 기능:**
            - 상품 유형 자동 분류 (Put, Call, Bonus Coupon 등)
            - 만기별 필터링 (미만기, M+1, M+2, M+3)
            - Token Amount 합계 계산
            - CSV 다운로드
            """
        )
        if st.button("Option Classifier 열기", key="open_classifier", use_container_width=True):
            st.switch_page("pages/1_Option_Classifier.py")

    with col2:
        st.markdown("### 💹 Xunke Support")
        st.markdown(
            """
            OKX API를 통해 실시간 가격과 거래일 종가를 조회합니다.

            **주요 기능:**
            - OKX 실시간 현재가 조회
            - 거래일 종가 (1D 캔들) 조회
            - Qty * Month (USD) 계산
            - API 디버그 도구
            """
        )
        if st.button("Xunke Support 열기", key="open_xunke", use_container_width=True):
            st.switch_page("pages/2_Xunke_Support.py")

    with col3:
        st.markdown("### 📈 Custom Candle Returns")
        st.markdown(
            """
            OKX API를 통해 커스텀 캔들 수익률을 시각화합니다.

            **주요 기능:**
            - 커스텀 캔들 사이즈 설정 (1~90일)
            - 기간별 수익률 차트
            - 통계 요약 (평균, 최대, 최소)
            - CSV 다운로드
            """
        )
        if st.button("Custom Candle Returns 열기", key="open_candle", use_container_width=True):
            st.switch_page("pages/3_Custom_Candle_Returns.py")

    with col4:
        st.markdown("### 💰 Strategy PnL Summary")
        st.markdown(
            """
            전략별 PnL을 DTD/MTD/YTD 기준으로 분석합니다.

            **주요 기능:**
            - 전략별 DTD/MTD/YTD PnL 조회
            - 카테고리별 집계 및 Top Movers
            - 날짜별 조회 및 필터링
            - CSV 다운로드
            """
        )
        if st.button("Strategy PnL Summary 열기", key="open_strategy_pnl", use_container_width=True):
            st.switch_page("pages/4_Strategy_PnL_Summary.py")

    with col5:
        st.markdown("### 📄 Create Invoice")
        st.markdown(
            """
            거래소 체결내역 CSV를 Invoice 템플릿으로 변환합니다.

            **주요 기능:**
            - 다중 거래소 지원 (Binance, OKX, Bybit 등)
            - 거래소 자동 감지
            - Invoice 템플릿 자동 생성
            - XLSX 다운로드
            """
        )
        if st.button("Create Invoice 열기", key="open_create_invoice", use_container_width=True):
            st.switch_page("pages/5_Create_Invoice.py")

    with col6:
        st.markdown("### 📒 Ledger Tool")
        st.markdown(
            """
            클라이언트 렛저를 관리하고 추적합니다.

            **주요 기능:**
            - Coupon Received / Reclass / AES Trade 기록
            - Asset × Bucket 요약 대시보드
            - Undo (Reversal) 기능
            - Excel / PDF 내보내기
            """
        )
        if st.button("Ledger Tool 열기", key="open_ledger_tool", use_container_width=True):
            st.switch_page("pages/6_Ledger_Tool.py")

    st.markdown("---")

    # Quick Stats
    st.markdown("### 빠른 정보")
    col_s1, col_s2, col_s3 = st.columns(3)

    with col_s1:
        st.metric("데이터 소스", "OKX")

    with col_s2:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        KST = ZoneInfo("Asia/Seoul")
        st.metric("현재 시간 (KST)", datetime.now(KST).strftime("%Y-%m-%d %H:%M"))

    with col_s3:
        st.metric("상태", "정상 운영")


# ================== Main ==================
if not st.session_state.auth_ok:
    show_login_page()
else:
    show_dashboard()
