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

        col_f1, col_f2 = st.columns(2)

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

    # Main content
    st.markdown('<h1 class="dashboard-title">VERONICA Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="welcome-text">환영합니다! 좌측 사이드바 또는 아래 카드에서 원하는 기능을 선택하세요.</p>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

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
