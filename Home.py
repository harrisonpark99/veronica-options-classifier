# -*- coding: utf-8 -*-
"""
VERONICA - In-house Workflow Tools
Main entry point / Dashboard
"""

import streamlit as st

# Page config must be first Streamlit command
st.set_page_config(
    page_title="VERONICA",
    page_icon="V",
    layout="wide",
    initial_sidebar_state="expanded"
)

from utils.auth import require_auth, logout

# Require authentication
require_auth()

# Main content
st.title("VERONICA")
st.caption("인하우스 워크플로우 도구")

st.markdown("---")

# Feature cards
st.subheader("기능 선택")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Options Classifier

    CSV 옵션 딜 분류 및 분석 도구

    - OKX 실시간/과거 가격 조회
    - 상품 유형별 자동 분류
    - 만기별 필터링 (미만기, M+1~M+3)
    - Counterparty별 집계
    """)
    if st.button("Options Classifier 열기", type="primary", use_container_width=True):
        st.switch_page("pages/1_Options_Classifier.py")

with col2:
    st.markdown("""
    ### Cold Call Assistant

    콜드콜 업무 지원 도구

    - (준비 중)
    """)
    if st.button("Cold Call Assistant 열기", type="secondary", use_container_width=True, disabled=False):
        st.switch_page("pages/2_Cold_Call_Assistant.py")

st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("VERONICA")
    st.caption("v2.0 - Multipage Edition")

    st.markdown("---")

    if st.button("로그아웃", use_container_width=True):
        logout()
        st.rerun()

    st.markdown("---")
    st.caption("Copyright 2024. All rights reserved.")
