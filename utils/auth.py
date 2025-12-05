# -*- coding: utf-8 -*-
"""
VERONICA - Authentication Module
"""

import os
import streamlit as st


def get_app_password() -> str:
    """Get app password from secrets or environment."""
    return st.secrets.get("APP_PASSWORD", os.environ.get("APP_PASSWORD", ""))


def check_auth() -> bool:
    """
    Check if user is authenticated.
    Returns True if authenticated, False otherwise.
    """
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False
    return st.session_state.auth_ok


def show_login_page():
    """Display the login page and handle authentication."""
    APP_PASSWORD = get_app_password()

    st.title("VERONICA")
    st.caption("인하우스 워크플로우 도구")
    st.markdown("---")
    st.subheader("로그인")

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


def logout():
    """Logout the user."""
    st.session_state.auth_ok = False
    # Clear other session data
    for key in list(st.session_state.keys()):
        if key != "auth_ok":
            del st.session_state[key]


def require_auth():
    """
    Decorator-style function to require authentication.
    Call this at the start of each page that requires auth.
    Returns True if authenticated, stops execution otherwise.
    """
    if not check_auth():
        show_login_page()
        st.stop()
    return True
