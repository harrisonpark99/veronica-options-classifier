# -*- coding: utf-8 -*-
"""VERONICA Authentication Module"""

import os
import streamlit as st


def get_app_password() -> str:
    """Get the app password from secrets or environment variables."""
    return st.secrets.get("APP_PASSWORD", os.environ.get("APP_PASSWORD", ""))


def check_auth() -> bool:
    """Check if user is authenticated."""
    return st.session_state.get("auth_ok", False)


def require_auth():
    """
    Require authentication to access the page.
    If not authenticated, shows login form and stops execution.
    """
    if check_auth():
        return True

    st.warning("이 페이지에 접근하려면 먼저 로그인이 필요합니다.")
    st.info("홈 페이지에서 로그인해주세요.")

    # Show login form on the current page as well
    with st.expander("여기서 로그인", expanded=True):
        pw = st.text_input("Password", type="password", placeholder="패스워드 입력", key="page_auth_pw")
        if st.button("로그인", key="page_auth_btn"):
            app_password = get_app_password()
            if pw == app_password and app_password:
                st.session_state.auth_ok = True
                st.rerun()
            else:
                st.error("패스워드가 올바르지 않습니다.")

    st.stop()
    return False


def logout():
    """Log out the current user."""
    st.session_state.auth_ok = False
    # Clear other session data
    for key in ["df_raw", "file_hash", "last_result", "last_keys"]:
        st.session_state.pop(key, None)


def show_logout_button():
    """Show logout button in sidebar."""
    if check_auth():
        if st.sidebar.button("로그아웃", use_container_width=True):
            logout()
            st.rerun()
