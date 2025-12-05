# -*- coding: utf-8 -*-
"""
VERONICA - Cold Call Assistant Page
콜드콜 업무 지원 도구
"""

import streamlit as st

# Page config
st.set_page_config(
    page_title="Cold Call Assistant - VERONICA",
    page_icon="V",
    layout="wide",
    initial_sidebar_state="expanded"
)

from utils.auth import require_auth

# Require authentication
require_auth()

# Title
st.title("Cold Call Assistant")
st.caption("콜드콜 업무 지원 도구")

st.markdown("---")

# Placeholder content
st.info("이 기능은 현재 개발 중입니다.")

st.markdown("""
### 예정 기능

이 페이지에서 구현할 기능들:

- [ ] 콜드콜 대상 리스트 관리
- [ ] 통화 스크립트 템플릿
- [ ] 통화 기록 및 메모
- [ ] 후속 조치 스케줄링
- [ ] 성과 분석 대시보드

---

**개발 요청사항이 있으시면 알려주세요!**
""")

# Sidebar
with st.sidebar:
    st.header("Cold Call Assistant")
    st.caption("(개발 중)")

    st.markdown("---")

    if st.button("← 홈으로", use_container_width=True):
        st.switch_page("Home.py")
