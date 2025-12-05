# -*- coding: utf-8 -*-
"""
VERONICA - Configuration Module
"""

import os
from dataclasses import dataclass, asdict
from zoneinfo import ZoneInfo

import certifi
import streamlit as st

# TLS 인증서 경로 자동 설정 (requests가 신뢰 루트 못 찾는 환경 대응)
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

# Timezone constants
KST = ZoneInfo("Asia/Seoul")
OKX_BASE = "https://www.okx.com"


@dataclass
class AppConfig:
    """Options Classifier configuration."""
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


@dataclass
class ColdCallConfig:
    """Cold Call Assistant configuration (placeholder for future use)."""
    # Add configuration fields as needed
    pass

    def save_to_session(self):
        st.session_state.cold_call_config = asdict(self)

    @classmethod
    def load_from_session(cls):
        return cls(**st.session_state.get("cold_call_config", asdict(cls())))
