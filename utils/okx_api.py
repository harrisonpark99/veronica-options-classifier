# -*- coding: utf-8 -*-
"""VERONICA OKX API Module"""

import os
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date, timezone

import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

import requests
import pandas as pd
import streamlit as st

from .common import norm_inst_id

OKX_BASE = "https://www.okx.com"


# ================== OKX Market API ==================
@st.cache_data(show_spinner=False, ttl=10)
def fetch_okx_tickers_bulk_spot() -> Dict[str, float]:
    """OKX 스팟 전종목 티커 일괄 조회 → {INSTID: last}"""
    out: Dict[str, float] = {}
    try:
        r = requests.get(
            f"{OKX_BASE}/api/v5/market/tickers",
            params={"instType": "SPOT"},
            timeout=10,
            verify=certifi.where(),
        )
        if r.status_code != 200:
            return out
        js = r.json()
        if js.get("code") != "0":
            return out
        for row in js.get("data", []):
            inst = norm_inst_id(row.get("instId", ""))
            last = row.get("last")
            if inst and last not in (None, ""):
                try:
                    out[inst] = float(last)
                except Exception:
                    pass
    except Exception:
        pass
    return out


@st.cache_data(show_spinner=False, ttl=30)
def fetch_okx_ticker_price(inst_id: str) -> Optional[float]:
    """OKX 현재가 단건 — /market/ticker (재시도 포함)"""
    iid = norm_inst_id(inst_id)
    if not iid:
        return None
    if iid == "USDT":
        return 1.0
    params = {"instId": iid}
    for _ in range(2):
        try:
            r = requests.get(f"{OKX_BASE}/api/v5/market/ticker",
                             params=params, timeout=8, verify=certifi.where())
            if r.status_code != 200:
                continue
            js = r.json()
            if js.get("code") == "0" and js.get("data"):
                last = js["data"][0].get("last")
                if last not in (None, ""):
                    return float(last)
        except Exception:
            pass
    return None


@st.cache_data(show_spinner=False, ttl=15)
def get_batch_current_prices_okx(inst_ids: List[str]) -> Dict[str, Optional[float]]:
    """벌크 → 누락분 개별 폴백."""
    uniq = sorted({norm_inst_id(s) for s in inst_ids if s})
    results: Dict[str, Optional[float]] = {iid: (1.0 if iid == "USDT" else None) for iid in uniq}
    if not uniq:
        return {}

    bulk = fetch_okx_tickers_bulk_spot()
    for iid in uniq:
        if iid != "USDT" and iid in bulk:
            results[iid] = bulk[iid]

    missing = [iid for iid in uniq if iid != "USDT" and results[iid] is None]
    if missing:
        with ThreadPoolExecutor(max_workers=4) as ex:
            fut_map = {ex.submit(fetch_okx_ticker_price, iid): iid for iid in missing}
            for fut in as_completed(fut_map):
                iid = fut_map[fut]
                try:
                    px = fut.result()
                    if px is not None:
                        results[iid] = px
                except Exception:
                    pass
    return results


# ---- 캔들 페이징 (공용) ----
@st.cache_data(show_spinner=False, ttl=900)
def _okx_fetch_candles_page(inst_id: str, *, bar: str = "1D",
                            limit: int = 200, before: Optional[int] = None,
                            use_history: bool = False) -> List[List]:
    """
    OKX 1페이지 캔들 조회.
    - use_history=False  → /market/candles (최근 구간)
    - use_history=True   → /market/history-candles (과거 아카이브)
    반환: [[ts_ms, o, h, l, c, ...], ...]  (보통 최신→오래된 순)
    """
    iid = norm_inst_id(inst_id)
    if not iid:
        return []

    endpoint = "/api/v5/market/history-candles" if use_history else "/api/v5/market/candles"
    params = {"instId": iid, "bar": bar, "limit": limit}
    if before is not None:
        params["before"] = int(before)

    try:
        r = requests.get(f"{OKX_BASE}{endpoint}", params=params, timeout=10, verify=certifi.where())
        if r.status_code != 200:
            return []
        js = r.json()
        if js.get("code") != "0":
            return []
        data = js.get("data", [])
        try:
            data.sort(key=lambda k: int(k[0]), reverse=True)
        except Exception:
            pass
        return data
    except Exception:
        return []


# ---- 거래일(UTC) 종가 조회 (페이지네이션 포함) ----
@st.cache_data(show_spinner=False, ttl=3600)
def get_okx_daily_close_for_date(inst_id: str, utc_day: date) -> Tuple[Optional[float], str]:
    """
    특정 UTC 일자(00:00~23:59:59)의 1D 종가를 OKX에서 조회.
    1) /market/candles(최근)에서 먼저 찾고
    2) 없으면 /market/history-candles 를 before 커서로 여러 페이지 내려가며 찾는다.
    - 개선: 선택 캔들의 ts로 in-range 판정, ±2일 허용, 조기 종료 완화
    """
    iid = norm_inst_id(inst_id)
    if not iid:
        return None, "empty_symbol"
    if iid == "USDT":
        return 1.0, "stablecoin"

    # 타깃 일자 범위(ms)
    day_start = datetime(utc_day.year, utc_day.month, utc_day.day, tzinfo=timezone.utc)
    start_ms = int(day_start.timestamp() * 1000)
    end_ms = start_ms + 86_400_000 - 1
    mid_ms = start_ms + 43_200_000
    NEAREST_TOL = 172_800_000  # ±2 days

    def pick_from_batch(batch: List[List]) -> Tuple[Optional[float], Optional[int], Optional[int], Optional[int]]:
        """
        배치에서 정확 일자 매칭 or 근접값(±2d) 선택
        반환: (price, matched_ts, min_ts, max_ts)
        """
        if not batch:
            return None, None, None, None
        min_ts = None
        max_ts = None
        best = None
        best_dist = 1e18
        matched_ts = None

        for row in batch:
            try:
                ts_ms = int(row[0])
                close_px = float(row[4])
            except Exception:
                continue
            if max_ts is None or ts_ms > max_ts:
                max_ts = ts_ms
            if min_ts is None or ts_ms < min_ts:
                min_ts = ts_ms

            # 정확 매칭
            if start_ms <= ts_ms <= end_ms:
                matched_ts = ts_ms
                return close_px, matched_ts, min_ts, max_ts

            # 근접(±2일) 보정
            dist = abs(ts_ms - mid_ms)
            if dist < best_dist and dist <= NEAREST_TOL:
                best = close_px
                best_dist = dist
                matched_ts = ts_ms

        return best, matched_ts, min_ts, max_ts

    # 1) 최신 구간에서 시도
    recent = _okx_fetch_candles_page(iid, bar="1D", limit=200, use_history=False)
    px, matched_ts, min_ts, max_ts = pick_from_batch(recent)
    in_range = (matched_ts is not None) and (start_ms <= matched_ts <= end_ms)
    if px is not None:
        return (px, "ok" if in_range else "nearest")

    # 2) history로 페이지네이션 (과거 내려가며 탐색)
    anchor = (min_ts - 1) if min_ts else int(datetime.now(timezone.utc).timestamp() * 1000)
    MAX_PAGES = 20  # 20 * 200일 ≈ 4000일(10년+) 커버
    for _ in range(MAX_PAGES):
        batch = _okx_fetch_candles_page(iid, bar="1D", limit=200, before=anchor, use_history=True)
        px, matched_ts, batch_min, batch_max = pick_from_batch(batch)
        in_range_hist = (matched_ts is not None) and (start_ms <= matched_ts <= end_ms)
        if px is not None:
            return (px, "ok" if in_range_hist else "nearest")
        if not batch:
            break
        # 다음 페이지를 위해 더 과거로 anchor 이동
        anchor = (batch_min - 1) if batch_min else (anchor - 86_400_000 * 200)

    return None, "not_found"


@st.cache_data(show_spinner=False, ttl=3600)
def get_batch_okx_closes(pair_date_pairs: List[Tuple[str, date]]) -> Dict[Tuple[str, date], float]:
    """여러 (instId, UTC일자)의 1D 종가 배치 조회(페이지네이션 포함)."""
    uniq = list({(norm_inst_id(sym), d) for sym, d in pair_date_pairs})
    out: Dict[Tuple[str, date], float] = {}
    if not uniq:
        return out
    with ThreadPoolExecutor(max_workers=6) as ex:
        fut = {ex.submit(get_okx_daily_close_for_date, sym, d): (sym, d) for sym, d in uniq}
        for f in as_completed(fut):
            key = fut[f]
            try:
                px, _ = f.result()
                if px is not None:
                    out[key] = px
            except Exception:
                pass
    return out


# ---- Debug (OKX) ----
@st.cache_data(show_spinner=False, ttl=10)
def debug_fetch_okx_ticker(inst_id: str) -> Dict[str, object]:
    iid = norm_inst_id(inst_id)
    try:
        r = requests.get(f"{OKX_BASE}/api/v5/market/ticker",
                         params={"instId": iid}, timeout=8, verify=certifi.where())
        return {
            "OKX instId": iid, "endpoint": "/market/ticker",
            "ticker_status": r.status_code, "ticker_ok": (r.status_code == 200),
            "ticker_preview": r.text[:300]
        }
    except Exception as e:
        return {
            "OKX instId": iid, "endpoint": "/market/ticker",
            "ticker_status": "EXC", "ticker_ok": False,
            "ticker_preview": str(e)[:300]
        }


def build_current_price_debug_table_okx(inst_ids: List[str]) -> pd.DataFrame:
    ids = sorted(set(norm_inst_id(s) for s in inst_ids if s))
    rows = []
    for s in ids:
        r1 = debug_fetch_okx_ticker(s)
        rows.append(r1)
    return pd.DataFrame(rows)
