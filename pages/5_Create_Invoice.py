# -*- coding: utf-8 -*-
"""
Create Invoice - Simplified UI
Converts trade execution CSV to invoice Excel template.
"""

import io
import copy
from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st
from openpyxl import load_workbook
from openpyxl.worksheet.worksheet import Worksheet

# -----------------------------
# Auth Check
# -----------------------------
if "auth_ok" not in st.session_state or not st.session_state.auth_ok:
    st.warning("로그인이 필요합니다.")
    st.stop()


# -----------------------------
# Binance Configuration
# -----------------------------
BINANCE_REQUIRED_COLS = [
    "symbol", "id", "orderId", "orderListId", "price", "qty", "quoteQty",
    "commission", "commissionAsset", "time", "isBuyer", "isMaker", "isBestMatch",
]

BINANCE_OPTIONAL_COLS = [
    "fill_time(UTC+8)", "fill_value", "sum_fill_amount", "sum_fill_value", "ave_fill_price"
]

BINANCE_NUMERIC_COLS = ["price", "qty", "quoteQty", "commission"]

BINANCE_HEADER_MAP = {
    "symbol": ["symbol"],
    "id": ["id"],
    "orderId": ["orderid", "order id"],
    "orderListId": ["orderlistid", "order list id"],
    "price": ["price"],
    "qty": ["qty", "quantity"],
    "quoteQty": ["quoteqty", "quote qty", "amount"],
    "commission": ["commission", "fee"],
    "commissionAsset": ["commissionasset", "fee asset"],
    "time": ["time", "timestamp"],
    "isBuyer": ["isbuyer", "buyer", "side"],
    "isMaker": ["ismaker", "maker"],
    "isBestMatch": ["isbestmatch", "bestmatch"],
    "fill_time(UTC+8)": ["fill_time(utc+8)", "fill time(utc+8)", "fill time"],
    "fill_value": ["fill_value", "fill value"],
    "sum_fill_amount": ["sum_fill_amount", "sum fill amount"],
    "sum_fill_value": ["sum_fill_value", "sum fill value"],
    "ave_fill_price": ["ave_fill_price", "ave fill price", "avg fill price"],
}

# Summary labels for template
SUMMARY_LABELS = {
    "filled amount": "filled_amount",
    "filled value": "filled_value",
    "average filled price": "avg_price",
    "fee": "fee_amount",
    "fee rate": "fee_rate",
    "net of fee": "net",
    "buy order amount": "gross_buy",
    "sell order amount": "gross_sell",
    "rebate": "rebate",
}

SUMMARY_LABEL_VARIANTS = {
    "filled amount(btc)": "filled_amount",
    "filled amount (btc)": "filled_amount",
    "filled value(usdt)": "filled_value",
    "filled value (usdt)": "filled_value",
    "avg filled price": "avg_price",
    "average price": "avg_price",
    "net": "net",
}


# -----------------------------
# Helper Functions
# -----------------------------
def norm(s: str) -> str:
    return str(s).strip().lower()


def compute_totals(df: pd.DataFrame) -> Tuple[float, float, float]:
    filled_amount = float(pd.to_numeric(df["qty"], errors="coerce").fillna(0).sum())
    filled_value = float(pd.to_numeric(df["quoteQty"], errors="coerce").fillna(0).sum())
    avg_price = (filled_value / filled_amount) if filled_amount != 0 else 0.0
    return filled_amount, filled_value, avg_price


def copy_row_style(ws: Worksheet, src_row: int, dst_row: int, max_col: int):
    for c in range(1, max_col + 1):
        src = ws.cell(row=src_row, column=c)
        dst = ws.cell(row=dst_row, column=c)
        if src.has_style:
            dst._style = copy.copy(src._style)
        dst.number_format = src.number_format
        dst.font = copy.copy(src.font)
        dst.fill = copy.copy(src.fill)
        dst.border = copy.copy(src.border)
        dst.alignment = copy.copy(src.alignment)
        dst.protection = copy.copy(src.protection)


def find_sheet_with_trading_data(wb) -> Optional[Worksheet]:
    for ws in wb.worksheets:
        for row in ws.iter_rows(values_only=True):
            for v in row:
                if isinstance(v, str) and norm(v) == "trading data":
                    return ws
    for ws in wb.worksheets:
        if norm(ws.title) != "summary":
            return ws
    return wb.worksheets[0] if wb.worksheets else None


def find_header_row_and_map(ws: Worksheet) -> Tuple[int, Dict[str, int], int]:
    max_row = min(ws.max_row, 300)
    max_col = min(ws.max_column, 60)

    best_row = None
    best_score = -1
    key_headers = ["symbol", "price", "qty", "quoteqty", "commission", "time"]

    for r in range(1, max_row + 1):
        vals = []
        for c in range(1, max_col + 1):
            v = ws.cell(row=r, column=c).value
            if v is None:
                vals.append("")
            else:
                vals.append(norm(v) if isinstance(v, str) else norm(str(v)))
        score = sum(1 for k in key_headers if k in vals)
        if score > best_score:
            best_score = score
            best_row = r

    header_row = best_row if best_row is not None else 14

    header_vals = {}
    for c in range(1, max_col + 1):
        v = ws.cell(row=header_row, column=c).value
        if v is None:
            continue
        header_vals[norm(v)] = c

    mapping: Dict[str, int] = {}
    for df_col, variants in BINANCE_HEADER_MAP.items():
        for v in variants:
            if v in header_vals:
                mapping[df_col] = header_vals[v]
                break

    return header_row, mapping, max_col


def clear_table_area(ws: Worksheet, start_row: int, start_col: int, end_row: int, end_col: int):
    for r in range(start_row, end_row + 1):
        for c in range(start_col, end_col + 1):
            ws.cell(row=r, column=c).value = None


def write_trading_table(ws: Worksheet, df: pd.DataFrame) -> Tuple[int, Dict[str, int]]:
    header_row, colmap, max_col = find_header_row_and_map(ws)
    data_start = header_row + 1

    writable_cols = [c for c in df.columns if c in colmap]

    if not writable_cols:
        all_cols = BINANCE_REQUIRED_COLS + [c for c in BINANCE_OPTIONAL_COLS if c in df.columns]
        writable_cols = [c for c in all_cols if c in df.columns]
        colmap = {col: i + 1 for i, col in enumerate(writable_cols)}

    end_row = min(ws.max_row, data_start + 5000)
    end_col = max(colmap.values()) if colmap else 20
    clear_table_area(ws, data_start, 1, end_row, end_col)

    style_src_row = data_start

    for i, (_, row) in enumerate(df.iterrows()):
        r = data_start + i
        copy_row_style(ws, style_src_row, r, max(end_col, 20))

        for col in writable_cols:
            c = colmap[col]
            val = row[col]
            if pd.isna(val):
                val = None
            elif isinstance(val, pd.Timestamp):
                val = val.to_pydatetime()
            ws.cell(row=r, column=c).value = val

    return data_start, colmap


def set_value_next_to_label(ws: Worksheet, label_key: str, value) -> bool:
    target = norm(label_key)
    for r in range(1, ws.max_row + 1):
        for c in range(1, ws.max_column + 1):
            v = ws.cell(row=r, column=c).value
            if isinstance(v, str) and target == norm(v):
                ws.cell(row=r, column=c + 1).value = value
                return True
    return False


def set_value_by_label_fuzzy(ws: Worksheet, label_text: str, value) -> bool:
    needle = norm(label_text)
    for r in range(1, ws.max_row + 1):
        for c in range(1, ws.max_column + 1):
            v = ws.cell(row=r, column=c).value
            if isinstance(v, str):
                if needle in norm(v):
                    ws.cell(row=r, column=c + 1).value = value
                    return True
    return False


def update_summary_sheets(wb, side: str, fee_rate: float, filled_amount: float,
                          filled_value: float, avg_price: float):
    rebate = 0.0

    if side == "BUY":
        net = filled_value
        gross = net / (1 - fee_rate) if (1 - fee_rate) != 0 else net
        fee_amount = gross - net
        gross_buy = gross
        gross_sell = None
    else:  # SELL
        gross = filled_value
        fee_amount = gross * fee_rate
        net = gross - fee_amount
        gross_buy = None
        gross_sell = gross

    for ws in wb.worksheets:
        if ws.max_row < 5:
            continue

        payload = {
            "filled_amount": filled_amount,
            "filled_value": filled_value,
            "avg_price": avg_price,
            "fee_rate": fee_rate,
            "fee_amount": fee_amount,
            "net": net,
            "rebate": rebate,
            "gross_buy": gross_buy,
            "gross_sell": gross_sell,
        }

        for lbl, key in SUMMARY_LABELS.items():
            val = payload.get(key)
            if val is not None:
                set_value_next_to_label(ws, lbl, val)

        for lbl, key in SUMMARY_LABEL_VARIANTS.items():
            val = payload.get(key)
            if val is not None:
                set_value_by_label_fuzzy(ws, lbl, val)


def write_totals_in_table(ws: Worksheet, first_data_row: int, colmap: Dict[str, int],
                          filled_amount: float, filled_value: float, avg_price: float):
    def set_if_exists(col_name: str, value) -> bool:
        if col_name in colmap:
            ws.cell(row=first_data_row, column=colmap[col_name]).value = value
            return True
        return False

    ok1 = set_if_exists("sum_fill_amount", filled_amount)
    ok2 = set_if_exists("sum_fill_value", filled_value)
    ok3 = set_if_exists("ave_fill_price", avg_price)

    if not (ok1 or ok2 or ok3):
        ws.cell(row=first_data_row, column=16).value = filled_amount
        ws.cell(row=first_data_row, column=17).value = filled_value
        ws.cell(row=first_data_row, column=18).value = avg_price


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Create Invoice", layout="wide")
st.title("Create Invoice")
st.caption("체결내역 CSV를 Invoice 템플릿으로 변환")

# Sidebar - Settings
with st.sidebar:
    st.header("설정")

    # 1. Order Side Selection
    st.subheader("1. 주문 유형")
    order_side = st.radio(
        "매수/매도 선택",
        options=["BUY (매수)", "SELL (매도)"],
        horizontal=True
    )
    selected_side = "BUY" if "BUY" in order_side else "SELL"

    st.markdown("---")

    # 2. CEX Selection
    st.subheader("2. 거래소 선택")
    cex_option = st.radio(
        "CEX 선택",
        options=["Binance", "Others"],
        horizontal=True
    )

    st.markdown("---")

    # Fee rate (only for Binance)
    if cex_option == "Binance":
        fee_rate = st.number_input(
            "Fee rate (예: 0.25% = 0.0025)",
            value=0.0025,
            step=0.0001,
            format="%.6f"
        )

# Main content
if cex_option == "Others":
    st.warning("현재 Binance만 지원됩니다. 다른 거래소 지원은 준비 중입니다.")
    st.stop()

# Binance flow
st.subheader("파일 업로드")

col1, col2 = st.columns(2)
with col1:
    csv_file = st.file_uploader("체결내역 CSV", type=["csv"])
with col2:
    template_file = st.file_uploader("Invoice 템플릿 XLSX", type=["xlsx"])

if csv_file and template_file:
    # Read CSV
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        st.error(f"CSV 읽기 실패: {e}")
        st.stop()

    # Validate columns
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in BINANCE_REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"CSV 필수 컬럼이 없습니다 (Binance 형식): {missing}")
        st.stop()

    # Convert numeric columns
    for col in BINANCE_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute totals
    filled_amount, filled_value, avg_price = compute_totals(df)

    # Display summary
    st.subheader("입력 CSV 요약")
    a, b, c, d = st.columns(4)
    a.metric("Rows", f"{len(df):,}")
    b.metric("Side", selected_side)
    c.metric("Filled Amount", f"{filled_amount:,.8f}")
    d.metric("Filled Value", f"{filled_value:,.2f}")
    st.caption(f"Average Filled Price: {avg_price:,.2f}")

    # Load template
    try:
        wb = load_workbook(template_file)
    except Exception as e:
        st.error(f"템플릿 XLSX 로드 실패: {e}")
        st.stop()

    trading_ws = find_sheet_with_trading_data(wb)
    if trading_ws is None:
        st.error("Trading Data 시트를 찾지 못했습니다.")
        st.stop()

    st.write(f"선택된 Trading 시트: **{trading_ws.title}**")

    # Write trading table
    first_data_row, colmap = write_trading_table(trading_ws, df)

    # Write totals
    write_totals_in_table(trading_ws, first_data_row, colmap,
                          filled_amount, filled_value, avg_price)

    # Update summaries with user-selected side
    update_summary_sheets(wb, side=selected_side, fee_rate=float(fee_rate),
                          filled_amount=filled_amount, filled_value=filled_value, avg_price=avg_price)

    # Output
    out = io.BytesIO()
    wb.save(out)
    out.seek(0)

    # Filename
    base_name = f"invoice_binance_{selected_side.lower()}"
    if "symbol" in df.columns and df["symbol"].nunique() == 1:
        base_name += f"_{df['symbol'].iloc[0]}"
    if "fill_time(UTC+8)" in df.columns:
        try:
            s = str(df["fill_time(UTC+8)"].iloc[0])
            base_name += f"_{s[:10].replace('-', '').replace('/', '')}"
        except Exception:
            pass
    base_name += ".xlsx"

    st.success("인보이스 파일 생성 완료!")
    st.download_button(
        label="인보이스 XLSX 다운로드",
        data=out.getvalue(),
        file_name=base_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

elif csv_file or template_file:
    st.info("CSV 파일과 템플릿 XLSX 파일을 모두 업로드해주세요.")
else:
    st.info("체결내역 CSV와 Invoice 템플릿 XLSX를 업로드하면 인보이스가 생성됩니다.")
