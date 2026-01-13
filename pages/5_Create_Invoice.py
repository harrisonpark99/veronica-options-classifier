# -*- coding: utf-8 -*-
"""
Create Invoice - Auto-generate invoice from trade execution CSV
"""

import io
from datetime import datetime
from typing import Tuple

import pandas as pd
import streamlit as st
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, Border, Side, PatternFill
from openpyxl.utils.dataframe import dataframe_to_rows

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

BINANCE_NUMERIC_COLS = ["price", "qty", "quoteQty", "commission"]


# -----------------------------
# Helper Functions
# -----------------------------
def compute_totals(df: pd.DataFrame) -> Tuple[float, float, float, float]:
    """Compute filled_amount, filled_value, avg_price, total_commission."""
    filled_amount = float(pd.to_numeric(df["qty"], errors="coerce").fillna(0).sum())
    filled_value = float(pd.to_numeric(df["quoteQty"], errors="coerce").fillna(0).sum())
    avg_price = (filled_value / filled_amount) if filled_amount != 0 else 0.0
    total_commission = float(pd.to_numeric(df["commission"], errors="coerce").fillna(0).sum())
    return filled_amount, filled_value, avg_price, total_commission


def create_invoice_workbook(
    df: pd.DataFrame,
    side: str,
    fee_rate: float,
    filled_amount: float,
    filled_value: float,
    avg_price: float,
    total_commission: float
) -> Workbook:
    """Create invoice Excel workbook from trade data."""
    wb = Workbook()

    # ----- Summary Sheet -----
    ws_summary = wb.active
    ws_summary.title = "Summary"

    # Styles
    title_font = Font(bold=True, size=14)
    header_font = Font(bold=True, size=11)
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    header_font_white = Font(bold=True, size=11, color="FFFFFF")
    border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )

    # Title
    ws_summary["A1"] = "Trade Invoice"
    ws_summary["A1"].font = title_font

    # Get symbol if available
    symbol = df["symbol"].iloc[0] if "symbol" in df.columns and len(df) > 0 else "N/A"

    # Order Info
    ws_summary["A3"] = "Order Information"
    ws_summary["A3"].font = header_font

    info_data = [
        ("Symbol", symbol),
        ("Side", side),
        ("Exchange", "Binance"),
        ("Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    ]

    for i, (label, value) in enumerate(info_data, start=4):
        ws_summary[f"A{i}"] = label
        ws_summary[f"B{i}"] = value
        ws_summary[f"A{i}"].font = Font(bold=True)

    # Summary Section
    ws_summary["A10"] = "Order Summary"
    ws_summary["A10"].font = header_font

    # Calculate fee/net based on side
    if side == "BUY":
        gross = filled_value / (1 - fee_rate) if (1 - fee_rate) != 0 else filled_value
        fee_amount = gross - filled_value
        net = filled_value
    else:  # SELL
        gross = filled_value
        fee_amount = gross * fee_rate
        net = gross - fee_amount

    summary_data = [
        ("Filled Amount", f"{filled_amount:,.8f}"),
        ("Filled Value", f"{filled_value:,.2f} USDT"),
        ("Average Price", f"{avg_price:,.2f} USDT"),
        ("Total Commission", f"{total_commission:,.8f}"),
        ("Fee Rate", f"{fee_rate * 100:.4f}%"),
        ("Gross Amount", f"{gross:,.2f} USDT"),
        ("Fee Amount", f"{fee_amount:,.2f} USDT"),
        ("Net Amount", f"{net:,.2f} USDT"),
    ]

    for i, (label, value) in enumerate(summary_data, start=11):
        ws_summary[f"A{i}"] = label
        ws_summary[f"B{i}"] = value
        ws_summary[f"A{i}"].font = Font(bold=True)
        ws_summary[f"A{i}"].border = border
        ws_summary[f"B{i}"].border = border

    # Adjust column widths
    ws_summary.column_dimensions["A"].width = 20
    ws_summary.column_dimensions["B"].width = 25

    # ----- Trading Data Sheet -----
    ws_trading = wb.create_sheet("Trading Data")

    # Select columns to include
    display_cols = [c for c in BINANCE_REQUIRED_COLS if c in df.columns]
    df_display = df[display_cols].copy()

    # Write headers
    for col_idx, col_name in enumerate(display_cols, start=1):
        cell = ws_trading.cell(row=1, column=col_idx, value=col_name)
        cell.font = header_font_white
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")
        cell.border = border

    # Write data
    for row_idx, row in enumerate(df_display.itertuples(index=False), start=2):
        for col_idx, value in enumerate(row, start=1):
            cell = ws_trading.cell(row=row_idx, column=col_idx, value=value)
            cell.border = border

    # Adjust column widths
    for col_idx, col_name in enumerate(display_cols, start=1):
        max_length = max(len(str(col_name)), 12)
        ws_trading.column_dimensions[chr(64 + col_idx) if col_idx <= 26 else f"{chr(64 + col_idx // 26)}{chr(64 + col_idx % 26)}"].width = max_length + 2

    return wb


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Create Invoice", layout="wide")
st.title("Create Invoice")
st.caption("체결내역 CSV로 Invoice 자동 생성")

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
st.subheader("체결내역 업로드")
csv_file = st.file_uploader("체결내역 CSV 파일", type=["csv"])

if csv_file:
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
    filled_amount, filled_value, avg_price, total_commission = compute_totals(df)

    # Display summary
    st.subheader("체결내역 요약")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("거래 건수", f"{len(df):,}")
    col2.metric("주문 유형", selected_side)
    col3.metric("체결 수량", f"{filled_amount:,.8f}")
    col4.metric("체결 금액", f"{filled_value:,.2f} USDT")

    st.caption(f"평균 체결가: {avg_price:,.2f} USDT | 총 수수료: {total_commission:,.8f}")

    # Preview data
    with st.expander("체결내역 미리보기", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)

    st.markdown("---")

    # Generate invoice
    if st.button("인보이스 생성", type="primary", use_container_width=True):
        wb = create_invoice_workbook(
            df=df,
            side=selected_side,
            fee_rate=float(fee_rate),
            filled_amount=filled_amount,
            filled_value=filled_value,
            avg_price=avg_price,
            total_commission=total_commission
        )

        # Save to bytes
        out = io.BytesIO()
        wb.save(out)
        out.seek(0)

        # Filename
        symbol = df["symbol"].iloc[0] if "symbol" in df.columns and df["symbol"].nunique() == 1 else "trade"
        date_str = datetime.now().strftime("%Y%m%d")
        file_name = f"invoice_{symbol}_{selected_side.lower()}_{date_str}.xlsx"

        st.success("인보이스 생성 완료!")
        st.download_button(
            label="인보이스 XLSX 다운로드",
            data=out.getvalue(),
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

else:
    st.info("Binance 체결내역 CSV 파일을 업로드하면 인보이스가 자동 생성됩니다.")
