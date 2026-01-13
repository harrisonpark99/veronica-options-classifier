# -*- coding: utf-8 -*-
"""
Create Invoice - Multi-Exchange Support
Converts trade execution CSV to invoice Excel template.
"""

import io
import copy
from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st
from openpyxl import load_workbook
from openpyxl.worksheet.worksheet import Worksheet

# Import exchange configuration module
from exchange_configs import (
    ExchangeConfig,
    EXCHANGE_REGISTRY,
    detect_exchange,
    get_exchange_config,
    get_exchange_display_names,
    default_side_detector,
    default_totals_calculator,
)

# -----------------------------
# Auth Check
# -----------------------------
if "auth_ok" not in st.session_state or not st.session_state.auth_ok:
    st.warning("로그인이 필요합니다.")
    st.stop()


# -----------------------------
# Summary Labels (Template-side)
# -----------------------------
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
    """Normalize string for comparison."""
    return str(s).strip().lower()


def detect_side(df: pd.DataFrame, config: ExchangeConfig) -> str:
    """Determine BUY/SELL/MIXED using exchange-specific logic."""
    if config.side_detector:
        return config.side_detector(df)
    # Default: use isBuyer column
    return default_side_detector(df)


def compute_totals(df: pd.DataFrame, config: ExchangeConfig) -> Tuple[float, float, float]:
    """Compute filled_amount, filled_value, avg_price using exchange-specific logic."""
    if config.totals_calculator:
        return config.totals_calculator(df)
    # Default: use qty/quoteQty
    return default_totals_calculator(df)


def copy_row_style(ws: Worksheet, src_row: int, dst_row: int, max_col: int):
    """Copy cell styles from src_row to dst_row for columns 1..max_col."""
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
    """Pick the sheet that likely contains the Trading Data table."""
    for ws in wb.worksheets:
        for row in ws.iter_rows(values_only=True):
            for v in row:
                if isinstance(v, str) and norm(v) == "trading data":
                    return ws
    # Fallback: choose first non-Summary
    for ws in wb.worksheets:
        if norm(ws.title) != "summary":
            return ws
    return wb.worksheets[0] if wb.worksheets else None


def find_header_row_and_map(ws: Worksheet, config: ExchangeConfig) -> Tuple[int, Dict[str, int], int]:
    """
    Find header row containing expected columns.
    Returns: header_row_idx, mapping(df_col -> excel_col_idx), max_used_col
    """
    max_row = min(ws.max_row, 300)
    max_col = min(ws.max_column, 60)

    best_row = None
    best_score = -1

    # Key columns to look for (use config's header mapping keys)
    key_headers = ["symbol", "price", "qty", "quoteqty", "commission", "time"]
    # Add exchange-specific keys
    for col in config.required_columns[:6]:
        key_headers.append(norm(col))

    # Scan rows to find header candidates
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

    # Build mapping: df_col -> column idx by matching header variants
    header_vals = {}
    for c in range(1, max_col + 1):
        v = ws.cell(row=header_row, column=c).value
        if v is None:
            continue
        header_vals[norm(v)] = c

    mapping: Dict[str, int] = {}
    for df_col, variants in config.header_mapping.items():
        for v in variants:
            if v in header_vals:
                mapping[df_col] = header_vals[v]
                break

    return header_row, mapping, max_col


def clear_table_area(ws: Worksheet, start_row: int, start_col: int, end_row: int, end_col: int):
    """Clear values in a rectangular area."""
    for r in range(start_row, end_row + 1):
        for c in range(start_col, end_col + 1):
            ws.cell(row=r, column=c).value = None


def write_trading_table(ws: Worksheet, df: pd.DataFrame, config: ExchangeConfig) -> Tuple[int, Dict[str, int]]:
    """Write DataFrame to the trading table area in the worksheet."""
    header_row, colmap, max_col = find_header_row_and_map(ws, config)
    data_start = header_row + 1

    # Determine which columns we will write (intersection)
    writable_cols = [c for c in df.columns if c in colmap]

    if not writable_cols:
        # Fallback: assume default layout starting at column A
        all_cols = config.required_columns + [c for c in config.optional_columns if c in df.columns]
        writable_cols = [c for c in all_cols if c in df.columns]
        colmap = {col: i + 1 for i, col in enumerate(writable_cols)}

    # Clear previous table area
    end_row = min(ws.max_row, data_start + 5000)
    end_col = max(colmap.values()) if colmap else 20
    clear_table_area(ws, data_start, 1, end_row, end_col)

    # Copy style template row
    style_src_row = data_start

    # Write rows
    for i, (_, row) in enumerate(df.iterrows()):
        r = data_start + i
        copy_row_style(ws, style_src_row, r, max(end_col, 20))

        for col in writable_cols:
            c = colmap[col]
            val = row[col]
            # Convert pandas types
            if pd.isna(val):
                val = None
            elif isinstance(val, pd.Timestamp):
                val = val.to_pydatetime()
            ws.cell(row=r, column=c).value = val

    return data_start, colmap


def set_value_next_to_label(ws: Worksheet, label_key: str, value) -> bool:
    """Find a label cell and set the cell to the right with 'value'."""
    target = norm(label_key)
    for r in range(1, ws.max_row + 1):
        for c in range(1, ws.max_column + 1):
            v = ws.cell(row=r, column=c).value
            if isinstance(v, str) and target == norm(v):
                ws.cell(row=r, column=c + 1).value = value
                return True
    return False


def set_value_by_label_fuzzy(ws: Worksheet, label_text: str, value) -> bool:
    """Fuzzy match: if label_text appears within a cell string, set right neighbor."""
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
    """Update 'Summary' sheet and trading sheet's order summary area."""
    rebate = 0.0

    if side == "BUY":
        net = filled_value
        gross = net / (1 - fee_rate) if (1 - fee_rate) != 0 else net
        fee_amount = gross - net
        gross_buy = gross
        gross_sell = None
    elif side == "SELL":
        gross = filled_value
        fee_amount = gross * fee_rate
        net = gross - fee_amount
        gross_buy = None
        gross_sell = gross
    else:
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
                          filled_amount: float, filled_value: float, avg_price: float,
                          config: ExchangeConfig):
    """Put totals into sum columns if present, or fallback to P/Q/R."""
    def set_if_exists(col_name: str, value) -> bool:
        if col_name in colmap:
            ws.cell(row=first_data_row, column=colmap[col_name]).value = value
            return True
        return False

    # Try standard column names first
    ok1 = set_if_exists("sum_fill_amount", filled_amount)
    ok2 = set_if_exists("sum_fill_value", filled_value)
    ok3 = set_if_exists("ave_fill_price", avg_price)

    if not (ok1 or ok2 or ok3):
        # Fallback to P/Q/R = 16/17/18
        ws.cell(row=first_data_row, column=16).value = filled_amount
        ws.cell(row=first_data_row, column=17).value = filled_value
        ws.cell(row=first_data_row, column=18).value = avg_price


def validate_csv(df: pd.DataFrame, config: ExchangeConfig) -> Tuple[bool, list]:
    """Validate that CSV has required columns for the exchange."""
    df_cols_lower = set(c.lower() for c in df.columns)
    missing = []

    for req_col in config.required_columns:
        if req_col not in df.columns and req_col.lower() not in df_cols_lower:
            missing.append(req_col)

    return len(missing) == 0, missing


def normalize_columns(df: pd.DataFrame, config: ExchangeConfig) -> pd.DataFrame:
    """Convert numeric columns to proper types."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    for col in config.numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Create Invoice", layout="wide")
st.title("Create Invoice")
st.caption("체결내역 CSV -> Invoice 템플릿 자동 생성 (다중 거래소 지원)")

# Sidebar
with st.sidebar:
    st.header("설정")

    # Exchange selection
    exchange_names = get_exchange_display_names()
    exchange_options = ["자동 감지"] + [f"{v} ({k})" for k, v in exchange_names.items()]
    selected_exchange = st.selectbox("거래소 선택", exchange_options)

    st.markdown("---")

    fee_rate = st.number_input(
        "Fee rate (예: 0.25% = 0.0025)",
        value=0.0025,
        step=0.0001,
        format="%.6f"
    )
    st.caption("템플릿에 이미 수식이 있어도, 요약 셀을 라벨 기반으로 직접 업데이트합니다.")

    st.markdown("---")
    st.markdown("**지원 거래소**")
    for name, display in exchange_names.items():
        st.markdown(f"- {display}")

    st.markdown("---")
    st.caption("새 거래소 추가가 필요하면 exchange_configs.py를 수정하세요.")

# Main content
col1, col2 = st.columns(2)
with col1:
    csv_file = st.file_uploader("1) 체결내역 CSV 업로드", type=["csv"])
with col2:
    template_file = st.file_uploader("2) Invoice 템플릿 XLSX 업로드", type=["xlsx"])

if csv_file and template_file:
    # Read CSV
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        st.error(f"CSV 읽기 실패: {e}")
        st.stop()

    # Determine exchange
    if selected_exchange == "자동 감지":
        detected = detect_exchange(df)
        if detected:
            st.info(f"자동 감지된 거래소: **{exchange_names.get(detected, detected)}**")
            config = get_exchange_config(detected)
        else:
            st.warning("거래소를 자동 감지할 수 없습니다. 수동으로 선택해주세요.")
            st.stop()
    else:
        # Parse selection like "Binance (binance)"
        exchange_key = selected_exchange.split("(")[-1].rstrip(")")
        config = get_exchange_config(exchange_key)
        if not config:
            st.error(f"알 수 없는 거래소: {exchange_key}")
            st.stop()

    # Validate CSV
    is_valid, missing_cols = validate_csv(df, config)
    if not is_valid:
        st.error(f"CSV 필수 컬럼이 없습니다 ({config.display_name}): {missing_cols}")
        st.stop()

    # Normalize columns
    df = normalize_columns(df, config)

    # Compute metrics
    side = detect_side(df, config)
    filled_amount, filled_value, avg_price = compute_totals(df, config)

    # Display summary
    st.subheader("입력 CSV 요약")
    a, b, c, d = st.columns(4)
    a.metric("Rows", f"{len(df):,}")
    b.metric("Side", side)
    c.metric("Filled Amount", f"{filled_amount:,.8f}")
    d.metric("Filled Value", f"{filled_value:,.2f}")
    st.caption(f"Average Filled Price: {avg_price:,.2f} | Exchange: {config.display_name}")

    # Load template workbook
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

    # Write Trading table
    first_data_row, colmap = write_trading_table(trading_ws, df, config)

    # Write totals into table
    write_totals_in_table(trading_ws, first_data_row, colmap,
                          filled_amount, filled_value, avg_price, config)

    # Update summaries
    update_summary_sheets(wb, side=side, fee_rate=float(fee_rate),
                          filled_amount=filled_amount, filled_value=filled_value, avg_price=avg_price)

    # Output
    out = io.BytesIO()
    wb.save(out)
    out.seek(0)

    # Filename suggestion
    base_name = f"generated_invoice_{config.name}"

    # Try to get symbol from config's symbol column
    symbol_col = config.column_mapping.get("symbol", ["symbol"])[0]
    if symbol_col in df.columns and df[symbol_col].nunique() == 1:
        base_name += f"_{df[symbol_col].iloc[0]}"

    # Try to get date
    time_col = config.column_mapping.get("time", ["time"])[0]
    if time_col in df.columns:
        try:
            s = str(df[time_col].iloc[0])
            base_name += f"_{s[:10].replace('-', '').replace('/', '')}"
        except Exception:
            pass
    base_name += ".xlsx"

    st.success("완성 인보이스 파일 생성 완료!")
    st.download_button(
        label="인보이스 XLSX 다운로드",
        data=out.getvalue(),
        file_name=base_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.ml",
    )

else:
    st.info("좌측에 CSV와 템플릿 XLSX를 업로드하면, 자동으로 인보이스가 생성됩니다.")

    # Show supported formats
    with st.expander("지원 CSV 형식 보기"):
        for name, config in EXCHANGE_REGISTRY.items():
            st.markdown(f"### {config.display_name}")
            st.markdown(f"**필수 컬럼:** `{', '.join(config.required_columns)}`")
            if config.optional_columns:
                st.markdown(f"**선택 컬럼:** `{', '.join(config.optional_columns)}`")
            st.markdown("---")
