# -*- coding: utf-8 -*-
"""
VERONICA - Strategy PnL Summary Dashboard
Upload strategy_ytd_history.csv and analyze Strategy-level DTD/MTD/YTD
"""

import os
import sys
import pandas as pd
import streamlit as st

# Import utilities
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.auth import require_auth, show_logout_button

# ================== Page Config ==================
st.set_page_config(
    page_title="Strategy PnL Summary - VERONICA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== Auth Check ==================
require_auth()


# =========================
# Helpers
# =========================
def load_history(uploaded_file):
    """Load and process uploaded CSV file."""
    try:
        df = pd.read_csv(uploaded_file)
        df["date"] = pd.to_datetime(df["date"])
        df["pnl_usd"] = pd.to_numeric(df["pnl_usd"], errors="coerce")
        df["strategy_name"] = df["strategy_name"].astype(str).str.strip()
        df = df[~df["strategy_name"].str.upper().eq("TOTAL")]
        return df, None
    except Exception as e:
        return None, f"Failed to parse CSV: {e}"


def pivot_ytd(df_hist):
    wide = df_hist.pivot_table(
        index="date",
        columns="strategy_name",
        values="pnl_usd",
        aggfunc="first"
    ).sort_index()
    return wide


def get_prev_available_date(dates, current_date):
    available = [d for d in dates if d < current_date]
    return max(available) if available else None


def get_last_month_end(dates, current_date):
    month_start = current_date.replace(day=1)
    available = [d for d in dates if d < month_start]
    return max(available) if available else None


def compute_dtd_mtd_ytd(wide_df, target_date):
    dates = list(wide_df.index)

    if target_date not in wide_df.index:
        return None, None, None

    prev_date = get_prev_available_date(dates, target_date)
    last_month_end = get_last_month_end(dates, target_date)

    ytd_today = wide_df.loc[target_date]

    # DTD
    if prev_date is not None:
        dtd = ytd_today - wide_df.loc[prev_date]
    else:
        dtd = pd.Series(index=wide_df.columns, data=[pd.NA] * len(wide_df.columns))

    # MTD (based on last month end)
    if last_month_end is not None:
        mtd = ytd_today - wide_df.loc[last_month_end]
    else:
        mtd = pd.Series(index=wide_df.columns, data=[pd.NA] * len(wide_df.columns))

    out = pd.DataFrame({
        "strategy_name": wide_df.columns,
        "DTD": dtd.values,
        "MTD": mtd.values,
        "YTD": ytd_today.values
    })

    for col in ["DTD", "MTD", "YTD"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out, prev_date, last_month_end


def infer_category(strategy_name: str) -> str:
    if not isinstance(strategy_name, str) or not strategy_name.strip():
        return "other"
    s = strategy_name.strip()
    if "_" in s:
        return s.split("_", 1)[0].lower()
    return "other"


def add_category(df_strategy):
    df_strategy = df_strategy.copy()
    df_strategy["category"] = df_strategy["strategy_name"].apply(infer_category)
    return df_strategy


def category_summary(df_with_cat):
    df = df_with_cat.copy()
    for col in ["DTD", "MTD", "YTD"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df_cat = (
        df.groupby("category")[["DTD", "MTD", "YTD"]]
        .sum()
        .reset_index()
        .sort_values("YTD", ascending=False)
    )
    return df_cat


def format_money(x):
    if pd.isna(x):
        return ""
    return f"{x:,.2f}"


def build_top_table(df, col, n=10, ascending=False):
    df2 = df[["strategy_name", "category", col]].copy()
    df2[col] = pd.to_numeric(df2[col], errors="coerce")
    df2 = df2.dropna(subset=[col])
    df2 = df2.sort_values(col, ascending=ascending).head(n)
    df2[col] = df2[col].apply(format_money)
    return df2


# =========================
# UI
# =========================
st.title("Strategy PnL Summary Dashboard")
st.caption("Upload strategy_ytd_history.csv to analyze Strategy-level DTD/MTD/YTD PnL")

# Sidebar
with st.sidebar:
    st.header("VERONICA")
    show_logout_button()
    st.markdown("---")

# File upload
st.subheader("CSV Upload")
uploaded_file = st.file_uploader(
    "Upload strategy_ytd_history.csv",
    type=["csv"],
    help="CSV must have columns: date, strategy_name, pnl_usd"
)

if uploaded_file is None:
    st.info("CSV 파일을 업로드해주세요. (필수 컬럼: date, strategy_name, pnl_usd)")
    st.stop()

# Load data
df_hist, error = load_history(uploaded_file)

if error:
    st.error(error)
    st.stop()

# Validate required columns
required_cols = ["date", "strategy_name", "pnl_usd"]
missing_cols = [c for c in required_cols if c not in df_hist.columns]
if missing_cols:
    st.error(f"Missing required columns: {missing_cols}")
    st.stop()

wide = pivot_ytd(df_hist)

available_dates = list(wide.index)
latest_date = max(available_dates)

# Sidebar options
with st.sidebar:
    st.header("Options")

    target_date = st.date_input(
        "Select Date",
        value=latest_date.date(),
        min_value=min(available_dates).date(),
        max_value=latest_date.date()
    )
    target_date = pd.to_datetime(target_date)

    search = st.text_input("Search Strategy (substring)", "")
    sort_by = st.selectbox("Sort By", ["YTD", "MTD", "DTD"])
    ascending = st.checkbox("Ascending Order", value=False)
    top_n = st.slider("Top N Strategies to Show", 10, 300, 80)

df_strategy, prev_date, last_month_end = compute_dtd_mtd_ytd(wide, target_date)

if df_strategy is None:
    st.warning("No data available for the selected date.")
    st.stop()

df_strategy = add_category(df_strategy)

# Filter by search
if search.strip():
    df_strategy = df_strategy[df_strategy["strategy_name"].str.contains(search, case=False, na=False)]

df_strategy_full = df_strategy.copy()
df_strategy_view = df_strategy.sort_values(sort_by, ascending=ascending).head(top_n)
df_cat = category_summary(df_strategy_full)

st.divider()

# Header metrics
c1, c2, c3 = st.columns(3)
c1.metric("Selected Date", target_date.strftime("%Y-%m-%d"))
c2.metric("DTD Baseline", prev_date.strftime("%Y-%m-%d") if prev_date else "N/A")
c3.metric("MTD Baseline", last_month_end.strftime("%Y-%m-%d") if last_month_end else "N/A")

st.divider()

# Total summary
st.subheader("Total (All Strategies)")
total_dtd = df_strategy_full["DTD"].sum(skipna=True)
total_mtd = df_strategy_full["MTD"].sum(skipna=True)
total_ytd = df_strategy_full["YTD"].sum(skipna=True)

t1, t2, t3 = st.columns(3)
t1.metric("Total DTD", format_money(total_dtd))
t2.metric("Total MTD", format_money(total_mtd))
t3.metric("Total YTD", format_money(total_ytd))

st.divider()

# Category Summary
st.subheader("Category Summary (prefix before '_')")
df_cat_display = df_cat.copy()
for col in ["DTD", "MTD", "YTD"]:
    df_cat_display[col] = df_cat_display[col].apply(format_money)
st.dataframe(df_cat_display, use_container_width=True)

st.divider()

# Top Movers
st.subheader("Top Movers")
colA, colB, colC = st.columns(3)
with colA:
    st.markdown("### Top DTD")
    st.dataframe(build_top_table(df_strategy_full, "DTD", n=10, ascending=False), use_container_width=True)
with colB:
    st.markdown("### Top MTD")
    st.dataframe(build_top_table(df_strategy_full, "MTD", n=10, ascending=False), use_container_width=True)
with colC:
    st.markdown("### Top YTD")
    st.dataframe(build_top_table(df_strategy_full, "YTD", n=10, ascending=False), use_container_width=True)

st.divider()

# Strategy Table
st.subheader("Strategy-Level Summary")
df_display = df_strategy_view.copy()
for col in ["DTD", "MTD", "YTD"]:
    df_display[col] = df_display[col].apply(format_money)

st.dataframe(df_display[["strategy_name", "category", "DTD", "MTD", "YTD"]], use_container_width=True)

st.divider()

# Export
st.subheader("Export")
export_df = df_strategy_full.copy()
st.download_button(
    label="Download Strategy Summary CSV (All Strategies)",
    data=export_df.to_csv(index=False).encode("utf-8"),
    file_name=f"strategy_summary_{target_date.strftime('%Y%m%d')}.csv",
    mime="text/csv",
)

st.caption("TOTAL rows are automatically excluded. Category is auto-classified from the prefix before '_' in strategy_name.")
