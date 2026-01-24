# pages/6_Ledger_Tool.py
# Streamlit + SQLite ledger tool
# Features:
# - Global filters (date range, account, asset, bucket)
# - Quick Add:
#   1) Coupon Received
#   2) Reclass (Coupon -> AES 등)
#   3) AES Trade Done (USDT 기준, bucket=AES)
# - Undo (Reverse by group_id, append-only)
# - Export:
#   - Excel: Filters + Asset×Bucket Summary + Ledger_filtered + Pivot_like
#   - PDF:   Filters + Asset×Bucket Summary (Top 20)
# - Pagination for large datasets

import sqlite3
import uuid
from datetime import datetime, timezone
from io import BytesIO

import pandas as pd
import streamlit as st

# ================== Auth Check ==================
if "auth_ok" not in st.session_state or not st.session_state.auth_ok:
    st.warning("로그인이 필요합니다.")
    st.switch_page("app.py")
    st.stop()

# PDF (reportlab)
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

DB_PATH = "ledger.db"
ROWS_PER_PAGE = 50  # Pagination setting

# -----------------------------
# Helpers
# -----------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def make_gid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"

def normalize_iso(s: str) -> str:
    """
    Accepts:
      - "2026-01-24"
      - "2026-01-24T10:30:00"
      - "2026-01-24T10:30:00Z"
    Returns an ISO-ish string that sorts lexicographically.
    If only date is provided, treat as start-of-day UTC.
    """
    s = (s or "").strip()
    if not s:
        return ""
    if "T" not in s:
        return s + "T00:00:00Z"
    if s.endswith("Z"):
        return s
    return s + "Z"

# -----------------------------
# Pagination Helper
# -----------------------------
def paginate_dataframe(df: pd.DataFrame, page_key: str) -> pd.DataFrame:
    """
    Display pagination controls and return the sliced dataframe for current page.
    """
    if df.empty:
        return df

    total_rows = len(df)
    total_pages = (total_rows + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE

    # Initialize page state
    if page_key not in st.session_state:
        st.session_state[page_key] = 1

    current_page = st.session_state[page_key]

    # Ensure current page is valid
    if current_page > total_pages:
        current_page = total_pages
        st.session_state[page_key] = current_page
    if current_page < 1:
        current_page = 1
        st.session_state[page_key] = current_page

    # Pagination controls
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])

    with col1:
        if st.button("⏮ First", key=f"{page_key}_first", disabled=(current_page == 1)):
            st.session_state[page_key] = 1
            st.rerun()

    with col2:
        if st.button("◀ Prev", key=f"{page_key}_prev", disabled=(current_page == 1)):
            st.session_state[page_key] = current_page - 1
            st.rerun()

    with col3:
        start_row = (current_page - 1) * ROWS_PER_PAGE + 1
        end_row = min(current_page * ROWS_PER_PAGE, total_rows)
        st.markdown(f"**Page {current_page} / {total_pages}** ({start_row}-{end_row} of {total_rows} rows)")

    with col4:
        if st.button("Next ▶", key=f"{page_key}_next", disabled=(current_page == total_pages)):
            st.session_state[page_key] = current_page + 1
            st.rerun()

    with col5:
        if st.button("Last ⏭", key=f"{page_key}_last", disabled=(current_page == total_pages)):
            st.session_state[page_key] = total_pages
            st.rerun()

    # Slice dataframe
    start_idx = (current_page - 1) * ROWS_PER_PAGE
    end_idx = start_idx + ROWS_PER_PAGE

    return df.iloc[start_idx:end_idx]

# -----------------------------
# DB
# -----------------------------
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    with get_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS ledger_entries (
          id                INTEGER PRIMARY KEY AUTOINCREMENT,
          ts                TEXT NOT NULL,                 -- ISO8601 string (recommended Z)
          account           TEXT NOT NULL,
          asset             TEXT NOT NULL,
          bucket            TEXT NOT NULL,                 -- "Coupon", "AES", "Margin", "Coupon Swap", ...
          amount            REAL NOT NULL,                 -- signed (+/-)
          memo              TEXT DEFAULT NULL,
          ref               TEXT DEFAULT NULL,
          group_id          TEXT DEFAULT NULL,             -- reclass_id / trade_id / etc
          group_type        TEXT DEFAULT NULL,             -- "reclass", "aes_trade", ...
          leg               TEXT DEFAULT NULL,             -- optional: "credit/debit/pay/receive/fee"
          created_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
          created_by        TEXT DEFAULT NULL,
          is_reversal       INTEGER NOT NULL DEFAULT 0,
          reversed_group_id TEXT DEFAULT NULL
        );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ledger_ts      ON ledger_entries(ts);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ledger_acc     ON ledger_entries(account);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ledger_asset   ON ledger_entries(asset);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ledger_bucket  ON ledger_entries(bucket);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ledger_group   ON ledger_entries(group_id);")

def insert_entries(rows: list[dict]):
    """
    Insert multiple ledger rows atomically.
    Each row must include: ts, account, asset, bucket, amount
    Optional: memo, ref, group_id, group_type, leg, created_by, is_reversal, reversed_group_id
    """
    with get_conn() as conn:
        conn.execute("BEGIN;")
        try:
            conn.executemany("""
                INSERT INTO ledger_entries
                (ts, account, asset, bucket, amount, memo, ref, group_id, group_type, leg,
                 created_by, is_reversal, reversed_group_id)
                VALUES
                (:ts, :account, :asset, :bucket, :amount, :memo, :ref, :group_id, :group_type, :leg,
                 :created_by, :is_reversal, :reversed_group_id)
            """, rows)
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise

def fetch_distinct(col: str) -> list[str]:
    with get_conn() as conn:
        cur = conn.execute(f"SELECT DISTINCT {col} FROM ledger_entries WHERE {col} IS NOT NULL ORDER BY {col};")
        return [r[0] for r in cur.fetchall() if r[0] is not None]

def query_ledger(date_from=None, date_to=None, accounts=None, assets=None, buckets=None) -> pd.DataFrame:
    q = "SELECT * FROM ledger_entries WHERE 1=1"
    params = []

    if date_from:
        q += " AND ts >= ?"
        params.append(date_from)
    if date_to:
        q += " AND ts <= ?"
        params.append(date_to)

    def add_in_filter(field, values):
        nonlocal q, params
        if values:
            placeholders = ",".join(["?"] * len(values))
            q += f" AND {field} IN ({placeholders})"
            params.extend(values)

    add_in_filter("account", accounts)
    add_in_filter("asset", assets)
    add_in_filter("bucket", buckets)

    q += " ORDER BY ts ASC, id ASC"

    with get_conn() as conn:
        df = pd.read_sql_query(q, conn, params=params)
    return df

# -----------------------------
# Actions (Quick Add)
# -----------------------------
def add_coupon_received(ts, account, asset, amount, memo=None, created_by=None) -> str:
    gid = make_gid("cpn")
    rows = [{
        "ts": ts, "account": account, "asset": asset, "bucket": "Coupon",
        "amount": float(amount),
        "memo": memo, "ref": None,
        "group_id": gid, "group_type": "coupon_received", "leg": "credit",
        "created_by": created_by, "is_reversal": 0, "reversed_group_id": None
    }]
    insert_entries(rows)
    return gid

def add_reclass(ts, account, asset, from_bucket, to_bucket, amount, memo=None, created_by=None) -> str:
    gid = make_gid("rc")
    amt = float(amount)
    rows = [
        {"ts": ts, "account": account, "asset": asset, "bucket": from_bucket,
         "amount": -amt, "memo": memo, "ref": None, "group_id": gid, "group_type": "reclass", "leg": "debit",
         "created_by": created_by, "is_reversal": 0, "reversed_group_id": None},
        {"ts": ts, "account": account, "asset": asset, "bucket": to_bucket,
         "amount": +amt, "memo": memo, "ref": None, "group_id": gid, "group_type": "reclass", "leg": "credit",
         "created_by": created_by, "is_reversal": 0, "reversed_group_id": None},
    ]
    insert_entries(rows)
    return gid

def add_aes_trade(
    ts, account, side, traded_asset, qty, usdt_amount,
    fee_asset=None, fee_amount=None, memo=None, created_by=None
) -> str:
    """
    AES Trade Done:
      - counter fixed to USDT
      - all legs recorded under bucket="AES"
      - BUY:  USDT - , Asset +
      - SELL: Asset - , USDT +
    """
    gid = make_gid("aes")
    qty = float(qty)
    usdt_amount = float(usdt_amount)

    rows = []
    if side == "BUY":
        rows += [
            {"ts": ts, "account": account, "asset": "USDT", "bucket": "AES",
             "amount": -usdt_amount, "memo": memo, "ref": None, "group_id": gid, "group_type": "aes_trade", "leg": "pay",
             "created_by": created_by, "is_reversal": 0, "reversed_group_id": None},
            {"ts": ts, "account": account, "asset": traded_asset, "bucket": "AES",
             "amount": +qty, "memo": memo, "ref": None, "group_id": gid, "group_type": "aes_trade", "leg": "receive",
             "created_by": created_by, "is_reversal": 0, "reversed_group_id": None},
        ]
    else:  # SELL
        rows += [
            {"ts": ts, "account": account, "asset": traded_asset, "bucket": "AES",
             "amount": -qty, "memo": memo, "ref": None, "group_id": gid, "group_type": "aes_trade", "leg": "deliver",
             "created_by": created_by, "is_reversal": 0, "reversed_group_id": None},
            {"ts": ts, "account": account, "asset": "USDT", "bucket": "AES",
             "amount": +usdt_amount, "memo": memo, "ref": None, "group_id": gid, "group_type": "aes_trade", "leg": "collect",
             "created_by": created_by, "is_reversal": 0, "reversed_group_id": None},
        ]

    if fee_asset and fee_amount and float(fee_amount) != 0.0:
        rows.append({
            "ts": ts, "account": account, "asset": fee_asset, "bucket": "Fee",
            "amount": -abs(float(fee_amount)),
            "memo": memo, "ref": None,
            "group_id": gid, "group_type": "aes_trade_fee", "leg": "fee",
            "created_by": created_by, "is_reversal": 0, "reversed_group_id": None
        })

    insert_entries(rows)
    return gid

def reverse_group(group_id: str, created_by=None) -> str:
    """
    Append-only reversal:
      - fetch all rows in group_id
      - insert new rows with amount negated
      - tag is_reversal=1 and reversed_group_id = original group_id
    """
    with get_conn() as conn:
        df = pd.read_sql_query(
            "SELECT ts, account, asset, bucket, amount, memo, ref, group_id, group_type FROM ledger_entries WHERE group_id = ? ORDER BY id ASC",
            conn, params=[group_id]
        )
    if df.empty:
        raise ValueError(f"No rows found for group_id={group_id}")

    new_gid = f"rev_{uuid.uuid4().hex[:12]}"
    now_iso = utc_now_iso()
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "ts": now_iso,
            "account": r["account"],
            "asset": r["asset"],
            "bucket": r["bucket"],
            "amount": float(-r["amount"]),
            "memo": f"[REVERSAL of {group_id}] " + (r["memo"] if r["memo"] else ""),
            "ref": r["ref"],
            "group_id": new_gid,
            "group_type": "reversal",
            "leg": "reversal",
            "created_by": created_by,
            "is_reversal": 1,
            "reversed_group_id": group_id
        })
    insert_entries(rows)
    return new_gid

def import_csv(csv_df: pd.DataFrame, account: str, created_by=None) -> tuple[int, str]:
    """
    Import ledger entries from CSV.
    Expected columns: Date, Type, ccy, amount, remarks
    Maps to: ts, bucket, asset, amount, memo
    Returns: (count of imported rows, group_id)
    """
    gid = make_gid("csv")
    rows = []

    for _, r in csv_df.iterrows():
        # Parse date - handle various formats
        date_val = r.get("Date", "")
        if pd.isna(date_val) or str(date_val).strip() == "":
            continue

        # Convert date to ISO format
        try:
            if isinstance(date_val, str):
                # Try parsing common formats
                for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"]:
                    try:
                        parsed = datetime.strptime(date_val.strip(), fmt)
                        ts = parsed.strftime("%Y-%m-%dT00:00:00Z")
                        break
                    except ValueError:
                        continue
                else:
                    ts = normalize_iso(date_val)
            else:
                # Assume it's a datetime object
                ts = pd.Timestamp(date_val).strftime("%Y-%m-%dT00:00:00Z")
        except Exception:
            ts = utc_now_iso()

        bucket = str(r.get("Type", "")).strip() or "Unknown"
        asset = str(r.get("ccy", "")).strip() or "Unknown"

        try:
            amt_val = r.get("amount", 0)
            if pd.isna(amt_val) or str(amt_val).strip() == "":
                amount = 0.0
            else:
                amount = float(amt_val)
        except (ValueError, TypeError):
            amount = 0.0

        memo = str(r.get("remarks", "")).strip() if pd.notna(r.get("remarks")) else None

        rows.append({
            "ts": ts,
            "account": account,
            "asset": asset,
            "bucket": bucket,
            "amount": amount,
            "memo": memo,
            "ref": None,
            "group_id": gid,
            "group_type": "csv_import",
            "leg": "import",
            "created_by": created_by,
            "is_reversal": 0,
            "reversed_group_id": None
        })

    if rows:
        insert_entries(rows)

    return len(rows), gid

# -----------------------------
# Summary + Export
# -----------------------------
def compute_asset_bucket_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot style: index=asset, columns=bucket, values=sum(amount), plus Total
    """
    if df.empty:
        return pd.DataFrame()
    pv = (df.groupby(["asset", "bucket"])["amount"].sum()
            .reset_index()
            .pivot(index="asset", columns="bucket", values="amount")
            .fillna(0.0))
    pv["Total"] = pv.sum(axis=1)
    cols = ["Total"] + [c for c in pv.columns if c != "Total"]
    return pv[cols].sort_values("Total", ascending=False)

def df_to_excel_bytes(df_filtered: pd.DataFrame, summary: pd.DataFrame, filters: dict) -> bytes:
    """
    Excel 구성(요청 반영: KPI 제외):
      - Summary: Filters + Asset×Bucket Summary
      - Ledger_filtered: 필터 적용 원장
      - Pivot_like: account×asset×bucket 집계
    """
    output = BytesIO()

    # Pivot_like
    if df_filtered.empty:
        pivot_like = pd.DataFrame()
    else:
        pivot_like = (
            df_filtered.groupby(["account", "asset", "bucket"])["amount"]
            .sum().reset_index()
            .sort_values(["account", "asset", "bucket"])
        )

    filters_rows = [
        {"key": "date_from", "value": filters.get("date_from") or ""},
        {"key": "date_to", "value": filters.get("date_to") or ""},
        {"key": "accounts", "value": ", ".join(filters.get("accounts") or [])},
        {"key": "assets", "value": ", ".join(filters.get("assets") or [])},
        {"key": "buckets", "value": ", ".join(filters.get("buckets") or [])},
        {"key": "generated_at_utc", "value": utc_now_iso()},
    ]
    filters_df = pd.DataFrame(filters_rows)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        start_row = 0

        # Filters
        filters_df.to_excel(writer, sheet_name="Summary", index=False, startrow=start_row)
        start_row += len(filters_df) + 2

        # Asset × Bucket Summary
        if summary is None or summary.empty:
            pd.DataFrame({"note": ["No data for summary under current filters."]}).to_excel(
                writer, sheet_name="Summary", index=False, startrow=start_row
            )
        else:
            summary_out = summary.reset_index().rename(columns={"index": "asset"})
            summary_out.to_excel(writer, sheet_name="Summary", index=False, startrow=start_row)

        # Ledger_filtered
        df_filtered.to_excel(writer, sheet_name="Ledger_filtered", index=False)

        # Pivot_like
        pivot_like.to_excel(writer, sheet_name="Pivot_like", index=False)

    return output.getvalue()

def df_to_pdf_bytes(df_filtered: pd.DataFrame, summary: pd.DataFrame, filters: dict) -> bytes:
    """
    PDF는 요약 리포트 (KPI 제외):
      - Filters
      - Asset × Bucket Summary (Top 20 by Total)
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Ledger Report (Filtered)", styles["Title"]))
    story.append(Spacer(1, 12))

    # Filters
    story.append(Paragraph("Filters", styles["Heading2"]))
    filter_lines = [
        f"Date from: {filters.get('date_from') or ''}",
        f"Date to: {filters.get('date_to') or ''}",
        f"Accounts: {', '.join(filters.get('accounts') or [])}",
        f"Assets: {', '.join(filters.get('assets') or [])}",
        f"Buckets: {', '.join(filters.get('buckets') or [])}",
        f"Generated (UTC): {utc_now_iso()}",
    ]
    for line in filter_lines:
        story.append(Paragraph(line, styles["BodyText"]))
    story.append(Spacer(1, 12))

    # Summary table
    story.append(Paragraph("Asset × Bucket Summary (Top 20 by Total)", styles["Heading2"]))
    if summary is None or summary.empty:
        story.append(Paragraph("No data under current filters.", styles["BodyText"]))
    else:
        summary_out = summary.reset_index().rename(columns={"index": "asset"})
        if "Total" in summary_out.columns:
            summary_out = summary_out.sort_values("Total", ascending=False).head(20)
        else:
            summary_out = summary_out.head(20)

        cols = list(summary_out.columns)
        preferred = ["asset", "Total", "AES", "Coupon", "Margin", "Coupon Swap", "Fee"]
        cols = [c for c in preferred if c in cols] + [c for c in cols if c not in preferred]
        cols = cols[:7]  # keep PDF readable

        pdf_df = summary_out[cols].copy()
        for c in cols:
            if c != "asset":
                pdf_df[c] = pdf_df[c].apply(lambda x: f"{float(x):,.6f}")

        data = [cols] + pdf_df.values.tolist()
        tbl = Table(data, hAlign="LEFT")
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("PADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(tbl)

    doc.build(story)
    return buffer.getvalue()

# -----------------------------
# Streamlit UI
# -----------------------------
init_db()

st.title("Ledger Tool (SQLite)")

# Sidebar nav
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Ledger", "Reclass", "Export"], index=0)

# Sidebar filters
st.sidebar.header("Global Filters")
date_from_in = st.sidebar.text_input("Date from (YYYY-MM-DD or ISO)", value="")
date_to_in = st.sidebar.text_input("Date to (YYYY-MM-DD or ISO)", value="")

date_from = normalize_iso(date_from_in) if date_from_in.strip() else None
date_to = normalize_iso(date_to_in) if date_to_in.strip() else None

accounts_all = fetch_distinct("account")
assets_all = fetch_distinct("asset")
buckets_all = fetch_distinct("bucket")

# Sensible fallbacks if DB is empty
if not accounts_all:
    accounts_all = ["Deribit master"]
if not assets_all:
    assets_all = ["USDT", "BTC", "ETH", "KAIA"]
if not buckets_all:
    buckets_all = ["Coupon", "AES", "Margin", "Coupon Swap", "Fee"]

accounts = st.sidebar.multiselect("Account", options=accounts_all, default=accounts_all[:1])
assets = st.sidebar.multiselect("Asset", options=assets_all, default=[])
buckets = st.sidebar.multiselect("Bucket/Type", options=buckets_all, default=[])

# Pagination settings in sidebar
st.sidebar.header("Display Settings")
rows_per_page = st.sidebar.selectbox("Rows per page", options=[25, 50, 100, 200], index=1)
ROWS_PER_PAGE = rows_per_page

df = query_ledger(
    date_from=date_from,
    date_to=date_to,
    accounts=accounts or None,
    assets=assets or None,
    buckets=buckets or None
)

# Shared for exports
filters_for_export = {
    "date_from": date_from_in.strip() or None,
    "date_to": date_to_in.strip() or None,
    "accounts": accounts or [],
    "assets": assets or [],
    "buckets": buckets or [],
}

# -----------------------------
# Pages
# -----------------------------
if page == "Dashboard":
    st.subheader("Dashboard (filtered)")

    summary = compute_asset_bucket_summary(df)
    st.markdown("### Asset × Bucket Summary")
    st.dataframe(summary, use_container_width=True)

    st.markdown("### Filtered Ledger (preview)")
    st.caption(f"Total: {len(df)} rows")
    df_page = paginate_dataframe(df, "dashboard_page")
    st.dataframe(df_page, use_container_width=True)

elif page == "Ledger":
    st.subheader("Ledger (filtered)")

    # Quick Add buttons
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("+ Coupon Received", use_container_width=True):
            st.session_state["open_quick"] = "coupon"
    with c2:
        if st.button("+ Reclass", use_container_width=True):
            st.session_state["open_quick"] = "reclass"
    with c3:
        if st.button("+ AES Trade Done", use_container_width=True):
            st.session_state["open_quick"] = "aes"
    with c4:
        if st.button("📤 CSV Upload", use_container_width=True):
            st.session_state["open_quick"] = "csv_upload"

    quick = st.session_state.get("open_quick")

    if quick == "coupon":
        st.markdown("#### Quick Add: Coupon Received")
        with st.form("coupon_form", clear_on_submit=True):
            ts = st.text_input("ts (ISO8601)", value=utc_now_iso())
            account = st.selectbox("Account", options=accounts_all, index=0)
            # default USDT if exists
            usdt_idx = assets_all.index("USDT") if "USDT" in assets_all else 0
            asset = st.selectbox("Asset", options=assets_all, index=usdt_idx)
            amount = st.number_input("Amount", min_value=0.0, value=0.0)
            memo = st.text_input("Memo", value="")
            submitted = st.form_submit_button("Post")
        if submitted:
            gid = add_coupon_received(ts, account, asset, amount, memo=memo or None)
            st.success(f"Posted. group_id={gid}")
            st.session_state["open_quick"] = None
            st.rerun()

    elif quick == "reclass":
        st.markdown("#### Quick Add: Reclass (Bucket Conversion)")
        with st.form("reclass_form", clear_on_submit=True):
            ts = st.text_input("ts (ISO8601)", value=utc_now_iso())
            account = st.selectbox("Account", options=accounts_all, index=0)
            usdt_idx = assets_all.index("USDT") if "USDT" in assets_all else 0
            asset = st.selectbox("Asset", options=assets_all, index=usdt_idx)

            # defaults: Coupon -> AES if present
            from_default = buckets_all.index("Coupon") if "Coupon" in buckets_all else 0
            to_default = buckets_all.index("AES") if "AES" in buckets_all else 0

            from_bucket = st.selectbox("From Bucket", options=buckets_all, index=from_default)
            to_bucket = st.selectbox("To Bucket", options=buckets_all, index=to_default)
            amount = st.number_input("Amount", min_value=0.0, value=0.0)
            memo = st.text_input("Memo", value="")
            st.caption("Posts two ledger rows: From bucket (-) and To bucket (+), same asset.")
            submitted = st.form_submit_button("Post")
        if submitted:
            gid = add_reclass(ts, account, asset, from_bucket, to_bucket, amount, memo=memo or None)
            st.success(f"Posted. group_id={gid}")
            st.session_state["open_quick"] = None
            st.rerun()

    elif quick == "aes":
        st.markdown("#### Quick Add: AES Trade Done (USDT counter, bucket=AES)")
        with st.form("aes_form", clear_on_submit=True):
            ts = st.text_input("ts (ISO8601)", value=utc_now_iso())
            account = st.selectbox("Account", options=accounts_all, index=0)
            side = st.radio("Side", options=["BUY", "SELL"], horizontal=True)

            traded_options = [a for a in assets_all if a != "USDT"]
            if not traded_options:
                traded_options = ["KAIA"]
            traded_asset = st.selectbox("Traded Asset", options=traded_options, index=0)

            qty = st.number_input("Quantity (asset units)", min_value=0.0, value=0.0)
            usdt_amount = st.number_input("USDT Amount (total)", min_value=0.0, value=0.0)

            fee_on = st.checkbox("Include fee", value=False)
            fee_asset, fee_amount = None, None
            if fee_on:
                fee_asset = st.selectbox("Fee Asset", options=assets_all, index=(assets_all.index("USDT") if "USDT" in assets_all else 0))
                fee_amount = st.number_input("Fee Amount", min_value=0.0, value=0.0)

            memo = st.text_input("Memo", value="")
            st.caption("Posts two AES bucket rows (and optional Fee row). BUY: USDT(-), Asset(+). SELL: Asset(-), USDT(+).")
            submitted = st.form_submit_button("Post")
        if submitted:
            gid = add_aes_trade(
                ts, account, side, traded_asset, qty, usdt_amount,
                fee_asset=fee_asset if fee_on else None,
                fee_amount=fee_amount if fee_on else None,
                memo=memo or None
            )
            st.success(f"Posted. group_id={gid}")
            st.session_state["open_quick"] = None
            st.rerun()

    elif quick == "csv_upload":
        st.markdown("#### CSV Upload (Bulk Import)")
        st.caption("CSV 컬럼: Date, Type, ccy, amount, remarks")

        uploaded_file = st.file_uploader("CSV 파일 선택", type=["csv"], key="csv_uploader")
        account_for_import = st.selectbox("Import할 Account 선택", options=accounts_all, index=0, key="csv_account")

        if uploaded_file is not None:
            try:
                csv_df = pd.read_csv(uploaded_file)
                st.markdown("**Preview (first 10 rows):**")
                st.dataframe(csv_df.head(10), use_container_width=True)
                st.caption(f"Total rows in CSV: {len(csv_df)}")

                # Check required columns
                required_cols = ["Date", "Type", "ccy", "amount"]
                missing_cols = [c for c in required_cols if c not in csv_df.columns]

                if missing_cols:
                    st.error(f"필수 컬럼 누락: {missing_cols}")
                else:
                    if st.button("Import CSV", type="primary", use_container_width=True):
                        try:
                            count, gid = import_csv(csv_df, account_for_import)
                            st.success(f"Imported {count} rows. group_id={gid}")
                            st.session_state["open_quick"] = None
                            st.rerun()
                        except Exception as e:
                            st.error(f"Import 실패: {e}")
            except Exception as e:
                st.error(f"CSV 읽기 실패: {e}")

        if st.button("Cancel", use_container_width=True):
            st.session_state["open_quick"] = None
            st.rerun()

    st.markdown("### Filtered ledger table")
    st.caption(f"Total: {len(df)} rows")
    df_page = paginate_dataframe(df, "ledger_page")
    st.dataframe(df_page, use_container_width=True)

    st.markdown("### Undo (Reverse by group_id)")
    gid_in = st.text_input("group_id to reverse", value="")
    if st.button("Reverse group_id") and gid_in.strip():
        try:
            new_gid = reverse_group(gid_in.strip())
            st.success(f"Reversed. new_group_id={new_gid}")
            st.rerun()
        except Exception as e:
            st.error(str(e))

elif page == "Reclass":
    st.subheader("Reclass (Bucket Conversion)")
    st.info("현재는 Ledger 탭의 '+ Reclass' Quick Add와 동일 기능입니다. 필요하면 Reclass 탭을 별도 UX(Preview/History)로 확장할 수 있어요.")

elif page == "Export":
    st.subheader("Export (filtered)")

    summary = compute_asset_bucket_summary(df)

    st.markdown("### Options")
    base_name = st.text_input("File base name", value="ledger_report")
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    xlsx_name = f"{base_name}_{stamp}.xlsx"
    pdf_name = f"{base_name}_{stamp}.pdf"

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Excel (Filters + Summary + Ledger + Pivot)")
        if st.button("Generate Excel", use_container_width=True):
            try:
                xlsx_bytes = df_to_excel_bytes(df, summary, filters_for_export)
                st.session_state["xlsx_bytes"] = xlsx_bytes
                st.success("Excel generated.")
            except Exception as e:
                st.error(str(e))

        if "xlsx_bytes" in st.session_state:
            st.download_button(
                "Download Excel",
                data=st.session_state["xlsx_bytes"],
                file_name=xlsx_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    with col2:
        st.markdown("#### PDF (Filters + Summary only)")
        if st.button("Generate PDF", use_container_width=True):
            try:
                pdf_bytes = df_to_pdf_bytes(df, summary, filters_for_export)
                st.session_state["pdf_bytes"] = pdf_bytes
                st.success("PDF generated.")
            except Exception as e:
                st.error(str(e))

        if "pdf_bytes" in st.session_state:
            st.download_button(
                "Download PDF",
                data=st.session_state["pdf_bytes"],
                file_name=pdf_name,
                mime="application/pdf",
                use_container_width=True
            )

    st.markdown("### Preview (filtered ledger)")
    st.caption(f"Total: {len(df)} rows")
    df_page = paginate_dataframe(df, "export_page")
    st.dataframe(df_page, use_container_width=True)
