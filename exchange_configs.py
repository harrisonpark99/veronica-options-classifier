# -*- coding: utf-8 -*-
"""
Exchange Configuration Module
Extensible structure for supporting multiple exchange CSV formats.

To add a new exchange:
1. Create a new config dict following the ExchangeConfig structure
2. Add it to EXCHANGE_REGISTRY
3. (Optional) Add detection patterns to help auto-detect the exchange
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
import pandas as pd


@dataclass
class ExchangeConfig:
    """Configuration for a specific exchange's CSV format."""

    # Exchange identifier
    name: str
    display_name: str

    # Required columns in the CSV (will fail if missing)
    required_columns: List[str]

    # Optional columns (may or may not exist)
    optional_columns: List[str] = field(default_factory=list)

    # Column mapping: standard_name -> list of possible CSV column names
    # Standard names: symbol, trade_id, order_id, price, qty, quote_qty,
    #                 commission, commission_asset, time, is_buyer, is_maker
    column_mapping: Dict[str, List[str]] = field(default_factory=dict)

    # Header mapping for Excel template: df_column -> possible excel header names
    header_mapping: Dict[str, List[str]] = field(default_factory=dict)

    # Columns that should be numeric
    numeric_columns: List[str] = field(default_factory=list)

    # How to detect if a CSV is from this exchange (column patterns)
    detection_columns: List[str] = field(default_factory=list)

    # Custom side detection function (returns "BUY", "SELL", "MIXED", or "UNKNOWN")
    # If None, uses default isBuyer-based detection
    side_detector: Optional[Callable[[pd.DataFrame], str]] = None

    # Custom totals computation function
    # If None, uses default qty/quoteQty sum
    totals_calculator: Optional[Callable[[pd.DataFrame], tuple]] = None


def default_side_detector(df: pd.DataFrame, buyer_column: str = "isBuyer") -> str:
    """Default side detection using isBuyer column."""
    if buyer_column not in df.columns:
        return "UNKNOWN"
    vals = df[buyer_column].dropna().astype(bool).unique()
    if len(vals) == 1:
        return "BUY" if vals[0] else "SELL"
    return "MIXED"


def default_totals_calculator(df: pd.DataFrame, qty_col: str = "qty", quote_qty_col: str = "quoteQty") -> tuple:
    """Default totals calculation."""
    filled_amount = float(pd.to_numeric(df[qty_col], errors="coerce").fillna(0).sum())
    filled_value = float(pd.to_numeric(df[quote_qty_col], errors="coerce").fillna(0).sum())
    avg_price = (filled_value / filled_amount) if filled_amount != 0 else 0.0
    return filled_amount, filled_value, avg_price


# =============================================================================
# BINANCE Configuration
# =============================================================================
BINANCE_CONFIG = ExchangeConfig(
    name="binance",
    display_name="Binance",
    required_columns=[
        "symbol", "id", "orderId", "orderListId", "price", "qty", "quoteQty",
        "commission", "commissionAsset", "time", "isBuyer", "isMaker", "isBestMatch",
    ],
    optional_columns=[
        "fill_time(UTC+8)", "fill_value", "sum_fill_amount", "sum_fill_value", "ave_fill_price"
    ],
    column_mapping={
        "symbol": ["symbol"],
        "trade_id": ["id"],
        "order_id": ["orderId"],
        "price": ["price"],
        "qty": ["qty"],
        "quote_qty": ["quoteQty"],
        "commission": ["commission"],
        "commission_asset": ["commissionAsset"],
        "time": ["time"],
        "is_buyer": ["isBuyer"],
        "is_maker": ["isMaker"],
    },
    header_mapping={
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
    },
    numeric_columns=["price", "qty", "quoteQty", "commission"],
    detection_columns=["isBuyer", "isMaker", "isBestMatch", "quoteQty"],
)


# =============================================================================
# OKX Configuration (Example - adjust based on actual OKX export format)
# =============================================================================
def okx_side_detector(df: pd.DataFrame) -> str:
    """OKX uses 'side' column with 'buy'/'sell' values."""
    if "side" not in df.columns:
        return "UNKNOWN"
    vals = df["side"].dropna().str.lower().unique()
    if len(vals) == 1:
        return "BUY" if vals[0] == "buy" else "SELL"
    return "MIXED"


def okx_totals_calculator(df: pd.DataFrame) -> tuple:
    """OKX totals calculation - adjust column names as needed."""
    qty_col = "sz" if "sz" in df.columns else "fillSz"
    value_col = "fillPx" if "fillPx" in df.columns else "px"

    filled_amount = float(pd.to_numeric(df[qty_col], errors="coerce").fillna(0).sum())

    # For OKX, quote value might need to be calculated
    if "fillNotionalUsd" in df.columns:
        filled_value = float(pd.to_numeric(df["fillNotionalUsd"], errors="coerce").fillna(0).sum())
    else:
        prices = pd.to_numeric(df[value_col], errors="coerce").fillna(0)
        qtys = pd.to_numeric(df[qty_col], errors="coerce").fillna(0)
        filled_value = float((prices * qtys).sum())

    avg_price = (filled_value / filled_amount) if filled_amount != 0 else 0.0
    return filled_amount, filled_value, avg_price


OKX_CONFIG = ExchangeConfig(
    name="okx",
    display_name="OKX",
    required_columns=[
        "instId", "ordId", "tradeId", "fillPx", "fillSz", "side", "fillTime"
    ],
    optional_columns=[
        "fee", "feeCcy", "fillNotionalUsd", "execType", "posSide"
    ],
    column_mapping={
        "symbol": ["instId"],
        "trade_id": ["tradeId"],
        "order_id": ["ordId"],
        "price": ["fillPx", "px"],
        "qty": ["fillSz", "sz"],
        "quote_qty": ["fillNotionalUsd"],
        "commission": ["fee"],
        "commission_asset": ["feeCcy"],
        "time": ["fillTime", "ts"],
        "is_buyer": ["side"],  # needs custom processing
    },
    header_mapping={
        "instId": ["instid", "inst id", "symbol", "instrument"],
        "ordId": ["ordid", "ord id", "order id", "orderid"],
        "tradeId": ["tradeid", "trade id", "id"],
        "fillPx": ["fillpx", "fill px", "price", "fill price"],
        "fillSz": ["fillsz", "fill sz", "qty", "quantity", "size"],
        "side": ["side", "direction"],
        "fillTime": ["filltime", "fill time", "time", "timestamp"],
        "fee": ["fee", "commission"],
        "feeCcy": ["feeccy", "fee ccy", "fee currency", "commission asset"],
    },
    numeric_columns=["fillPx", "fillSz", "fee", "fillNotionalUsd"],
    detection_columns=["instId", "ordId", "fillPx", "fillSz"],
    side_detector=okx_side_detector,
    totals_calculator=okx_totals_calculator,
)


# =============================================================================
# BYBIT Configuration (Example - adjust based on actual Bybit export format)
# =============================================================================
def bybit_side_detector(df: pd.DataFrame) -> str:
    """Bybit uses 'Side' column with 'Buy'/'Sell' values."""
    side_col = None
    for col in ["Side", "side", "SIDE"]:
        if col in df.columns:
            side_col = col
            break
    if side_col is None:
        return "UNKNOWN"
    vals = df[side_col].dropna().str.lower().unique()
    if len(vals) == 1:
        return "BUY" if vals[0] == "buy" else "SELL"
    return "MIXED"


BYBIT_CONFIG = ExchangeConfig(
    name="bybit",
    display_name="Bybit",
    required_columns=[
        "Symbol", "OrderId", "TradeId", "Price", "Qty", "Side", "TradeTime"
    ],
    optional_columns=[
        "Fee", "FeeCurrency", "ExecValue", "OrderType"
    ],
    column_mapping={
        "symbol": ["Symbol"],
        "trade_id": ["TradeId"],
        "order_id": ["OrderId"],
        "price": ["Price"],
        "qty": ["Qty"],
        "quote_qty": ["ExecValue"],
        "commission": ["Fee"],
        "commission_asset": ["FeeCurrency"],
        "time": ["TradeTime"],
        "is_buyer": ["Side"],
    },
    header_mapping={
        "Symbol": ["symbol"],
        "OrderId": ["orderid", "order id"],
        "TradeId": ["tradeid", "trade id", "id"],
        "Price": ["price"],
        "Qty": ["qty", "quantity"],
        "Side": ["side", "direction"],
        "TradeTime": ["tradetime", "trade time", "time", "timestamp"],
        "Fee": ["fee", "commission"],
        "FeeCurrency": ["feecurrency", "fee currency", "commission asset"],
        "ExecValue": ["execvalue", "exec value", "value", "quote qty"],
    },
    numeric_columns=["Price", "Qty", "Fee", "ExecValue"],
    detection_columns=["Symbol", "OrderId", "TradeId", "ExecValue"],
    side_detector=bybit_side_detector,
)


# =============================================================================
# Exchange Registry
# =============================================================================
EXCHANGE_REGISTRY: Dict[str, ExchangeConfig] = {
    "binance": BINANCE_CONFIG,
    "okx": OKX_CONFIG,
    "bybit": BYBIT_CONFIG,
}


def detect_exchange(df: pd.DataFrame) -> Optional[str]:
    """
    Auto-detect which exchange the CSV is from based on column patterns.
    Returns exchange name or None if unknown.
    """
    df_cols = set(c.lower() for c in df.columns)

    best_match = None
    best_score = 0

    for name, config in EXCHANGE_REGISTRY.items():
        # Check how many detection columns match
        detection_cols = [c.lower() for c in config.detection_columns]
        score = sum(1 for c in detection_cols if c in df_cols)

        # Also check required columns
        required_cols = [c.lower() for c in config.required_columns]
        score += sum(0.5 for c in required_cols if c in df_cols)

        if score > best_score:
            best_score = score
            best_match = name

    # Only return if we have reasonable confidence
    return best_match if best_score >= 3 else None


def get_exchange_config(exchange_name: str) -> Optional[ExchangeConfig]:
    """Get configuration for a specific exchange."""
    return EXCHANGE_REGISTRY.get(exchange_name.lower())


def list_supported_exchanges() -> List[str]:
    """List all supported exchange names."""
    return list(EXCHANGE_REGISTRY.keys())


def get_exchange_display_names() -> Dict[str, str]:
    """Get display names for all exchanges."""
    return {name: config.display_name for name, config in EXCHANGE_REGISTRY.items()}
